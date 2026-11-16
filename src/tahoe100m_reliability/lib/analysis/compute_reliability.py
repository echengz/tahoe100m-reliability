"""
Functions to support computing reliability metrics.

Most of this is derived from the original framework.
"""
import logging
import numpy as np
import os
import pandas as pd
import psutil
import time

from typing import Any
from sklearn.metrics import pairwise_distances


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def log_memory_usage(msg: str = None):
    # Get the ID of the current process
    process = psutil.Process(os.getpid())
    
    # Get memory usage in bytes, then convert to Megabytes (MB)
    mem_bytes = process.memory_info().rss
    mem_gb = process.memory_info().rss / (1024 ** 3)
    if msg:
        log.info(f"[PID {process.pid}] {msg}: {mem_gb:.4f} GB")
    else:
        log.info(f"[PID {process.pid}] CURRENT MEMORY USAGE: {mem_gb:.4f} GB")


def parse_condition(condition: str) -> tuple[str, str, str]:
    """Parse condition 'cell_line|drug|dosage' into the 
    context and perturbation.
    """
    parts = condition.split('|')
    return parts[0], '|'.join(parts[1:])


def cp10k_normalization(counts):
    library_size = counts.sum(axis=1, keepdims=True)
    scale = np.divide(
        10_000.0,
        library_size,
        out=np.zeros_like(library_size, dtype=float),
        where=library_size > 0,
    )
    return np.log1p(counts * scale)


def batch_pearsonr(X, Y):
    """
    Row-wise Pearson correlation between corresponding rows of X and Y.

    Parameters
    ----------
    X, Y : ndarray, shape (R, G)

    Returns
    -------
    r : ndarray, shape (R,)
    """
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    num = (X * Y).sum(axis=1)
    denom = np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    denom[denom == 0] = 1.0
    result = np.clip(num / denom, -1.0, 1.0)
    return result


def spearman_brown(r):
    """
    Both Spearman-Brown variants are stored: SB(mean_r), the classical
    estimator of E[r], and SB(median_r), robust to outlier splits.
    The reported reliability is SB(median_r). Per-repeat SB is also
    kept to characterise the full distribution.

    Spearman-Brown prophecy with rho clipped to [0, 1].
    Negative values indicate no reliable signal (field convention:
    reliability = signal_var / total_var ∈ [0, 1] by definition).
    """
    d = 1.0 + r
    rho = np.where(np.abs(d) > 1e-10, 2.0 * r / d, 0.0)
    return np.maximum(rho, 0.0)


def compute_split_half_reliability(
    condition,
    X_stim,
    X_ctrl,
    ctrl_expr_idx,
    n_stim,
    n_half_max,
    n_half_grid_feasible,
    n_repeats,
    task_seed,
) -> list[dict[str, Any]]:
    """
    Compute pearson split-half reliability for one (context, perturbation) condition.
    """
    context, perturbation = parse_condition(condition)
    n_ctrl = X_ctrl.shape[0]
    n_genes = X_ctrl.shape[1]

    rng = np.random.default_rng(task_seed)
    rows = []

    try:
        # expr_idx is precomputed per-context (top-k genes by mean ctrl expression)
        # — no per-condition gene selection needed

        # ---- Collect all n values to compute ----
        # Primary: max split
        # Secondary: fixed values that are feasible
        all_n_configs = []

        # Primary (max) mode
        all_n_configs.append((n_half_max, "max"))

        # Secondary (fixed) modes
        for n_fixed in n_half_grid_feasible:
            if n_fixed != n_half_max:  # avoid duplicate if max happens to equal a fixed value
                all_n_configs.append((n_fixed, "fixed"))
            else:
                # Still compute it, but tag as both
                all_n_configs.append((n_fixed, "fixed"))  # redundant but keeping in for now

        # ---- Compute reliability for each n ----
        for n, n_mode in all_n_configs:
            if n < 4:  # minimum for meaningful correlation
                continue

            # Pre-allocate
            deltas_1 = np.empty((n_repeats, n_genes), dtype=np.float32)
            deltas_2 = np.empty((n_repeats, n_genes), dtype=np.float32)

            for r in range(n_repeats):
                perm_stim = rng.choice(n_stim, size=2 * n, replace=False)
                half_A = perm_stim[:n]
                half_B = perm_stim[n:]

                perm_ctrl = rng.choice(n_ctrl, size=2 * n, replace=False)
                half_C = perm_ctrl[:n]
                half_D = perm_ctrl[n:]

                deltas_1[r] = X_stim[half_A].mean(axis=0) - X_ctrl[half_C].mean(axis=0)
                deltas_2[r] = X_stim[half_B].mean(axis=0) - X_ctrl[half_D].mean(axis=0)

            # ---- Correlations ----
            r_all = batch_pearsonr(deltas_1, deltas_2)
            r_expr = batch_pearsonr(deltas_1[:, ctrl_expr_idx], deltas_2[:, ctrl_expr_idx])
            expr_k_label = f"top{len(ctrl_expr_idx)}_expr"
            del deltas_1, deltas_2

            # ---- Store results for both gene modes ----
            for gene_mode, r_values in [("all_genes", r_all),
                                        (expr_k_label, r_expr)]:
                median_r = float(np.median(r_values))
                mean_r = float(np.mean(r_values))

                sb_from_mean = float(spearman_brown(mean_r))
                sb_from_median = float(spearman_brown(median_r))

                # Per-repeat SB for distribution diagnostics
                sb_per_repeat = spearman_brown(r_values)
                sb_mean_of_per_repeat = float(np.mean(sb_per_repeat))
                sb_median_of_per_repeat = float(np.median(sb_per_repeat))

                # Skewness of raw r distribution (diagnostic)
                r_std = float(np.std(r_values))
                if r_std > 1e-10:
                    skewness = float(np.mean(((r_values - mean_r) / r_std) ** 3))
                else:
                    skewness = 0.0

                rows.append({
                    "condition": condition,
                    "context": context,
                    "perturbation": perturbation,
                    "n_stim_cells": n_stim,
                    "n_half": n,
                    "n_mode": n_mode,       # "max" or "fixed"
                    "gene_mode": gene_mode,  # "all_genes" or "topK_expr"
                    "n_repeats": int(n_repeats),
                    # Raw split-half r distribution
                    "mean_r": mean_r,
                    "std_r": r_std,
                    "median_r": median_r,
                    "q25_r": float(np.percentile(r_values, 25)),
                    "q75_r": float(np.percentile(r_values, 75)),
                    "min_r": float(np.min(r_values)),
                    "max_r": float(np.max(r_values)),
                    "skewness_r": skewness,
                    # Spearman-Brown — multiple approaches stored
                    "sb_from_mean": sb_from_mean,         # SB(mean(r_i))
                    "sb_from_median": sb_from_median,     # SB(median(r_i))
                    "sb_mean_per_repeat": sb_mean_of_per_repeat,   # mean(SB(r_i))
                    "sb_median_per_repeat": sb_median_of_per_repeat, # median(SB(r_i))
                    # Ceilings from each approach
                    "ceiling_from_mean": float(np.sqrt(max(0.0, sb_from_mean))),
                    "ceiling_from_median": float(np.sqrt(max(0.0, sb_from_median))),
                })

            del r_all, r_expr

    except Exception as e:
        log.exception(f"[{context}] compute_split_half_reliability failed")
        rows.append({
            "condition": condition,
            "context": context,
            "perturbation": perturbation,
            "n_stim_cells": n_stim,
            "n_half": -1,
            "n_mode": "ERROR",
            "gene_mode": "ERROR",
            "n_repeats": 0,
            "mean_r": np.nan, "std_r": np.nan, "median_r": np.nan,
            "q25_r": np.nan, "q75_r": np.nan,
            "min_r": np.nan, "max_r": np.nan,
            "skewness_r": np.nan,
            "sb_from_mean": np.nan, "sb_from_median": np.nan,
            "sb_mean_per_repeat": np.nan, "sb_median_per_repeat": np.nan,
            "ceiling_from_mean": np.nan, "ceiling_from_median": np.nan,
            "error": str(e),
        })

    return rows
