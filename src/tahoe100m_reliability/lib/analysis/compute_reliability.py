"""
Functions to support computing reliability metrics.
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


# ============================================================
# PEARSON SPLIT-WISE RELIABILITY
# ============================================================

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
    use_old_sampling: bool = False  # for regression testing
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
                # for regression testing, want to use the old sampling method
                if use_old_sampling:
                    perm_stim = rng.permutation(n_stim)
                    half_A = perm_stim[:n]
                    half_B = perm_stim[n:2 * n]

                    perm_ctrl = rng.permutation(n_ctrl)
                    half_C = perm_ctrl[:n]
                    half_D = perm_ctrl[n:2 * n]
                else:
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


def summarize_split_half_results(
    cluster, all_rows, expr_k, output_dir, dataset_name: str = 'tahoe100m'
) -> None:
    """Summarize the pearson split half reliability results."""

    # ---- Build DataFrames ----
    df_long = pd.DataFrame(all_rows)
    del all_rows

    df_long.insert(0, "cluster", cluster)

    # Check errors
    errors = df_long[df_long["n_mode"] == "ERROR"]
    if len(errors) > 0:
        log.error(f"\n  ⚠ {len(errors)} condition(s) had errors:")
        for _, row in errors.head(5).iterrows():
            log.error(f"    {row['condition']}: {row.get('error', 'unknown')}")

    # ---- Summary per condition ----
    df_valid = df_long[df_long["n_mode"] != "ERROR"].copy()
    summary_rows = []

    for cond, cdata in df_valid.groupby("condition"):
        n_stim = cdata["n_stim_cells"].iloc[0]
        ctx = cdata["context"].iloc[0]
        pert = cdata["perturbation"].iloc[0]

        row = {
            "cluster": cluster,
            "condition": cond,
            "context": ctx,
            "perturbation": pert,
            "n_stim_cells": n_stim,
        }

        # Primary (max) result
        max_data = cdata[cdata["n_mode"] == "max"]
        for gene_mode in max_data["gene_mode"].unique():
            gm_data = max_data[max_data["gene_mode"] == gene_mode]
            if len(gm_data) == 0:
                continue
            gm_row = gm_data.iloc[0]
            suffix = f"_{gene_mode}"
            row[f"n_half_max{suffix}"] = int(gm_row["n_half"])
            row[f"median_r_max{suffix}"] = gm_row["median_r"]
            row[f"mean_r_max{suffix}"] = gm_row["mean_r"]
            row[f"std_r_max{suffix}"] = gm_row["std_r"]
            row[f"skewness_r_max{suffix}"] = gm_row["skewness_r"]
            # All 4 SB approaches
            row[f"sb_from_mean{suffix}"] = gm_row["sb_from_mean"]
            row[f"sb_from_median{suffix}"] = gm_row["sb_from_median"]
            row[f"sb_mean_per_repeat{suffix}"] = gm_row["sb_mean_per_repeat"]
            row[f"sb_median_per_repeat{suffix}"] = gm_row["sb_median_per_repeat"]
            row[f"ceiling_from_mean{suffix}"] = gm_row["ceiling_from_mean"]
            row[f"ceiling_from_median{suffix}"] = gm_row["ceiling_from_median"]

        # Fixed curve: smallest n where median_r >= 0.5 (n*)
        fixed_data = cdata[cdata["n_mode"] == "fixed"]
        for gene_mode in fixed_data["gene_mode"].unique():
            gm_data = (fixed_data[fixed_data["gene_mode"] == gene_mode]
                       .sort_values("n_half"))
            if len(gm_data) == 0:
                continue
            suffix = f"_{gene_mode}"
            reliable_rows = gm_data[gm_data["median_r"] >= 0.5]
            if len(reliable_rows) > 0:
                row[f"n_star{suffix}"] = int(reliable_rows.iloc[0]["n_half"])
            else:
                row[f"n_star{suffix}"] = np.nan

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    del df_valid, summary_rows

    # ---- Print summary stats ----
    log.info(f"\n  ── Summary for cluster {cluster} ──")
    # log.info(f"  Total conditions found: {total_conditions}")
    log.info(f"  Conditions processed: {len(df_summary)}")
    # log.info(f"  Skipped (few ctrl): {total_skipped_ctrl}")
    # log.info(f"  Skipped (few stim): {total_skipped_stim}")

    for gene_mode in ["all_genes", f"top{expr_k}_expr"]:
        col_sb_mean = f"sb_from_mean_{gene_mode}"
        col_sb_median = f"sb_from_median_{gene_mode}"
        col_ceil = f"ceiling_from_mean_{gene_mode}"
        col_skew = f"skewness_r_max_{gene_mode}"
        if col_sb_mean in df_summary.columns:
            vals_mean = df_summary[col_sb_mean].dropna()
            vals_median = df_summary[col_sb_median].dropna()
            skew_vals = df_summary[col_skew].dropna()
            n_reliable = (vals_mean >= 0.5).sum()
            log.info(f"  [{gene_mode}] Reliable (SB≥0.5 from mean): {n_reliable}/{len(vals_mean)} "
                  f"({100 * n_reliable / len(vals_mean):.1f}%)")
            log.info(f"    SB(mean_r) median: {vals_mean.median():.3f}, "
                  f"SB(median_r) median: {vals_median.median():.3f}")
            if len(skew_vals) > 0:
                log.info(f"    Skewness of r dist — median: {skew_vals.median():.3f}, "
                      f"mean: {skew_vals.mean():.3f}")
            if col_ceil in df_summary.columns:
                ceil_vals = df_summary[col_ceil].dropna()
                if len(ceil_vals) > 0:
                    log.info(f"    Ceiling median: {ceil_vals.median():.3f}")

    # write to output
    cur_time = time.strftime('%Y%m%d-%H%M%S')
    df_long.to_csv(output_dir / f'reliability_{dataset_name}_cluster_{cluster}_{cur_time}.csv', index=False)
    df_summary.to_csv(output_dir / f'reliability_summary_{dataset_name}_cluster_{cluster}_{cur_time}.csv', index=False)
    log.info(f"Wrote Pearson split-half reliability results for cluster {cluster}")

    return df_long, df_summary


# ============================================================
# E-DISTANCE RELIABILITY
# ============================================================

def euclidean_upper_sum(X, norms = None, block_size: int = 1024):
    """Sum Euclidean distances for all unique pairs of rows in X.
    Sum in blocks for memory efficiency.

    Returns:
        sum_{i < j} ||X[i] - X[j]||_2
    """
    n = X.shape[0]

    # squared norm of each cell
    if norms is None:
        norms = np.einsum("ij,ij->i", X, X)

    total = 0.0

    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)

        Xi = X[i:i_end]
        ni = norms[i:i_end]

        # Only calculate j >= i.
        for j in range(i, n, block_size):
            j_end = min(j + block_size, n)

            Xj = X[j:j_end]
            nj = norms[j:j_end]

            # ||x-y||² = ||x||² + ||y||² - 2 x·y
            D2 = ni[:, None] + nj[None, :] - 2.0 * (Xi @ Xj.T)

            # Protect against tiny negative values from floating-point error.
            np.maximum(D2, 0, out=D2)

            np.sqrt(D2, out=D2)

            if i == j:
                # Same block: only take the strict upper triangle.
                total += np.triu(D2, k=1).sum(dtype=np.float64)
            else:
                # Off-diagonal block: every pair is unique.
                total += D2.sum(dtype=np.float64)

    return total


def euclidean_cross_sum(
    X,
    Y,
    y_norms,
    block_size=1024,
):
    """Euclidean distance of X and Y."""
    x_norms = np.einsum("ij,ij->i", X, X)

    total = 0.0

    for i in range(0, X.shape[0], block_size):
        i_end = min(i + block_size, X.shape[0])

        Xi = X[i:i_end]
        xi_norms = x_norms[i:i_end]

        for j in range(0, Y.shape[0], block_size):
            j_end = min(j + block_size, Y.shape[0])

            Yj = Y[j:j_end]
            yj_norms = y_norms[j:j_end]

            D2 = (
                xi_norms[:, None]
                + yj_norms[None, :]
                - 2.0 * (Xi @ Yj.T)
            )

            np.maximum(D2, 0, out=D2)
            np.sqrt(D2, out=D2)

            total += D2.sum(dtype=np.float64)

    return total


# Cell-count grid for rho_E(N) curves (n_half values).
N_GRID = [8, 16, 32, 64, 128, 256, 512]

def edistance_from_blocks(d_xy_sum, d_xx_sum, d_yy_sum, n_x, n_y):
    """
    Given precomputed sums of pairwise L2 distances between blocks X and Y
    and within each block, return the energy distance.

      E = 2 * mean_xy - mean_xx - mean_yy
    where mean_xy = d_xy_sum / (n_x * n_y), etc.
    """
    mean_xy = d_xy_sum / (n_x * n_y)
    mean_xx = d_xx_sum / (n_x * n_x)
    mean_yy = d_yy_sum / (n_y * n_y)
    return 2.0 * mean_xy - mean_xx - mean_yy

def _sum_block_chunked(D, rows, cols, row_chunk=1024):
    total = 0.0
    for i in range(0, len(rows), row_chunk):
        r = rows[i:i + row_chunk]
        total += D[np.ix_(r, cols)].sum(dtype=np.float64)
    return float(total)


def _sum_within_chunked(D, idx, row_chunk=1024):
    return _sum_block_chunked(D, idx, idx, row_chunk)


def _sum_block(D, rows, cols):
    """Sum of D[rows][:, cols]. Both rows, cols are 1d index arrays."""
    return float(D[np.ix_(rows, cols)].sum())


def _sum_within(D, idx):
    """Sum of D[idx][:, idx], including diagonal (which is 0 for L2)."""
    return float(D[np.ix_(idx, idx)].sum())


def compute_e_distance_reliability(
    perturbation,
    X_pert,
    X_ctrl,
    D_cc_sum,
    ctrl_norms,
    task_seed,
    n_grid = N_GRID,
    n_seeds: int = 100
) -> list[dict[str, Any]]:
    """Computes the e-distance reliability metric. Adapted from the
    compute_e_reliability.py script.

    Compute ρ_E(N) at each N in n_grid + {full} for one perturbation.

    Returns a list of dict rows (one per N), with the full pass tagged
    n_half = -1.
    """
    n_ctrl = X_ctrl.shape[0]
    n_pert = X_pert.shape[0]

    # Precompute within-perturbation and perturbation-vs-control pairwise distances.
    # D_cc_sum is passed in already (one-shot per condition).

    # We do need D_pp directly, so keep
    #D_pp_sum = 2 * euclidean_upper_sum(X_pert)
    D_pp = pairwise_distances(X_pert, X_pert, metric="euclidean").astype(np.float64)
    D_pp_sum = D_pp.sum(dtype=np.float64)
    D_pc_sum = euclidean_cross_sum(X_pert, X_ctrl, ctrl_norms)

    e_between = edistance_from_blocks(
        D_pc_sum, D_pp_sum, D_cc_sum, n_pert, n_ctrl,
    )

    rng = np.random.default_rng(task_seed)
    rows = []

    # N values to evaluate: each grid N at most once, plus the full pass.
    N_full = n_pert // 2
    if N_full < 4:
        return rows
    grid_Ns = [N for N in n_grid if 4 <= N < N_full]
    work_Ns = grid_Ns + [N_full]

    # Degenerate effect (E_between <= 0): emit rho_E = 0 rows.
    if not np.isfinite(e_between) or e_between <= 0:
        for N in work_Ns:
            rows.append(dict(
                perturbation=perturbation,
                n_half=N if N != N_full else -1,
                e_within_med=np.nan,
                e_between=e_between,
                rho_E=0.0,
                n_cells=n_pert,
            ))
        return rows

    for N in work_Ns:
        ratios = np.empty(n_seeds, dtype=np.float64)
        for s in range(n_seeds):
            perm = rng.permutation(n_pert)
            h1 = perm[:N]
            h2 = perm[N:2 * N]

            if N == N_full and 2 * N == n_pert:
                d12 = _sum_block_chunked(D_pp, h1, h2)
                e_within = (4.0 * d12 - D_pp_sum) / (N * N)
            else:
                d11 = _sum_within_chunked(D_pp, h1)
                d22 = _sum_within_chunked(D_pp, h2)
                d12 = _sum_block_chunked(D_pp, h1, h2)
                e_within = edistance_from_blocks(d12, d11, d22, N, N)

            ratios[s] = e_within / e_between
        e_within_med = float(np.median(ratios) * e_between)
        rho = float(np.clip(1.0 - np.median(ratios), -1.0, 1.0))
        rows.append(dict(
            perturbation=perturbation,
            n_half=N if N != N_full else -1,
            e_within_med=e_within_med,
            e_between=e_between,
            rho_E=rho,
            n_cells=n_pert,
        ))

    del D_pp
    return rows


def summarize_edist_results(cluster, all_rows, output_dir) -> None:
    """Summarize the edist results."""
    df = pd.DataFrame(all_rows)
    del all_rows

    df.insert(0, "cluster", cluster)
    cur_time = time.strftime('%Y%m%d-%H%M%S')
    out = output_dir / f"e_reliability_tahoe100m_cluster_{cluster}_{cur_time}.csv"
    df.to_csv(out, index=False)

    log.info(f"Wrote e-distance reliability results for cluster {cluster}")
