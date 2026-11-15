"""
Runs the reliability regression step. Runs current framework
on prior Wei et al datasets to ensure output is still the
same/consistent with the original.

Focusing on cellular context as that is the version the
current Tahoe100m version was based on.

Basically takes the dataset loading code of the original
framework but uses the reliability computation functions
in the new one, since input data types are different but
core reliability functions should work the same.

NOTE: the original datasets already underwent CP10K normalization,
and as thus does not need to be applied/recalculated here. If possible,
it would be interesting to compute reliability on the raw datasets, but
that data is not publically available.
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from pathlib import Path

from tahoe100m_reliability.lib.step.base_reliability_step import BaseReliabilityStep

from tahoe100m_reliability.lib.analysis.compute_reliability import (
    compute_split_half_reliability,
)
from .datasets import CELLULAR_DATASETS, CTRL_LABEL_CANDIDATES

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


_G_CTRL_DENSE = None


def compute_split_half_reliability_wrapper(args):
    condition, X_stim, X_ctrl, ctrl_expr_idx, n_stim, n_half_max, n_half_grid_feasible, n_repeats, task_seed, output_dir = args

    # write delta-p for processing later
    # compute delta_p for systema
    stim_mean = X_stim.mean(axis=0)
    ctrl_mean = X_ctrl.mean(axis=0)
    delta_p = stim_mean - ctrl_mean

    return compute_split_half_reliability(
        condition, X_stim, X_ctrl, ctrl_expr_idx, n_stim,
        n_half_max, n_half_grid_feasible, n_repeats, task_seed,
        use_old_sampling=True
    ), delta_p, len(X_stim)


class RegressionReliabilityStep(BaseReliabilityStep):

    def __init__(
        self,
        input_dir: Path,
        baseline_dir: Path,
        **kwargs,
    ) -> None:
        """Initialize reliability step."""
        super().__init__(**kwargs)
        self.input_dir = input_dir
        self.baseline_dir = baseline_dir

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "RegressionReliabilityStep":
        """Create RegressionReliabilityStep from arguments."""
        return cls(**cls.clean_args(args))

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for the regression reliability step to the given parser."""
        parser.add_argument(
            '-i', '--input-dir', dest='input_dir', required=True, type=Path,
            help='Input directory containing h5ad files for the original cellular datasets'
        )
        parser.add_argument(
            '-b', '--baseline-dir', dest='baseline_dir', required=True, type=Path,
            help='Baseline directory containing results to compare to; should contain '
                 + 'reliability_cc_* results from the original framework'
        )

        super().add_cli_args(parser)

    def regression_split_half_reliability(
        self, ds_name: str, h5ad_path: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process the given dataset to regression test split-half reliability
        and systema results."""
        log.info(f"Running regression test for {ds_name}...")

        cfg = CELLULAR_DATASETS[ds_name]
        context_col = cfg["context_col"]
        condition_col = cfg["condition_col"]
        ctrl_label = cfg["ctrl_label"]

        # ---- Step 1: Load ----
        adata = ad.read_h5ad(h5ad_path)

        # ---- Step 2: Extract expression matrix ----

        X = adata.X
        if sp.issparse(X):
            X_norm = X.tocsr().astype(np.float32)
            is_sparse = True
            density = X_norm.nnz / (X_norm.shape[0] * X_norm.shape[1]) * 100
        else:
            X_norm = np.asarray(X, dtype=np.float32)
            is_sparse = False

        # Sanity check
        sample = X_norm[:100].toarray() if is_sparse else X_norm[:100]
        xmax = sample.max()
        if xmax > 30:
            log.info(f"        ⚠ WARNING: max={xmax:.2f} seems high for log-normalized data!")

        # ---- Step 3: Validate columns ----

        if context_col not in adata.obs.columns:
            raise ValueError(f"context_col '{context_col}' not in obs columns: "
                             f"{list(adata.obs.columns)}. Run --dataset discover first!")
        if condition_col not in adata.obs.columns:
            raise ValueError(f"condition_col '{condition_col}' not in obs columns: "
                             f"{list(adata.obs.columns)}. Run --dataset discover first!")

        contexts = adata.obs[context_col].astype(str).str.strip().values
        conditions = adata.obs[condition_col].astype(str).str.strip().values

        del adata

        unique_contexts = sorted(np.unique(contexts))
        unique_conditions = sorted(np.unique(conditions))

        unique_contexts = sorted(np.unique(contexts))
        unique_conditions = sorted(np.unique(conditions))

        # Validate control label (with fuzzy fallback)
        if ctrl_label not in unique_conditions:
            ctrl_found = None
            for cand in CTRL_LABEL_CANDIDATES:
                if cand in unique_conditions:
                    ctrl_found = cand
                    break
            if ctrl_found:
                log.info(f"        Note: ctrl_label '{ctrl_label}' not found, using '{ctrl_found}'")
                ctrl_label = ctrl_found
            else:
                raise ValueError(f"ctrl_label '{ctrl_label}' not in condition values: "
                                 f"{unique_conditions}")

        stim_conditions = [c for c in unique_conditions if c != ctrl_label]

        # ---- Step 4: Build tasks per context ----
        # This is analogous to the condition_groups loop in the genetic v2 script
        # (used for Frangieh), but here EVERY dataset has context stratification.
        all_rows = []
        delta_p_results = []
        total_conditions = 0
        total_skipped_ctrl = 0
        total_skipped_stim = 0

        for ctx in unique_contexts:
            ctx_mask = contexts == ctx

            # ---- Controls for this context ----
            ctrl_mask = ctx_mask & (conditions == ctrl_label)
            ctrl_idx = np.where(ctrl_mask)[0]
            n_ctrl = len(ctrl_idx)

            if n_ctrl < 16:
                n_stim_here = sum(1 for sc in stim_conditions
                                  if (ctx_mask & (conditions == sc)).any())
                log.info(f"        Context '{ctx}': only {n_ctrl} controls, skipping "
                      f"({n_stim_here} conditions lost)")
                total_skipped_ctrl += n_stim_here
                continue

            # Precompute dense control matrix
            # Use all controls (field convention: Virtual Cell Challenge, scPerturBench,
            # Ahlmann-Eltze 2025, PerturBench, Replogle 2022 all use full control pools
            # for pseudo-bulk mean Delta — no subsampling).
            if is_sparse:
                ctrl_dense = X_norm[ctrl_idx].toarray().astype(np.float32)
            else:
                ctrl_dense = X_norm[ctrl_idx].astype(np.float32)

            ctrl_n_effective = ctrl_dense.shape[0]

            # ---- Compute top-k gene index by mean control expression ----
            # (Ahlmann-Eltze et al. 2025 approach: rank by expression, not effect size)
            # Computed ONCE per context from control cells → non-circular
            ctrl_mean_expr = ctrl_dense.mean(axis=0)
            expr_k = min(self.expr_k, ctrl_dense.shape[1])
            expr_idx = np.argpartition(ctrl_mean_expr, -expr_k)[-expr_k:]
            expr_idx = expr_idx[np.argsort(ctrl_mean_expr[expr_idx])[::-1]]

            # ---- Build tasks for all perturbations in this context ----
            tasks = []
            n_skipped = 0
            cell_count_dist = []

            # Set globals for worker access (sequential mode uses these directly)
            global _G_CTRL_DENSE
            _G_CTRL_DENSE = ctrl_dense

            for i, stim_cond in enumerate(stim_conditions):
                stim_mask = ctx_mask & (conditions == stim_cond)
                stim_idx = np.where(stim_mask)[0]
                n_stim = len(stim_idx)

                if n_stim == 0:
                    continue  # this (context, perturbation) combo doesn't exist

                total_conditions += 1
                cell_count_dist.append(n_stim)

                # Max n_half: min of half-stim and half-ctrl (same as v2)
                n_half_max = min(n_stim // 2, ctrl_n_effective // 2)

                if n_half_max < 4:
                    n_skipped += 1
                    total_skipped_stim += 1
                    continue

                # Fixed n values that are feasible
                n_half_fixed_feasible = [
                    n for n in self.n_half_grid
                    if n <= n_half_max
                ]

                cond_name = f"{ctx}|{stim_cond}"
                task_seed = self.seed + i

                # Extract dense stim submatrix HERE in parent (avoids sending
                # full X_norm to workers — critical for macOS spawn)
                X_sub = X_norm[stim_idx]
                if sp.issparse(X_sub):
                    X_stim_dense = X_sub.toarray().astype(np.float32)
                elif not isinstance(X_sub, np.ndarray):
                    X_stim_dense = np.array(X_sub, dtype=np.float32)
                else:
                    X_stim_dense = (X_sub.astype(np.float32)
                                    if X_sub.dtype != np.float32 else X_sub)

                tasks.append((
                    cond_name,
                    X_stim_dense,
                    _G_CTRL_DENSE,
                    expr_idx,
                    n_stim,
                    n_half_max,
                    n_half_fixed_feasible,
                    self.n_repeats,
                    task_seed,
                    self.output_dir
                ))

            if not tasks:
                log.info(f"        Context '{ctx}': {n_ctrl:,} ctrl (eff {ctrl_n_effective:,}), "
                      f"0 tasks" +
                      (f", {n_skipped} skipped" if n_skipped else ""))
                continue

            # ---- Step 5: Compute ----
            n_workers = min(self.workers, len(tasks))
            if n_workers <= 1:
                ctx_rows = []
                for i, task in enumerate(tasks):
                    result, delta_p, n_stim_cells = compute_split_half_reliability_wrapper(task)
                    ctx_rows.extend(result)
                    delta_p_results.append({
                        'condition': result[0]['condition'],
                        'n_stim_cells': n_stim_cells,
                        'delta_p': delta_p.astype(np.float32)
                    })

            else:
                ctx_rows = []
                chunk_size = max(1, len(tasks) // (n_workers * 10))
                # Only ctrl_dense goes via initializer (small per context).
                # Stim submatrices are in each task tuple (small per condition).
                # This avoids pickling the full X_norm matrix to each worker.
                with Pool(
                    processes=n_workers,
                    initializer=_init_worker,
                    initargs=(ctrl_dense,),
                ) as pool:
                    results_iter = pool.imap_unordered(
                        compute_split_half_reliability_wrapper,
                        tasks,
                        chunksize=chunk_size,
                    )
                    done = 0
                    for (result, delta_p, n_stim_cells) in results_iter:
                        ctx_rows.extend(result)
                        delta_p_results.append({
                            'condition': result['condition'],
                            'n_stim_cells': n_stim_cells,
                            'delta_p': delta_p.astype(np.float32)
                        })

                        done += 1
                        if done % 20 == 0 or done == len(tasks):
                            rate = done / elapsed if elapsed > 0 else 0
                            eta = (len(tasks) - done) / rate if rate > 0 else 0

            # Parse condition name into context + perturbation
            for row in ctx_rows:
                parts = row["condition"].split("|", 1)
                row["context"] = parts[0]
                row["perturbation"] = parts[1] if len(parts) > 1 else ""
            all_rows.extend(ctx_rows)

            # Reset cell_count_dist for next context
            cell_count_dist = []

        # Clean up
        _G_CTRL_DENSE = None

        # save delta_p results for downstream analysis
        if delta_p_results:
            delta_p_filename = f"{ds_name}_delta_p.npz"

            self.save_delta_p_results(
                delta_p_results=delta_p_results,
                filename=delta_p_filename
            )

            del delta_p_results
        else:
            raise ValueError("expected systema data")

        # summarize reliability results
        return self.summarize_split_half_results(
            cluster=0,
            all_rows=all_rows,
            dataset_name=ds_name,
        )

    def compare_baseline(self, ds_name: str, df_long: pd.DataFrame, df_summary: pd.DataFrame) -> None:
        """Compare baseline results."""

        # get baseline (original) results
        baseline_df_long = pd.read_csv(self.baseline_dir / f'reliability_cc_{ds_name}.csv')
        baseline_df_summary = pd.read_csv(self.baseline_dir / f'reliability_cc_summary_{ds_name}.csv')

        # remove dataset/cluster columns, meaning different between two versions
        df_long = df_long.drop(columns=['cluster'])
        df_summary = df_summary.drop(columns=['cluster'])
        baseline_df_long = baseline_df_long.drop(columns=['dataset'])
        baseline_df_summary = baseline_df_summary.drop(columns=['dataset'])

        # Check columns match, ignoring order
        assert set(baseline_df_long.columns) == set(df_long.columns), (
            f"Long columns differ: "
            f"missing={set(baseline_df_long.columns) - set(df_long.columns)}, "
            f"extra={set(df_long.columns) - set(baseline_df_long.columns)}"
        )

        assert set(baseline_df_summary.columns) == set(df_summary.columns), (
            f"Summary columns differ: "
            f"missing={set(baseline_df_summary.columns) - set(df_summary.columns)}, "
            f"extra={set(df_summary.columns) - set(baseline_df_summary.columns)}"
        )

        # Reorder new DataFrames to match baseline column order
        df_long = df_long[baseline_df_long.columns]
        df_summary = df_summary[baseline_df_summary.columns]

        # ensure type matches on context, and sort on some columns to ensure consistent order
        for df in [df_long, baseline_df_long, df_summary, baseline_df_summary]:
            df['context'] = df['context'].astype('string')

        sort_cols = [
            'condition',
            'gene_mode',
            'mean_r',
            'std_r',
            'median_r'
        ]
        df_long = df_long.sort_values(sort_cols).reset_index(drop=True)
        baseline_df_long = baseline_df_long.sort_values(sort_cols).reset_index(drop=True)

        sort_cols = [
            'condition',
            'median_r_max_all_genes',
            'mean_r_max_all_genes',
            'std_r_max_all_genes',
        ]
        df_summary = df_summary.sort_values(sort_cols).reset_index(drop=True)
        baseline_df_summary = baseline_df_summary.sort_values(sort_cols).reset_index(drop=True)

        pd.testing.assert_frame_equal(baseline_df_long, df_long)
        pd.testing.assert_frame_equal(baseline_df_summary, df_summary)

        log.info(f"{ds_name} passed reliability regression test!")


    def run(self) -> None:
        """Run reliability regression step."""

        # run all datasets for regression
        datasets_to_run = list(CELLULAR_DATASETS.keys())
        all_results = {}

        for ds_name in datasets_to_run:
            h5ad_path = self.input_dir / f"{ds_name}.h5ad"

            if not h5ad_path.exists():
                log.info(f"\n  ✗ SKIPPING {ds_name}: not found at {h5ad_path}")
                continue

            df_long, df_summary = self.regression_split_half_reliability(ds_name, h5ad_path)

            # compare results with baseline
            self.compare_baseline(ds_name, df_long, df_summary)

            all_results[ds_name] = df_long

        # print out split half results for report; identical to how Tahoe-100M's heatmap
        # is constructed
        for ds_name, df in all_results.items():
            gene_mode = "all_genes"

            # Fixed split-half sizes
            fixed = df[
                (df["n_mode"] == "fixed") &
                (df["gene_mode"] == gene_mode)
            ]

            fixed_by_condition = (
                fixed.groupby(["condition", "n_half"])["median_r"]
                     .median()
                     .unstack("n_half")
            )

            summary = fixed_by_condition.mean()

            # Maximum available sample size
            max_available = df[
                (df["n_mode"] == "max") &
                (df["gene_mode"] == gene_mode)
            ]

            summary["Max Available"] = (
                max_available.groupby("condition")["median_r"]
                             .median()
                             .mean()
            )

            # Display as a latex table
            columns = [8, 16, 32, 64, 128, 256, 512, "Max Available"]
            values = summary.reindex(columns)
            row = (
                f"{ds_name} & Wei et al & "
                + " & ".join(f"${v:.2f}$" for v in values)
                + r" \\"
            )
            print(row)
