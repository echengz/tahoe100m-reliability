"""
Base reliability step, to be used by both
Tahoe100M and Regression pipelines.
"""
import logging
import numpy as np
import pandas as pd
import time

from pathlib import Path

from .pipeline_step import PipelineStep


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Fixed n_half values for the scaling curve.
# Original 7-point grid (powers of 2). n_total = 2 * n_half spans 16-1024.
DEFAULT_N_HALF_GRID = [8, 16, 32, 64, 128, 256, 512]


class BaseReliabilityStep(PipelineStep):

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize reliability step."""
        self.args = args
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_half_grid = sorted(
            {int(x.strip()) for x in self.args.n_half_grid.split(",") if x.strip()}
        )
        self.delta_p_dir = self.args.output_dir / 'delta_p_raw'

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for the reliability step to the given parser."""

        parser.add_argument(
            '--output-dir', dest='output_dir', required=True, type=Path,
            help='Output directory to write results to'
        )

        parser.add_argument(
            '--n-repeats', dest='n_repeats', type=int, default=100,
            help='Split-half repetitions (default: 100)'
        )

        parser.add_argument(
            '--expr-k', dest='expr_k', type=int, default=1000,
            help='Number of top expressed genes to restrict to (set to -1 for None)'
                + '(Ahlmann-Eltze approach; default: 1000)'
        )
        parser.add_argument(
            '--workers', dest='workers', type=int, default=1,
            help='Parallel workers to use (default: 1)'
        )
        parser.add_argument(
            '--seed', dest='seed', type=int, default=42,
            help='Random seed (default: 42)'
        )

        n_half_default = ",".join(str(n) for n in DEFAULT_N_HALF_GRID)
        parser.add_argument(
            "--n-half-grid", dest="n_half_grid", type=str, default=n_half_default,
            help="Comma-separated fractions of stimulated cells per split half "
                + f"for the fixed-n curve. Default: {n_half_default}.",
        )

    def save_delta_p_results(
        self,
        delta_p_results: list[dict],
        filename: str,
        group: str | None = None,
    ) -> None:
        """Saves delta_p results for downstream systematic analysis."""
        output_dir = self.delta_p_dir
        if group:
            output_dir = output_dir / group

        output_dir.mkdir(parents=True, exist_ok=True)

        conditions_stack = np.array(
            [x["condition"] for x in delta_p_results],
            dtype=str,
        )

        n_stim_stack = np.array(
            [x["n_stim_cells"] for x in delta_p_results],
            dtype=np.int32,
        )

        delta_p_stack = np.stack(
            [x["delta_p"] for x in delta_p_results],
            axis=0,
        )

        np.savez(
            output_dir / filename,
            conditions=conditions_stack,
            n_stim_cells=n_stim_stack,
            delta_p=delta_p_stack,
        )

    def summarize_split_half_results(self, cluster: int, all_rows: list, dataset_name: str) -> None:
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
        log.info(f"  Conditions processed: {len(df_summary)}")

        for gene_mode in ["all_genes", f"top{self.args.expr_k}_expr"]:
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

        df_long.to_csv(
            self.args.output_dir / f'reliability_{dataset_name}_cluster_{cluster}_{cur_time}.csv',
            index=False
        )
        df_summary.to_csv(
            self.args.output_dir / f'reliability_summary_{dataset_name}_cluster_{cluster}_{cur_time}.csv',
            index=False
        )
        log.info(f"Wrote Pearson split-half reliability results for cluster {cluster}")

        return df_long, df_summary
