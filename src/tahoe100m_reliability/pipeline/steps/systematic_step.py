"""
Runs step 3 of the pipeline: systematic score

For Tahoe-100M-specific analysis, also adds influence calculations:
    1. Gene influence: leave-one-out gene
    2. Pathway influence: leave-one-pathway out influence. Pathways derived from GSEA scores
"""
import argparse
import logging
import numpy as np
import pandas as pd
import time

from pathlib import Path
from tqdm import tqdm
from typing import Any

from tahoe100m_reliability.lib.analysis.influence_accumulator import (
    GeneInfluenceAccumulatorHandler,
    GSEAInfluenceAccumulatorHandler,
)
from tahoe100m_reliability.lib.analysis.compute_systematic import (
    score_perturbation,
)

from tahoe100m_reliability.lib.ingest.intermediate_loaders import (
    get_combined_cluster_split_half_reliability_df,
    load_delta_p_results,
    load_reliable_conditions,
    RELIABILITY_THRESH,
    SPECIFICITY_THRESH,
    REL_COL_FALLBACKS,
)
from tahoe100m_reliability.lib.step.base_systematic_step import BaseSystematicStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SystematicStep(BaseSystematicStep):

    def __init__(
        self,
        name: str,
        datasets: list[str],
        **kwargs,
    ) -> None:
        """Initializer.

        Arguments:
            args: command line args
            name: name for the group of data being evaluated
            datasets: the datasets being evaluated
        """
        super().__init__(**kwargs)
        self.name = name
        self.datasets = datasets

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "SystematicStep":
        """Create SystematicStep from arguments."""
        return cls(**cls.clean_args(args))

    def post_process_perturbation_score(
        self,
        mode: str,
        cell_line_id: str,
        condition: str,
        delta_p,
        delta_avg,
        score_results,
        working_results,
    ) -> None:
        """Post process perturbation score as needed."""
        pass

    def post_process_cell_line_results(self, cell_line_id: str, cell_line_rows: list) -> None:
        """Post process cell line results as needed."""
        pass

    def compute_systema_metrics(
        self, mode: str, ds_name: str, delta_p_dir: Path, reliability_dir: Path
    ) -> pd.DataFrame:
        """
        Compute Systema's systematic variation metric for all (context, perturbation)
        conditions in a cellular context dataset.
        """
        t_start = time.time()
        log.info(f"Computing systema metrics from {delta_p_dir}")

        # ---- Load reliability (used to filter which perts define δ_avg) ----
        log.info("  [0/3] Loading reliability data for stim data...")
        reliable_conds, all_conds = load_reliable_conditions(
            ds_name, reliability_dir, self.clusters
        )

        # ---- Load precomputed delta values for each (reliable) cell line ----
        log.info("  [1/3] Loading precomputed delta_p values for reliable data...")
        delta_p_data = load_delta_p_results(
            ds_name=ds_name,
            delta_p_dir=delta_p_dir,
            target_conds=all_conds,
            clusters=self.clusters
        )

        # ---- Compute Systema per cell line ----
        log.info("  [2/3] Computing cosine similarities and influence scores per cell line...")
        all_rows = []

        total_conditions = 0
        total_skipped = 0

        for cell_line_id, data in tqdm(
            delta_p_data.items(), desc=f'Processing cell lines ({mode})', unit='cell_line'
        ):
            conditions = data["conditions"]
            n_stim_cells = data["n_stim_cells"]
            delta_p_all = data["delta_p"]

            if len(conditions) == 0:
                log.info(
                    f"Cell line '{cell_line_id}': no conditions, skipping"
                )
                continue

            cell_line_rows = []

            # Group conditions by context
            contexts = np.asarray([
                self.parse_condition(condition)[0]
                for condition in conditions
            ])

            for context_group in np.unique(contexts):
                context_mask = contexts == context_group

                context_conditions = conditions[context_mask]
                context_weights = n_stim_cells[context_mask]
                context_deltas = delta_p_all[context_mask]

                # Count unique perturbations in this context
                unique_perts = {
                    self.parse_condition(condition)[1]
                    for condition in context_conditions
                }

                n_perts = len(unique_perts)
                if n_perts == 0:
                    log.info(
                        f"Context group '{context_group}': no perturbations, skipping"
                    )
                    continue

                # ---- Step 1: identify reliable perturbations ----
                reliable_mask = np.array(
                    [condition in reliable_conds for condition in context_conditions],
                    dtype=bool,
                )
                n_reliable = reliable_mask.sum()

                # ---- Step 2: δ_avg from reliable perturbations ----
                if n_reliable == 0:
                    log.warning(f"Context {context_group} has no reliability conditions; skipping")

                    delta_avg = np.zeros_like(context_deltas[0], dtype=np.float64)
                    norm_delta_avg = 0.0
                else:
                    reliable_deltas = context_deltas[reliable_mask]
                    reliable_weights = context_weights[reliable_mask]

                    # Cell-weighted average perturbation effect.
                    delta_avg = np.average(
                        reliable_deltas.astype(np.float64), axis=0, weights=reliable_weights,
                    )
                    norm_delta_avg = float(np.linalg.norm(delta_avg))

                # ---- Step 3: For each perturbation, compute δ_p; cos_sim only if reliable ----
                for condition, n_stim, delta_p, is_rel in zip(
                    context_conditions,
                    context_weights,
                    context_deltas,
                    reliable_mask,
                ):
                    # skip perturbations without enough stim cells
                    if n_stim < 4:
                        total_skipped += 1
                        continue

                    # score the condition
                    total_conditions += 1
                    context, perturbation = self.parse_condition(condition)
                    row = {
                        'dataset': ds_name,
                        'condition': condition,
                        'context': context,
                        'perturbation': perturbation,
                        'n_stim_cells': n_stim,
                        'n_perts_in_context': n_perts,
                    }

                    score_results, working_results = score_perturbation(
                        delta_p=delta_p,
                        delta_avg=delta_avg,
                        norm_delta_avg=norm_delta_avg,
                        is_rel=is_rel
                    )

                    row.update(score_results)
                    cell_line_rows.append(row)

                    # Optional post-processing of perturbation results
                    self.post_process_perturbation_score(
                        mode,
                        cell_line_id,
                        condition,
                        delta_p,
                        delta_avg,
                        score_results,
                        working_results,
                    )

            # Optional post processing of cell line results
            self.post_process_cell_line_results(cell_line_id, cell_line_rows)
            all_rows.extend(cell_line_rows)

        # build final dataframes
        df = pd.DataFrame(all_rows)
        del all_rows

        # ---- Summary statistics ----
        elapsed_total = time.time() - t_start
        log.info(f"\n  ── Summary for {ds_name} ──")
        log.info(f"  Conditions computed: {total_conditions}")
        log.info(f"  Skipped (few cells): {total_skipped}")
        log.info(f"  cos_sim — mean: {df.cos_sim.mean():.3f} ± {df.cos_sim.std():.3f}, "
              f"median: {df.cos_sim.median():.3f}")
        log.info(f"  cos_sim — min: {df.cos_sim.min():.3f}, "
              f"max: {df.cos_sim.max():.3f}")
        log.info(f"  frac with cos_sim > 0.5: "
              f"{(df.cos_sim > 0.5).sum()}/{len(df)} "
              f"({100*(df.cos_sim > 0.5).mean():.1f}%)")
        log.info(f"  frac with cos_sim > 0.7: "
              f"{(df.cos_sim > 0.7).sum()}/{len(df)} "
              f"({100*(df.cos_sim > 0.7).mean():.1f}%)")

        # Per-context breakdown
        log.info(f"\n  Per-context cos_sim (median):")
        for ctx, gdf in df.groupby("context"):
            log.info(f"    {ctx:20s}: {gdf.cos_sim.median():.3f} "
                  f"(n={len(gdf)}, n_perts_in_ctx={gdf.n_perts_in_context.iloc[0]})")

        log.info(f"  Total time: {elapsed_total:.1f}s")

        # Save Systema-only results
        df.to_csv(
            self.output_dir / f"systema_cc_{mode}_{ds_name}_{self.cur_time}.csv",
            index=False
        )
        return df

    def merge_with_reliability(
        self, mode: str, ds_name: str, systema_df: pd.DataFrame, reliability_dir: Path,
    ) -> pd.DataFrame:
        """
        Merge Systema cos_sim with pre-computed split-half reliability
        from the cellular context reliability script.

        Merges on the 'condition' column (format: "context|perturbation").
        """
        rel_df = get_combined_cluster_split_half_reliability_df(
            reliability_dir, self.clusters, ds_name=ds_name
        )
        log.info(f"  Loaded reliability: {len(rel_df)} conditions")

        # Merge on condition (= "context|perturbation")
        merged = systema_df.merge(
            rel_df,
            on="condition",
            how="inner",
            suffixes=("", "_rel"),
        )

        log.info(f"  Merged: {len(merged)} conditions "
              f"(Systema: {len(systema_df)}, Reliability: {len(rel_df)})")

        if len(merged) < len(systema_df):
            missing = set(systema_df["condition"]) - set(merged["condition"])
            if len(missing) <= 10:
                log.info(f"  Unmatched conditions: {sorted(missing)}")
            else:
                log.info(f"  Unmatched conditions: {len(missing)} "
                      f"(first 5: {sorted(missing)[:5]})")

        # ---- 2D classification ----
        # Use sb_from_median_all_genes as the reliability metric
        rel_col = "sb_from_median_all_genes"
        if rel_col not in merged.columns:
            rel_col = "sb_from_mean_all_genes"

        if rel_col in merged.columns:
            rho = merged[rel_col]
            cos = merged["cos_sim"]

            merged["reliable"] = rho >= RELIABILITY_THRESH
            merged['high_systematic'] = merged['frac_systematic_var'] >= SPECIFICITY_THRESH

            # 2D triage categories
            conditions = [
                (~merged["reliable"]),
                (merged["reliable"]) & (merged["high_systematic"]),
                (merged["reliable"]) & (~merged["high_systematic"]),
            ]
            choices = ["Unreliable", "Shared", "Specific"]
            merged["triage_2d"] = np.select(conditions, choices, default="other")

            # log.info 2D triage summary
            log.info(f"\n  ── 2D Triage Summary ──")
            log.info(f"  Reliability metric: {rel_col}")
            for cat in ["Unreliable", "Shared", "Specific"]:
                n = (merged["triage_2d"] == cat).sum()
                pct = 100 * n / len(merged) if len(merged) > 0 else 0
                log.info(f"    {cat:20s}: {n:>5} ({pct:5.1f}%)")

            # Ceiling on genuine signal
            ceiling_col = "ceiling_from_median_all_genes"
            if ceiling_col not in merged.columns:
                ceiling_col = "ceiling_from_mean_all_genes"
            if ceiling_col in merged.columns:
                genuine = merged[merged["triage_2d"] == "Specific"]
                if len(genuine) > 0:
                    log.info(f"    Genuine signal: mean ceiling = "
                          f"{genuine[ceiling_col].mean():.3f}, "
                          f"mean cos_sim = {genuine['cos_sim'].mean():.3f}")
        else:
            log.info(f"  ⚠ Reliability column '{rel_col}' not found in reliability data")

        # save to CSV
        merged.to_csv(
            self.output_dir / f"systema_cc_{mode}_reliability_2d_{ds_name}_{self.cur_time}.csv",
            index=False
        )

        return merged

    def generate_full_summaries(self, all_summary_rows: dict[str, Any]) -> None:
        """Generate the full summaries."""
        for mode, summary_rows in all_summary_rows.items():
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(
                self.output_dir / f'systema_cc_{mode}_summary_{self.name}_{self.cur_time}.csv',
                index=False
            )

            log.info(f"\n{'='*70}")
            log.info(f"  Cross-Dataset Summary (Cellular Context), mode {mode}")
            log.info(f"{'='*70}")
            log.info(summary_df.to_string(index=False))

            # Grand totals
            if "n_Unreliable" in summary_df.columns:
                total = summary_df["n_conditions"].sum()
                total_unmeas = summary_df.get("n_Unreliable", pd.Series([0])).sum()
                total_falsely = summary_df.get("n_Shared", pd.Series([0])).sum()
                total_genuine = summary_df.get("n_Specific", pd.Series([0])).sum()
                total_2d = total_unmeas + total_falsely + total_genuine

                if total_2d > 0:
                    log.info(f"\n  ── Grand Totals (across all datasets) ──")
                    log.info(f"  Total conditions (Systema):  {int(total)}")
                    log.info(f"  Total conditions (merged):   {int(total_2d)}")
                    log.info(f"    Unmeasurable:    {int(total_unmeas):>5} "
                          f"({100*total_unmeas/total_2d:.1f}%)")
                    log.info(f"    Falsely solved:  {int(total_falsely):>5} "
                          f"({100*total_falsely/total_2d:.1f}%)")
                    log.info(f"    Genuine signal:  {int(total_genuine):>5} "
                          f"({100*total_genuine/total_2d:.1f}%)")

                    log.info(f"\n  ★ Effective benchmark size: {int(total_genuine)} conditions "
                          f"({100*total_genuine/total_2d:.1f}% of evaluated)")

    def post_per_dataset_processing(
        self, mode: str, ds_name: str, systema_df: pd.DataFrame, merged_df: pd.DataFrame
    ) -> None:
        """Post-processing after generating results for a given df."""
        pass

    def run(self) -> None:
        """Run the systematic step."""
        log.info("Running Systema Systematic Variation + Reliability 2D")

        # run for each mode (raw and CP10K normalization)
        all_summary_rows = {}
        for mode, delta_p_mode in self.mode_mapping.items():
            reliability_dir = delta_p_mode.reliability_dir
            delta_p_dir = delta_p_mode.delta_p_dir

            if not delta_p_dir or not delta_p_dir.is_dir():
                log.warning(f"{mode} delta_p dir not found, skipping mode")
                continue

            log.info(f"Evaluating {mode}")
            all_summary_rows[mode] = []

            for ds_name in self.datasets:
                systema_df = self.compute_systema_metrics(
                    mode, ds_name, delta_p_dir, reliability_dir
                )

                # Merge with reliability
                merged_df = self.merge_with_reliability(
                    mode, ds_name, systema_df, reliability_dir
                )

                # any post processing as needed by the class
                self.post_per_dataset_processing(mode, ds_name, systema_df, merged_df)

                # Cross-dataset summary row
                summary_row = {
                    "dataset": ds_name,
                    "n_conditions": len(systema_df),
                    "n_contexts": systema_df["context"].nunique(),
                    "n_perturbations": systema_df["perturbation"].nunique(),
                    "cos_sim_mean": systema_df.cos_sim.mean(),
                    "cos_sim_std": systema_df.cos_sim.std(),
                    "cos_sim_median": systema_df.cos_sim.median(),
                    "frac_cos_gt_0.5": (systema_df.cos_sim > 0.5).mean(),
                    "frac_cos_gt_0.7": (systema_df.cos_sim > 0.7).mean(),
                }

                # Add 2D triage counts if available
                if "triage_2d" in merged_df.columns:
                    for cat in ["Unreliable", "Shared", "Specific"]:
                        n = (merged_df["triage_2d"] == cat).sum()
                        summary_row[f"n_{cat}"] = n
                        summary_row[f"frac_{cat}"] = (
                            n / len(merged_df) if len(merged_df) > 0 else np.nan
                        )

                all_summary_rows[mode].append(summary_row)

        self.generate_full_summaries(all_summary_rows)


class SystematicInfluenceStep(SystematicStep):
    """Evaluate systematic and influence scores."""

    def __init__(
        self,
        gsea_dir: Path | None = None,
        **kwargs,
    ) -> None:
        """Initializer."""
        super().__init__(name='tahoe100m', datasets=['tahoe100m'], **kwargs)

        self.drug_to_moa = self.load_drug_moa_mapping()

        self.gene_influences = GeneInfluenceAccumulatorHandler(
            self.gene_metadata,
            self.output_dir,
            self.cur_time,
        )

        self.gsea_influences = GSEAInfluenceAccumulatorHandler(
            gsea_dir,
            self.gene_metadata,
            self.output_dir,
            self.cur_time,
        )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "SystematicInfluenceStep":
        """Create SystematicInfluenceStep from arguments."""
        return cls(**cls.clean_args(args))

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this systematic step to the given parser."""
        BaseSystematicStep.add_cli_args(parser=parser)

        parser.add_argument(
            '--gsea-dir', dest='gsea_dir', type=Path,
            help='Path to directory containing GSEA results. If provided, will add influence '
                 + 'scores to the corresponding lead genes'
        )

    def parse_condition(self, condition: str) -> tuple[str, str]:
        """Parse condition into context and perturbation."""
        parts = condition.split("|")

        if len(parts) != 3:
            raise ValueError(f"Unexpected condition format for Tahoe-100M: {condition}")
        context, drug, dosage = parts
        perturbation = f'{drug}|{dosage}'

        return context, perturbation

    def post_process_perturbation_score(
        self,
        mode: str,
        cell_line_id: str,
        condition: str,
        delta_p,
        delta_avg,
        score_results,
        working_results,
    ) -> None:
        """Post process perturbation score - computes gene accumulator."""
        # gene accumulation; make sure this is a reliable/finate condition
        frac_systematic = score_results['frac_systematic_var']
        if not np.isfinite(frac_systematic):
            return

        drug = self.parse_condition(condition)[1].split('|')[0]
        moa = self.drug_to_moa.get(drug, 'unclear')

        for handler in [self.gene_influences, self.gsea_influences]:
            # if encountering new mode/cell line combo, add accumulators for it
            handler.ensure_exists(mode, cell_line_id, moa, self.ctrl_gene_expressions_dir)

            # accumulate influence
            handler.accumulate_influence(
                mode, moa, cell_line_id, delta_p, delta_avg, working_results, frac_systematic
            )

    def post_process_cell_line_results(
        self, cell_line_id: str, cell_line_rows: list
    ) -> None:
        """Accumulate cell line-specific influence results."""
        pass

    def __summarize_merged_results(
        self, mode: str, ds_name: str, merged_df: pd.DataFrame
    ) -> None:
        """Summarize Tahoe-100m merged results by specific groupings."""
        df = merged_df.copy()

        # Extract cell_line_id, drug, and dosage from condition.
        parts = df["condition"].astype(str).str.split(
            "|", regex=False, expand=True
        )

        if parts.shape[1] != 3:
            raise ValueError(
                "Expected 3 condition components "
                f"(cell_line|drug|dosage), found {parts.shape[1]}"
            )

        df["cell_line_id"] = parts[0]
        df["drug"] = parts[1]
        df["dosage"] = parts[2]

        # Select available reliability and ceiling columns
        rel_col = next(
            (c for c in REL_COL_FALLBACKS if c in df.columns),
            None,
        )

        ceiling_col = next(
            (
                c for c in [
                    "ceiling_from_median_all_genes",
                    "ceiling_from_mean_all_genes",
                ]
                if c in df.columns
            ),
            None,
        )

        # Map each drug to its corresponding MOA
        df["moa"] = (
            df["drug"]
            .astype(str)
            .str.strip()
            .map(self.drug_to_moa)
        )

        # Define the groupings
        groupings = {
            "cell_line_id": ["cell_line_id"],
            "drug": ["drug"],
            "dosage": ["dosage"],
            "drug_x_dosage": ["drug", "dosage"],
            "moa": ['moa']
        }
        log.info(f"Grouping merged_df by the following categories: {list(groupings.keys())}")

        for grouping_name, group_cols in groupings.items():
            agg = {
                "condition": ["nunique"],
                "cos_sim": ["count", "mean", "std", "median"],
                "n_stim_cells": ["sum", "median"],
            }

            if rel_col:
                agg[rel_col] = ["mean", "median"]

            if ceiling_col:
                agg[ceiling_col] = ["mean", "median"]

            summary = df.groupby(group_cols, dropna=False).agg(agg)

            # Flatten the MultiIndex columns.
            summary.columns = [
                "_".join(str(x) for x in col if x)
                for col in summary.columns
            ]
            summary = summary.reset_index()

            summary = summary.rename(columns={
                "condition_nunique": "n_conditions",
                "cos_sim_count": "n_scored",
                "cos_sim_mean": "cos_sim_mean",
                "cos_sim_std": "cos_sim_std",
                "cos_sim_median": "cos_sim_median",
                "n_stim_cells_sum": "total_stim_cells",
                "n_stim_cells_median": "median_stim_cells",
            })

            # Add 2D classification counts and proportions
            if "triage_2d" in df.columns:
                categories = ["Unreliable", "Shared", "Specific"]

                condition_keys = ["condition", *group_cols]
                condition_triage = df[condition_keys + ["triage_2d"]].drop_duplicates()

                labels_per_condition = (
                    condition_triage
                    .groupby(condition_keys, dropna=False)["triage_2d"]
                    .nunique(dropna=False)
                )
                if (labels_per_condition > 1).any():
                    log.warning(
                        "%d conditions have conflicting triage_2d labels "
                        "for grouping %s",
                        int((labels_per_condition > 1).sum()),
                        grouping_name,
                    )

                # Only count conditions with exactly one valid category
                condition_triage = condition_triage[
                    condition_triage["triage_2d"].isin(categories)
                ].drop_duplicates(condition_keys)

                counts = (
                    condition_triage
                    .groupby(group_cols, dropna=False)["triage_2d"]
                    .value_counts()
                    .unstack(fill_value=0)
                    .reindex(columns=categories, fill_value=0)
                )

                counts = counts.rename(columns={
                    "Unreliable": "n_Unreliable",
                    "Shared": "n_Shared",
                    "Specific": "n_Specific",
                })

                counts["n_classified"] = counts[
                    ["n_Unreliable", "n_Shared", "n_Specific"]
                ].sum(axis=1)

                for category in categories:
                    counts[f"frac_{category}"] = (
                        counts[f"n_{category}"]
                        / counts["n_classified"].replace(0, np.nan)
                    )

                summary = summary.drop(
                    columns=[
                        c for c in [
                            "n_Unreliable", "n_Shared", "n_Specific",
                            "n_classified", "frac_Unreliable",
                            "frac_Shared", "frac_Specific",
                        ]
                        if c in summary.columns
                    ],
                    errors="ignore",
                )

                summary = summary.merge(
                    counts.reset_index(),
                    on=group_cols,
                    how="left",
                )

            # Standardize reliability and ceiling column names
            if rel_col:
                summary = summary.rename(columns={
                    f"{rel_col}_mean": "reliability_mean",
                    f"{rel_col}_median": "reliability_median",
                })

            if ceiling_col:
                summary = summary.rename(columns={
                    f"{ceiling_col}_mean": "ceiling_mean",
                    f"{ceiling_col}_median": "ceiling_median",
                })

            # Save CSV for this grouping.
            out_path = (
                self.output_dir
                / f"systema_cc_{mode}_reliability_2d_summary_{ds_name}_"
                  f"by_{grouping_name}_{self.cur_time}.csv"
            )

            summary.to_csv(out_path, index=False)

    def post_per_dataset_processing(
        self, mode: str, ds_name: str, systema_df: pd.DataFrame, merged_df: pd.DataFrame
    ) -> None:
        """Post-dataset processing."""
        self.__summarize_merged_results(mode, ds_name, merged_df)

        # summarize influence scores
        self.gene_influences.finalize_influence(mode)
        self.gsea_influences.finalize_influence(mode)
