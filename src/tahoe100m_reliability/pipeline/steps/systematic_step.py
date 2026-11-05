"""
Runs step 2 of the pipeline: systematic.
"""
import argparse
import logging
import numpy as np
import pandas as pd
import time

from pathlib import Path
from tqdm import tqdm

from tahoe100m_reliability.lib.tahoe100m_dataset import (
    get_drug_metadata,
)

from .step_helpers.intermediate_loaders import (
    get_combined_split_half_reliability_df,
    load_gene_ids,
    load_reliable_conditions,
    load_systema_deltas,
    RELIABILITY_THRESH,
    SPECIFICITY_THRESH,
    REL_COL_FALLBACKS,
)
from .step_helpers.influence_accumulator import (
    InfluenceAccumulator,
    summarize_perturbation_influence,
)
from .step_helpers.pipeline_step import BaseSystematicStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def cosine_similarity(a, b):
    """Cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def batch_cosine_similarity(deltas, avg_delta):
    """
    Cosine similarity between each row of `deltas` and a single vector.

    Parameters
    ----------
    deltas    : ndarray, shape (N_perts, G)
    avg_delta : ndarray, shape (G,)

    Returns
    -------
    cos_sims : ndarray, shape (N_perts,)
    """
    dots = deltas @ avg_delta
    norms_delta = np.linalg.norm(deltas, axis=1)
    norm_avg = np.linalg.norm(avg_delta)
    denom = norms_delta * norm_avg
    denom[denom < 1e-12] = 1.0
    return dots / denom


def calculate_frac_systematic(
    delta_p, delta_avg, norm_delta_avg, is_rel
):
    norm_delta_p = float(np.linalg.norm(delta_p))

    if is_rel and norm_delta_avg > 1e-12:
        cos_sim = cosine_similarity(
            delta_p,
            delta_avg,
        )

        proj_scalar = (
            np.dot(delta_p, delta_avg)
            / (norm_delta_avg ** 2)
        )

        delta_specific = (
            delta_p
            - proj_scalar * delta_avg
        )

        norm_delta_specific = float(
            np.linalg.norm(delta_specific)
        )

        frac_systematic = (
            float(cos_sim ** 2)
            if norm_delta_p > 1e-12
            else np.nan
        )

    else:
        cos_sim = np.nan
        norm_delta_specific = np.nan
        frac_systematic = np.nan

    return cos_sim, norm_delta_p, norm_delta_specific, frac_systematic


def score_perturbation(
    ds_name,
    condition,
    n_perts,
    n_stim,
    delta_p,
    delta_avg,
    norm_delta_avg,
    is_rel,
    gene_ids,
):
    condition = str(condition)
    n_stim = int(n_stim)

    if n_stim < 4:
        return None, None, None

    parts = condition.split("|")

    if ds_name == 'tahoe100m':
        if len(parts) != 3:
            raise ValueError(f"Unexpected condition format: {condition}")
        context, drug, dosage = parts
        perturbation = f'{drug}|{dosage}'
    else:
        context, perturbation = parts

    delta_p = np.asarray(delta_p, dtype=np.float64)
    delta_avg = np.asarray(delta_avg, dtype=np.float64)

    # Calculate shared quantities
    dot_pa = np.dot(delta_p, delta_avg)
    norm_p_sq = np.dot(delta_p, delta_p)
    norm_a_sq = norm_delta_avg ** 2

    norm_delta_p = np.sqrt(norm_p_sq)

    if is_rel and norm_a_sq > 1e-24 and norm_delta_p > 1e-12:
        cos_sim = dot_pa / (norm_delta_p * norm_delta_avg) \
            if norm_delta_p > 1e-12 else np.nan

        frac_systematic = cos_sim ** 2

        norm_delta_specific = np.sqrt(max(
            norm_p_sq - dot_pa**2 / norm_a_sq,
            0.0
        ))

        # Leave-one-gene-out calculations
        dot_without = np.empty_like(delta_p)
        np.multiply(delta_p, delta_avg, out=dot_without)
        np.subtract(dot_pa, dot_without, out=dot_without)

        p_without = np.empty_like(delta_p)
        np.square(delta_p, out=p_without)
        np.subtract(norm_p_sq, p_without, out=p_without)

        a_without = np.empty_like(delta_avg)
        np.square(delta_avg, out=a_without)
        np.subtract(norm_a_sq, a_without, out=a_without)

        np.maximum(p_without, 0.0, out=p_without)
        np.maximum(a_without, 0.0, out=a_without)

        valid = (p_without > 1e-24) & (a_without > 1e-24)
        np.multiply(p_without, a_without, out=p_without)

        np.square(dot_without, out=dot_without)
        np.divide(
            dot_without,
            p_without,
            out=dot_without,
            where=valid,
        )

        influences = np.full(len(gene_ids), np.nan)
        influences[valid] = frac_systematic - dot_without[valid]

    else:
        cos_sim = np.nan
        norm_delta_specific = np.nan
        frac_systematic = np.nan
        valid = None
        influences = None

    row = {
        "condition": condition,
        "context": context,
        "perturbation": perturbation,
        "n_stim_cells": n_stim,
        "n_perts_in_context": n_perts,
        "cos_sim": cos_sim,
        "norm_delta_p": norm_delta_p,
        "norm_delta_avg": norm_delta_avg,
        "norm_delta_specific": norm_delta_specific,
        "frac_systematic_var": frac_systematic,
    }

    return row, influences, valid


class SystematicStep(BaseSystematicStep):

    def __init__(self, args: argparse.Namespace, regression: bool = False) -> None:
        """Initializer."""
        super().__init__(args=args)

        if regression:
            self.datasets = list(CELLULAR_DATASETS.keys())
        else:
            self.datasets = ['tahoe100m']

        self.regression = regression

    def summarize_tahoe100m_by_group(
        self,
        mode: str,
        df: pd.DataFrame,
        cur_time: str,
    ) -> None:
        """Summarize Tahoe-100m merged results by specific groupings."""
        df = df.copy()

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

        # Load drug mappings for MOA grouping
        moa_mapping = self.load_drug_moa_mapping()

        # Map each drug to its corresponding MOA
        df["moa"] = (
            df["drug"]
            .astype(str)
            .str.strip()
            .map(moa_mapping)
        )

        # Define the groupings
        groupings = {
            "cell_line_id": ["cell_line_id"],
            "drug": ["drug"],
            "dosage": ["dosage"],
            "drug_x_dosage": ["drug", "dosage"],
            "moa": ['moa']
        }

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

            summary = (
                df.groupby(group_cols, dropna=False)
                .agg(agg)
            )

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

                condition_keys = [
                    "condition",
                    *group_cols,
                ]
                condition_triage = df[
                    condition_keys + ["triage_2d"]
                ].drop_duplicates()

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

            # Save one CSV for this grouping.
            out_path = (
                self.output_dir
                / f"systema_{mode}_cc_reliability_2d_summary_tahoe100m_"
                  f"by_{grouping_name}_{cur_time}.csv"
            )

            summary.to_csv(out_path, index=False)
            log.info("Saved %s summary: %s", grouping_name, out_path)

    def compute_systema_metrics(
        self, ds_name: str, mode: str, systema_dir: Path, cur_time: str
    ) -> pd.DataFrame:
        """
        Compute Systema's systematic variation metric for all (context, perturbation)
        conditions in a cellular context dataset.

        Key properties of this implementation:
          • δ_avg is computed PER CONTEXT (not global)
          • δ_avg uses cells from RELIABLE perts in that context only
          • cos_sim is computed only for reliable perts; unreliable perts get NaN.

        Returns
        -------
        df : DataFrame with columns:
            dataset, condition, context, perturbation, n_stim_cells,
            cos_sim, norm_delta_p, norm_delta_avg, norm_delta_specific,
            frac_systematic_var
        """
        t_start = time.time()
        log.info(f"Computing systema metrics from {systema_dir}")

        # ---- Load reliability (used to filter which perts define δ_avg) ----
        log.info("  [0/3] Loading reliability data for stim data...")
        reliable_conds, all_conds = load_reliable_conditions(
            ds_name, self.reliability_dir, self.clusters
        )

        # ---- Load precomputed delta values for each (reliable) cell line ----
        log.info("  [1/3] Loading precomputed delta_p values for reliable data...")
        systema_data = load_systema_deltas(
            ds_name=ds_name,
            systema_dir=systema_dir,
            target_conds=all_conds,
            clusters=self.clusters
        )

        # ---- Compute Systema per cell line ----
        log.info("  [2/3] Computing cosine similarities and influence scores per cell line...")
        all_rows = []
        all_influence_rows_by_cell_line = {}
        all_influence_rows_by_moa = {}
        drug_to_moa = self.load_drug_moa_mapping()

        total_conditions = 0
        total_skipped = 0

        for cell_line_id, data in tqdm(
            systema_data.items(),
            desc='Processing cell lines',
            unit='cell_line'
        ):
            conditions = data["conditions"]
            n_stim_cells = data["n_stim_cells"]
            delta_p_all = data["delta_p"]

            n_perts = len(conditions)

            if n_perts == 0:
                log.info(
                    f"Cell line '{cell_line_id}': no perturbations, skipping"
                )
                continue

            # gene accumulator for influence scores
            gene_ids = load_gene_ids(self.ctrl_gene_expressions_dir, cell_line_id)
            acc = InfluenceAccumulator(gene_ids)
            moa_accs = {}

            # Group conditions by context.
            if ds_name == 'tahoe100m':
                contexts = np.asarray([
                    condition.split("|")[0]
                    for condition in conditions
                ])
            else:
                contexts = np.asarray([
                    condition.split("|")[0]
                    for condition in conditions
                ])

            for context_group in np.unique(contexts):
                context_mask = contexts == context_group

                context_conditions = conditions[context_mask]
                context_weights = n_stim_cells[context_mask]
                context_deltas = delta_p_all[context_mask]

                # ---- Step 1: identify reliable perturbations ----
                reliable_mask = np.array(
                    [condition in reliable_conds for condition in context_conditions],
                    dtype=bool,
                )
                n_reliable = reliable_mask.sum()

                # ---- Step 2: δ_avg from reliable perturbations ----
                if n_reliable == 0:
                    delta_avg = np.zeros(
                        delta_p_all.shape[1],
                        dtype=np.float64,
                    )
                    norm_delta_avg = 0.0
                else:
                    reliable_deltas = context_deltas[reliable_mask]
                    reliable_weights = context_weights[reliable_mask]

                    # Cell-weighted average perturbation effect.
                    delta_avg = np.average(
                        reliable_deltas.astype(np.float64),
                        axis=0,
                        weights=reliable_weights,
                    )

                    norm_delta_avg = float(np.linalg.norm(delta_avg))

                    # ---- Step 3: score each perturbation ----
                    for condition, n_stim, delta_p, is_rel in zip(
                        context_conditions,
                        context_weights,
                        context_deltas,
                        reliable_mask,
                    ):
                        row, influences, valid = score_perturbation(
                            ds_name,
                            condition,
                            n_perts,
                            n_stim,
                            delta_p,
                            delta_avg,
                            norm_delta_avg,
                            is_rel,
                            gene_ids,
                        )

                        if row is None:
                            total_skipped += 1
                            continue

                        total_conditions += 1
                        all_rows.append(row)

                        if influences is not None and valid is not None:
                            acc.accumulate(
                                cell_line_id,
                                gene_ids,
                                delta_p,
                                influences,
                                valid,
                                row['frac_systematic_var'],
                            )

                            # also group by MOA
                            moa = drug_to_moa.get(condition.split("|")[1])
                            if moa not in moa_accs:
                                moa_accs[moa] = InfluenceAccumulator(gene_ids)

                            moa_accs[moa].accumulate(
                                cell_line_id,
                                gene_ids,
                                delta_p,
                                influences,
                                valid,
                                row['frac_systematic_var'],
                            )

            # aggregate accumulator results by cell line and MOA
            # by cell line
            all_influence_rows_by_cell_line[cell_line_id] = acc.finalize_by_gene()

            # by MOA
            for moa, moa_acc in moa_accs.items():
                moa_gene_rows = moa_acc.finalize_by_gene()

                if moa not in all_influence_rows_by_moa:
                    all_influence_rows_by_moa[moa] = {}

                for gene_id, influence in moa_gene_rows.items():

                    if gene_id not in all_influence_rows_by_moa[moa]:
                        all_influence_rows_by_moa[moa][gene_id] = {
                            key: 0
                            for key in influence
                            if key != 'cell_lines'
                        }
                        all_influence_rows_by_moa[moa][gene_id]['cell_lines'] = set()

                    global_gene = all_influence_rows_by_moa[moa][gene_id]

                    for key, value in influence.items():
                        if key == 'cell_lines':
                            global_gene['cell_lines'].update(value)
                        else:
                            global_gene[key] += value

        # merge gene influence across cell lines, and also save top genes per cell line
        global_influence_rows = {}
        top_genes_by_cell_line_df = pd.DataFrame()
        n_genes_per_grouping = 200

        for cell_line_id, gene_rows in all_influence_rows_by_cell_line.items():

            # global accumulation
            for gene_id, influence in gene_rows.items():

                if gene_id not in global_influence_rows:
                    global_influence_rows[gene_id] = {
                        key: 0
                        for key in influence
                        if key != 'cell_lines'
                    }
                    global_influence_rows[gene_id]['cell_lines'] = set()

                global_gene = global_influence_rows[gene_id]

                # Sum all additive metrics
                for key, value in influence.items():
                    if key == 'cell_lines':
                        global_gene['cell_lines'].update(value)
                    else:
                        global_gene[key] += value

            # rank genes within this cell line
            cell_line_influence_summary = []
            for gene_id, influence in gene_rows.items():
                row = summarize_perturbation_influence(gene_id, influence)
                row['cell_line_id'] = cell_line_id
                cell_line_influence_summary.append(row)
            
            # keep top genes per cell line
            cell_line_df = pd.DataFrame(cell_line_influence_summary).nlargest(n_genes_per_grouping, 'mean_abs_influence')
            cell_line_df = cell_line_df.merge(
                self.gene_metadata[['token_id', 'gene_symbol', 'ensembl_id']],
                on='token_id',
                how='left',
                validate='many_to_one',
            )
            
            top_genes_by_cell_line_df = pd.concat(
                [top_genes_by_cell_line_df, cell_line_df],
                ignore_index=True,
            )

        del all_influence_rows_by_cell_line

        top_genes_by_cell_line_df.to_csv(
            self.output_dir / f'systema_{mode}_top{n_genes_per_grouping}_influential_genes_by_cell_line_id_{cur_time}.csv',
            index=False
        )
        del top_genes_by_cell_line_df

        # by MOA
        top_genes_by_moa_df = pd.DataFrame()
        for moa, gene_rows in all_influence_rows_by_moa.items():
            moa_influence_summary = []
            for gene_id, influence in gene_rows.items():
                row = summarize_perturbation_influence(gene_id, influence)
                row['moa'] = moa
                moa_influence_summary.append(row)

            # keep top genes per MOA
            moa_df = pd.DataFrame(moa_influence_summary).nlargest(n_genes_per_grouping, 'mean_abs_influence')
            moa_df = moa_df.merge(
                self.gene_metadata[['token_id', 'gene_symbol', 'ensembl_id']],
                on='token_id',
                how='left',
                validate='many_to_one',
            )

            top_genes_by_moa_df = pd.concat(
                [top_genes_by_moa_df, moa_df],
                ignore_index=True,
            )
        del all_influence_rows_by_moa

        top_genes_by_moa_df.to_csv(
            self.output_dir / f'systema_{mode}_top{n_genes_per_grouping}_influential_genes_by_moa_{cur_time}.csv',
            index=False
        )
        del top_genes_by_moa_df

        # Build influence summary
        influence_summary = []

        for gene_id, influence in global_influence_rows.items():
            influence_summary.append(summarize_perturbation_influence(gene_id, influence))

        del global_influence_rows

        influence_df = pd.DataFrame(influence_summary)
        # map to gene_metadata
        influence_df = influence_df.merge(
            self.gene_metadata[['token_id', 'gene_symbol', 'ensembl_id']],
            on='token_id',
            how='left',
            validate='many_to_one',
        )

        # Keep the top 1000 genes by mean absolute influence
        influence_df.nlargest(1000, 'mean_abs_influence').to_csv(
            self.output_dir / f'systema_{mode}_top1000_influential_genes_{cur_time}.csv',
            index=False
        )
        del influence_summary

        # Build DataFrames
        df = pd.DataFrame(all_rows)
        del all_rows

        if len(df) == 0:
            raise ValueError(f"  ⚠ No results for Tahoe-100m")

        # ---- Summary statistics ----
        elapsed_total = time.time() - t_start
        log.info(f"\n  ── Summary for Tahoe-100m ──")
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
        return df

    def merge_with_reliability(
        self,
        ds_name: str,
        systema_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Systema cos_sim with pre-computed split-half reliability
        from the cellular context reliability script.

        Merges on the 'condition' column (format: "context|perturbation").
        """
        rel_df = get_combined_split_half_reliability_df(
            self.reliability_dir, self.clusters, summary=True, ds_name=ds_name
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

        return merged

    def run(self) -> None:
        """Run the full pipeline."""
        log.info(f"Systema Systematic Variation + Reliability 2D (Cellular Context)")
        log.info(f"  Clusters:\t{', '.join(self.clusters) if self.clusters else 'all'}")
        log.info(f"  Reliability dir:\t{self.reliability_dir}")
        log.info(f"  Output:\t{self.output_dir}")

        cur_time = time.strftime('%Y%m%d-%H%M%S')
        all_summary_rows = {}

        try:
            for mode, systema_dir in self.mode_mapping.items():
                if not systema_dir or not systema_dir.is_dir():
                    log.warning(f"{mode} dir not found, skipping mode")
                    continue

                log.info(f"Evaluating {mode}")
                all_summary_rows[mode] = []

                for ds_name in self.datasets:
                    systema_df = self.compute_systema_metrics(ds_name, mode, systema_dir, cur_time)

                    # Save Systema-only results
                    systema_path = self.output_dir / f"systema_{mode}_{ds_name}_{cur_time}.csv"
                    systema_df.to_csv(systema_path, index=False)
                    log.info(f"\n  Saved: {systema_path}")

                    # ---- Merge with reliability ----
                    merged_df = self.merge_with_reliability(ds_name, systema_df)

                    merged_path = self.output_dir / f"systema_{mode}_cc_reliability_2d_{ds_name}_{cur_time}.csv"
                    merged_df.to_csv(merged_path, index=False)
                    log.info(f"  Saved: {merged_path}")

                    if ds_name == 'tahoe100m':
                        self.summarize_tahoe100m_by_group(mode, merged_df, cur_time)

                    # ---- Cross-dataset summary row ----
                    summary_row = {
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

        except Exception as e:
            log.info(f"\n  ✗ FAILED: {e}")
            raise e

        # ---- Cross-dataset summary ----
        for mode, summary_rows in all_summary_rows.items():
            summary_df = pd.DataFrame(summary_rows)

            if self.regression:
                summary_path = self.output_dir / f"systema_{mode}_cc_summary_regression_{cur_time}.csv"
            else:
                summary_path = self.output_dir / f"systema_{mode}_cc_summary_tahoe100m_{cur_time}.csv"

            summary_df.to_csv(summary_path, index=False)

            log.info(f"\n{'='*70}")
            log.info(f"  Cross-Dataset Summary (Cellular Context), mode {mode}")
            log.info(f"{'='*70}")
            log.info(summary_df.to_string(index=False))
            log.info(f"\n  Saved: {summary_path}")

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
