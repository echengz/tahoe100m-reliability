
"""
Runs step 3 of the pipeline: systematic.

Preserves the original leave-one-gene-out analysis and adds:
    1. Pathway-level influence (remove each significant MOA GSEA pathway).
    2. Union-level influence (remove all significant pathway lead genes together).
"""
import argparse
import logging
import numpy as np
import pandas as pd
import time

from pathlib import Path
from tqdm import tqdm

from tahoe100m_reliability.lib.analysis.influence_accumulator import (
    InfluenceAccumulator,
    GroupInfluenceAccumulator,
    summarize_perturbation_influence,
)
from tahoe100m_reliability.lib.ingest.intermediate_loaders import (
    get_combined_cluster_split_half_reliability_df,
    load_delta_p_results,
    load_gene_ids,
    load_reliable_conditions,
    RELIABILITY_THRESH,
    SPECIFICITY_THRESH,
    REL_COL_FALLBACKS,
)
from tahoe100m_reliability.lib.step.base_systematic_step import BaseSystematicStep

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


def calculate_frac_systematic(delta_p, delta_avg, norm_delta_avg, is_rel):
    """Calculate baseline systematic fraction and associated metrics."""
    norm_delta_p = float(np.linalg.norm(delta_p))

    if is_rel and norm_delta_avg > 1e-12:
        cos_sim = cosine_similarity(delta_p, delta_avg)

        proj_scalar = (
            np.dot(delta_p, delta_avg) / (norm_delta_avg ** 2)
        )

        delta_specific = delta_p - proj_scalar * delta_avg
        norm_delta_specific = float(np.linalg.norm(delta_specific))

        frac_systematic = (
            float(cos_sim ** 2)
            if norm_delta_p > 1e-12
            else np.nan
        )
    else:
        cos_sim = np.nan
        norm_delta_specific = np.nan
        frac_systematic = np.nan

    return (
        cos_sim,
        norm_delta_p,
        norm_delta_specific,
        frac_systematic,
    )


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
    """
    Score a perturbation and calculate the original vectorized
    leave-one-gene-out influence scores.

    The gene-level calculation is deliberately preserved:
        influence_g = phi_before - phi_without_gene_g
    """
    condition = str(condition)
    n_stim = int(n_stim)

    if n_stim < 4:
        return None, None, None

    parts = condition.split("|")

    if ds_name == "tahoe100m":
        if len(parts) != 3:
            raise ValueError(
                f"Unexpected condition format: {condition}"
            )
        context, drug, dosage = parts
        perturbation = f"{drug}|{dosage}"
    else:
        if len(parts) != 2:
            raise ValueError(
                f"Unexpected condition format: {condition}"
            )
        context, perturbation = parts

    delta_p = np.asarray(delta_p, dtype=np.float64)
    delta_avg = np.asarray(delta_avg, dtype=np.float64)

    dot_pa = np.dot(delta_p, delta_avg)
    norm_p_sq = np.dot(delta_p, delta_p)
    norm_a_sq = norm_delta_avg ** 2
    norm_delta_p = np.sqrt(norm_p_sq)

    if (
        is_rel
        and norm_a_sq > 1e-24
        and norm_delta_p > 1e-12
    ):
        cos_sim = dot_pa / (norm_delta_p * norm_delta_avg)
        frac_systematic = cos_sim ** 2

        norm_delta_specific = np.sqrt(
            max(norm_p_sq - dot_pa ** 2 / norm_a_sq, 0.0)
        )

        # Original vectorized leave-one-gene-out calculation.
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

        valid = (
            (p_without > 1e-24)
            & (a_without > 1e-24)
        )

        np.multiply(p_without, a_without, out=p_without)
        np.square(dot_without, out=dot_without)

        np.divide(
            dot_without,
            p_without,
            out=dot_without,
            where=valid,
        )

        influences = np.full(len(gene_ids), np.nan)
        influences[valid] = (
            frac_systematic - dot_without[valid]
        )

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


def _is_significant(series):
    """Interpret common CSV boolean representations."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def _merge_accumulator_rows(target, source):
    """Merge additive accumulator summaries keyed by group/gene ID."""
    for key, values in source.items():
        if key not in target:
            target[key] = {
                field: value
                for field, value in values.items()
                if field != "cell_lines"
            }
            target[key]["cell_lines"] = set(values.get("cell_lines", set()))
            continue

        existing = target[key]

        for field, value in values.items():
            if field == "cell_lines":
                existing["cell_lines"].update(value)
            elif field == "gene_ids":
                existing["gene_ids"] = sorted(
                    set(existing.get("gene_ids", [])) | set(value)
                )
            else:
                existing[field] += value


def _summarize_group_rows(group_rows, group_name):
    """Convert group accumulator output into a summary DataFrame."""
    rows = []

    for group_id, values in group_rows.items():
        row = summarize_perturbation_influence(group_id, values)

        row["group_id"] = group_id
        row["gene_ids"] = ";".join(
            str(gene) for gene in values.get("gene_ids", [])
        )
        row["cell_lines"] = ";".join(
            sorted(values.get("cell_lines", set()))
        )
        row["group_type"] = group_name

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Group IDs are pathway terms or the union label, not gene token IDs.
    return df.drop(columns=["token_id"], errors="ignore")


class SystematicStep(BaseSystematicStep):

    def __init__(
        self,
        args: argparse.Namespace,
        regression: bool = False,
    ) -> None:
        super().__init__(args=args)

        if regression:
            # Keep the dataset selection supplied by the base pipeline.
            self.datasets = getattr(
                self, "datasets", ["tahoe100m"]
            )
        else:
            self.datasets = ["tahoe100m"]

        self.regression = regression

        # Keep both existing CLI arguments.
        self.gsea_global_path = args.gsea_global_shared_axis_hallmark
        self.gsea_moa_path = args.gsea_moa_shared_axis_hallmark

    def load_moa_gsea_pathways(self, mode):
        """
        Load significant pathway results from the MOA GSEA file.

        Expected columns:
            mode, cell_line, moa, term, lead_genes, significant

        lead_genes is expected to be semicolon-separated gene symbols.
        """
        required = [
            "mode",
            "cell_line",
            "moa",
            "term",
            "lead_genes",
            "significant",
        ]

        if self.gsea_moa_path is None:
            log.info("No MOA GSEA file supplied; skipping group influence analyses")
            return pd.DataFrame(columns=required)

        if not self.gsea_moa_path.is_file():
            log.warning(
                "MOA GSEA file does not exist: %s",
                self.gsea_moa_path,
            )
            return pd.DataFrame(columns=required)

        df = pd.read_csv(self.gsea_moa_path)

        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(
                f"MOA GSEA file is missing required columns: {sorted(missing)}"
            )

        df = df.loc[
            (df["mode"].astype(str) == str(mode))
            # & _is_significant(df["significant"])
        ].copy()

        df["lead_genes"] = df["lead_genes"].fillna("").astype(str)
        df["term"] = df["term"].fillna("").astype(str)
        df["moa"] = df["moa"].fillna("").astype(str)
        df["cell_line"] = df["cell_line"].fillna("").astype(str)

        return df

    def build_pathway_masks(
        self,
        cell_line_id,
        gene_ids,
        mode,
        gsea_df,
    ):
        """
        Build pathway and union masks aligned to delta_p gene ordering.

        Returns
        -------
        pathway_data:
            dict keyed by MOA. Each entry contains pathway IDs, gene IDs,
            and a boolean pathway mask.
        """
        if gsea_df.empty:
            return {}

        # Match GSEA cell lines to the cell line currently being processed.
        cell_mask = (
            gsea_df["cell_line"].astype(str)
            == str(cell_line_id)
        )
        cell_df = gsea_df.loc[cell_mask]

        if cell_df.empty:
            return {}

        metadata = self.gene_metadata[
            ["gene_symbol", "token_id"]
        ].dropna().drop_duplicates()

        symbol_to_token = dict(
            zip(
                metadata["gene_symbol"].astype(str),
                metadata["token_id"],
            )
        )

        token_to_index = {
            token.item() if hasattr(token, "item") else token: i
            for i, token in enumerate(gene_ids)
        }

        pathway_data = {}

        for moa, moa_df in cell_df.groupby("moa", sort=False):
            pathway_gene_sets = {}

            for _, pathway_row in moa_df.iterrows():
                term = str(pathway_row["term"]).strip()
                if not term:
                    continue

                symbols = {
                    symbol.strip()
                    for symbol in str(pathway_row["lead_genes"]).split(";")
                    if symbol.strip()
                }

                tokens = {
                    symbol_to_token[symbol]
                    for symbol in symbols
                    if symbol in symbol_to_token
                }

                indices = {
                    token_to_index[
                        token.item() if hasattr(token, "item") else token
                    ]
                    for token in tokens
                    if (
                        token.item() if hasattr(token, "item") else token
                    ) in token_to_index
                }

                # Merge duplicate pathway terms rather than overwriting them.
                pathway_gene_sets.setdefault(term, set()).update(indices)

            if not pathway_gene_sets:
                continue

            group_ids = list(pathway_gene_sets.keys())
            group_gene_ids = []
            pathway_mask = np.zeros(
                (len(group_ids), len(gene_ids)),
                dtype=bool,
            )

            for i, group_id in enumerate(group_ids):
                indices = sorted(pathway_gene_sets[group_id])
                pathway_mask[i, indices] = True
                group_gene_ids.append(
                    [gene_ids[index].item() for index in indices]
                )

            # Union of all significant pathway lead genes for this MOA.
            union_indices = sorted(
                set().union(*pathway_gene_sets.values())
            )
            union_mask = np.zeros(
                (1, len(gene_ids)),
                dtype=bool,
            )
            union_mask[0, union_indices] = True

            union_gene_ids = [
                gene_ids[index].item()
                for index in union_indices
            ]

            pathway_data[moa] = {
                "group_ids": group_ids,
                "group_gene_ids": group_gene_ids,
                "mask": pathway_mask,
                "union_gene_ids": union_gene_ids,
                "union_mask": union_mask,
            }

        return pathway_data

    def summarize_gene_rows(
        self,
        gene_rows,
        extra_fields=None,
        n_genes_per_grouping=200,
    ):
        """Summarize gene-level LOO results and attach gene metadata."""
        rows = []

        for gene_id, influence in gene_rows.items():
            row = summarize_perturbation_influence(
                gene_id, influence
            )
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        df = df.merge(
            self.gene_metadata[
                ["token_id", "gene_symbol", "ensembl_id"]
            ],
            on="token_id",
            how="left",
            validate="many_to_one",
        )

        df["influence_rank"] = (
            df["mean_abs_influence"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        if extra_fields:
            for key, value in extra_fields.items():
                df[key] = value

        return df.nlargest(
            n_genes_per_grouping,
            "mean_abs_influence",
        )

    def summarize_tahoe100m_by_group(
        self,
        mode,
        df,
        cur_time,
    ):
        """Summarize Tahoe-100m merged results by specific groupings."""
        df = df.copy()

        parts = df["condition"].astype(str).str.split(
            "|", regex=False, expand=True
        )

        if parts.shape[1] != 3:
            raise ValueError(
                "Expected 3 condition components "
                "(cell_line|drug|dosage), "
                f"found {parts.shape[1]}"
            )

        df["cell_line_id"] = parts[0]
        df["drug"] = parts[1]
        df["dosage"] = parts[2]

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

        moa_mapping = self.load_drug_moa_mapping()
        df["moa"] = (
            df["drug"].astype(str).str.strip().map(moa_mapping)
        )

        groupings = {
            "cell_line_id": ["cell_line_id"],
            "drug": ["drug"],
            "dosage": ["dosage"],
            "drug_x_dosage": ["drug", "dosage"],
            "moa": ["moa"],
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

            summary = df.groupby(
                group_cols, dropna=False
            ).agg(agg)

            summary.columns = [
                "_".join(str(x) for x in col if x)
                for col in summary.columns
            ]
            summary = summary.reset_index()

            summary = summary.rename(columns={
                "condition_nunique": "n_conditions",
                "cos_sim_count": "n_scored",
                "n_stim_cells_sum": "total_stim_cells",
                "n_stim_cells_median": "median_stim_cells",
            })

            if "triage_2d" in df.columns:
                categories = ["Unreliable", "Shared", "Specific"]
                condition_keys = ["condition", *group_cols]

                condition_triage = df[
                    condition_keys + ["triage_2d"]
                ].drop_duplicates()

                labels_per_condition = (
                    condition_triage.groupby(
                        condition_keys, dropna=False
                    )["triage_2d"].nunique(dropna=False)
                )

                if (labels_per_condition > 1).any():
                    log.warning(
                        "%d conditions have conflicting triage labels "
                        "for grouping %s",
                        int((labels_per_condition > 1).sum()),
                        grouping_name,
                    )

                condition_triage = condition_triage[
                    condition_triage["triage_2d"].isin(categories)
                ].drop_duplicates(condition_keys)

                counts = (
                    condition_triage.groupby(
                        group_cols, dropna=False
                    )["triage_2d"]
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

                summary = summary.merge(
                    counts.reset_index(),
                    on=group_cols,
                    how="left",
                )

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

            out_path = (
                self.output_dir
                / f"systema_{mode}_cc_reliability_2d_summary_tahoe100m_"
                  f"by_{grouping_name}_{cur_time}.csv"
            )
            summary.to_csv(out_path, index=False)
            log.info("Saved %s summary: %s", grouping_name, out_path)

    def compute_systema_metrics(
        self,
        ds_name,
        mode,
        delta_p_dir,
        reliability_dir,
        cur_time,
    ):
        """Compute baseline Systema metrics and all influence analyses."""
        t_start = time.time()
        log.info("Computing Systema metrics from %s", delta_p_dir)

        log.info("[0/3] Loading reliability data...")
        reliable_conds, all_conds = load_reliable_conditions(
            ds_name, reliability_dir, self.clusters
        )

        log.info("[1/3] Loading delta_p values...")
        delta_p_data = load_delta_p_results(
            ds_name=ds_name,
            delta_p_dir=delta_p_dir,
            target_conds=all_conds,
            clusters=self.clusters,
        )

        log.info("[2/3] Computing metrics and influence scores...")

        gsea_df = self.load_moa_gsea_pathways(mode)
        drug_to_moa = self.load_drug_moa_mapping()

        all_rows = []
        all_influence_rows_by_cell_line = {}
        all_influence_rows_by_moa = {}

        # New group-level results are merged across cell lines.
        all_pathway_rows_by_moa = {}
        all_union_rows_by_moa = {}

        by_cell_line_moa_dir = (
            self.output_dir / "influence_by_cell_line_moa"
        )
        by_cell_line_moa_dir.mkdir(
            parents=True, exist_ok=True
        )

        total_conditions = 0
        total_skipped = 0

        for cell_line_id, data in tqdm(
            delta_p_data.items(),
            desc="Processing cell lines",
            unit="cell_line",
        ):
            conditions = data["conditions"]
            n_stim_cells = data["n_stim_cells"]
            delta_p_all = data["delta_p"]
            n_perts = len(conditions)

            if n_perts == 0:
                continue

            gene_ids = np.asarray(
                load_gene_ids(
                    self.ctrl_gene_expressions_dir,
                    cell_line_id,
                )
            )

            if delta_p_all.shape[1] != len(gene_ids):
                raise ValueError(
                    f"{cell_line_id}: delta_p has "
                    f"{delta_p_all.shape[1]} genes but load_gene_ids "
                    f"returned {len(gene_ids)} IDs"
                )

            # Original leave-one-gene-out accumulators.
            gene_acc = InfluenceAccumulator(gene_ids)
            moa_gene_accs = {}

            # New pathway and union accumulators for this cell line.
            pathway_data = self.build_pathway_masks(
                cell_line_id,
                gene_ids,
                mode,
                gsea_df,
            )

            pathway_accs = {}
            union_accs = {}

            for moa, group_data in pathway_data.items():
                pathway_accs[moa] = GroupInfluenceAccumulator(
                    group_data["group_ids"],
                    group_data["group_gene_ids"],
                )

                union_accs[moa] = GroupInfluenceAccumulator(
                    ["all_significant_pathway_lead_genes"],
                    [group_data["union_gene_ids"]],
                )

            contexts = np.asarray([
                str(condition).split("|")[0]
                for condition in conditions
            ])

            for context_group in np.unique(contexts):
                context_mask = contexts == context_group

                context_conditions = conditions[context_mask]
                context_weights = n_stim_cells[context_mask]
                context_deltas = delta_p_all[context_mask]

                reliable_mask = np.asarray([
                    condition in reliable_conds
                    for condition in context_conditions
                ], dtype=bool)

                n_reliable = int(reliable_mask.sum())

                if n_reliable == 0:
                    delta_avg = np.zeros(
                        delta_p_all.shape[1],
                        dtype=np.float64,
                    )
                    norm_delta_avg = 0.0
                    continue

                reliable_deltas = context_deltas[reliable_mask]
                reliable_weights = context_weights[reliable_mask]

                delta_avg = np.average(
                    reliable_deltas.astype(np.float64),
                    axis=0,
                    weights=reliable_weights,
                )
                norm_delta_avg = float(np.linalg.norm(delta_avg))

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

                    # Preserve original LOO gene analysis.
                    if influences is None or valid is None:
                        continue

                    gene_acc.accumulate(
                        cell_line_id,
                        gene_ids,
                        delta_p,
                        influences,
                        valid,
                        row["frac_systematic_var"],
                    )

                    parts = str(condition).split("|")
                    drug = parts[1] if len(parts) > 1 else None
                    moa = drug_to_moa.get(drug)

                    if moa not in moa_gene_accs:
                        moa_gene_accs[moa] = InfluenceAccumulator(
                            gene_ids
                        )

                    moa_gene_accs[moa].accumulate(
                        cell_line_id,
                        gene_ids,
                        delta_p,
                        influences,
                        valid,
                        row["frac_systematic_var"],
                    )

                    # New pathway and union analyses.
                    if moa not in pathway_data:
                        continue

                    group_data = pathway_data[moa]

                    pathway_influences = (
                        GroupInfluenceAccumulator.calculate_group_influences(
                            delta_p,
                            delta_avg,
                            group_data["mask"],
                        )
                    )

                    pathway_valid = np.isfinite(pathway_influences)
                    n_pathways = len(group_data["group_ids"])

                    pathway_accs[moa].accumulate(
                        cell_line_id,
                        group_data["group_ids"],
                        delta_p,
                        group_data["mask"],
                        pathway_influences,
                        pathway_valid,
                        np.full(
                            n_pathways,
                            row["frac_systematic_var"],
                        ),
                    )

                    union_influence = (
                        GroupInfluenceAccumulator.calculate_group_influences(
                            delta_p,
                            delta_avg,
                            group_data["union_mask"],
                        )
                    )

                    union_accs[moa].accumulate(
                        cell_line_id,
                        ["all_significant_pathway_lead_genes"],
                        delta_p,
                        group_data["union_mask"],
                        union_influence,
                        np.isfinite(union_influence),
                        np.asarray([
                            row["frac_systematic_var"]
                        ]),
                    )

            # Preserve original cell-line gene-level output.
            cell_line_gene_rows = gene_acc.finalize_by_gene()
            all_influence_rows_by_cell_line[cell_line_id] = (
                cell_line_gene_rows
            )

            # Preserve original MOA gene-level accumulation.
            for moa, moa_acc in moa_gene_accs.items():
                moa_gene_rows = moa_acc.finalize_by_gene()

                if moa not in all_influence_rows_by_moa:
                    all_influence_rows_by_moa[moa] = {}

                _merge_accumulator_rows(
                    all_influence_rows_by_moa[moa],
                    moa_gene_rows,
                )

                # cell_line_moa_df = self.summarize_gene_rows(
                #     moa_gene_rows,
                #     extra_fields={
                #         "cell_line_id": cell_line_id,
                #         "moa": moa,
                #     },
                # )

                # cell_line_moa_df.to_csv(
                #     by_cell_line_moa_dir
                #     / f"{mode}_{cell_line_id}_moa_{moa}_"
                #       f"top200_influential_genes_{cur_time}.csv",
                #     index=False,
                # )

            # Merge pathway results across cell lines.
            for moa, pathway_acc in pathway_accs.items():
                pathway_rows = pathway_acc.finalize_by_group()

                if moa not in all_pathway_rows_by_moa:
                    all_pathway_rows_by_moa[moa] = {}

                _merge_accumulator_rows(
                    all_pathway_rows_by_moa[moa],
                    pathway_rows,
                )

            # Merge union results across cell lines.
            for moa, union_acc in union_accs.items():
                union_rows = union_acc.finalize_by_group()

                if moa not in all_union_rows_by_moa:
                    all_union_rows_by_moa[moa] = {}

                _merge_accumulator_rows(
                    all_union_rows_by_moa[moa],
                    union_rows,
                )

        # ------------------------------------------------------------
        # Original gene-level outputs.
        # ------------------------------------------------------------
        global_gene_rows = {}

        for gene_rows in all_influence_rows_by_cell_line.values():
            _merge_accumulator_rows(
                global_gene_rows,
                gene_rows,
            )

        top_genes_by_cell_line_df = pd.concat(
            [
                self.summarize_gene_rows(
                    gene_rows,
                    extra_fields={"cell_line_id": cell_line_id},
                )
                for cell_line_id, gene_rows
                in all_influence_rows_by_cell_line.items()
                if gene_rows
            ],
            ignore_index=True,
        ) if all_influence_rows_by_cell_line else pd.DataFrame()

        top_genes_by_cell_line_df.to_csv(
            self.output_dir
            / f"systema_{mode}_top200_influential_genes_"
              f"by_cell_line_id_{cur_time}.csv",
            index=False,
        )

        top_genes_by_moa_df = pd.concat(
            [
                self.summarize_gene_rows(
                    gene_rows,
                    extra_fields={"moa": moa},
                )
                for moa, gene_rows in all_influence_rows_by_moa.items()
                if gene_rows
            ],
            ignore_index=True,
        ) if all_influence_rows_by_moa else pd.DataFrame()

        top_genes_by_moa_df.to_csv(
            self.output_dir
            / f"systema_{mode}_top200_influential_genes_"
              f"by_moa_{cur_time}.csv",
            index=False,
        )

        global_gene_df = self.summarize_gene_rows(
            global_gene_rows,
            n_genes_per_grouping=1000,
        )

        global_gene_df.to_csv(
            self.output_dir
            / f"systema_{mode}_top1000_influential_genes_{cur_time}.csv",
            index=False,
        )

        # ------------------------------------------------------------
        # New pathway-level output: one row per MOA x pathway.
        # ------------------------------------------------------------
        pathway_dfs = []

        for moa, group_rows in all_pathway_rows_by_moa.items():
            df_moa = _summarize_group_rows(
                group_rows,
                "pathway",
            )
            if df_moa.empty:
                continue

            df_moa["moa"] = moa
            df_moa["pathway"] = df_moa["group_id"]
            pathway_dfs.append(df_moa)

        pathway_summary = (
            pd.concat(pathway_dfs, ignore_index=True)
            if pathway_dfs
            else pd.DataFrame()
        )

        pathway_summary.to_csv(
            self.output_dir
            / f"systema_{mode}_pathway_influence_by_moa_{cur_time}.csv",
            index=False,
        )

        # ------------------------------------------------------------
        # New union-level output: one row per MOA.
        # ------------------------------------------------------------
        union_dfs = []

        for moa, group_rows in all_union_rows_by_moa.items():
            df_moa = _summarize_group_rows(
                group_rows,
                "pathway_union",
            )
            if df_moa.empty:
                continue

            df_moa["moa"] = moa
            union_dfs.append(df_moa)

        union_summary = (
            pd.concat(union_dfs, ignore_index=True)
            if union_dfs
            else pd.DataFrame()
        )

        union_summary.to_csv(
            self.output_dir
            / f"systema_{mode}_all_pathway_lead_gene_union_"
              f"influence_by_moa_{cur_time}.csv",
            index=False,
        )

        log.info(
            "Saved pathway influence summary with %d rows",
            len(pathway_summary),
        )
        log.info(
            "Saved pathway-union influence summary with %d rows",
            len(union_summary),
        )

        # ------------------------------------------------------------
        # Baseline Systema results.
        # ------------------------------------------------------------
        df = pd.DataFrame(all_rows)

        if df.empty:
            raise ValueError("No Systema results were computed")

        elapsed_total = time.time() - t_start

        log.info("\n  ── Summary for %s ──", ds_name)
        log.info("  Conditions computed: %d", total_conditions)
        log.info("  Skipped (few cells): %d", total_skipped)
        log.info(
            "  cos_sim — mean: %.3f ± %.3f, median: %.3f",
            df.cos_sim.mean(),
            df.cos_sim.std(),
            df.cos_sim.median(),
        )
        log.info(
            "  cos_sim — min: %.3f, max: %.3f",
            df.cos_sim.min(),
            df.cos_sim.max(),
        )
        log.info(
            "  frac with cos_sim > 0.5: %d/%d (%.1f%%)",
            (df.cos_sim > 0.5).sum(),
            len(df),
            100 * (df.cos_sim > 0.5).mean(),
        )
        log.info(
            "  frac with cos_sim > 0.7: %d/%d (%.1f%%)",
            (df.cos_sim > 0.7).sum(),
            len(df),
            100 * (df.cos_sim > 0.7).mean(),
        )

        log.info("  Per-context cos_sim (median):")
        for ctx, group_df in df.groupby("context"):
            log.info(
                "    %s: %.3f (n=%d, n_perts_in_ctx=%d)",
                ctx,
                group_df.cos_sim.median(),
                len(group_df),
                group_df.n_perts_in_context.iloc[0],
            )

        log.info("  Total time: %.1fs", elapsed_total)
        return df

    def merge_with_reliability(
        self,
        ds_name,
        systema_df,
        reliability_dir,
    ):
        """Merge Systema cosine similarity with split-half reliability."""
        rel_df = get_combined_cluster_split_half_reliability_df(
            reliability_dir,
            self.clusters,
            ds_name=ds_name,
        )

        log.info("Loaded reliability: %d conditions", len(rel_df))

        merged = systema_df.merge(
            rel_df,
            on="condition",
            how="inner",
            suffixes=("", "_rel"),
        )

        log.info(
            "Merged: %d conditions (Systema: %d, Reliability: %d)",
            len(merged),
            len(systema_df),
            len(rel_df),
        )

        if len(merged) < len(systema_df):
            missing = (
                set(systema_df["condition"])
                - set(merged["condition"])
            )
            if len(missing) <= 10:
                log.info("Unmatched conditions: %s", sorted(missing))
            else:
                log.info(
                    "Unmatched conditions: %d (first 5: %s)",
                    len(missing),
                    sorted(missing)[:5],
                )

        rel_col = "sb_from_median_all_genes"
        if rel_col not in merged.columns:
            rel_col = "sb_from_mean_all_genes"

        if rel_col in merged.columns:
            rho = merged[rel_col]

            merged["reliable"] = rho >= RELIABILITY_THRESH
            merged["high_systematic"] = (
                merged["frac_systematic_var"] >= SPECIFICITY_THRESH
            )

            conditions = [
                ~merged["reliable"],
                merged["reliable"] & merged["high_systematic"],
                merged["reliable"] & ~merged["high_systematic"],
            ]
            choices = ["Unreliable", "Shared", "Specific"]

            merged["triage_2d"] = np.select(
                conditions,
                choices,
                default="other",
            )

            log.info("\n  ── 2D Triage Summary ──")
            log.info("  Reliability metric: %s", rel_col)

            for category in ["Unreliable", "Shared", "Specific"]:
                n = (merged["triage_2d"] == category).sum()
                pct = 100 * n / len(merged) if len(merged) else 0
                log.info(
                    "    %s: %d (%.1f%%)",
                    category,
                    n,
                    pct,
                )

            ceiling_col = "ceiling_from_median_all_genes"
            if ceiling_col not in merged.columns:
                ceiling_col = "ceiling_from_mean_all_genes"

            if ceiling_col in merged.columns:
                genuine = merged[
                    merged["triage_2d"] == "Specific"
                ]

                if len(genuine):
                    log.info(
                        "    Genuine signal: mean ceiling = %.3f, "
                        "mean cos_sim = %.3f",
                        genuine[ceiling_col].mean(),
                        genuine["cos_sim"].mean(),
                    )
        else:
            log.warning(
                "Reliability column '%s' not found",
                rel_col,
            )

        return merged

    @classmethod
    def add_cli_args(cls, parser):
        """Add CLI args for this systematic step."""
        BaseSystematicStep.add_cli_args(parser=parser)

        parser.add_argument(
            "--gsea-global-shared-axis-hallmark",
            dest="gsea_global_shared_axis_hallmark",
            type=Path,
            help="Path to global shared-axis GSEA results.",
        )

        parser.add_argument(
            "--gsea-moa-shared-axis-hallmark",
            dest="gsea_moa_shared_axis_hallmark",
            type=Path,
            help=(
                "Path to MOA shared-axis GSEA results. Significant "
                "pathway lead genes are used for pathway and union "
                "influence analyses."
            ),
        )

    def run(self):
        """Run the full systematic pipeline."""
        log.info(
            "Running Systema Systematic Variation + Reliability 2D"
        )

        cur_time = time.strftime("%Y%m%d-%H%M%S")
        all_summary_rows = {}

        try:
            for mode, delta_p_mode in self.mode_mapping.items():
                reliability_dir = delta_p_mode.reliability_dir
                delta_p_dir = delta_p_mode.delta_p_dir

                if not delta_p_dir or not delta_p_dir.is_dir():
                    log.warning(
                        "%s delta_p dir not found, skipping mode",
                        mode,
                    )
                    continue

                log.info("Evaluating %s", mode)
                all_summary_rows[mode] = []

                for ds_name in self.datasets:
                    systema_df = self.compute_systema_metrics(
                        ds_name,
                        mode,
                        delta_p_dir,
                        reliability_dir,
                        cur_time,
                    )

                    systema_path = (
                        self.output_dir
                        / f"systema_{mode}_{ds_name}_{cur_time}.csv"
                    )
                    systema_df.to_csv(systema_path, index=False)
                    log.info("Saved: %s", systema_path)

                    merged_df = self.merge_with_reliability(
                        ds_name,
                        systema_df,
                        reliability_dir,
                    )

                    merged_path = (
                        self.output_dir
                        / f"systema_{mode}_cc_reliability_2d_"
                          f"{ds_name}_{cur_time}.csv"
                    )
                    merged_df.to_csv(merged_path, index=False)
                    log.info("Saved: %s", merged_path)

                    if ds_name == "tahoe100m":
                        self.summarize_tahoe100m_by_group(
                            mode,
                            merged_df,
                            cur_time,
                        )

                    summary_row = {
                        "n_conditions": len(systema_df),
                        "n_contexts": systema_df["context"].nunique(),
                        "n_perturbations": (
                            systema_df["perturbation"].nunique()
                        ),
                        "cos_sim_mean": systema_df.cos_sim.mean(),
                        "cos_sim_std": systema_df.cos_sim.std(),
                        "cos_sim_median": systema_df.cos_sim.median(),
                        "frac_cos_gt_0.5": (
                            systema_df.cos_sim > 0.5
                        ).mean(),
                        "frac_cos_gt_0.7": (
                            systema_df.cos_sim > 0.7
                        ).mean(),
                    }

                    if "triage_2d" in merged_df.columns:
                        for category in [
                            "Unreliable",
                            "Shared",
                            "Specific",
                        ]:
                            n = (
                                merged_df["triage_2d"] == category
                            ).sum()
                            summary_row[f"n_{category}"] = n
                            summary_row[f"frac_{category}"] = (
                                n / len(merged_df)
                                if len(merged_df)
                                else np.nan
                            )

                    all_summary_rows[mode].append(summary_row)

        except Exception:
            log.exception("Systematic step failed")
            raise

        # Cross-dataset summary.
        for mode, summary_rows in all_summary_rows.items():
            summary_df = pd.DataFrame(summary_rows)

            if self.regression:
                summary_path = (
                    self.output_dir
                    / f"systema_{mode}_cc_summary_regression_{cur_time}.csv"
                )
            else:
                summary_path = (
                    self.output_dir
                    / f"systema_{mode}_cc_summary_tahoe100m_{cur_time}.csv"
                )

            summary_df.to_csv(summary_path, index=False)

            log.info("\n%s", "=" * 70)
            log.info(
                "Cross-Dataset Summary (Cellular Context), mode %s",
                mode,
            )
            log.info("%s", "=" * 70)
            log.info("\n%s", summary_df.to_string(index=False))
            log.info("Saved: %s", summary_path)

            if "n_Unreliable" in summary_df.columns:
                total = summary_df["n_conditions"].sum()
                total_unmeas = summary_df["n_Unreliable"].sum()
                total_shared = summary_df["n_Shared"].sum()
                total_specific = summary_df["n_Specific"].sum()
                total_2d = (
                    total_unmeas + total_shared + total_specific
                )

                if total_2d > 0:
                    log.info("\n  ── Grand Totals ──")
                    log.info("  Total conditions: %d", int(total))
                    log.info(
                        "  Unreliable: %d (%.1f%%)",
                        int(total_unmeas),
                        100 * total_unmeas / total_2d,
                    )
                    log.info(
                        "  Shared: %d (%.1f%%)",
                        int(total_shared),
                        100 * total_shared / total_2d,
                    )
                    log.info(
                        "  Specific: %d (%.1f%%)",
                        int(total_specific),
                        100 * total_specific / total_2d,
                    )
                    log.info(
                        "  Effective benchmark size: %d conditions",
                        int(total_specific),
                    )
