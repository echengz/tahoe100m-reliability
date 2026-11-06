"""
Runs step 3 of the pipeline: GSEA.
"""
import argparse
import blitzgsea as blitz
import logging
import gseapy as gp
import numpy as np
import pandas as pd
import re

from pathlib import Path
from tqdm import tqdm
from typing import Any

from tahoe100m_reliability.lib.tahoe100m_dataset import (
    get_cell_line_metadata,
)

from .step_helpers.intermediate_loaders import (
    get_combined_split_half_reliability_df,
    load_gene_ids,
    load_reliable_conditions,
    load_systema_deltas,
)
from .step_helpers.pipeline_step import BaseSystematicStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

logging.disable(logging.WARNING)
import gseapy as gp


FDR_THRESHOLD = 0.05


def load_hallmark_sets():
    """Load hallmark sets."""
    library = blitz.enrichr.get_library("MSigDB_Hallmark_2020")
    return {
        term: [gene.upper() for gene in genes]
        for term, genes in library.items()
    }


def gsea_nes(vec, genes_upper, hallmark_sets):
    """Runs preranked GSEA using all genes."""
    series = pd.Series(
        np.asarray(vec, dtype=float),
        index=genes_upper
    )
    series = series[~series.index.duplicated()].sort_values(ascending=False)

    # skip degenerate rankings (near-constant => gseapy Rust backend panics);
    # residual~0 perts correctly -> ns
    if (
        series.shape[0] < 200
        or series.std() < 1e-12
        or (series.value_counts().iloc[0] / series.shape[0]) > 0.5
    ):
        return {}

    try:
        result = gp.prerank(
            rnk=series,
            gene_sets=hallmark_sets,
            min_size=5,
            max_size=500,
            permutation_num=1000,
            threads=4,
            seed=0,
            no_plot=True,
            outdir=None,
            verbose=False,
        ).res2d

        return {
            str(row.Term).replace("HALLMARK_", ""): {
                "nes": float(row['NES']),
                "fdr_qval": float(row['FDR q-val']),
                "tag_pct": row['Tag %'],
                "gene_pct": row['Gene %'],
                "lead_genes": row['Lead_genes']
            }
            for _, row in result.iterrows()
        }

    except Exception as exc:
        log.info(f"  WARNING: GSEA failed/skipped: {exc}")
        return {}


class GseaStep(BaseSystematicStep):

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializer."""
        super().__init__(args=args)
        self.ds_name = 'tahoe100m'
        self.num_top_pathways = args.num_top_pathways
        self.moa_shared_axis = args.moa_shared_axis

        self.gene_metadata["token_id"] = self.gene_metadata["token_id"].astype(np.int64)
        self.id_to_symbol = dict(
            zip(self.gene_metadata["token_id"],
                self.gene_metadata["gene_symbol"].astype(str)
            )
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this systematic step to the given parser."""
        BaseSystematicStep.add_cli_args(parser=parser)

        parser.add_argument(
            '--num-top-pathways', dest='num_top_pathways', required=False, type=int, default=10,
            help='Number of top positive/negative NES pathways to retain'
        )

        parser.add_argument(
            '--moa-shared-axis', dest='moa_shared_axis', action='store_true',
            help='Whether or not to calculate the MOA shared axis results'
        )

    def __run_gsea(
        self,
        mode: str,
        cell_line: str,
        output_rows: list[Any],
        conds,
        deltas,
        weights,
        gene_symbols,
        hallmark_sets,
        moa: str | None = None
    ):
        """Runs GSEA on shared axis for specified conditions."""
        rel_weights = np.asarray([weights[c] for c in conds], dtype=np.float64)
        rel_deltas = np.stack([deltas[c] for c in conds]).astype(np.float64)
        shared = np.average(rel_deltas, axis=0, weights=rel_weights)

        genes_upper = [g.upper() for g in gene_symbols]
        gsea = gsea_nes(shared, genes_upper, hallmark_sets)

        for term, stats in gsea.items():
            row = {
                "mode": mode,
                "cell_line": cell_line,
                "term": term,
                "nes": stats["nes"],
                "fdr_qval": stats["fdr_qval"],
                "tag_pct": stats["tag_pct"],
                "gene_pct": stats["gene_pct"],
                "lead_genes": stats["lead_genes"],
                "n_lead_genes": len(stats["lead_genes"].split(";")),
                "n_reliable_conditions": len(conds),
                "n_genes_ranked": len(gene_symbols),
            }

            if moa is not None:
                row['moa'] = moa

            output_rows.append(row)

    def run(self) -> None:
        """Run the step."""
        reliable_conds, _ = load_reliable_conditions(
            self.ds_name, self.reliability_dir, self.clusters
        )

        hallmark_sets = load_hallmark_sets()
        drug_to_moa = self.load_drug_moa_mapping() if self.moa_shared_axis else {}

        # track globally and by drug
        all_rows = []
        moa_rows = []

        for mode, systema_dir in self.mode_mapping.items():
            candidate_files = sorted(systema_dir.rglob("*.npz"))
            cell_lines = sorted({
                re.sub(r"_CP10K", "", re.sub(r"_cluster\d+\.npz$", "", f.name, flags=re.I), flags=re.I)
                for f in candidate_files
                if (("CP10K" in f.name.upper()) == (mode == "cp10k"))
            })

            log.info(f"\n[{mode}] {len(cell_lines)} cell line(s) to process")
            for cell_line in tqdm(cell_lines, desc=f"Processing cell lines for mode {mode}"):
                # Load deltas
                systema_data = load_systema_deltas(
                    ds_name=self.ds_name,
                    systema_dir=systema_dir,
                    target_conds=reliable_conds,  # only care about reliable conditions in this context
                    clusters=self.clusters,
                    cell_lines={cell_line},
                )

                if cell_line not in systema_data:
                    log.warning(f"{cell_line}: no matching NPZ files; skipping")
                    continue

                data = systema_data[cell_line]

                conditions = data["conditions"].astype(str)
                n_stim_cells = data["n_stim_cells"]
                delta_p = data["delta_p"]

                conds = sorted(conditions)

                if not conds:
                    log.warning(f"{cell_line}: no reliable conditions found; skipping")
                    continue

                deltas = {
                    condition: delta_p[i]
                    for i, condition in enumerate(conditions)
                }

                weights = {
                    condition: int(n_stim_cells[i])
                    for i, condition in enumerate(conditions)
                }

                # load gene IDs
                gene_ids = load_gene_ids(self.ctrl_gene_expressions_dir, cell_line)
                gene_symbols = [self.id_to_symbol.get(int(gid), "") for gid in gene_ids]

                vector_lengths = {len(deltas[c]) for c in conds}
                if len(vector_lengths) != 1:
                    raise ValueError(f"{cell_line}/{mode}: inconsistent delta vector lengths")

                n_genes = next(iter(vector_lengths))
                if len(gene_ids) != n_genes:
                    raise ValueError(
                        f"{cell_line}/{mode}: delta_p has {n_genes} genes but control parquet "
                        f"provides {len(gene_ids)} gene IDs. Check gene order/source."
                    )

                # Run GSEA
                # Global shared axis: n_stim_cells - weighted average of reliable perturbation deltas
                self.__run_gsea(
                    mode,
                    cell_line,
                    all_rows,
                    conds,
                    deltas,
                    weights,
                    gene_symbols,
                    hallmark_sets,
                )

                # MOA-specific axis - groups reliability perturbations for a specific MOA
                # and computes along that shared axis
                if self.moa_shared_axis:
                    moas_map = {}
                    for condition in conds:
                        parts = condition.split("|")
                        drug = parts[1]  # TODO - do something with dosage?

                        moa = drug_to_moa.get(drug, "Unknown MOA")
                        if moa not in moas_map:
                            moas_map[moa] = []

                        moas_map[moa].append(condition)

                    for moa, moa_conds in moas_map.items():
                        self.__run_gsea(
                            mode,
                            cell_line,
                            moa_rows,
                            moa_conds,
                            deltas,
                            weights,
                            gene_symbols,
                            hallmark_sets,
                            moa=moa,
                        )

        # summarize GSEA results
        groups = {
            'global': all_rows,
            'moa': moa_rows
        }

        metadata = get_cell_line_metadata(to_pandas=True)
        metadata["cellosaurus_id"] = metadata["Cell_ID_Cellosaur"].astype(str).str.strip()
        metadata["organ"] = metadata["Organ"].astype(str).str.strip()
        organ_by_cell_line = metadata.set_index("cellosaurus_id")["organ"].to_dict()

        for axis, rows in {'global': all_rows, 'moa': moa_rows}.items():
            if not rows:
                continue

            df = pd.DataFrame(rows)
            df['significant'] = df['fdr_qval'] < FDR_THRESHOLD
            df.to_csv(self.output_dir / f"{axis}_shared_axis_hallmark_gsea.csv", index=False)

            # get the top positive and negative NES pathways
            # for every cell line and mode
            log.info("Summarizing GSEA results...")
            group_columns = ['mode', 'cell_line']
            if axis == 'moa':
                group_columns.append('moa')

            summaries = []
            for _, group in df.groupby(group_columns):
                group = group.dropna(subset=["nes"])

                positive = group[group["nes"] > 0].nlargest(self.num_top_pathways, "nes").copy()
                positive["direction"] = "positive"

                negative = group[group["nes"] < 0].nsmallest(self.num_top_pathways, "nes").copy()
                negative["direction"] = "negative"

                summaries.extend([positive, negative])

            if summaries:
                summary = pd.concat(summaries, ignore_index=True)
                summary.sort_values(
                    group_columns + ['direction', 'fdr_qval']
                ).to_csv(self.output_dir / f"{axis}_top_{self.num_top_pathways}_pathways.csv", index=False)

            # Select pathways significant in at least one result.
            significant_terms = sorted(
                df.loc[df["significant"], "term"].dropna().unique()
            )

            # If none pass the threshold, show the 30 pathways with
            # the smallest minimum FDR instead.
            if not significant_terms:
                significant_terms = (
                    df.groupby("term")["fdr_qval"]
                    .min()
                    .nsmallest(30)
                    .index.tolist()
                )

            # Save significant results
            df[df["significant"]].sort_values(
                group_columns + ['fdr_qval']
            ).to_csv(self.output_dir / f"{axis}_significant_pathways.csv", index=False)

            log.info(f"Total rows: {len(df)}")
            log.info(f"Rows with FDR < {FDR_THRESHOLD}: {df['significant'].sum()}")
