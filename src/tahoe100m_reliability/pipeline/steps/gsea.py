"""
Runs step 3 of the pipeline: GSEA.
"""
import argparse
import blitzgsea as blitz
import gseapy as gp
import logging
import numpy as np
import pandas as pd
import re

from pathlib import Path

from .figures.gsea_figures import (
    plot_gsea,
    FDR_THRESHOLD,
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
                "NES": float(row.NES),
                "FDR_qval": float(row["FDR q-val"]),
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

        self.gene_metadata["token_id"] = self.gene_metadata["token_id"].astype(np.int64)
        self.id_to_symbol = dict(
            zip(self.gene_metadata["token_id"],
                self.gene_metadata["gene_symbol"].astype(str)
            )
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this base systematic step to the given parser."""
        BaseSystematicStep.add_cli_args(parser=parser)

        parser.add_argument(
            '--num-top-pathways', dest='num_top_pathways', required=False, type=int, default=5,
            help='Number of top positive/negative NES pathways to retain'
        )


    def run(self) -> None:
        """Run the step."""
        reliable_conds, _ = load_reliable_conditions(
            self.ds_name, self.reliability_dir, self.clusters
        )

        hallmark_sets = load_hallmark_sets()

        all_rows = []

        for mode, systema_dir in self.mode_mapping.items():
            candidate_files = sorted(systema_dir.rglob("*.npz"))
            cell_lines = sorted({
                re.sub(r"_CP10K", "", re.sub(r"_cluster\d+\.npz$", "", f.name, flags=re.I), flags=re.I)
                for f in candidate_files
                if (("CP10K" in f.name.upper()) == (mode == "cp10k"))
            })

            log.info(f"\n[{mode}] {len(cell_lines)} cell line(s) to process")
            for cell_line in cell_lines:
                # Load deltas
                systema_data = load_systema_deltas(
                    ds_name=self.ds_name,
                    systema_dir=systema_dir,
                    target_conds=reliable_conds,  # only care about reliable conditions in this context
                    clusters=self.clusters,
                    cell_lines={cell_line},
                )

                if cell_line not in systema_data:
                    print(f"  {cell_line}: no matching NPZ files; skipping")
                    continue

                data = systema_data[cell_line]

                conditions = data["conditions"].astype(str)
                n_stim_cells = data["n_stim_cells"]
                delta_p = data["delta_p"]

                conds = sorted(conditions)

                if not conds:
                    print(f"  {cell_line}: no reliable conditions found; skipping")
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

                # run GSEA
                # shared axis: n_stim_cells - weighted average of reliable perturbation deltas
                rel_weights = np.asarray([weights[c] for c in conds], dtype=np.float64)
                rel_deltas = np.stack([deltas[c] for c in conds]).astype(np.float64)
                shared = np.average(rel_deltas, axis=0, weights=rel_weights)

                genes_upper = [g.upper() for g in gene_symbols]
                gsea = gsea_nes(shared, genes_upper, hallmark_sets)

                for term, stats in gsea.items():
                    all_rows.append({
                        "mode": mode, "cell_line": cell_line, "term": term,
                        "NES": stats["NES"], "FDR_qval": stats["FDR_qval"],
                        "n_reliable_conditions": len(conds),
                        "n_genes_ranked": len(gene_symbols),
                    })

                del deltas, weights, rel_deltas, shared

        # summarize GSEA results
        df = pd.DataFrame(all_rows)
        df['significant'] = df['FDR_qval'] < FDR_THRESHOLD
        df.to_csv(self.output_dir / "shared_axis_hallmark_gsea.csv", index=False)

        # plot
        log.info("Plotting GSEA results...")
        plot_gsea(df, self.output_dir)

        # get the top positive and negative NES pathways
        # for every cell line and mode
        log.info("Summarizing GSEA results...")
        summaries = []
        for (mode, cell_line), group in df.groupby(["mode", "cell_line"]):
            group = group.dropna(subset=["NES"])

            positive = group[group["NES"] > 0].nlargest(self.num_top_pathways, "NES").copy()
            positive["direction"] = "positive"

            negative = group[group["NES"] < 0].nsmallest(self.num_top_pathways, "NES").copy()
            negative["direction"] = "negative"

            summaries.extend([positive, negative])

        if summaries:
            summary = pd.concat(summaries, ignore_index=True)
            summary.sort_values(
                ["mode", "cell_line", "direction", "FDR_qval"]
            ).to_csv(self.output_dir / f"top_{self.num_top_pathways}_pathways.csv", index=False)

        # Select pathways significant in at least one result.
        significant_terms = sorted(
            df.loc[df["significant"], "term"].dropna().unique()
        )

        # If none pass the threshold, show the 30 pathways with
        # the smallest minimum FDR instead.
        if not significant_terms:
            significant_terms = (
                df.groupby("term")["FDR_qval"]
                .min()
                .nsmallest(30)
                .index.tolist()
            )

        # Save significant results
        df[df["significant"]].sort_values(
            ["mode", "cell_line", "FDR_qval"]
        ).to_csv(self.output_dir / "significant_pathways.csv", index=False)

        log.info(f"Total rows: {len(df)}")
        log.info(f"Rows with FDR < {FDR_THRESHOLD}: {df['significant'].sum()}")
