"""
Runs step 2 of the pipeline: GSEA.
"""
import argparse
import logging
import gseapy as gp
import numpy as np
import pandas as pd
import re

from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Any

from tahoe100m_reliability.lib.ingest.intermediate_loaders import (
    load_delta_p_results,
    load_gene_ids,
    load_reliable_conditions,
)
from tahoe100m_reliability.lib.ingest.msigdb_hallmark import (
    load_hallmark_sets,
)
from tahoe100m_reliability.lib.ingest.tahoe100m_dataset import (
    get_organ_by_cell_line,
)
from tahoe100m_reliability.lib.step.base_systematic_step import BaseSystematicStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

logging.getLogger("gseapy").setLevel(logging.ERROR)

FDR_THRESHOLD = 0.05
CLUSTER_MIN_PATHWAY_COVERAGE = 0.90


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

    def __init__(
        self,
        num_top_pathways: int = 10,
        num_moa_clusters: int = 4,
        skip_moa_shared_axis: bool = False,
        **kwargs,
    ) -> None:
        """Initializer."""
        super().__init__(**kwargs)
        self.ds_name = 'tahoe100m'
        self.num_top_pathways = num_top_pathways
        self.skip_moa_shared_axis = skip_moa_shared_axis

        self.gene_metadata["token_id"] = self.gene_metadata["token_id"].astype(np.int64)
        self.id_to_symbol = dict(
            zip(self.gene_metadata["token_id"],
                self.gene_metadata["gene_symbol"].astype(str)
            )
        )

        self.num_moa_clusters = num_moa_clusters
        self.hallmark_sets = load_hallmark_sets()

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "GseaStep":
        """Create GseaStep from arguments."""
        return cls(**cls.clean_args(args))

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this GSEA step to the given parser."""
        BaseSystematicStep.add_cli_args(parser=parser)

        parser.add_argument(
            '--num-top-pathways', dest='num_top_pathways', type=int, default=10,
            help='Number of top positive/negative NES pathways to retain'
        )

        parser.add_argument(
            '--num-moa-clusters', dest='num_moa_clusters', type=int, default=4,
            help='Number of clusters to generate from MOA results'
        )

        parser.add_argument(
            '--skip-moa-shared-axis', dest='skip_moa_shared_axis', action='store_true',
            help='Whether or not to skip calculation of the MOA shared axis results'
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
        moa: str | None = None
    ):
        """Runs GSEA on the shared axis for specified conditions."""
        rel_weights = np.asarray([weights[c] for c in conds], dtype=np.float64)
        rel_deltas = np.stack([deltas[c] for c in conds]).astype(np.float64)
        shared = np.average(rel_deltas, axis=0, weights=rel_weights)

        genes_upper = [g.upper() for g in gene_symbols]
        gsea = gsea_nes(shared, genes_upper, self.hallmark_sets)

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

    def __load_deltas_and_weights(
        self,
        cell_line: str,
        delta_p_dir: Path,
        reliable_conds: set[str], 
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load deltas and their corresponding weights, indexed by the condition."""
        delta_p_dir = load_delta_p_results(
            ds_name=self.ds_name,
            delta_p_dir=delta_p_dir,
            target_conds=reliable_conds,  # only care about reliable conditions in this context
            clusters=self.clusters,
            cell_lines={cell_line},
        )

        if cell_line not in delta_p_dir:
            raise FileNotFoundError(f"{cell_line}: no matching NPZ files")

        data = delta_p_dir[cell_line]

        conditions = data["conditions"].astype(str)
        n_stim_cells = data["n_stim_cells"]
        delta_p = data["delta_p"]

        if not sorted(conditions):
            raise ValueError(f"{cell_line}: no reliable conditions found")

        deltas = {
            condition: delta_p[i]
            for i, condition in enumerate(conditions)
        }

        weights = {
            condition: int(n_stim_cells[i])
            for i, condition in enumerate(conditions)
        }

        return deltas, weights

    def __summarize_gsea_results(
        self, all_rows: list, moa_rows: list,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Summarize GSEA results, also add organ information."""
        # add organ information
        organ_by_cell_line = get_organ_by_cell_line()
        results = {}

        for axis, rows in {"global": all_rows, "moa": moa_rows}.items():
            df = pd.DataFrame(rows)
            df["significant"] = df["fdr_qval"] < FDR_THRESHOLD
            df["organ"] = df["cell_line"].map(organ_by_cell_line)

            df.to_csv(self.output_dir / f"{axis}_shared_axis_hallmark_gsea.csv", index=False)

            results[axis] = df

        return results["global"], results["moa"]

    def __moa_pca_and_cluster(self, mode: str, df: pd.DataFrame) -> None:
        """Run PCA on MOA GSEA results and cluster based on centroids.

        Constructs a PCA matrix where each row is a specific cell line x MOA observation,
        with biological pathways as the columna features and NES scores as the values. 
        """
        output_dir = self.output_dir / "moa_clustering" / mode
        output_dir.mkdir(parents=True, exist_ok=True)

        # get fields of interest
        fields = ['moa', 'cell_line', 'organ', 'term', 'nes']
        df = df[fields].copy()
        df['nes'] = pd.to_numeric(df['nes'], errors='coerce')
        df = df.dropna(subset=fields)

        # group one row per MOA x cell line/organ x pathways
        # in theory should be no duplicates, but take the average just in case
        df = df.groupby(['moa', 'cell_line', 'organ', 'term'], as_index=False)["nes"].mean()

        # generate PCA matrix; keep observations with enough coverage
        matrix = df.pivot(index=['moa', 'cell_line', 'organ'], columns='term', values='nes')
        matrix = matrix.loc[:, matrix.notna().mean(axis=0) >= CLUSTER_MIN_PATHWAY_COVERAGE]

        # Handle missing data
        if matrix.isna().any().any():
            log.warning("Missing NES values found. Using pathway-wise mean imputation.")
            matrix = matrix.fillna(matrix.mean(axis=0))

        # Remove pathways with no variation.
        matrix = matrix.loc[:, matrix.std(axis=0) > 1e-12]

        if matrix.shape[0] < 3 or matrix.shape[1] < 2:
            raise ValueError("Not enough contexts or pathways for PCA.")

        # run PCA
        X = matrix.to_numpy(dtype=float)
        X = X - X.mean(axis=0)  # center features

        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)

        scores = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        log.info("Explained variance:")
        log.info(f"PC1: {explained[0] * 100:.2f}%")
        log.info(f"PC2: {explained[1] * 100:.2f}%")

        # Save all PCA coordinates
        scores_df = pd.DataFrame(
            scores,
            columns=[f"PC{i+1}" for i in range(scores.shape[1])],
            index=matrix.index,
        ).reset_index()

        scores_df.to_csv(output_dir / f"{mode}_pca_context_coordinates.csv", index=False)

        # Save explained variance
        pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(explained))],
            "explained_variance_ratio": explained,
        }).to_csv(output_dir / f"{mode}_pca_explained_variance.csv", index=False)

        # save pathway loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            index=matrix.columns,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )

        loadings.to_csv(output_dir / f"{mode}_pca_pathway_loadings.csv")

        # Perform hierarchical clustering based on the PCA centroids
        # include enough PCs to capture most of variation
        pc_cols = [c for c in scores_df.columns if c.startswith("PC")]
        pc_cols = sorted(pc_cols, key=lambda x: int(x[2:]))

        cumulative = np.cumsum(explained)
        n_pcs = int(np.searchsorted(cumulative, 0.80) + 1)
        selected_pcs = pc_cols[:n_pcs]
        log.info(f"Using {len(selected_pcs)} PCs: {selected_pcs}")

        # find MOA centroids and perform clustering
        moa_centroids = scores_df.groupby("moa")[selected_pcs].mean()
        Z = linkage(
            moa_centroids.values,
            method="ward",
            metric="euclidean"
        )
        cluster_labels = fcluster(
            Z,
            t=self.num_moa_clusters,
            criterion="maxclust"
        )

        moa_clusters = pd.DataFrame({
            "moa": moa_centroids.index,
            "cluster": cluster_labels
        })

        moa_clusters.to_csv(output_dir / f"{mode}_moa_cluster_assignments.csv", index=False)

    def run(self) -> None:
        """Run the step."""
        drug_to_moa = self.load_drug_moa_mapping() if not self.skip_moa_shared_axis else {}

        # track globally and by MOA for each cell line
        all_rows = []
        moa_rows = []

        for mode, delta_p_mode in self.mode_mapping.items():
            reliability_dir = delta_p_mode.reliability_dir
            delta_p_dir = delta_p_mode.delta_p_dir

            if not delta_p_dir or not delta_p_dir.is_dir():
                log.warning(f"{mode} delta_p dir not found, skipping mode")
                continue

            reliable_conds, _ = load_reliable_conditions(
                self.ds_name, reliability_dir, self.clusters
            )

            delta_p_files = sorted(delta_p_dir.rglob("*.npz"))
            cell_lines = sorted({
                re.sub(r"_CP10K", "", re.sub(r"_cluster\d+\.npz$", "", f.name, flags=re.I), flags=re.I)
                for f in delta_p_files
                if (("CP10K" in f.name.upper()) == (mode == "cp10k"))
            })

            log.info(f"\n[{mode}] {len(cell_lines)} cell line(s) to process")
            for cell_line in tqdm(cell_lines, desc=f"Processing cell lines for mode: {mode}"):
                deltas, weights = self.__load_deltas_and_weights(
                    cell_line,
                    delta_p_dir,
                    reliable_conds,
                )

                # load gene IDs
                gene_ids = load_gene_ids(self.ctrl_gene_expressions_dir, cell_line)
                gene_symbols = [self.id_to_symbol.get(int(gid), "") for gid in gene_ids]

                # sanity check that genes align with deltas
                conds = sorted(list(deltas.keys()))
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
                )

                # MOA-specific axis - groups reliability perturbations for a specific MOA
                # and computes along that shared axis
                if not self.skip_moa_shared_axis:
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
                            moa=moa,
                        )

        # summarize GSEA results
        _, moa_df = self.__summarize_gsea_results(all_rows, moa_rows)

        # for skipping above and going straight to clustering
        # moa_df = pd.read_csv('./outputs/gsea/moa_shared_axis_hallmark_gsea.csv')

        # cluster MOA results
        if not self.skip_moa_shared_axis:
            for mode, mode_df in moa_df.groupby("mode"):
                self.__moa_pca_and_cluster(mode, mode_df)
