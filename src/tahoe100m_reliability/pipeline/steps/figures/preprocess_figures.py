"""
Helps with generating preprocess figures.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from tahoe100m_reliability.lib.ingest.treatment_set import TreatmentSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PreprocessFigures:

    def __init__(
        self,
        treatment_sets: list[TreatmentSet],
        clustered_treatments_csv: Path,
        output_dir: Path,
    ) -> None:
        """Initializes class to generate preprocessing figuers."""

        # We don't care about the controls for these, so filter out
        self.__treatment_sets = [
            treatment for treatment in treatment_sets
            if treatment.drug != 'DMSO_TF'
        ]
        self.__clustered_treatments = pd.read_csv(clustered_treatments_csv)
        self.__output_dir = output_dir

    def __generate_shard_count_figure(self) -> None:
        """1. See how many shards each drug/dosage combo lives in."""
        rows = []
        for treatment in self.__treatment_sets:
            rows.append({
                'drug': treatment.drug,
                'dosage': treatment.dosage,
                'n_shards': len(treatment.parquet_files),
                'n_samples': len(treatment.samples)
            })

        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 4))

        # Top 5 drug × dosage combinations and average
        top_combos = df.nlargest(10, "n_shards")
        avg_shards = df["n_shards"].mean()

        legend_handles = [
            Line2D(
                [], [],
                linestyle="none",
                marker="",
                label=f"{row['drug']}|{row['dosage']} = {row['n_shards']} shards"
            )
            for _, row in top_combos.iterrows()
        ]

        ax.legend(
            handles=legend_handles,
            title="Top 10 drug×dosage permutations",
            loc="upper right",
            fontsize=9,
            title_fontsize=10,
            framealpha=0.85,
            handlelength=0,
            handletextpad=0,
        )

        ax.bar(range(len(df)), df['n_shards'], bottom=0, width=1.1)
        ax.axhline(
            avg_shards,
            linestyle="--",
            linewidth=1,
            color='red'
        )
        
        ax.set_xlabel("Number of parquet shards")
        ax.set_ylabel("Drug x dosage (permutations)")
        ax.set_title(f"Number of parquet shards per drug/dosage (avg = {avg_shards:.3f})")
        plt.tight_layout()
        plt.savefig(self.__output_dir / 'parquet_per_drug_dosage.png')

    def __generate_unique_perturbations_per_shard(self) -> None:
        """2. For each parquet shard, how many unique drug x dosage
        (perturbation)s are in it
        """
        drugdosage_per_shard = defaultdict(set)
        for treatment in self.__treatment_sets:
            for p in treatment.parquet_files:
                drugdosage_per_shard[p.split('/')[-1]].add(f'{treatment.drug}_{treatment.dosage}')

        df = pd.DataFrame({
            "parquet_file": list(drugdosage_per_shard.keys()),
            "n_drugdosages": [len(drugs) for drugs in drugdosage_per_shard.values()],
        })
        df['shard'] = (
            df["parquet_file"]
            .str.extract(r"train-(\d+)-of-03388")[0]
            .astype(int)
        )
        df = df.sort_values('shard')

        #                                            parquet_file  n_drugs
        # 217   hf://datasets/tahoebio/Tahoe-100m/data/train-0...       94
        # 283   hf://datasets/tahoebio/Tahoe-100m/data/train-0...       94
        # 191   hf://datasets/tahoebio/Tahoe-100m/data/train-0...       94
        # 246   hf://datasets/tahoebio/Tahoe-100m/data/train-0...       94
        # 824   hf://datasets/tahoebio/Tahoe-100m/data/train-0...       91
        log.info(df.sort_values("n_drugdosages", ascending=False).head(10))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df["shard"], df["n_drugdosages"], bottom=0, width=1.1)
        avg_drugs = df["n_drugdosages"].mean()
        ax.axhline(
            avg_drugs,
            linestyle="--",
            linewidth=1,
            color='red'
        )

        ax.set_xlabel("Parquet shard")
        ax.set_ylabel("Number of unique drug x dosage perturbations")
        ax.set_title(f"Number of unique drug x dosage perturbations per parquet shard (avg = {avg_drugs:.3f})")
        plt.tight_layout()
        plt.savefig(self.__output_dir / 'drug_dosage_per_parquet.png')

    def __generate_perturbation_heatmaps(self) -> None:
        """3. Generate heatmaps of the perturbation spread."""
        rows = []
        for treatment in self.__treatment_sets:
            shards = {
                int(p.split("train-")[1].split("-of-")[0])
                for p in treatment.parquet_files
            }

            rows.append({
                'drug': treatment.drug,
                'dosage': treatment.dosage,
                'shards': shards,
            })

        log.info(f"{len(rows)} unique perturbations")

        df = pd.DataFrame(rows)
        # shard statistics
        df["n_shards"] = df["shards"].str.len()
        df["min_shard"] = df["shards"].apply(min)
        df["max_shard"] = df["shards"].apply(max)

        # binary matrix
        matrix = np.zeros((len(df), 3389), dtype=np.uint8)
        for i, shards in enumerate(df["shards"]):
            matrix[i, list(shards)] = 1

        # sort by where the drug/dosage starts in shard list
        order = np.argsort(df["min_shard"].values)
        matrix = matrix[order]
        df = df.iloc[order].reset_index(drop=True)

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            matrix,
            aspect="auto",
            cmap=ListedColormap(["grey", "yellow"]),
            interpolation="none",
        )

        ax.set_xlabel("Parquet shard")
        ax.set_ylabel("Drug x dosage (perturbation)")
        plt.tight_layout()
        plt.savefig(self.__output_dir / 'drug_dosage_distribution.png')

        # Add precomputed cluster assignments
        df = df.merge(
            self.__clustered_treatments[["drug", "dosage", "cluster"]],
            on=["drug", "dosage"],
            how="left",
            validate="one_to_one",
        )

        # Check that every treatment has a cluster
        if df["cluster"].isna().any():
            raise ValueError("Some treatments have no cluster assignment")

        clusters = (
            df.groupby('cluster').agg(
                n_drug_dosages=("drug", "size"),
                min_shard=("min_shard", "min"),
                max_shard=("max_shard", "max")
            ).sort_values("min_shard")
        )

        # color code the heatmap
        cluster_matrix = np.zeros_like(matrix, dtype=np.int16)

        for i in range(len(df)):
            cluster_matrix[i, matrix[i] == 1] = df.iloc[i]["cluster"]

        colors = ["grey"] + list(plt.cm.turbo(np.linspace(0, 1, n_clusters)))
        cmap = ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            cluster_matrix,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=0,
            vmax=n_clusters,
        )

        cluster_order = df["cluster"].drop_duplicates().tolist()
        n_parquets = {
            cluster: np.sum(np.any(cluster_matrix == cluster, axis=0))
            for cluster in cluster_order
        }

        # Create legend entries
        legend_handles = [
            Patch(
                facecolor=colors[cluster],
                label=(
                    f"Cluster {cluster} "
                    f"({clusters.loc[cluster, 'n_drug_dosages']} perturbations, "
                    f"{n_parquets[cluster]} parquets)"
                )
            )
            for cluster in cluster_order
        ]

        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=8.5,
            title_fontsize=10,
            framealpha=0.8,
        )

        ax.set_xlabel("Parquet shard")
        ax.set_ylabel("Drug x dosage (perturbation)")
        plt.tight_layout()
        plt.savefig(self.__output_dir / 'drug_dosage_distribution_clustered.png')

        # color heatmap by dosage
        # Map dosage -> integer
        dosage_to_id = {
            0.05: 1,
            0.50: 2,
            5.00: 3,
        }

        # Same shape as matrix
        dosage_matrix = np.zeros_like(matrix, dtype=int)

        for i in range(len(df)):
            dosage_id = dosage_to_id[df.iloc[i]["dosage"]]

            # Only color the shards where this drug/dosage is present
            dosage_matrix[i, matrix[i] == 1] = dosage_id

        cmap = ListedColormap([
            "grey",
            "red",  # 0.05 uM
            "yellow",  # 0.50 uM
            "blue",  # 5.00 uM
        ])

        legend_elements = [
            Patch(facecolor="red", label="0.05 uM"),
            Patch(facecolor="yellow", label="0.50 uM"),
            Patch(facecolor="blue", label="5.00 uM"),
        ]

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            dosage_matrix,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=0,
            vmax=3,
        )

        ax.legend(
            handles=legend_elements,
            title="Dosage",
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
        )

        ax.set_xlabel("Parquet shard")
        ax.set_ylabel("Drug x dosage (perturbation)")
        plt.tight_layout()
        plt.savefig(self.__output_dir / 'dosage_distribution.png')

        # color heatmap by drug
        fig, ax = plt.subplots(figsize=(8, 8))
        # Drug order as they appear in the current matrix / df
        drugs_in_order = df["drug"].drop_duplicates().tolist()

        drug_to_id = {
            drug: i + 1
            for i, drug in enumerate(drugs_in_order)
        }

        drug_matrix = np.zeros_like(matrix, dtype=int)

        for i in range(len(df)):
            drug_id = drug_to_id[df.iloc[i]["drug"]]
            drug_matrix[i, matrix[i] == 1] = drug_id

        cmap = plt.colormaps["turbo"].resampled(len(drugs_in_order) + 1)

        colors = cmap(np.arange(len(drugs_in_order) + 1))
        grey_val = 128 / 255
        colors[0] = [grey_val, grey_val, grey_val, 1]  # zero = white

        cmap = ListedColormap(colors)

        ax.imshow(
            drug_matrix,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=0,
            vmax=len(drugs_in_order),
        )
        ax.set_xlabel("Parquet shard")
        ax.set_ylabel("Drug x dosage (perturbation)")
        plt.tight_layout()
        plt.savefig(self.__output_dir / 'drug_distribution.png')

    def generate_figures(self) -> None:
        """
        Generates figures to evaluate the distribution of treatments
        across parquet shards; will be useful for knowing how to iterate
        over them in the next part
        """

        # 1. See how many shards each drug/dosage combo lives in
        self.__generate_shard_count_figure()

        # 2. For each parquet shard, how many unique drug x dosage (perturbation)s are in it
        self.__generate_unique_perturbations_per_shard()

        # 3. Generate heatmap of each drug x dosage (perturbation) per parquet
        self.__generate_perturbation_heatmaps()
