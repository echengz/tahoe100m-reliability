"""
Helps generate reliability figures
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap

from pathlib import Path

from tahoe100m_reliability.lib.step.base_reliability_step import DEFAULT_N_HALF_GRID
from tahoe100m_reliability.lib.ingest.intermediate_loaders import (
    get_combined_cluster_split_half_reliability_df,
)

log = logging.getLogger(__name__)


class ReliabilityFigures:

    def __init__(self, reliability_dir: Path, output_dir: Path, mode: str) -> None:
        """Initializer."""
        self.__reliability_dir = reliability_dir
        self.__output_dir = output_dir
        self.__mode = mode

    def __plot_reliability_heatmap(
        self,
        df,
        heat_data,
        heat,
        title,
        outfile_name,
        figsize=(10, 16),
        n_label=None,
        y_label='Perturbation conditions, ordered by sample size',
        count_max_available: bool = True
    ):
        # Put in a background for missing values
        cmap = plt.colormaps["RdBu_r"].copy()
        cmap.set_bad("grey")

        fig, ax = plt.subplots(figsize=figsize)

        # Leave a visible gap before the Max available column
        gap = 0.15
        positions = list(range(len(DEFAULT_N_HALF_GRID))) + [
            len(DEFAULT_N_HALF_GRID) + gap
        ]

        # Draw fixed sample-size columns
        im = ax.imshow(
            heat[DEFAULT_N_HALF_GRID].to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=(-0.5, len(DEFAULT_N_HALF_GRID) - 0.5, len(heat) - 0.5, -0.5)
        )

        # Draw the separate Max available column at its own position
        max_col = heat[["Max available"]].to_numpy()
        ax.imshow(
            max_col,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=(positions[-1] - 0.5, positions[-1] + 0.5,
                    len(heat) - 0.5, -0.5)
        )

        # add average reliability across conditions for each sample size
        avg_reliability = heat.mean()
        for i, n in enumerate(DEFAULT_N_HALF_GRID):
            ax.text(
                i, 1.01,
                f"{avg_reliability[n]:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8
            )

        # Mean reliability above the Max available column
        ax.text(
            positions[-1], 1.01,
            f"{avg_reliability['Max available']:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8
        )

        if count_max_available:
            n_stim = (
                df.groupby("condition")["n_stim_cells"]
                  .max()
                  .reindex(heat.index)
                  .dropna()
            )

            xticklabels = (
                [str(s) for s in DEFAULT_N_HALF_GRID]
                + [
                    f"Max available\n"
                    f"Mean: {n_stim.mean() // 2}\n"
                    f"Median: {n_stim.median() // 2}"
                ]
            )
        else:
            xticklabels = (
                [str(s) for s in DEFAULT_N_HALF_GRID] + ["Max available"]
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(-0.5, positions[-1] + 0.5)

        if n_label is None:
            ax.set_yticks([])
            ax.set_ylabel(y_label, labelpad=25)
        else:
            ax.set_yticks([])

            if 'bottom20' in outfile_name:
                # Split condition names into cell line, drug, and dosage
                labels = heat.index.to_series().str.split("|", expand=True)
                labels.columns = ["Cell line", "Drug", "Dosage"]

                # Add aligned labels to the left of the heatmap
                for i, (cell_line, drug, dosage) in enumerate(labels.itertuples(index=False, name=None)):
                    drug = textwrap.fill(drug, width=18)

                    ax.text(-0.02, i, dosage, transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=7, clip_on=False)
                    ax.text(-0.09, i, drug, transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=7, clip_on=False)
                    ax.text(-0.24, i, cell_line, transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=7, clip_on=False)

                ax.set_ylabel(y_label, labelpad=158)
            else:
                ax.set_yticks(range(len(heat)))
                ax.set_yticklabels(heat.index, fontsize=7)
                ax.set_ylabel(y_label, labelpad=25)

        ax.set_xlabel("Cells per half (fixed sizes)", labelpad=15)
        ax.set_title(title, pad=35)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Median split-half correlation", labelpad=15)

        plt.subplots_adjust(top=0.82)
        plt.tight_layout()

        ax.axvspan(
            len(DEFAULT_N_HALF_GRID) - 0.5,
            positions[-1] - 0.5,
            color="grey",
            zorder=0
        )
        plt.savefig(self.__output_dir / outfile_name)
        plt.close(fig)


    def plot_reliability_by_sample_size(self, rel_df: pd.DataFrame, gene_mode: str) -> None:
        """
        Plot reliability trends by sample size.

        Generates two plots:
            - Line graph summarizing how, on average, reliability
                increases with the sample size. 
            - Heatmap highlighting reliability for every perturbation
        """
        df = rel_df[
            (rel_df["n_mode"] == "fixed") &
            (rel_df["gene_mode"] == gene_mode)
        ].copy()

        # --- HEATMAP ---
        heat_data = (
            df.groupby(["condition", "n_half"])["median_r"]
               .median()
               .unstack("n_half")
               .reindex(columns=DEFAULT_N_HALF_GRID)
        )

        # Get each condition's reliability at its largest available sample size
        max_df = rel_df[
            (rel_df["n_mode"] == "max") &
            (rel_df["gene_mode"] == gene_mode)
        ]

        counts = max_df.groupby("condition").size()

        if not (counts == 1).all():
            raise ValueError(
                f"Expected exactly one 'max' row per condition for "
                f"gene_mode={gene_mode!r}, but found missing or duplicate rows."
            )

        max_conditions = set(max_df["condition"])
        heat_conditions = set(heat_data.index)

        if max_conditions != heat_conditions:
            missing = heat_conditions - max_conditions
            extra = max_conditions - heat_conditions

            log.warning(
                f"Conditions mismatch for gene_mode={gene_mode!r}. "
                f"Missing from max_df: {sorted(missing)[:10]} "
                f"(total: {len(missing)}). "
                f"Extra in max_df: {sorted(extra)[:10]} "
                f"(total: {len(extra)})."
            )

        max_available = (
            max_df.set_index("condition")["median_r"]
                  .reindex(heat_data.index)
        )

        # Add the max-available values as a separate column
        heat_data["Max available"] = max_available

        # add perturbations with no fixed data
        heat_data = heat_data.reindex(
            heat_data.index.union(max_df["condition"])
        )

        heat_data["Max available"] = (
            max_df.set_index("condition")["median_r"]
                  .reindex(heat_data.index)
        )

        # Sort conditions by their correlation at n_half=16
        heat = heat_data.join(
            df.groupby("condition")["n_stim_cells"].max()
        ).sort_values(
            by="n_stim_cells",
            ascending=False,
            na_position="last"
        ).drop(columns="n_stim_cells")

        self.__plot_reliability_heatmap(
            max_df,
            heat_data,
            heat,
            title=f"Split-half reliability across {heat.shape[0]} conditions (cell line x drug x dosage)",
            outfile_name=f"reliability_{self.__mode}_by_sample_size_heatmap_{gene_mode}.png"
        )

        # next, group by drug
        # Extract drug name from condition: cell_line|drug|dosage
        df["drug"] = df["condition"].str.split("|").str[1]

        # First calculate one reliability value per condition and sample size
        condition_data = (
            df.groupby(["condition", "drug", "n_half"])["median_r"]
              .median()
              .reset_index()
        )

        # Then summarize across conditions for each drug and sample size
        drug_data = (
            condition_data.groupby(["drug", "n_half"])["median_r"]
                          .median()
                          .unstack("n_half")
                          .reindex(columns=DEFAULT_N_HALF_GRID)
        )

        max_per_condition = (
            max_df[["condition", "median_r"]]
            .assign(drug=lambda x: x["condition"].str.split("|").str[1])
        )

        max_by_drug = max_per_condition.groupby("drug")["median_r"].median()

        drug_heat_data = drug_data.copy()
        drug_heat_data["Max available"] = max_by_drug.reindex(
            drug_heat_data.index
        )

        max_by_drug = max_per_condition.groupby("drug")["median_r"].median()

        drug_heat_data = drug_data.copy()
        drug_heat_data["Max available"] = max_by_drug.reindex(drug_heat_data.index)

        n_stim_by_drug = (
            df.drop_duplicates("condition")
              .groupby("drug")["n_stim_cells"]
              .sum()
              .rename("n_stim_cells")
        )

        drug_heat = (
            drug_heat_data.join(n_stim_by_drug)
              .sort_values("n_stim_cells", ascending=False, na_position="last")
              .drop(columns="n_stim_cells")
        )

        self.__plot_reliability_heatmap(
            max_df,
            drug_heat_data,
            drug_heat,
            title=f'Split-half reproducibility across {drug_heat.shape[0]} drugs',
            outfile_name=f"reliability_{self.__mode}_heatmap_by_drug_{gene_mode}.png",
            y_label='Drugs, ordered by total sample size',
        )

        # Extract cell-line ID from condition: cell_line_id|drug|dosage
        df["cell_line"] = df["condition"].str.split("|").str[0]

        # First calculate one reliability value per condition and sample size
        condition_data = (
            df.groupby(["condition", "cell_line", "n_half"])["median_r"]
              .median()
              .reset_index()
        )

        # Summarize across conditions for each cell line and sample size
        cell_line_data = (
            condition_data.groupby(["cell_line", "n_half"])["median_r"]
                          .median()
                          .unstack("n_half")
                          .reindex(columns=DEFAULT_N_HALF_GRID)
        )

        # Reliability at each condition's largest available sample size,
        # then summarize across conditions for each cell line
        max_per_condition = (
            df.sort_values("n_half")
              .groupby("condition")
              .tail(1)[["condition", "cell_line", "median_r"]]
        )

        max_by_cell_line = (
            max_per_condition.groupby("cell_line")["median_r"].median()
        )

        cell_line_heat_data = cell_line_data.copy()
        cell_line_heat_data["Max available"] = (
            max_by_cell_line.reindex(cell_line_heat_data.index)
        )

        # Estimate total stimulated-cell counts across unique conditions
        n_stim_by_cell_line = (
            df.drop_duplicates("condition")
              .groupby("cell_line")["n_stim_cells"]
              .sum()
              .rename("n_stim_cells")
        )

        cell_line_heat = (
            cell_line_heat_data.join(n_stim_by_cell_line)
                               .sort_values(
                                   "n_stim_cells",
                                   ascending=False,
                                   na_position="last",
                               )
                               .drop(columns="n_stim_cells")
        )

        cell_line_heat.index.name = "cell_line"

        self.__plot_reliability_heatmap(
            max_df,
            cell_line_heat_data,
            cell_line_heat,
            title=f"Split-half reproducibility across {cell_line_heat.shape[0]} cell lines",
            outfile_name=f"reliability_{self.__mode}_heatmap_by_cell_line_{gene_mode}.png",
            n_label=len(cell_line_heat),
            y_label="Cell lines, ordered by total sample size",
            count_max_available=False
        )

        # also output the csv
        cell_line_heat_data.to_csv(
            self.__output_dir / f"reliability_{self.__mode}_heatmap_by_cell_line_{gene_mode}.csv"
        )

    def generate_figures(self):
        """Generate figures."""
        rel_df = get_combined_cluster_split_half_reliability_df(
            self.__reliability_dir, None, summary=False, ds_name='tahoe100m'
        )

        # plot reliability by sample size
        self.plot_reliability_by_sample_size(rel_df, gene_mode='all_genes')

        # optionally: look at top 1000 expressed gene results
        # self.plot_reliability_by_sample_size(rel_df, gene_mode='top1000_expr')
