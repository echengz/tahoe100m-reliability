
"""
Helps generate regression reliability figures.
"""

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from pathlib import Path

from tahoe100m_reliability.lib.step.base_reliability_step import (
    DEFAULT_N_HALF_GRID,
)


class RegressionReliabilityFigures:

    def __init__(self, output_dir: Path) -> None:
        """Initializer."""
        self.__output_dir = output_dir

    def _summarize_reliability(self, df: pd.DataFrame):
        fixed = (
            df[df["n_mode"] == "fixed"]
            .groupby(
                ["dataset", "normalization", "condition", "n_half"]
            )["median_r"]
            .median()
            .reset_index()
            .groupby(
                ["dataset", "normalization", "n_half"]
            )["median_r"]
            .mean()
            .reset_index()
        )

        max_available = (
            df[df["n_mode"] == "max"]
            .groupby(
                ["dataset", "normalization", "condition"]
            )["median_r"]
            .first()
            .reset_index()
            .groupby(
                ["dataset", "normalization"]
            )["median_r"]
            .mean()
            .rename("Max Available")
        )

        return fixed, max_available

    def _summarize_delta_reliability(self, df: pd.DataFrame):
        pair_cols = [
            "dataset",
            "cluster",
            "condition",
            "context",
            "perturbation",
            "n_mode",
            "n_half",
        ]

        paired = (
            df.pivot_table(
                index=pair_cols,
                columns="normalization",
                values="median_r",
                aggfunc="median",
            )
            .dropna(subset=["raw", "cp10k"])
            .reset_index()
        )

        paired["delta_r"] = paired["cp10k"] - paired["raw"]

        fixed = (
            paired[paired["n_mode"] == "fixed"]
            .groupby(["dataset", "condition", "n_half"])["delta_r"]
            .median()
            .reset_index()
            .groupby(["dataset", "n_half"])["delta_r"]
            .median()
            .reset_index()
        )

        max_delta = (
            paired[paired["n_mode"] == "max"]
            .groupby(["dataset", "condition"])["delta_r"]
            .median()
            .reset_index()
            .groupby("dataset")["delta_r"]
            .median()
            .rename("Max Available")
        )

        return fixed, max_delta

    def plot_reliability_vs_sample_size(
        self, df: pd.DataFrame
    ) -> None:
        """
        Plot the change in split-half reliability (CP10K - raw)
        against the number of cells per half.

        Fixed-mode observations are connected by lines.
        Max-mode observations are shown as separate diamond markers.
        """

        # Pair raw and CP10K observations.
        pair_cols = [
            "dataset",
            "cluster",
            "condition",
            "context",
            "perturbation",
            "n_mode",
            "n_half",
        ]

        paired = (
            df.pivot_table(
                index=pair_cols,
                columns="normalization",
                values="median_r",
                aggfunc="median",
            )
            .dropna(subset=["raw", "cp10k"])
            .reset_index()
        )

        if paired.empty:
            print("No paired raw/CP10K observations available to plot.")
            return

        paired["delta_r"] = paired["cp10k"] - paired["raw"]

        # First summarize within each condition, then across conditions
        fixed, max_delta = self._summarize_delta_reliability(df)

        fixed = fixed.rename(columns={"n_half": "plot_size"})
        fixed["point_type"] = "fixed"

        max_size = (
            df[df["n_mode"] == "max"]
            .groupby("dataset")["n_stim_cells"]
            .max()
            .div(2)
        )

        max_summary = (
            pd.concat(
                [
                    max_delta.rename("delta_r"),
                    max_size.rename("plot_size"),
                ],
                axis=1,
            )
            .dropna()
            .reset_index()
        )
        max_summary["point_type"] = "max"

        plot_df = pd.concat([fixed, max_summary], ignore_index=True)

        if plot_df.empty:
            print("No reliability summaries available to plot.")
            return

        # Put Tahoe-100M first in the legend.
        datasets = sorted(plot_df["dataset"].unique())
        if "Tahoe-100M" in datasets:
            datasets.remove("Tahoe-100M")
            datasets.insert(0, "Tahoe-100M")

        # Assign one color per dataset.
        cmap = plt.get_cmap("tab20")
        colors = {
            dataset: cmap(i % 20)
            for i, dataset in enumerate(datasets)
        }

        fig, ax = plt.subplots(figsize=(12, 6))

        for dataset in datasets:
            dataset_df = plot_df[plot_df["dataset"] == dataset]
            color = colors[dataset]

            fixed = (
                dataset_df[dataset_df["point_type"] == "fixed"]
                .sort_values("plot_size")
            )
            max_point = dataset_df[dataset_df["point_type"] == "max"]

            # Fixed sample sizes: connected points.
            if not fixed.empty:
                ax.plot(
                    fixed["plot_size"],
                    fixed["delta_r"],
                    marker="o",
                    linestyle="-",
                    color=color,
                    alpha=0.8,
                    linewidth=1.5,
                    markersize=6,
                )

            # Max available: separate diamond.
            if not max_point.empty:
                ax.scatter(
                    max_point["plot_size"],
                    max_point["delta_r"],
                    marker="D",
                    color=color,
                    alpha=0.8,
                    s=65,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=4,
                )

        # Reference line.
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
        )

        # Axes and labels.
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEFAULT_N_HALF_GRID)
        ax.set_xticklabels(DEFAULT_N_HALF_GRID)

        ax.set_xlabel("Cells per half", labelpad=15)
        ax.set_ylabel("Δ reliability (cp10k - raw)")
        ax.set_title("Effect of normalization on split-half reliability")

        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle=":",
            alpha=0.35,
        )

        # Dataset legend.
        dataset_handles = [
            Line2D(
                [0], [0],
                color=colors[dataset],
                marker="o",
                linestyle="-",
                linewidth=1.8,
                markersize=6,
                label=dataset,
            )
            for dataset in datasets
        ]

        dataset_legend = ax.legend(
            handles=dataset_handles,
            title="Dataset",
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
            fontsize=9,
            title_fontsize=10,
        )
        dataset_legend.get_frame().set_linewidth(0.8)

        # Sample-size mode legend.
        legend_color = to_rgba("black", alpha=0.5)

        mode_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="-",
                color=legend_color,
                markerfacecolor=legend_color,
                markeredgecolor=legend_color,
                label="Fixed",
                markersize=6,
            ),
            Line2D(
                [0], [0],
                marker="D",
                linestyle="None",
                color=legend_color,
                markerfacecolor=legend_color,
                markeredgecolor=legend_color,
                label="Max available",
                markersize=7,
            ),
        ]

        mode_legend = ax.legend(
            handles=mode_handles,
            title="Sample-size mode",
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
            fontsize=9,
            title_fontsize=10,
        )
        mode_legend.get_frame().set_linewidth(0.8)

        # Preserve both legends.
        ax.add_artist(dataset_legend)

        fig.tight_layout()
        fig.savefig(
            self.__output_dir / "reliability_vs_sample_size.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    def print_delta_latex_table(self, df: pd.DataFrame) -> None:
        """Print LaTeX table for delta reliability."""

        fixed, max_delta = self._summarize_delta_reliability(df)

        table = (
            fixed.pivot(
                index="dataset",
                columns="n_half",
                values="delta_r",
            )
            .reindex(columns=DEFAULT_N_HALF_GRID)
        )

        table["Max Available"] = max_delta

        datasets = sorted(table.index)
        if "Tahoe-100M" in datasets:
            datasets.remove("Tahoe-100M")
            datasets.insert(0, "Tahoe-100M")

        table = table.reindex(datasets)


        for dataset, row in table.iterrows():
            values = [
                f"${value:.2f}$" if pd.notna(value) else "--"
                for value in row
            ]

            print(
                f"{dataset} & "
                + " & ".join(values)
                + r" \\"
            )

    def print_reliability_latex_table(
        self, df: pd.DataFrame
    ) -> None:
        """Print median reliability as LaTeX table rows."""

        fixed, max_available = self._summarize_reliability(df)

        table = (
            fixed.pivot(
                index=["dataset", "normalization"],
                columns="n_half",
                values="median_r",
            )
            .reindex(columns=DEFAULT_N_HALF_GRID)
        )

        table["Max Available"] = max_available

        # Order datasets, with Tahoe-100M first.
        datasets = sorted(table.index.get_level_values("dataset").unique())
        if "Tahoe-100M" in datasets:
            datasets.remove("Tahoe-100M")
            datasets.insert(0, "Tahoe-100M")

        ordered_index = pd.MultiIndex.from_product(
            [datasets, ["raw", "cp10k"]],
            names=["dataset", "normalization"],
        )
        table = table.reindex(ordered_index)

        # Print LaTeX rows.
        previous_dataset = None

        for (dataset, normalization), row in table.iterrows():
            if row.isna().all():
                continue

            if previous_dataset is not None and dataset != previous_dataset:
                print(r"\hline")

            normalization_label = (
                "Raw" if normalization == "raw" else "CP10K"
            )

            values = [
                f"${value:.2f}$" if pd.notna(value) else "--"
                for value in row
            ]

            print(
                f"{dataset} & {normalization_label} & "
                + " & ".join(values)
                + r" \\"
            )

            previous_dataset = dataset

        print(r"\hline")

    def generate_figures(self, combined_df: pd.DataFrame) -> None:
        """Generate figures."""
        # only interested in the all genes mode here
        combined_df = combined_df[
            combined_df["gene_mode"] == "all_genes"
        ]

        self.plot_reliability_vs_sample_size(combined_df)

        # for the report
        self.print_delta_latex_table(combined_df)
        print("")
        self.print_reliability_latex_table(combined_df)
