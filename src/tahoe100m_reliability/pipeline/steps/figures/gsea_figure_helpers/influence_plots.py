"""
Plots systematic influence x GSEA results
"""
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from .config import HALLMARK_GROUPINGS


class GSEAInfluenceFigures:

    def __init__(self, output_dir: Path, axis: str) -> None:
        """Initializer."""
        self.__axis = axis
        self.__output_dir = output_dir
        self.__output_dir.mkdir(parents=True, exist_ok=True)

    def __prepare_plot_data(
        self,
        df: pd.DataFrame,
        mode: str,
        target_pathways: list[str] | None = None,
    ) -> pd.DataFrame:
        """Prepare systematic influence x GSEA plot data."""
        columns = [
            'abs_nes', 'mean_influence', 'fdr_qval', 'term'
        ]

        if self.__axis == 'moa':
            columns.append('moa')

        plot_df = (
            df.loc[df['mode'] == mode, columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )

        # Keep only target pathways, if specified
        if target_pathways is not None:
            plot_df = plot_df[
                plot_df["term"].isin(target_pathways)
            ]

        if plot_df.empty:
            raise ValueError("systematic influence x GSEA plot_df is empty")

        return plot_df

    def __compute_group_correlations(
        self, plot_df: pd.DataFrame, group_by: str = 'term'
    ) -> tuple[list, dict]:
        """Compute group correlations.

        Returns tuple of list of sorted groups with highest spearman
        correlation in either direction and the correlations as a dict.
        """
        group_correlations = {}
        for term, term_df in plot_df.groupby(group_by):
            if len(term_df) >= 3:
                rho, pval = spearmanr(
                    term_df["abs_nes"],
                    term_df["mean_influence"]
                )
                group_correlations[term] = (rho, pval)

        if not group_correlations:
            raise ValueError(f"no {group_by} correlations found")

        # sort by those with the highest absolute spearman
        # correlation in either direction
        groups = sorted(
            group_correlations,
            key=lambda group: abs(group_correlations[group][0]),
            reverse=True
        )

        return groups, group_correlations

    def __plot_term_df(
        self, ax, term_df: pd.DataFrame, color_by: str, colors: dict, sig_alpha: float=0.9
    ) -> None:
        """Plot the provided term df on the given axis."""
        x = term_df["abs_nes"]
        y = term_df["mean_influence"]

        significant = term_df["fdr_qval"] < 0.05

        # color significant points by the given color_by column and colormap
        point_colors = [
            colors[group] if sig else 'lightgrey'
            for group, sig in zip(term_df[color_by], significant)
        ]

        # Plot non-significant points first
        ax.scatter(
            x[~significant],
            y[~significant],
            color="lightgrey",
            alpha=0.4,
            s=8,
            zorder=1,
        )

        # Plot significant points on top
        ax.scatter(
            x[significant],
            y[significant],
            c=[colors[group] for group in term_df.loc[significant, color_by]],
            alpha=sig_alpha,
            s=10,
            zorder=2,
        )

        # draw line at 0
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    def __add_legend(
        self, fig, title: str, groupings: list, colors: dict, fontsize: int = 7
    ) -> None:
        """Add the legend to the figure for the color by scheme."""
        legend_elements = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=colors[group],
                markersize=fontsize - 1,
                label=group,
            )
            for group in groupings
        ]

        fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(1, 0.95),
            bbox_transform=fig.transFigure,
            loc="upper left",
            fontsize=fontsize,
            title=title,
        )

    def generate_per_pathway_figures(
        self,
        df: pd.DataFrame,
        mode: str,
        filename: str,
        target_pathways: list[str] | None = None
    ) -> None:
        """
        Generate systematic influence x GSEA figures per pathway.
        Colored by MOA.
        """
        plot_df = self.__prepare_plot_data(df, mode, target_pathways=target_pathways)
        terms, term_correlations = self.__compute_group_correlations(plot_df)

        # MOA colours shared across all subplots
        moas = sorted(plot_df["moa"].unique())
        colors = plt.get_cmap("turbo", len(moas))

        moa_colors = {moa: colors(i) for i, moa in enumerate(moas)}

        ncols = 5
        nrows = int(np.ceil(len(terms) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(20, nrows * 3),
            sharex=True,
            sharey=True,
        )

        axes = axes.flatten()

        for ax, term in zip(axes, terms):
            term_df = plot_df[plot_df["term"] == term]
            self.__plot_term_df(ax, term_df, 'moa', moa_colors)

            # add title with the rho and pvals from spearmanr
            rho, pval = term_correlations[term]
            ax.set_title(
                f"{term}\n$r_s = {rho:.2f},\\ p = {pval:.2e}$",
                fontsize=8,
            )

        # Remove unused subplot axes
        for ax in axes[len(terms):]:
            ax.remove()

        self.__add_legend(fig, 'MOA', moas, moa_colors)

        fig.suptitle(
            f"Enrichment strength vs signed pathway influence ({mode})",
            fontsize=14,
        )

        fig.supxlabel("Absolute NES", fontsize=10)
        fig.supylabel("Mean influence", fontsize=10)
        fig.tight_layout(rect=[0.015, 0, 1, 0.98])

        fig.savefig(
            self.__output_dir / filename, bbox_inches="tight", dpi=300,
        )

    def __add_axis_legend(
        self,
        ax,
        title: str,
        pathways: list[str],
        colors: dict,
        fontsize: int = 7,
    ) -> None:
        """Add a pathway legend to an individual subplot."""

        legend_elements = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=colors[pathway],
                markersize=fontsize - 1,
                label=pathway,
            )
            for pathway in pathways
        ]

        ax.legend(
            handles=legend_elements,
            title=title,
            fontsize=fontsize,
            title_fontsize=fontsize,
            loc="upper left",
            frameon=False,
        )

    def generate_by_biological_group(
            self,
            df: pd.DataFrame,
            mode: str,
            filename: str,
            target_pathways: list[str] | None = None,
        ) -> None:
            """
            Generate systematic influence x GSEA figures per biological group.

            Each subplot represents one biological group and pathways
            within that group are colored individually.
            """
            plot_df = self.__prepare_plot_data(
                df, mode, target_pathways=target_pathways,
            )

            # add biological pathway groupings
            plot_df = plot_df.copy()
            plot_df["group"] = plot_df["term"].map(HALLMARK_GROUPINGS)

            # Calculate correlations per pathway, and order biological groups by max pathway correlation
            _, pathway_correlations = self.__compute_group_correlations(plot_df, group_by='term')

            group_correlations = {}

            for term, (rho, _) in pathway_correlations.items():
                group = HALLMARK_GROUPINGS.get(term, "Other")
                group_correlations.setdefault(group, []).append(abs(rho))

            group_correlations = {
                group: np.mean(values)
                for group, values in group_correlations.items()
            }

            groups = sorted(
                group_correlations,
                key=lambda group: group_correlations[group],
                reverse=True,
            )

            # Assign pathway colors using the turbo colormap
            cmap = plt.get_cmap("turbo")
            pathway_colors = {}

            for group in groups:
                group_terms = sorted(
                    plot_df.loc[plot_df["group"] == group, "term"].unique()
                )

                positions = np.linspace(0.05, 0.95, len(group_terms))

                for term, position in zip(group_terms, positions):
                    pathway_colors[term] = cmap(position)

            # Create subplot grid
            ncols = 2
            nrows = int(np.ceil(len(groups) / ncols))

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(16, nrows * 5),
                sharex=True,
                sharey=True,
            )

            axes = np.asarray(axes).flatten()

            for ax, group in zip(axes, groups):
                group_df = plot_df[plot_df["group"] == group]

                terms = sorted(group_df["term"].unique())

                for term in terms:
                    term_df = group_df[group_df["term"] == term]

                    self.__plot_term_df(
                        ax,
                        term_df,
                        "term",
                        pathway_colors,
                    )

                # Group title with average pathway correlation
                ax.set_title(
                    f"{group}\nMean |$r_s$| = "
                    f"{group_correlations[group]:.2f}",
                    fontsize=11,
                )

                self.__add_axis_legend(
                    ax,
                    "Pathway",
                    terms,
                    pathway_colors,
                )

            # Remove unused subplot axes
            for ax in axes[len(groups):]:
                ax.remove()

            fig.suptitle(
                f"Enrichment strength vs signed pathway influence ({mode})",
                fontsize=14,
            )

            fig.supxlabel("Absolute NES")
            fig.supylabel("Mean influence")

            fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])

            fig.savefig(
                self.__output_dir / filename, bbox_inches="tight", dpi=300,
            )

    def generate_figures(
        self, 
        df: pd.DataFrame,
        mode: str,
        file_prefix: str,
        target_pathways: list[str] | None = None,
    ) -> None:
        """Generate plots both by pathway and by aggregated pathways."""
        self.generate_per_pathway_figures(
            df, mode,
            filename=f'{file_prefix}_{mode}.png',
            target_pathways=target_pathways,
        )

        self.generate_by_biological_group(
            df,
            mode,
            filename=f"{file_prefix}_by_biological_group_{mode}.png",
            target_pathways=target_pathways,
        )
