"""
Plots systematic influence x GSEA results
"""
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import matplotlib.pyplot as plt


class GSEAInfluenceFigures:

    def __init__(self, output_dir: Path) -> None:
        """Initializer."""
        self.__output_dir = output_dir
        self.__output_dir.mkdir(parents=True, exist_ok=True)

    def generate_figures(
        self,
        df: pd.DataFrame,
        mode: str,
        filename: str,
        target_pathways: list[str] | None = None
    ) -> None:
        """Generate systematic influence x GSEA figures."""
        mode_df = df[df["mode"] == mode].copy()

        plot_df = mode_df[
            ["abs_nes", "mean_influence", "fdr_qval", "term", "moa"]
        ].replace([np.inf, -np.inf], np.nan).dropna()

        # Keep only target pathways, if specified
        if target_pathways is not None:
            plot_df = plot_df[
                plot_df["term"].isin(target_pathways)
            ]

        if plot_df.empty:
            return

        term_correlations = {}
        for term, term_df in plot_df.groupby('term'):
            if len(term_df) >= 3:
                rho, pval = spearmanr(
                    term_df["abs_nes"],
                    term_df["mean_influence"]
                )
                term_correlations[term] = (rho, pval)

        # sort by those with the highest absolute spearman
        # correlation in either direction
        terms = sorted(
            term_correlations,
            key=lambda term: abs(term_correlations[term][0]),
            reverse=True
        )

        # MOA colours shared across all subplots
        moas = sorted(plot_df["moa"].unique())
        colors = plt.get_cmap("turbo", len(moas))

        moa_colors = {
            moa: colors(i)
            for i, moa in enumerate(moas)
        }

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

            x = term_df["abs_nes"]
            y = term_df["mean_influence"]

            significant = term_df["fdr_qval"] < 0.05

            point_colors = [
                moa_colors[moa] if sig else "lightgrey"
                for moa, sig in zip(term_df["moa"], significant)
            ]

            # Non-significant points first
            ax.scatter(
                x[~significant],
                y[~significant],
                color="lightgrey",
                alpha=0.4,
                s=8,
                zorder=1,
            )

            # Significant points on top
            ax.scatter(
                x[significant],
                y[significant],
                c=[moa_colors[moa] for moa in term_df.loc[significant, "moa"]],
                alpha=0.9,
                s=10,
                zorder=2,
            )

            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

            rho, pval = term_correlations[term]
            ax.set_title(
                f"{term}\n$r_s = {rho:.2f},\\ p = {pval:.2e}$",
                fontsize=8,
            )

        # Remove unused subplot axes
        for ax in axes[len(terms):]:
            ax.remove()

        # Shared MOA legend
        legend_elements = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=moa_colors[moa],
                markersize=6,
                label=moa,
            )
            for moa in moas
        ]

        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="lightgrey",
                markersize=6,
                label="Non-significant",
            )
        )

        fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.005, 0.855),
            loc="upper left",
            fontsize=7,
            title="MOA",
        )

        fig.suptitle(
            f"Enrichment strength vs signed pathway influence ({mode})",
            fontsize=14,
        )

        fig.supxlabel("Absolute NES", fontsize=10)
        fig.supylabel("Mean influence", fontsize=10)

        fig.tight_layout(rect=[0.015, 0, 1, 0.98])

        fig.savefig(
            self.__output_dir / filename,
            bbox_inches="tight",
            dpi=300,
        )
