"""
Plots systematic influence x GSEA results
"""
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.lines import Line2D
from scipy.stats import linregress, spearmanr

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

        # flag significant NES values
        plot_df['significant'] = plot_df['fdr_qval'] < 0.05

        return plot_df

    def __compute_group_correlations(
        self, plot_df: pd.DataFrame, group_by: str = 'term'
    ) -> tuple[list, dict]:
        """Compute group correlations. Computes a score using the
        spearman rank correlation weighted by the number of significant
        NES terms.

        Returns tuple of list of sorted groups with highest scores as a dict.
        """
        group_correlations = {}
        for term, term_df in plot_df.groupby(group_by):
            if len(term_df) >= 3:
                rho, pval = spearmanr(
                    term_df["abs_nes"],
                    term_df["mean_influence"]
                )

                # weight by proportion of significant NES values
                n_total = len(term_df)
                n_sig = term_df["significant"].sum()
                score = abs(rho) * (np.log1p(n_sig) / np.log1p(n_total))

                group_correlations[term] = {
                    "rho": rho,
                    "pval": pval,
                    "n_sig": n_sig,
                    "n_total": n_total,
                    "score": score,
                }

        if not group_correlations:
            raise ValueError(f"no {group_by} correlations found")

        # sort by those with the highest score in either direction
        groups = sorted(
            group_correlations,
            key=lambda group: abs(group_correlations[group]['score']),
            reverse=True
        )

        return groups, group_correlations

    def __plot_term_df(
        self,
        ax,
        term_df: pd.DataFrame,
        color_by: str,
        colors: dict,
    ) -> None:
        """Plot the provided term df along with its best fit on the given axis,
        along with best fits if provided.
        """
        x = term_df["abs_nes"]
        y = term_df["mean_influence"]

        significant = term_df["significant"]

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
            alpha=0.9,
            s=10,
            zorder=3,
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

            # add title with the weighted spearmanr score
            results = term_correlations[term]
            ax.set_title(
                f"{term}\nweighted $r_s = {results['score']:.2f},\\ p = {results['pval']:.2e}$",
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

    def __apply_best_fit_to_terms(self, df: pd.DataFrame) -> dict:
        """Apply z-score normalized linear regression to significant
        observations of each pathway.
        """
        best_fits = {}

        for term, term_df in df.groupby("term"):
            sig_df = term_df.loc[term_df["significant"]]

            x = sig_df["abs_nes"]
            y = sig_df["mean_influence"]

            best_fits[term] = {
                'x_mean': x.mean(), 'x_std': x.std(), 'y_mean': y.mean(), 'y_std': y.std()
            }

            if len(x) < 3 or x.std() == 0 or y.std() == 0:
                best_fits[term].update({"slope": None, "intercept": None})
                continue

            x_norm = (x - x.mean()) / x.std()
            y_norm = (y - y.mean()) / y.std()

            slope, intercept, _, _, _ = linregress(x_norm, y_norm)
            best_fits[term].update({"slope": slope, "intercept": intercept})

        return best_fits

    def __plot_best_fits(
        self, ax, term_df: pd.DataFrame, best_fits: dict, colors: dict
    ) -> None:
        """Plot best fit lines."""
        for term in term_df["term"].unique():
            fit = best_fits.get(term)
            if fit is None or fit["slope"] is None:
                continue

            # Generate fitted coordinates to significant values
            sig_df = term_df.loc[
                (term_df["term"] == term) & term_df["significant"]
            ]

            x_fit = np.linspace(
                sig_df["abs_nes"].min(),
                sig_df["abs_nes"].max(),
                100,
            )

            x_fit_norm = (
                x_fit - fit["x_mean"]
            ) / fit["x_std"]

            y_fit_norm = (
                fit["slope"] * x_fit_norm + fit["intercept"]
            )

            # Transform fitted y values back to original scale
            y_fit = y_fit_norm * fit["y_std"] + fit["y_mean"]

            ax.plot(
                x_fit,
                y_fit,
                color=colors[term],
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                zorder=2,
            )

    def __add_axis_legend(
        self,
        ax,
        title: str,
        pathways: list[str],
        best_fits: list[float],
        colors: dict,
        fontsize: int = 8.5,
    ) -> None:
        """Add a pathway legend to an individual subplot."""

        legend_elements = []
        for pathway in pathways:
            slope = 'N/A'
            if pathway in best_fits and best_fits[pathway]['slope']:
                slope = f"{best_fits[pathway]['slope']:.2f}"

            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[pathway],
                    markersize=fontsize - 1,
                    label=f'{pathway}, $m_s$= {slope}',
                )
            )

        ax.legend(
            handles=legend_elements,
            title=title,
            fontsize=fontsize,
            title_fontsize=fontsize,
            loc="upper left",
            frameon=False,
            ncols=2 if len(pathways) > 9 else 1,
            columnspacing=1,
            handletextpad=0.4,
        )

    def generate_by_biological_group(
            self,
            df: pd.DataFrame,
            mode: str,
            filename: str,
            target_pathways: list[str] | None = None,
            num_top_groups: int | None = None
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

            for term, result in pathway_correlations.items():
                group = HALLMARK_GROUPINGS.get(term, "Other")
                group_correlations.setdefault(group, []).append(abs(result['score']))

            group_correlations = {
                group: np.mean(score)
                for group, score in group_correlations.items()
            }

            groups = sorted(
                group_correlations,
                key=lambda group: group_correlations[group],
                reverse=True,
            )

            if num_top_groups:
                groups = groups[:num_top_groups]

            # Assign pathway colors using the turbo colormap based on the best fits
            best_fits = self.__apply_best_fit_to_terms(plot_df)
            cmap = plt.get_cmap("turbo")
            pathway_colors = {}

            for group in groups:
                group_terms = plot_df.loc[
                    plot_df["group"] == group, "term"
                ].unique()

                sorted_terms = sorted(
                    group_terms,
                    key=lambda term: (
                        best_fits[term]["slope"]
                        if best_fits[term]["slope"] is not None
                        else float("-inf")
                    ),
                    reverse=True,
                )

                positions = np.linspace(0.05, 0.95, len(sorted_terms))

                for term, position in zip(sorted_terms, positions):
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

                # get terms and sort by their slope
                terms = sorted(group_df["term"].unique())
                sorted_terms = sorted(
                    terms,
                    key=lambda p: best_fits[p]['slope'] if best_fits[p]['slope']
                        is not None else float("-inf"),
                    reverse=True,
                )

                # plot in reverse order so highest slopes are plotted on top
                for term in reversed(sorted_terms):
                    term_df = group_df[group_df["term"] == term]

                    self.__plot_term_df(
                        ax, term_df, "term", pathway_colors,
                    )

                    self.__plot_best_fits(
                        ax, term_df, best_fits, pathway_colors
                    )

                # Group title with average pathway correlation
                ax.set_title(
                    f"{group}\nWeighted mean |$r_s$| = "
                    f"{group_correlations[group]:.2f}",
                    fontsize=11,
                )

                self.__add_axis_legend(
                    ax, "Pathway", sorted_terms, best_fits, pathway_colors,
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
            filename=f'{file_prefix}_all_pathways_{mode}.png',
            target_pathways=target_pathways,
        )

        self.generate_by_biological_group(
            df,
            mode,
            filename=f"{file_prefix}_by_biological_group_{mode}.png",
            target_pathways=target_pathways,
        )

        self.generate_by_biological_group(
            df,
            mode,
            filename=f"{file_prefix}_by_biological_group_{mode}_top4.png",
            target_pathways=target_pathways,
            num_top_groups=4,
        )
