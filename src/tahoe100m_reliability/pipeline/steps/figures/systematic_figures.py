"""
Helps generate systematic figures
"""
import re
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from matplotlib.lines import Line2D
from pathlib import Path

log = logging.getLogger(__name__)


SPECIFICITY_COLORS = {
    "Specific": "#1f77b4",
    "Shared": "#ff7f0e",
    "Unreliable": "#2ca02c",
}


class BaseSystematicFigures:
    """Base class for generating systematic figures."""

    def __init__(self, systema_dir: Path, output_dir: Path) -> None:
        """Initializer."""
        self.systema_dir = systema_dir
        self.output_dir = output_dir

    def grab_latest_systema_results(self, prefix: str) -> pd.DataFrame:
        """Grab the latest systema results."""
        prefix_path = Path(prefix)
        search_dir = self.systema_dir / prefix_path.parent
        filename_prefix = prefix_path.name

        dated_pattern = re.compile(
            rf"^{re.escape(filename_prefix)}_(\d{{8}}-\d{{6}})\.csv$"
        )

        latest_file = {"path": None, "timestamp": None}

        for path in search_dir.glob(f"{filename_prefix}_*.csv"):
            match = dated_pattern.match(path.name)

            if match:
                timestamp = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")

                if latest_file["timestamp"] is None or timestamp > latest_file["timestamp"]:
                    latest_file = {"path": path, "timestamp": timestamp}

        if latest_file["path"] is None:
            raise FileNotFoundError(f"Cannot find {prefix} file in {search_dir} (may be missing timestamp?)")

        log.info("Loading latest results from %s", latest_file["path"])
        return pd.read_csv(latest_file["path"])


class SystematicFigures(BaseSystematicFigures):
    """Plots systematic figures for each mode."""
    def __init__(self, systema_dir: Path, output_dir: Path, mode: str) -> None:
        """Initializer."""
        super().__init__(systema_dir, output_dir)
        self.mode = mode

    def plot_systematic_all(self) -> None:
        """Plot global specificity results."""
        df = self.grab_latest_systema_results(
            prefix=f'systema_cc_{self.mode}_summary_tahoe100m'
        )

        categories = ["Unreliable", "Shared", "Specific"]

        row = df.iloc[0]
        proportions = pd.Series({
            "Unreliable": row["frac_Unreliable"],
            "Shared": row["frac_Shared"],
            "Specific": row["frac_Specific"],
        })

        fig, ax = plt.subplots(figsize=(7, 5))

        bars = ax.bar(
            categories,
            proportions.values,
            color=[SPECIFICITY_COLORS[c] for c in categories],
            width=0.8,
        )

        ax.set_ylabel("Fraction of perturbations")
        ax.set_title("Perturbation classification")
        ax.set_ylim(0, 1)

        for bar, value in zip(bars, proportions.values):
            ax.annotate(
                f"{value:.1%}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"systema_{self.mode}_perturbation_classification.png",
            dpi=300,
            bbox_inches="tight",
        )

    def __plot_drug_x_dosage_specificity(self, df: pd.DataFrame) -> None:
        """Plot stacked specificity fractions grouped by drug and dosage."""
        df = df.copy()
        dosage_order = [0.05, 0.5, 5.0]

        df["dosage"] = pd.to_numeric(df["dosage"])
        df = df[df["dosage"].isin(dosage_order)].copy()

        # Sort drugs by mean Specific fraction across available dosages
        drug_order = (
            df.groupby("drug")["frac_Specific"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )

        drug_spacing = 0.35
        bar_width = 1.0
        group_width = len(dosage_order) * bar_width + drug_spacing

        drug_positions = {drug: i for i, drug in enumerate(drug_order)}
        dosage_positions = {dose: i for i, dose in enumerate(dosage_order)}

        df["x"] = (
            df["drug"].map(drug_positions) * group_width
            + df["dosage"].map(dosage_positions) * bar_width
        )

        fig, ax = plt.subplots(figsize=(16, 6))
        bottom = np.zeros(len(df))

        for specificity in ["Unreliable", "Shared", "Specific"]:
            values = df[f"frac_{specificity}"].fillna(0).to_numpy()

            ax.bar(
                df["x"],
                values,
                width=bar_width,
                bottom=bottom,
                color=SPECIFICITY_COLORS[specificity],
                label=specificity,
                linewidth=0,
            )

            bottom += values

        ax.set_xticks([])
        ax.set_xlabel("drug x dosage")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1)
        ax.set_title(f"Specificity by drug x dosage")

        ax.legend(title="Specificity")

        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        plt.savefig(
            self.output_dir / f"systema_{self.mode}_specificity_by_drug_x_dosage.png",
            dpi=300,
            bbox_inches="tight",
        )

    def plot_grouped_systematic(self):
        """Plot the specificities grouped by drug, drug x dosage, and cell line ID.""" 
        for group in ['drug', 'dosage', 'drug_x_dosage', 'cell_line_id', 'moa']:
            df = self.grab_latest_systema_results(
                prefix=f'systema_cc_{self.mode}_reliability_2d_summary_tahoe100m_by_{group}'
            )

            if df.empty:
                continue

            # Handle drug x dosage separately
            if group == "drug_x_dosage":
                self.__plot_drug_x_dosage_specificity(df)
                continue

            # Sort groups by fraction Specific
            if group == "dosage":
                df[group] = pd.to_numeric(df[group])
                df = df.sort_values(group, ascending=True)
                df[group] = df[group].astype(str)
            else:
                df = df.sort_values("frac_Specific", ascending=True)

            # Set up the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Draw stacked bars
            bottom = np.zeros(len(df))
            for specificity in ['Unreliable', 'Shared', 'Specific']:
                values = (
                    df[f"frac_{specificity}"]
                    .fillna(0)
                    .to_numpy()
                )

                ax.bar(
                    df[group].astype(str),
                    values,
                    bottom=bottom,
                    label=specificity,
                    color=SPECIFICITY_COLORS[specificity],
                )

                bottom += values

            # Format plot
            ax.set_xlabel(group)
            ax.set_ylabel("Fraction")
            ax.set_title(f"Specificity by {group}")
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.1, 0.1))

            ax.tick_params(axis="x", rotation=90)
            ax.legend(
                title="Specificity",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )

            # Hide labels for groupings with a lot of labels
            if group in ["drug"]:
                ax.set_xticks([])

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"systema_{self.mode}_specificity_by_{group}.png",
                dpi=300,
                bbox_inches="tight",
            )

    def plot_top_influential_genes(self, top_n: int = 10):
        """
        Plot three vertically stacked categorical scatter plots comparing
        group-specific gene influence against global influence.
        """
        global_df = self.grab_latest_systema_results(
            prefix=f'gene_influences/systema_{self.mode}_top1000_global'
        )

        ranking_col = 'total_influence_abs'
        influence_cols = [
            'mean_influence',
            'mean_positive_influence',
            'mean_negative_influence'
        ]

        # Select top N genes based on global influence
        top_genes_df = (
            global_df
            .sort_values(ranking_col, ascending=False)
            .drop_duplicates('group_id')
            .head(top_n)
            .copy()
        )

        top_genes = top_genes_df['group_id'].tolist()

        # Consistent colors across all subplots
        palette = dict(zip(
            top_genes,
            sns.color_palette('turbo', len(top_genes))
        ))

        for group in ['cell_line_id']:

            df = self.grab_latest_systema_results(
                prefix=f'gene_influences/systema_{self.mode}_top200_by_{group}'
            )

            # Only retain globally selected genes
            df = df[df['group_id'].isin(top_genes)].copy()

            if df.empty:
                raise ValueError(
                    f"Influential genes completely disjoint in {group}"
                )

            # Prepare categorical groups
            categories = sorted(
                df[group].dropna().astype(str).unique()
            )
            categories.append('Global')

            # Create figure with three vertically stacked subplots
            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=(max(10, len(categories) * 0.3), 15),
                sharex=True,
            )

            for ax, influence_col in zip(axes, influence_cols):

                # Group-specific observations
                group_plot_df = df[
                    ['group_id', group, influence_col]
                ].dropna().copy()

                group_plot_df = group_plot_df.rename(
                    columns={group: 'group'}
                )

                # Global observations
                global_plot_df = top_genes_df[
                    ['group_id', influence_col]
                ].copy()

                global_plot_df['group'] = 'Global'

                # Combine observations
                plot_df = pd.concat(
                    [group_plot_df, global_plot_df],
                    ignore_index=True
                )

                plot_df['group'] = pd.Categorical(
                    plot_df['group'].astype(str),
                    categories=categories,
                    ordered=True
                )

                sns.stripplot(
                    data=plot_df,
                    x='group',
                    y=influence_col,
                    hue='group_id',
                    hue_order=top_genes,
                    palette=palette,
                    jitter=0.1,
                    dodge=False,
                    alpha=0.85,
                    size=7,
                    ax=ax,
                    legend=False
                )

                ax.axhline(
                    y=0,
                    color='black',
                    linestyle='-',
                    linewidth=1,
                    alpha=0.9,
                    zorder=0
                )

                ax.set_title(influence_col.replace('_', ' ').title())
                ax.set_ylabel('Influence')
                ax.grid(axis='y', alpha=0.3)
                ax.set_xlabel('')

            # Shared x-axis
            axes[-1].set_xticks(range(len(categories)))
            axes[-1].set_xticklabels(categories, rotation=90)

            # Single shared legend
            handles = [
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=palette[gene],
                    markersize=7,
                    label=gene
                )
                for gene in top_genes
            ]

            fig.legend(
                handles=handles,
                labels=top_genes,
                title='Gene',
                bbox_to_anchor=(1.01, 0.95),
                loc='upper left',
                frameon=True
            )

            fig.suptitle(
                f'Top {top_n} globally influential genes by {group}',
                fontsize=16,
                y=1.01
            )

            sns.despine()
            plt.tight_layout()
            plt.savefig(
                self.output_dir /
                f"systema_{self.mode}_influence_by_{group}.png",
                dpi=300,
                bbox_inches='tight'
            )

            # Separate mean influence plot; mostly done to have more summarized
            # plots to add to the report
            influence_col = 'mean_influence'
            fig, ax = plt.subplots(figsize=(14, 5))

            group_plot_df = df[['group_id', group, influence_col]].dropna().copy()
            group_plot_df = group_plot_df.rename(columns={group: 'group'})

            global_plot_df = top_genes_df[['group_id', influence_col]].copy()
            global_plot_df['group'] = 'Global'

            plot_df = pd.concat(
                [group_plot_df, global_plot_df],
                ignore_index=True
            )

            plot_df['group'] = pd.Categorical(
                plot_df['group'].astype(str),
                categories=categories,
                ordered=True
            )

            sns.stripplot(
                data=plot_df,
                x='group',
                y=influence_col,
                hue='group_id',
                hue_order=top_genes,
                palette=palette,
                jitter=0.1,
                dodge=False,
                alpha=0.7,
                size=5,
                ax=ax
            )

            ax.axhline(
                y=0,
                color='black',
                linestyle='-',
                linewidth=1,
                alpha=0.9,
                zorder=0
            )

            ax.set_title(f'Mean Influence ({self.mode})')
            ax.set_ylabel('Mean Influence')
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', labelrotation=90, labelsize=8)

            ax.legend(
                title='Gene',
                bbox_to_anchor=(1.01, 1),
                loc='upper left',
                frameon=True
            )

            fig.tight_layout()
            plt.savefig(
                self.output_dir / f"systema_{self.mode}_{influence_col}_by_{group}.png",
                dpi=300,
                bbox_inches='tight'
            )

    def generate_figures(self) -> None:
        # Systematic results
        self.plot_systematic_all()
        self.plot_grouped_systematic()

        # Influence results
        self.plot_top_influential_genes()


class CombinedSystematicFigures(BaseSystematicFigures):
    """Plots combined systematic figures comparing the two modes."""
    def __init__(self, systema_dir: Path, output_dir: Path) -> None:
        super().__init__(systema_dir, output_dir)
        self.__colors = {
            'raw': 'tab:blue',
            'cp10k': 'tab:orange',
            'combined': 'tab:purple'
        }

    def plot_change_in_influence(self, df: pd.DataFrame, plot_df: pd.DataFrame) -> None:
        """Plots the change in influence scorse before and after normalization."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Reference line: equal influence in both runs
        lower = min(plot_df["mean_abs_influence_raw"].min(),
                    plot_df["mean_abs_influence_cp10k"].min())
        upper = max(plot_df["mean_abs_influence_raw"].max(),
                    plot_df["mean_abs_influence_cp10k"].max())
        ax.plot([lower, upper], [lower, upper], "--", color="lightgray", alpha=0.65)

        ax.scatter(
            plot_df["mean_abs_influence_raw"],
            plot_df["mean_abs_influence_cp10k"],
            alpha=0.65,
            s=24,
            color="grey"
        )

        # Identify top genes in each dataset, and label their union
        n_top_genes = 5
        top_raw = set(
            df.nlargest(n_top_genes, "mean_abs_influence_raw")["group_id"]
            .dropna()
            .tolist()
        )

        top_cp10k = set(
            df.nlargest(n_top_genes, "mean_abs_influence_cp10k")["group_id"]
            .dropna()
            .tolist()
        )

        for _, row in plot_df[
            plot_df["group_id"].isin(top_raw | top_cp10k)
        ].iterrows():
            xytext = (4, 4)

            # adjust for specific genes
            if row["group_id"] in ["MT-RNR1", "MT-ND1"]:
                xytext = (-30, 6)
            elif row["group_id"] in ["MT-ATP6", "FN1"]:
                xytext = (6, -2)
            elif row["group_id"] == "IGFBP3":
                xytext = (2, 6)
            elif row["group_id"] == "MT-CO1":
                xytext = (4, 2)

            ax.annotate(
                row["group_id"],
                (row["mean_abs_influence_raw"], row["mean_abs_influence_cp10k"]),
                xytext=xytext,
                textcoords="offset points",
                fontsize=8,
            )

        # Highlight top genes
        for genes, color in [
            (top_raw - top_cp10k, self.__colors['raw']),
            (top_cp10k - top_raw, self.__colors['cp10k']),
            (top_raw & top_cp10k, self.__colors['combined']),
        ]:
            subset = plot_df[plot_df["group_id"].isin(genes)]

            ax.scatter(
                subset["mean_abs_influence_raw"],
                subset["mean_abs_influence_cp10k"],
                alpha=0.65,
                s=24, 
                color=color,
            )

        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=self.__colors['raw'],
                   label=f"In top {n_top_genes} for Raw",
                   markersize=7
            ),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=self.__colors['cp10k'],
                   label=f"In top {n_top_genes} for CP10K",
                   markersize=7
            ),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=self.__colors['combined'],
                   label=f"In top {n_top_genes} for both",
                   markersize=7
            ),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="lightgray", label="Other genes", markersize=7),
            Line2D([0], [0], color="lightgray", linestyle="--", alpha=0.65,
                   label="Equal influence (y = x)"),
        ]

        ax.legend(handles=legend_elements, title="Gene influence", fontsize=8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean absolute influence - Raw")
        ax.set_ylabel("Mean absolute influence - CP10K")
        ax.set_title(f"Raw vs CP10K influence of top {len(plot_df)} shared influential genes")
        ax.grid(True, which="both", alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "gene_influence_raw_vs_cp10k.png", dpi=300, bbox_inches="tight"
        )

    def generate_figures(self) -> None:
        # Create combined DF
        influence_cols = [
            "mean_abs_influence",
            "mean_positive_influence",
            "mean_negative_influence",
        ]

        raw_df = self.grab_latest_systema_results(
            prefix="gene_influences/systema_raw_top1000_global"
        )
        cp10k_df = self.grab_latest_systema_results(
            prefix="gene_influences/systema_cp10k_top1000_global"
        )

        raw_df = raw_df[["group_id"] + influence_cols].rename(
            columns={col: f"{col}_raw" for col in influence_cols}
        )

        cp10k_df = cp10k_df[["group_id"] + influence_cols].rename(
            columns={col: f"{col}_cp10k" for col in influence_cols}
        )

        df = raw_df.merge(cp10k_df, on="group_id", how="inner")
        del raw_df, cp10k_df

        # Log axes require strictly positive mean absolute influence
        plot_df = df[
            (df["mean_abs_influence_raw"] > 0) &
            (df["mean_abs_influence_cp10k"] > 0)
        ].copy()

        self.plot_change_in_influence(df, plot_df)