"""
Plots GSEA specific residual results
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path


class GSEABarPlots:

    def __init__(self, output_dir: Path) -> None:
        """Initializer."""
        self.__output_dir = output_dir
        self.__output_dir.mkdir(parents=True, exist_ok=True)

    def generate_bar_plot(self, df: pd.DataFrame, group: str, filename: str) -> None:
        """Generate bar plot, where each bar is split by:
            - positive (significant)
            - negative (significant)
        """
        categories = {
            "Non-Significant Negative NES": "#8CB9D9",
            "Non-Signitican Positive NES": "#FFBE7A",
            "Significant Negative NES": "tab:blue",
            "Signitican Positive NES": "tab:orange",
        }

        df["category"] = np.select(
            [
                (df["nes"] < 0) & (df["significant"] == False & (df['mode'] == 'cp10k')),
                (df["nes"] > 0) & (df["significant"] == False & (df['mode'] == 'cp10k')),
                (df["nes"] < 0) & (df["significant"] == True & (df['mode'] == 'cp10k')),
                (df["nes"] > 0) & (df["significant"] == True & (df['mode'] == 'cp10k')),
            ],
            categories.keys(),
            default="Zero NES"
        )

        # Count perturbations per pathway and category, sort by # significant
        counts = (
            df.groupby(["term", "category"])
              .size()
              .unstack(fill_value=0)
        )

        counts = counts.reindex(columns=categories.keys(), fill_value=0)

        fig, ax = plt.subplots(figsize=(max(12, len(counts) * 0.35), 7))
        bottom = np.zeros(len(counts))

        for category, color in categories.items():
            values = counts[category].values

            ax.bar(
                counts.index,
                values,
                bottom=bottom,
                label=category,
                color=color,
                edgecolor="white",
                linewidth=0.3
            )

            bottom += values

        ax.set_xlabel("Pathway")
        ax.set_ylabel(f"# of NES Scores by {group}", labelpad=15)
        ax.set_title(f"NES Scores by {group} per pathway (CP10K)", pad=15)

        plt.xticks(rotation=90, ha="center", fontsize=8)
        ax.margins(x=0.01, y=0.01)

        handles, labels = ax.get_legend_handles_labels()

        ax.legend(
            handles[::-1],
            labels[::-1],
            title="Significance / NES",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(self.__output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def generate_figures(self, gsea_dir: Path) -> None:
        """Generate figures."""
        df = pd.read_csv(gsea_dir / 'global_shared_axis_hallmark_gsea.csv')
        self.generate_bar_plot(
            df, group='cell line (shared)', filename='shared_per_cell_line.png',
        )

        df = pd.read_csv(gsea_dir / 'moa_shared_axis_hallmark_gsea.csv')
        self.generate_bar_plot(
            df, group='cell line x MOA (shared)', filename='shared_per_cell_line_moa.png',
        )

        df = pd.read_csv(gsea_dir / 'cp10k_global_specific_axis_hallmark_gsea.csv')
        self.generate_bar_plot(
            df, group='perturbations (specific)', filename='specific_per_perturbation.png',
        )
