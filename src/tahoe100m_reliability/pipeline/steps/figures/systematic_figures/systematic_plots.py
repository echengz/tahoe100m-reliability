"""
Plots systematic results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from .base import BaseSystematicFigures


SPECIFICITY_COLORS = {
    "Specific": "#1f77b4",
    "Shared": "#ff7f0e",
    "Unreliable": "#2ca02c",
}


class SystematicFigures(BaseSystematicFigures):
    """Plots systematic figures for each mode."""
    def __init__(
        self, systema_dir: Path, output_dir: Path, mode: str, organ_by_cell_line: dict
    ) -> None:
        """Initializer."""
        super().__init__(systema_dir, output_dir)

        self.mode = mode
        self.organ_by_cell_line = organ_by_cell_line

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
        plt.close(fig)

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
        plt.close(fig)

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
            plt.close(fig)

    def generate_figures(self) -> None:
        # Systematic results
        self.plot_systematic_all()
        self.plot_grouped_systematic()
