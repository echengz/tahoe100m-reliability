"""
Helps generate systematic figures
"""
import re
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


SPECIFICITY_COLORS = {
    "Specific": "#1f77b4",
    "Shared": "#ff7f0e",
    "Unreliable": "#2ca02c",
}


def __grab_latest_systema_results(systema_dir: Path, prefix: str) -> pd.DataFrame:
    """Grab the latest systema results."""
    dated_pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d{{8}}-\d{{6}})\.csv$"
    )
    earliest_file = {
        'path': None,
        'timestamp': None
    }

    for path in systema_dir.glob(f"{prefix}*.csv"):
        match = dated_pattern.match(path.name)
        if match:
            timestamp = match.groups()[0]
            timestamp = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")

            if earliest_file['timestamp'] is None or timestamp < earliest_file['timestamp']:
                earliest_file = {
                    'path': path,
                    'timestamp': timestamp
                }

    if earliest_file['path'] is None:
        raise FileNotFoundError(f"Cannot find latest {prefix} file")

    return pd.read_csv(earliest_file['path'])


def plot_systematic_all(df: pd.DataFrame, output_dir: Path, mode: str):
    """Plot global specificity results."""
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
        output_dir / f"systema_{mode}_perturbation_classification.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_drug_x_dosage_specificity(
    df: pd.DataFrame,
    output_dir: Path,
    mode: str,
):
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
        output_dir / f"systema_{mode}_specificity_by_drug_x_dosage.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_grouped_systematic(systema_dir: Path, output_dir: Path, mode: str):
    """Plot the specificities grouped by drug, drug x dosage, and cell line ID.""" 
    for group in ['drug', 'dosage', 'drug_x_dosage', 'cell_line_id', 'moa']:
        df = __grab_latest_systema_results(
            systema_dir=systema_dir,
            prefix=f'systema_{mode}_cc_reliability_2d_summary_tahoe100m_by_{group}'
        )

        if df.empty:
            continue

        # Handle drug x dosage separately
        if group == "drug_x_dosage":
            plot_drug_x_dosage_specificity(df, output_dir, mode)
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
            output_dir / f"systema_{mode}_specificity_by_{group}.png",
            dpi=300,
            bbox_inches="tight",
        )


def plot_systematic_figures(systema_dir: Path, output_dir: Path, mode: str):
    df_global = __grab_latest_systema_results(
        systema_dir=systema_dir,
        prefix=f'systema_{mode}_cc_summary_tahoe100m'
    )
    plot_systematic_all(
        df=df_global,
        output_dir=output_dir,
        mode=mode
    )

    plot_grouped_systematic(
        systema_dir=systema_dir,
        output_dir=output_dir,
        mode=mode
    )
