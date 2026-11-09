"""
Help generate GSEA figures
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection

from tahoe100m_reliability.lib.ingest.tahoe100m_dataset import (
    get_cell_line_metadata,
)

# ============================================================
# SETTINGS
# ============================================================

FDR_THRESHOLD = 0.05
NON_SIGNIFICANT_ALPHA = 0.40

CELL_BORDER_COLOR = "white"
CELL_BORDER_WIDTH = 0.3

GROUP_SEPARATOR_COLOR = "white"
GROUP_SEPARATOR_WIDTH = 0.8

# Approximate display size per pathway column and cell-line row
COLUMN_WIDTH = 0.25
ROW_HEIGHT = 0.30

# Small white gap between organ groups
ORGAN_GAP = 0.8

# Compact left-side labels
ORGAN_LABEL_WIDTH = 0.1
CELL_LABEL_WIDTH = 0.1
HEATMAP_LEFT = 0.25

# Colorbar is positioned to right of heatmap
COLORBAR_GAP = 0.02
COLORBAR_WIDTH = 0.018

HALLMARK_GROUPINGS = {
    "Apical Junction":"Cellular\nComponent",
    "Apical Surface":"Cellular\nComponent",
    "Peroxisome":"Cellular\nComponent",
    "Adipogenesis":"Development",
    "Angiogenesis":"Development",
    "Epithelial Mesenchymal Transition":"Development",
    "Myogenesis":"Development",
    "Pancreas Beta Cells":"Development",
    "Spermatogenesis":"Development",
    "DNA Repair":"DNA Damage",
    "UV Response Dn":"DNA Damage",
    "UV Response Up":"DNA Damage",
    "Allograft Rejection":"Immune",
    "Coagulation":"Immune",
    "Complement":"Immune",
    "Inflammatory Response":"Immune",
    "Interferon Alpha Response":"Immune",
    "Interferon Gamma Response":"Immune",
    "Bile Acid Metabolism":"Metabolic",
    "Cholesterol Homeostasis":"Metabolic",
    "Fatty Acid Metabolism":"Metabolic",
    "Glycolysis":"Metabolic",
    "heme Metabolism":"Metabolic",
    "Oxidative Phosphorylation":"Metabolic",
    "Reactive Oxygen Species Pathway":"Metabolic",
    "Xenobiotic Metabolism":"Metabolic",
    "Androgen Response":"Pathway (Signaling)",
    "Estrogen Response Early":"Pathway (Signaling)",
    "Estrogen Response Late":"Pathway (Signaling)",
    "Hedgehog Signaling":"Pathway (Signaling)",
    "IL-2/STAT5 Signaling":"Pathway (Signaling)",
    "IL-6/JAK/STAT3 Signaling":"Pathway (Signaling)",
    "KRAS Signaling Dn":"Pathway (Signaling)",
    "KRAS Signaling Up":"Pathway (Signaling)",
    "mTORC1 Signaling":"Pathway (Signaling)",
    "Notch Signaling":"Pathway (Signaling)",
    "PI3K/AKT/mTOR  Signaling":"Pathway (Signaling)",
    "Protein Secretion":"Pathway (Signaling)",
    "TGF-beta Signaling":"Pathway (Signaling)",
    "TNF-alpha Signaling via NF-kB":"Pathway (Signaling)",
    "Wnt-beta Catenin Signaling":"Pathway (Signaling)",
    "Apoptosis":"Proliferation (Cell Cycle)",
    "E2F Targets":"Proliferation (Cell Cycle)",
    "G2-M Checkpoint":"Proliferation (Cell Cycle)",
    "Mitotic Spindle":"Proliferation (Cell Cycle)",
    "Myc Targets V1":"Proliferation (Cell Cycle)",
    "Myc Targets V2":"Proliferation (Cell Cycle)",
    "p53 Pathway":"Proliferation (Cell Cycle)",
    "Hypoxia":"Stress",
    "Unfolded Protein Response":"Stress"
}


# ============================================================
# BIOLOGICAL PATHWAY GROUPS
# ============================================================

def prepare_nes_matrices(df, organ_by_cell_line):
    """
    Use the complete configured Hallmark pathway set and shared
    cell-line order for raw and CP10k.
    """
    # Always include every configured Hallmark pathway, regardless of
    # significance or whether it appears in the input data
    terms = sorted(HALLMARK_GROUPINGS.keys())

    cell_lines = sorted(
        df["cell_line"].unique(),
        key=lambda cell_id: (
            organ_by_cell_line.get(
                cell_id, "Unknown organ"
            ).lower(),
            cell_id.lower(),
        ),
    )

    matrices = {}

    for mode in ["raw", "cp10k"]:
        subset = df[df["mode"] == mode].copy()

        nes = subset.pivot(
            index="term",
            columns="cell_line",
            values="NES",
        )

        fdr = subset.pivot(
            index="term",
            columns="cell_line",
            values="FDR_qval",
        )

        nes = nes.reindex(index=terms, columns=cell_lines)
        fdr = fdr.reindex(index=terms, columns=cell_lines)

        matrices[mode] = {
            "nes": nes,
            "fdr": fdr,
        }

    return matrices


def prepare_influence_matrices(gsea_df, influence_df, nes_matrix):
    gsea = gsea_df.copy()
    influence = influence_df.copy()

    # Harmonize gene symbols
    gsea["lead_genes"] = gsea["lead_genes"].fillna("")
    gsea["lead_genes"] = gsea["lead_genes"].str.split(";")

    influence["gene_symbol"] = (
        influence["gene_symbol"].str.upper().str.strip()
    )

    # One influence score per gene and cell line is expected
    if influence.duplicated(["cell_line_id", "gene_symbol"]).any():
        raise ValueError(
            "Duplicate gene/cell-line influence rows; "
            "aggregate them before joining."
        )

    # Expand each pathway's leading edge into individual genes
    long = gsea.explode("lead_genes").rename(
        columns={"lead_genes": "gene_symbol"}
    )

    long["gene_symbol"] = (
        long["gene_symbol"].str.upper().str.strip()
    )

    long = long[
        long["gene_symbol"].notna()
        & long["gene_symbol"].ne("")
    ]

    # Match leading-edge genes to their cell-line-specific scores
    merged = long.merge(
        influence[
            [
                "cell_line_id",
                "gene_symbol",
                "avg_influence",
                "mean_abs_influence",
            ]
        ],
        left_on=["cell_line", "gene_symbol"],
        right_on=["cell_line_id", "gene_symbol"],
        how="left",
        validate="many_to_one",
    )

    stats = (
        merged.groupby(
            ["mode", "cell_line", "term"],
            dropna=False,
        )
        .agg(
            mean_influence=("avg_influence", "mean"),
            median_influence=("avg_influence", "median"),
            mean_abs_influence=("mean_abs_influence", "mean"),
            pct_positive=(
                "avg_influence",
                lambda x: 100 * (x.dropna() > 0).mean()
                if x.notna().any() else np.nan,
            ),
            n_influence_genes=("avg_influence", "count"),
        )
        .reset_index()
    )

    matrices = {}

    for mode in ["raw", "cp10k"]:
        subset = stats[stats["mode"] == mode]

        matrices[mode] = {
            metric: (
                subset.pivot(
                    index="term",
                    columns="cell_line",
                    values=metric,
                )
                .reindex(
                    index=nes_matrix.index,
                    columns=nes_matrix.columns,
                )
            )
            for metric in [
                "mean_influence",
                "median_influence",
                "mean_abs_influence",
                "pct_positive",
                "n_influence_genes",
            ]
        }

    return matrices


def prepare_grouped_plot_data(nes, fdr):
    """
    Add blank display columns before pathway groups.

    The returned display arrays are separate from the CSV matrices.
    """
    terms = sorted(nes.index.tolist())
    cell_lines = nes.columns.tolist()

    plot_columns = []
    plot_nes_columns = []
    plot_fdr_columns = []
    column_types = []

    groups = set()
    for term in terms:
        groups.add(HALLMARK_GROUPINGS[term])

    for group in sorted(list(groups)):
        plot_columns.append(group)
        plot_nes_columns.append(np.full(len(cell_lines), np.nan))
        plot_fdr_columns.append(np.full(len(cell_lines), np.nan))
        column_types.append("group")

        for term in terms:
            if HALLMARK_GROUPINGS[term] != group:
                continue

            plot_columns.append(term)
            plot_nes_columns.append(
                nes.loc[term, cell_lines].to_numpy(dtype=float)
            )
            plot_fdr_columns.append(
                fdr.loc[term, cell_lines].to_numpy(dtype=float)
            )
            column_types.append("pathway")

    return (
        np.column_stack(plot_nes_columns),
        np.column_stack(plot_fdr_columns),
        plot_columns,
        column_types,
        cell_lines,
    )


def build_organ_layout(cell_lines, organ_by_cell_line):
    """
    Return cell-line y positions, organ spans, and total display height.

    A small blank space is inserted between organ blocks.
    """

    organ_labels = [
        organ_by_cell_line.get(cell_id, "Unknown organ").replace('/', ' /\n')
        for cell_id in cell_lines
    ]

    for i, organ in enumerate(organ_labels):
        if organ == 'Peripheral Nervous System':
            organ_labels[i] = 'Peripheral\nNervous System'

    row_positions = []
    organ_spans = []

    y = 0.0
    i = 0

    while i < len(cell_lines):
        organ = organ_labels[i]
        first_y = y
        j = i

        while j < len(cell_lines) and organ_labels[j] == organ:
            row_positions.append(y)
            y += 1.0
            j += 1

        organ_spans.append((organ, first_y, y - 1.0))

        if j < len(cell_lines):
            y += ORGAN_GAP

        i = j

    return np.asarray(row_positions), organ_spans, y


def plot_nes_heatmap(
    nes,
    fdr,
    title,
    output_path,
    vmax,
    organ_by_cell_line,
):
    """Plot cell lines as rows and grouped pathways as columns."""

    (
        values,
        qvalues,
        column_labels,
        column_types,
        cell_lines,
    ) = prepare_grouped_plot_data(nes, fdr)

    if len(cell_lines) == 0 or values.shape[1] == 0:
        return

    row_positions, organ_spans, total_height = build_organ_layout(
        cell_lines,
        organ_by_cell_line,
    )

    n_rows, n_cols = values.shape

    # Set NES color and FDR opacity
    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    cmap = plt.get_cmap("coolwarm")
    safe_values = np.where(np.isfinite(values), values, 0.0)
    rgba = cmap(norm(safe_values))

    significant = np.isfinite(qvalues) & (qvalues < FDR_THRESHOLD)

    rgba[:, :, 3] = np.where(
        np.isfinite(values),
        np.where(significant, 1.0, NON_SIGNIFICANT_ALPHA),
        1.0,
    )

    missing = ~np.isfinite(values)
    rgba[missing, :3] = [0.88, 0.88, 0.88]
    rgba[missing, 3] = 1.0

    # Blank pathway-group columns remain white
    for col, kind in enumerate(column_types):
        if kind == "group":
            rgba[:, col, :3] = 1.0
            rgba[:, col, 3] = 1.0

    # Set layout
    heatmap_width = max(10.0, n_cols * COLUMN_WIDTH)
    fig_width = heatmap_width + 1.0
    fig_height = max(4.5, min(12.0, total_height * ROW_HEIGHT + 1.5))

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use a compact, explicit layout.
    left_labels_left = 0.015
    organ_axis_width = 0.05
    cell_axis_left = 0.123
    cell_axis_width = 0.02

    heatmap_left = 0.14
    heatmap_right = 0.96
    heatmap_bottom = 0.24
    heatmap_top = 0.82

    ax = fig.add_axes([
        heatmap_left,
        heatmap_bottom,
        heatmap_right - heatmap_left,
        heatmap_top - heatmap_bottom,
    ])

    organ_ax = fig.add_axes([
        left_labels_left,
        heatmap_bottom,
        organ_axis_width,
        heatmap_top - heatmap_bottom,
    ])

    cell_label_ax = fig.add_axes([
        cell_axis_left,
        heatmap_bottom,
        cell_axis_width,
        heatmap_top - heatmap_bottom,
    ])

    # Heatmap cells, preserving white organ gaps
    polygons = []
    facecolors = []

    for r, y in enumerate(row_positions):
        for c in range(n_cols):
            polygons.append([
                (c - 0.5, y - 0.5),
                (c + 0.5, y - 0.5),
                (c + 0.5, y + 0.5),
                (c - 0.5, y + 0.5),
            ])
            facecolors.append(rgba[r, c])

    cells = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=CELL_BORDER_COLOR,
        linewidths=CELL_BORDER_WIDTH,
    )
    ax.add_collection(cells)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(total_height - 0.5, -0.5)

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=30,
    )

    # Cell-line IDs in a separate axis
    cell_label_ax.set_xlim(0, 1)
    cell_label_ax.set_ylim(total_height - 0.5, -0.5)
    cell_label_ax.axis("off")

    for cell_id, y in zip(cell_lines, row_positions):
        if cell_id in ['CVCL_1531', 'CVCL_1517', 'CVCL_1571']:
            cell_id = f'* {cell_id}'

        cell_label_ax.text(
            0.98,
            y,
            cell_id,
            ha="right",
            va="center",
            fontsize=8,
            clip_on=False,
        )

    # Organ labels on the left
    organ_ax.set_xlim(0, 1)
    organ_ax.set_ylim(total_height - 0.5, -0.5)
    organ_ax.axis("off")

    for organ, first_y, last_y in organ_spans:
        center = (first_y + last_y) / 2

        organ_ax.text(
            0.98,
            center,
            organ,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="medium",
            clip_on=False,
        )

    # Pathway labels along the bottom
    pathway_positions = [
        i for i, kind in enumerate(column_types)
        if kind == "pathway"
    ]

    pathway_labels = [column_labels[i] for i in pathway_positions]

    ax.set_xticks(pathway_positions)
    ax.set_xticklabels(
        pathway_labels,
        rotation=90,
        fontsize=7,
        ha="center",
    )
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Pathway group labels above the heatmap
    group_spans = []
    start = None
    current_group = None

    for col, kind in enumerate(column_types):
        if kind == "group":
            if current_group is not None:
                group_spans.append((current_group, start, col - 1))

            current_group = column_labels[col]
            start = col + 1

    if current_group is not None:
        group_spans.append((current_group, start, n_cols - 1))

    for group, first_col, last_col in group_spans:
        center = (first_col + last_col) / 2
        wrapped_group = group.replace(" / ", " /\n")

        ax.text(
            center,
            1.01,
            wrapped_group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            linespacing=1.1,
            clip_on=False,
        )

        # White separator between pathway groups
        if last_col < n_cols - 1:
            ax.axvline(
                last_col + 0.5,
                color=GROUP_SEPARATOR_COLOR,
                linewidth=GROUP_SEPARATOR_WIDTH,
                zorder=5,
            )

    # Add NES colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar_ax = fig.add_axes([
        heatmap_right + COLORBAR_GAP,
        heatmap_bottom,
        COLORBAR_WIDTH,
        heatmap_top - heatmap_bottom,
    ])

    colorbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    colorbar.set_label("Normalized Enrichment Score (NES)", fontsize=8, labelpad=5)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=2)

    # Significance note in the bottom margin
    fig.text(
        0.99,
        0.015,
        (
            f"Opaque: FDR < {FDR_THRESHOLD}  |  "
            f"Faded: FDR ≥ {FDR_THRESHOLD}  |  "
            "Gray: missing NES  |  "
            "*  Low cell coverage"
        ),
        ha="right",
        va="bottom",
        fontsize=7,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def characterize_nes_results(df: pd.DataFrame):
    """Characterize NES results to identify which pathways are worth looking at."""
    df = df.copy()
    df = df[np.isfinite(df["nes"]) & np.isfinite(df["fdr_qval"])]

    sig = df[df["fdr_qval"] < FDR_THRESHOLD]

    # Fraction of tested pathway-cell-line pairs that are significant
    significant_fraction = (
        len(sig) / len(df) if len(df) else np.nan
    )

    # How selectively pathways are enriched across cell lines
    pathway_prevalence = (
        sig.groupby("term")["cell_line"].nunique()
        / df.groupby("term")["cell_line"].nunique()
    )

    pathway_specificity = (
        (4 * pathway_prevalence * (1 - pathway_prevalence)).mean()
        if len(pathway_prevalence) else np.nan
    )

    # Pathways with significant enrichment in both directions
    directions = sig.assign(
        direction=np.sign(sig["nes"])
    ).groupby("term")["direction"].nunique()

    directional_contrast = (
        (directions > 1).sum() / len(directions)
        if len(directions) else 0
    )

    return pd.Series({
        "n_significant": len(sig),
        "significant_fraction": significant_fraction,
        "mean_abs_NES": sig["nes"].abs().mean(),
        "n_pathways": sig["term"].nunique(),
        "n_cell_lines": sig["cell_line"].nunique(),
        "pathway_specificity": pathway_specificity,
        "directional_contrast": directional_contrast,
    })


def plot_gsea_figures(
    gsea_df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str,
    organ_by_cell_line: dict[str, Any],
    title_postfix: str,
    influence_df: pd.DataFrame | None = None
) -> None:
    """Plot GSEA results.

    Assumes as input the DF generated from the GSEA step, saved as
    `{axis}_shared_axis_hallmark_gsea.csv`

    If influence DF is provided, also plots corresponding mean influence
    scores of the leading genes.
    """
    # fix Pperoxisome typo
    gsea_df.loc[gsea_df['term'] == 'Pperoxisome', 'term'] = 'Peroxisome'

    # prepare matrices
    nes_matrices = prepare_nes_matrices(gsea_df, organ_by_cell_line)

    # characterize the NES results to identify interesting pathways
    characterization = (
        gsea_df.groupby("mode")
        .apply(characterize_nes_results)
        .reset_index()
    )

    characterization.insert(0, "figure", filename_prefix)
    characterization.to_csv(
        output_dir / f"{filename_prefix}_characterization.csv",
        index=False
    )

    if influence_df is not None:
        inf_matrices = prepare_influence_matrices(gsea_df, influence_df, nes_matrices)
    else:
        inf_matrices = None

    # Same NES color scale for raw and CP10k
    all_nes = pd.concat(
        [
            nes_matrices["raw"]["nes"].stack(),
            nes_matrices["cp10k"]["nes"].stack(),
        ],
        ignore_index=True,
    )

    all_nes = all_nes.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if all_nes.empty:
        raise ValueError(
            "No finite NES values are available to plot."
        )

    vmax = float(np.abs(all_nes).max())
    vmax = max(vmax, 1e-6)

    for mode in ["raw", "cp10k"]:
        nes = nes_matrices[mode]["nes"]
        fdr = nes_matrices[mode]["fdr"]

        output_path = output_dir / f"{filename_prefix}_{mode}_nes_heatmap.png"

        plot_nes_heatmap(
            nes=nes,
            fdr=fdr,
            title=f"Shared-axis Hallmark GSEA ({mode.upper()}) - {title_postfix}",
            output_path=output_path,
            vmax=vmax,
            organ_by_cell_line=organ_by_cell_line,
        )

        # if inf_matrices:
        #     plot_influence_heatmap(
        #         gsea_df=gsea_df,
        #     )
