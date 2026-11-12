"""
Circular GSEA heatmaps using pyCirclize.

Geometry:
  - One angular sector per MOA.
  - One angular subdivision per cell line within each MOA sector.
  - One radial row per Hallmark pathway, grouped into biological categories.
  - Entry = NES; opacity = FDR significance.
"""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import IdentityTransform
from pycirclize import Circos


# ============================================================
# SETTINGS
# ============================================================

FDR_THRESHOLD = 0.05
NON_SIGNIFICANT_ALPHA = 0.40

CELL_BORDER_COLOR = "white"
CELL_BORDER_WIDTH = 0 # 0.18

INNER_RADIUS = 27.0
OUTER_RADIUS = 94.0

# Radial gaps between biological pathway groups
PATHWAY_GROUP_GAP = 0.55

# Angular gap reserved for category labels
TOP_GAP = 50  # degrees

# Gap between MOA sectors
SECTOR_GAP = 1.0


# ============================================================
# HALLMARK PATHWAY GROUPINGS
# ============================================================

HALLMARK_GROUPINGS = {
    "Apical Junction": "Cellular\nComponent",
    "Apical Surface": "Cellular\nComponent",
    "Peroxisome": "Cellular\nComponent",

    "Adipogenesis": "Development",
    "Angiogenesis": "Development",
    "Epithelial Mesenchymal Transition": "Development",
    "Myogenesis": "Development",
    "Pancreas Beta Cells": "Development",
    "Spermatogenesis": "Development",

    "DNA Repair": "DNA Damage",
    "UV Response Dn": "DNA Damage",
    "UV Response Up": "DNA Damage",

    "Allograft Rejection": "Immune",
    "Coagulation": "Immune",
    "Complement": "Immune",
    "Inflammatory Response": "Immune",
    "Interferon Alpha Response": "Immune",
    "Interferon Gamma Response": "Immune",

    "Bile Acid Metabolism": "Metabolic",
    "Cholesterol Homeostasis": "Metabolic",
    "Fatty Acid Metabolism": "Metabolic",
    "Glycolysis": "Metabolic",
    "heme Metabolism": "Metabolic",
    "Oxidative Phosphorylation": "Metabolic",
    "Reactive Oxygen Species Pathway": "Metabolic",
    "Xenobiotic Metabolism": "Metabolic",

    "Androgen Response": "Pathway\n(Signaling)",
    "Estrogen Response Early": "Pathway\n(Signaling)",
    "Estrogen Response Late": "Pathway\n(Signaling)",
    "Hedgehog Signaling": "Pathway\n(Signaling)",
    "IL-2/STAT5 Signaling": "Pathway\n(Signaling)",
    "IL-6/JAK/STAT3 Signaling": "Pathway\n(Signaling)",
    "KRAS Signaling Dn": "Pathway\n(Signaling)",
    "KRAS Signaling Up": "Pathway\n(Signaling)",
    "mTORC1 Signaling": "Pathway\n(Signaling)",
    "Notch Signaling": "Pathway\n(Signaling)",
    "PI3K/AKT/mTOR  Signaling": "Pathway\n(Signaling)",
    "Protein Secretion": "Pathway\n(Signaling)",
    "TGF-beta Signaling": "Pathway\n(Signaling)",
    "TNF-alpha Signaling via NF-kB": "Pathway\n(Signaling)",
    "Wnt-beta Catenin Signaling": "Pathway\n(Signaling)",

    "Apoptosis": "Proliferation\n(Cell Cycle)",
    "E2F Targets": "Proliferation\n(Cell Cycle)",
    "G2-M Checkpoint": "Proliferation\n(Cell Cycle)",
    "Mitotic Spindle": "Proliferation\n(Cell Cycle)",
    "Myc Targets V1": "Proliferation\n(Cell Cycle)",
    "Myc Targets V2": "Proliferation\n(Cell Cycle)",
    "p53 Pathway": "Proliferation\n(Cell Cycle)",

    "Hypoxia": "Stress",
    "Unfolded Protein Response": "Stress",
}


# ============================================================
# DATA HELPERS
# ============================================================


def _ordered_cell_lines(
    df: pd.DataFrame,
    organ_by_cell_line: Mapping[str, Any]
) -> list[str]:
    """Order cell lines by organ, then by cell-line name."""

    cell_lines = (
        df["cell_line"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(
        cell_lines,
        key=lambda cell_id: (
            str(
                organ_by_cell_line.get(
                    cell_id,
                    "Unknown organ"
                )
            ).lower(),
            cell_id.lower(),
        ),
    )


# ============================================================
# PATHWAY RADIAL LAYOUT
# ============================================================

def _pathway_layout(
    pathways: list[str],
    hallmark_groupings: Mapping[str, str]
):
    """
    Assign radial bounds to pathways.

    Pathways are concentric circular bands.
    Categories are separated by radial gaps.

    Returns:
        pathway_order
        category_spans: (category, first_index, last_index)
        pathway_bounds: pathway -> (inner_radius, outer_radius)
    """

    categories = sorted({
        hallmark_groupings[p]
        for p in pathways
    }, reverse=True)

    pathway_order = []
    category_spans = []

    for category in categories:

        members = sorted(
            [
                p for p in pathways
                if hallmark_groupings[p] == category
            ],
            reverse=True,
        )

        first = len(pathway_order)
        pathway_order.extend(members)
        last = len(pathway_order) - 1

        category_spans.append(
            (category, first, last)
        )

    n_pathways = len(pathway_order)
    n_gaps = max(0, len(category_spans) - 1)

    usable_radius = (
        OUTER_RADIUS
        - INNER_RADIUS
        - n_gaps * PATHWAY_GROUP_GAP
    )

    if usable_radius <= 0:
        raise ValueError("Pathway gaps leave no room for pathway rows.")

    row_height = usable_radius / n_pathways

    pathway_bounds = {}
    cursor = INNER_RADIUS

    for category, first, last in category_spans:

        for pathway in pathway_order[first:last + 1]:

            r0 = cursor
            r1 = cursor + row_height

            pathway_bounds[pathway] = (r0, r1)

            cursor = r1

        # Gap between biological categories
        cursor += PATHWAY_GROUP_GAP

    return pathway_order, category_spans, pathway_bounds


# ============================================================
# TILE COLOR
# ============================================================

def _make_tile_color(value, qvalue, norm, cmap):
    """
    NES determines color.
    FDR determines opacity.
    Missing NES is gray.
    """

    if pd.isna(value) or not np.isfinite(value):
        return (0.88, 0.88, 0.88, 1.0)

    rgba = list(cmap(norm(value)))

    if pd.notna(qvalue) and np.isfinite(qvalue) and qvalue < FDR_THRESHOLD:
        alpha = 1.0
    else:
        alpha = NON_SIGNIFICANT_ALPHA

    rgba[3] = alpha

    return tuple(rgba)


def _outline_moa_clusters(
    circos,
    moa_clusters,
    inner_radius=INNER_RADIUS,
    outer_radius=OUTER_RADIUS,
    color="black",
    linewidth=0.5,
):
    """Draw a continuous closed outline around each MOA cluster."""

    ax = circos.ax

    sectors = {s.name: s for s in circos.sectors}

    for cluster, moas in moa_clusters.items():

        # Preserve the actual circular plotting order
        cluster_sectors = [
            s for s in circos.sectors
            if s.name in moas
        ]

        if not cluster_sectors:
            continue

        # Check that all expected MOAs were found
        if len(cluster_sectors) != len(moas):
            missing = set(moas) - {
                s.name for s in cluster_sectors
            }
            print(f"Cluster {cluster}: missing MOAs {missing}")

        # Use pyCirclize's own radian coordinates
        start = cluster_sectors[0].rad_lim[0]
        end = cluster_sectors[-1].rad_lim[1]

        theta = np.linspace(start, end, 300)

        # Outer boundary
        ax.plot(
            theta,
            np.full_like(theta, outer_radius),
            color=color,
            linewidth=linewidth,
            zorder=30,
        )

        # Inner boundary
        ax.plot(
            theta,
            np.full_like(theta, inner_radius),
            color=color,
            linewidth=linewidth,
            zorder=30,
        )

        # Connecting boundaries
        for angle in (start, end):
            ax.plot(
                [angle, angle],
                [inner_radius, outer_radius],
                color=color,
                linewidth=linewidth,
                zorder=30,
            )


# ============================================================
# MAIN CIRCULAR HEATMAP
# ============================================================

def plot_moa_clusters(
    gsea_df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str,
    organ_by_cell_line: Mapping[str, Any],
    moa_col: str = "moa",
    moa_order: list[str] | None = None,
    hallmark_groupings: Mapping[str, str] = HALLMARK_GROUPINGS,
    vmax: float | None = None,
    sector_gap: float = SECTOR_GAP,
    figsize: tuple[float, float] = (15, 13),
    modes: tuple[str, ...] = ("cp10k",),
    title: str = 'Hallmark GSEA by MOA',
    moa_clusters: dict | None = None
) -> dict[str, Path]:
    """
    Make one circular plot for each requested mode.

    All MOAs in gsea_df appear as sectors in the same circle.
    Supply only the MOAs belonging to one biologically coherent
    MOA group per call.
    """

    required = {
        moa_col,
        "mode",
        "term",
        "cell_line",
        "nes",
        "fdr_qval",
    }

    missing = required.difference(gsea_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}.s"
        )

    df = gsea_df.copy()

    df["term"] = df["term"].replace({
        "Pperoxisome": "Peroxisome"
    })

    df[moa_col] = df[moa_col].astype(str)
    df["cell_line"] = df["cell_line"].astype(str)

    # Keep the complete configured pathway set
    pathways = sorted(hallmark_groupings)

    # Warn about terms in the data that are not in the mapping
    unknown_terms = (
        set(df["term"].dropna().unique())
        - set(pathways)
    )

    if unknown_terms:
        print(
            "Warning: these input pathways are not in "
            "HALLMARK_GROUPINGS:",
            sorted(unknown_terms)
        )

    pathway_order, category_spans, pathway_bounds = (
        _pathway_layout(
            pathways,
            hallmark_groupings
        )
    )

    # Cell-line ordering
    cell_lines = _ordered_cell_lines(
        df,
        organ_by_cell_line
    )

    if not cell_lines:
        raise ValueError("No cell lines found in the input.")

    # MOA ordering
    if moa_order is None:
        moas = sorted(df[moa_col].unique())
    else:
        moas = [str(m) for m in moa_order]

    if not moas:
        raise ValueError("No MOAs found in the input.")

    # Common NES scale across all MOAs and modes
    if vmax is None:

        finite = pd.to_numeric(
            df["nes"],
            errors="coerce"
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if finite.empty:
            raise ValueError(
                "No finite NES values are available to plot."
            )

        vmax = max(
            float(finite.abs().max()),
            1e-6
        )

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    cmap = plt.get_cmap("coolwarm")

    # Each MOA has one angular subdivision per cell line
    sectors = {
        moa: len(cell_lines)
        for moa in moas
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {}

    # ========================================================
    # MODE LOOP
    # ========================================================

    for mode in modes:

        mode_df = df[df["mode"] == mode].copy()

        if mode_df.empty:
            print(f"Skipping {mode}: no data found.")
            continue

        # Create a fresh Circos object for each mode
        # TOP_GAP leaves an angular opening for category labels.
        circos = Circos(
            sectors,
            start=0 + TOP_GAP / 2,
            end=360 - TOP_GAP / 2,
            space=sector_gap,
            endspace=False,
        )

        # ----------------------------------------------------
        # DRAW HEATMAP TILES
        # ----------------------------------------------------

        for sector in circos.sectors:

            moa = sector.name

            subset = mode_df[
                mode_df[moa_col] == moa
            ]

            # Duplicate values are ambiguous
            if subset.duplicated(
                ["term", "cell_line"]
            ).any():

                raise ValueError(
                    f"Duplicate term/cell_line rows for "
                    f"MOA={moa!r}, mode={mode!r}. "
                    "Resolve duplicates before plotting."
                )

            nes = subset.pivot(
                index="term",
                columns="cell_line",
                values="nes"
            )

            fdr = subset.pivot(
                index="term",
                columns="cell_line",
                values="fdr_qval"
            )

            nes = nes.reindex(
                index=pathway_order,
                columns=cell_lines
            )

            fdr = fdr.reindex(
                index=pathway_order,
                columns=cell_lines
            )

            # One angular subdivision per cell line
            for pathway in pathway_order:

                r0, r1 = pathway_bounds[pathway]

                for cell_idx, cell_line in enumerate(cell_lines):

                    value = nes.at[pathway, cell_line]
                    qvalue = fdr.at[pathway, cell_line]

                    facecolor = _make_tile_color(
                        value,
                        qvalue,
                        norm,
                        cmap
                    )

                    sector.rect(
                        start=cell_idx,
                        end=cell_idx + 1,
                        r_lim=(r0, r1),
                        fc=facecolor,
                        ec=CELL_BORDER_COLOR,
                        lw=CELL_BORDER_WIDTH,
                    )

            # MOA label outside the outer radius
            # give enough space for long MOA names
            if len(sectors) > 10:
                moa = moa.replace(' ', '\n')

            sector.text(
                moa,
                r=OUTER_RADIUS + 3 if len(sectors) <= 10 else OUTER_RADIUS + 1,
                size=7,
                orientation="horizontal",
                adjust_rotation=True,
                color="black",
            )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig = circos.plotfig()

        fig.set_size_inches(*figsize)

        # Position the circular heatmap to leave room for labels
        circos.ax.set_position([
            0.0,
            0.3,
            0.70,
            0.80,
        ])

        # ----------------------------------------------------
        # PATHWAY LABELS + CATEGORY BRACKETS
        # ----------------------------------------------------

        label_angle = -TOP_GAP / 2
        theta = np.deg2rad(label_angle)

        ax = circos.ax
        fig = ax.figure

        # ----------------------------------------------------
        # PATHWAY LABELS — draw once
        # ----------------------------------------------------

        pathway_texts = {}

        for pathway in pathway_order:

            r0, r1 = pathway_bounds[pathway]
            r_mid = (r0 + r1) / 2

            pathway_text = f"   {pathway}"

            # can only do this since it happens to be last
            if pathway_text == "   Unfolded Protein Response":
                pathway_text = "   Unfolded Protein\n   Response"
                r_mid -= 1

            pathway_texts[pathway] = ax.text(
                theta,
                r_mid,
                pathway_text,
                ha="left",
                va="center",
                rotation=-label_angle,
                rotation_mode="anchor",
                fontsize=5,
                clip_on=False,
            )

        # ----------------------------------------------------
        # CALCULATE THE ACTUAL ANGULAR ENDPOINTS OF THE LABELS
        # ----------------------------------------------------

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        text_direction = np.array([
            np.cos(np.deg2rad(-label_angle)),
            np.sin(np.deg2rad(-label_angle)),
        ])

        label_end_angles = []

        for pathway, txt in pathway_texts.items():

            r0, r1 = pathway_bounds[pathway]
            r_mid = (r0 + r1) / 2

            # Actual rendered text width, including leading spaces.
            text_width, _, _ = renderer.get_text_width_height_descent(
                txt.get_text(),
                txt.get_fontproperties(),
                ismath=False,
            )

            # Screen position of the pathway label anchor.
            anchor_px = ax.transData.transform((theta, r_mid))

            # Screen position at the end of the rendered text.
            endpoint_px = (
                anchor_px + text_direction * text_width
            )

            # Convert the endpoint back to polar data coordinates.
            endpoint_data = ax.transData.inverted().transform(
                endpoint_px
            )

            label_end_angles.append(endpoint_data[0])

        # Furthest angular endpoint of all pathway names.
        #bracket_theta = max(label_end_angles) + np.deg2rad(1.0)
        bracket_theta = 0

        # Category names sit just beyond the brackets.
        category_theta = bracket_theta + np.deg2rad(1.5)

        # Short bracket end caps.
        cap_angle = np.deg2rad(0.7)

        # ----------------------------------------------------
        # CATEGORY BRACKETS + CATEGORY NAMES
        # ----------------------------------------------------

        for category, first, last in category_spans:

            first_pathway = pathway_order[first]
            last_pathway = pathway_order[last]

            r_inner = pathway_bounds[first_pathway][0]
            r_outer = pathway_bounds[last_pathway][1]
            r_mid = (r_inner + r_outer) / 2

            # Radial bracket spine.
            ax.plot(
                [bracket_theta, bracket_theta],
                [r_inner, r_outer],
                color="black",
                linewidth=0.7,
                clip_on=False,
            )

            # Short end caps pointing back towards the pathways.
            for r in (r_inner, r_outer):

                ax.plot(
                    [bracket_theta - cap_angle, bracket_theta],
                    [r, r],
                    color="black",
                    linewidth=0.7,
                    clip_on=False,
                )

            # Category name after the bracket.
            ax.text(
                category_theta,
                r_mid,
                category,
                ha="left",
                va="center",
                # rotation=-label_angle,
                rotation_mode="anchor",
                fontsize=6,
                fontweight="bold",
                clip_on=False,
            )

        if moa_clusters is not None:
            _outline_moa_clusters(
                circos,
                moa_clusters,
                inner_radius=INNER_RADIUS,
                outer_radius=OUTER_RADIUS,
            )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        fig.suptitle(
            f"{title} ({mode.upper()})",
            y=0.95,
            fontsize=13,
            fontweight="bold",
        )

        # ----------------------------------------------------
        # SHARED NES COLORBAR
        # ----------------------------------------------------

        sm = ScalarMappable(
            norm=norm,
            cmap=cmap
        )
        sm.set_array([])

        cbar_width = 0.38
        cbar_height = 0.018

        cbar_ax = fig.add_axes([
            (1.0 - cbar_width) / 2,
            0.002,
            cbar_width,
            cbar_height,
        ])

        cbar = fig.colorbar(
            sm,
            cax=cbar_ax,
            orientation="horizontal",
        )

        cbar.set_label(
            "Normalized Enrichment Score (NES)",
            fontsize=9,
            labelpad=4,
        )

        cbar.ax.tick_params(
            labelsize=8,
            pad=1,
        )

        # ----------------------------------------------------
        # FOOTNOTE
        # ----------------------------------------------------

        fig.text(
            0.50,
            -0.055,
            (
                f"Opaque: FDR < {FDR_THRESHOLD}  |  "
                f"Faded: FDR ≥ {FDR_THRESHOLD}  |  "
                "Sectors: Mechanism of Action (MOA)  |  "
                "Radial axis: Biological Pathway Groups  |  "
                "Angular axis: Cell Lines"
            ),
            ha="center",
            va="bottom",
            fontsize=7,
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_path = (
            output_dir
            / f"{filename_prefix}_{mode}_moa_circos.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)

        outputs[mode] = output_path

    return outputs
