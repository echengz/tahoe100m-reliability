"""
Generates radial heatmaps with pyCirclize that allows
grouping multiple heatmaps together.
"""
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import IdentityTransform
from pycirclize import Circos

from .base import (
    GSEAHeatmaps,
    HALLMARK_GROUPINGS,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GSEARadialHeatmaps(GSEAHeatmaps):

    def __init__(self, detailed: bool = False, **kwargs) -> None:
        """Initializer."""
        super().__init__(**kwargs)

        self.__detailed = detailed
        self.__inner_radius = 27.0
        self.__outer_radius = 94.0

        self.__top_gap = 50 if detailed else 34
        self.__sector_gap = 1

    def __ordered_cell_lines(self, df: pd.DataFrame) -> list[str]:
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
                    self.organ_by_cell_line.get(
                        cell_id,
                        "Unknown organ"
                    )
                ).lower(),
                cell_id.lower(),
            ),
        )

    def __pathway_layout(self, pathways: list[str]):
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
            HALLMARK_GROUPINGS[p]
            for p in pathways
        }, reverse=True)

        pathway_order = []
        category_spans = []

        for category in categories:

            members = sorted(
                [
                    p for p in pathways
                    if HALLMARK_GROUPINGS[p] == category
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

        pathway_group_gap = 0.55
        usable_radius = (
            self.__outer_radius
            - self.__inner_radius
            - n_gaps * pathway_group_gap
        )

        if usable_radius <= 0:
            raise ValueError("Pathway gaps leave no room for pathway rows.")

        row_height = usable_radius / n_pathways

        pathway_bounds = {}
        cursor = self.__inner_radius

        for category, first, last in category_spans:

            for pathway in pathway_order[first:last + 1]:

                r0 = cursor
                r1 = cursor + row_height

                pathway_bounds[pathway] = (r0, r1)

                cursor = r1

            # Gap between biological categories
            cursor += pathway_group_gap

        return pathway_order, category_spans, pathway_bounds


    def __make_tile_color(self, value, qvalue, norm):
        """
        NES determines color.
        FDR determines opacity.
        Missing NES is gray.
        """
        if pd.isna(value) or not np.isfinite(value):
            return (0.88, 0.88, 0.88, 1.0)

        rgba = list(self.cmap(norm(value)))

        if pd.notna(qvalue) and np.isfinite(qvalue) and qvalue < self.fdr_threshold:
            alpha = 1.0
        else:
            alpha = self.non_significant_alpha

        rgba[3] = alpha

        return tuple(rgba)


    def __outline_moa_clusters(
        self,
        circos,
        moa_clusters,
        color="black",
        linewidth=None,
        label_radius=None,
    ):
        """Draw a continuous closed outline and label each MOA cluster."""
        ax = circos.ax

        if label_radius is None:
            label_radius = self.__outer_radius + 20
            # if self.__detailed:
            #     label_radius = self.__outer_radius + 15
            # else:
            #     label_radius = self.__outer_radius + 9

        if linewidth is None:
            if self.__detailed:
                linewidth = 0.5
            else:
                linewidth = 1

        for cluster, moas in moa_clusters.items():

            # Preserve the actual circular plotting order
            cluster_sectors = [
                s for s in circos.sectors
                if s.name in moas
            ]

            if not cluster_sectors:
                continue

            if len(cluster_sectors) != len(moas):
                missing = set(moas) - {
                    s.name for s in cluster_sectors
                }
                log.warning(f"Cluster {cluster}: missing MOAs {missing}")

            start = cluster_sectors[0].rad_lim[0]
            end = cluster_sectors[-1].rad_lim[1]

            theta = np.linspace(start, end, 300)

            # Outer boundary
            ax.plot(
                theta,
                np.full_like(theta, self.__outer_radius),
                color=color,
                linewidth=linewidth,
                zorder=30,
            )

            # Inner boundary
            ax.plot(
                theta,
                np.full_like(theta, self.__inner_radius),
                color=color,
                linewidth=linewidth,
                zorder=30,
            )

            # Connecting boundaries
            for angle in (start, end):
                ax.plot(
                    [angle, angle],
                    [self.__inner_radius, self.__outer_radius],
                    color=color,
                    linewidth=linewidth,
                    zorder=30,
                )

            # Cluster label at angular midpoint
            mid_angle = (start + end) / 2

            ax.text(
                mid_angle,
                # this should be based on where it is but just hardcoding for now
                label_radius if cluster != 2 else label_radius - 5,
                f'Cluster {cluster}',
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                rotation=0,
                rotation_mode="anchor",
                clip_on=False,
                zorder=40,
            )

    def plot_moa_radial_heatmaps(
        self,
        gsea_df: pd.DataFrame,
        mode: str,
        filename_prefix: str,
        moa_order: list[str] | None = None,
        vmax: float | None = None,
        title: str = 'Hallmark GSEA by MOA',
        moa_clusters: dict | None = None,
        target_pathways: list[str] = None,
    ) -> dict[str, Path]:
        """
        Make one circular plot for each group of MOAs.

        All MOAs in gsea_df appear as sectors in the same circle.
        Supply only the MOAs belonging to one biologically coherent
        MOA group per call.
        """
        required = {
            'moa',
            "mode",
            "term",
            "cell_line",
            'nes',
            "fdr_qval",
        }

        missing = required.difference(gsea_df.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}.s"
            )

        df = gsea_df.copy()

        df['moa'] = df['moa'].astype(str)
        df["cell_line"] = df["cell_line"].astype(str)

        # Keep the complete configured pathway set
        pathways = sorted(HALLMARK_GROUPINGS)

        # Raise error about terms in the data that are not in the mapping
        unknown_terms = set(df["term"].dropna().unique()) - set(pathways)

        if unknown_terms:
            raise ValueError(
                "These input pathways are not in HALLMARK_GROUPINGS:",
                sorted(unknown_terms)
            )

        # if targeting specific pathways, filter
        if target_pathways:
            pathways = [x for x in pathways if x in target_pathways]

        pathway_order, category_spans, pathway_bounds = self.__pathway_layout(pathways)

        # Cell-line ordering
        cell_lines = self.__ordered_cell_lines(df)

        if not cell_lines:
            raise ValueError("No cell lines found in the input.")

        # MOA ordering
        if moa_order is None:
            moas = sorted(df['moa'].unique())
        else:
            moas = [str(m) for m in moa_order]

        if not moas:
            raise ValueError("No MOAs found in the input.")

        # Common NES scale across all MOAs and modes
        if vmax is None:

            finite = pd.to_numeric(
                df['nes'],
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

        # Each MOA has one angular subdivision per cell line
        sectors = {
            moa: len(cell_lines)
            for moa in moas
        }

        # Select mode df
        mode_df = df[df["mode"] == mode].copy()

        if mode_df.empty:
            log.warning(f"Skipping {mode}: no data found.")
            return

        # Create a fresh Circos object for each mode
        # Top gap leaves an angular opening for category labels.
        if self.__detailed:
            start = 0 + self.__top_gap / 2
            end = 360 - self.__top_gap / 2
        else:
            start = 0 + self.__top_gap - 1
            end = 360 - 1

        circos = Circos(
            sectors,
            start=start,
            end=end,
            space=self.__sector_gap,
            endspace=False,
        )

        # Draw heatmap tiles
        for sector in circos.sectors:

            moa = sector.name

            subset = mode_df[
                mode_df['moa'] == moa
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
                values='nes'
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

                    facecolor = self.__make_tile_color(
                        value,
                        qvalue,
                        norm,
                    )

                    sector.rect(
                        start=cell_idx,
                        end=cell_idx + 1,
                        r_lim=(r0, r1),
                        fc=facecolor,
                    )

            # MOA label outside the outer radius
            # give enough space for long MOA names
            # if self.__detailed:
            if len(sectors) > 10:
                moa = moa.replace(' ', '\n')

            sector.text(
                moa,
                r=self.__outer_radius + 3 if len(sectors) <= 10 else self.__outer_radius + 1,
                size=10,
                orientation="horizontal",
                adjust_rotation=True,
                color="black",
                )

        # Set up the figure
        fig = circos.plotfig()

        fig.set_size_inches((15, 13))

        # Position the circular heatmap to leave room for labels
        circos.ax.set_position([
            0.0,
            0.3,
            0.70,
            0.80,
        ])

        # Set up pathway labels and categories
        label_angle = -self.__top_gap / 2
        theta = np.deg2rad(label_angle)

        ax = circos.ax
        fig = ax.figure

        # Pathway labels
        if self.__detailed:
            for pathway in pathway_order:
                r0, r1 = pathway_bounds[pathway]
                r_mid = (r0 + r1) / 2

                pathway_text = f"   {pathway}"

                # can only do this since it happens to be last
                if pathway_text == "   Unfolded Protein Response":
                    pathway_text = "   Unfolded Protein\n   Response"
                    r_mid -= 1

                ax.text(
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

        fig.canvas.draw()

        # Category labels
        # Furthest angular endpoint of all pathway names.
        bracket_theta = 0
        category_theta = bracket_theta + np.deg2rad(1.5)

        # Short bracket end caps.
        cap_angle = np.deg2rad(0.7)

        # Category labels
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
            if self.__detailed:
                category_label = category.replace(' ', '\n')
            else:
                category_label = category
                if category == 'Proliferation (Cell Cycle)':
                    category_label = 'Proliferation\n(Cell Cycle)'
                elif category == 'Pathway (Signaling)':
                    category_label = 'Pathway\n(Signaling)'

            ax.text(
                category_theta,
                r_mid,
                category_label,
                ha="left",
                va="center",
                rotation_mode="anchor",
                fontsize=8 if self.__detailed else 10,
                fontweight="bold",
                clip_on=False,
            )

        if moa_clusters is not None:
            self.__outline_moa_clusters(circos, moa_clusters)

        # set title
        fig.suptitle(
            f"{title} ({mode.upper()})",
            y=0.98,
            fontsize=13,
            fontweight="bold",
        )

        # Add NES colorbar
        sm = ScalarMappable(norm=norm, cmap=self.cmap)
        sm.set_array([])

        cbar_width = 0.38
        cbar_height = 0.018

        cbar_ax = fig.add_axes([
            (1.0 - cbar_width) / 2,
            -0.02,
            cbar_width,
            cbar_height,
        ])

        cbar = fig.colorbar(
            sm,
            cax=cbar_ax,
            orientation="horizontal",
        )

        cbar.set_label(
            f'Normalized Enrichment Score (NES)\n(Opaque: FDR < {self.fdr_threshold})',
            fontsize=10 if self.__detailed else 12,
            labelpad=4 if self.__detailed else 10,
        )

        cbar.ax.tick_params(
            labelsize=8,
            pad=1,
        )

        # Save results
        output_path = self.output_dir / f"{filename_prefix}_{mode}_moa_circos.png"
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
