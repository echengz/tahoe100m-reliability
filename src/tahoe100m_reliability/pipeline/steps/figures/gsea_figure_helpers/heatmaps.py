"""
Generates the individual heatmaps that display GSEA results
for exactly one grouping (e.g. global or a specific MOA).
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection

from .config import (
    GSEAHeatmaps,
    HALLMARK_GROUPINGS,
    FLAGGED_CELL_LINES,
)


class GSEAIndividualHeatmaps(GSEAHeatmaps):

    def __init__(self, modes: list = ['raw', 'cp10k'], **kwargs) -> None:
        """Initializer."""
        super().__init__(**kwargs)
        self.__modes = modes

    def __prepare_nes_matrices(self, df: pd.DataFrame) -> None:
        """
        Prepare NES matrices to use as basis for heatmap. Also
        compute a parallel FDR matrix to determine opacity of
        heatmap.
        """
        # include all hallmark groupings
        terms = sorted(HALLMARK_GROUPINGS.keys())

        cell_lines = sorted(
            df["cell_line"].unique(),
            key=lambda cell_id: (
                self.organ_by_cell_line.get(
                    cell_id, "Unknown organ"
                ).lower(),
                cell_id.lower(),
            ),
        )

        matrices = {}

        for mode in self.__modes:
            subset = df[df["mode"] == mode].copy()

            nes = subset.pivot(
                index="term",
                columns="cell_line",
                values="nes",
            )

            fdr = subset.pivot(
                index="term",
                columns="cell_line",
                values="fdr_qval",
            )

            nes = nes.reindex(index=terms, columns=cell_lines)
            fdr = fdr.reindex(index=terms, columns=cell_lines)

            matrices[mode] = {
                "nes": nes,
                "fdr": fdr,
            }

        return matrices

    def __prepare_grouped_plot_data(self, nes, fdr):
        """
        Add blank display columns before pathway groups to separate them.
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

    def __build_organ_layout(self, cell_lines: list) -> None:
        """
        Builds the organ layout, which groups each cell line by organ.

        Return cell-line y positions, organ spans, and total display height.
        """
        organ_labels = [
            self.organ_by_cell_line.get(cell_id, "Unknown organ").replace('/', ' /\n')
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
                y += 0.8

            i = j

        return np.asarray(row_positions), organ_spans, y

    def __plot_nes_heatmap(self, nes, fdr, title, output_path, vmax) -> None:
        """Plot heatmap; cell lines as rows and grouped pathways as columns."""
        values, qvalues, column_labels, column_types, cell_lines = \
            self.__prepare_grouped_plot_data(nes, fdr)

        if len(cell_lines) == 0 or values.shape[1] == 0:
            return

        row_positions, organ_spans, total_height = self.__build_organ_layout(cell_lines)

        n_rows, n_cols = values.shape

        # Set NES color and FDR opacity
        norm = TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0,
            vmax=vmax,
        )

        safe_values = np.where(np.isfinite(values), values, 0.0)
        rgba = self.cmap(norm(safe_values))

        significant = np.isfinite(qvalues) & (qvalues < self.fdr_threshold)

        rgba[:, :, 3] = np.where(
            np.isfinite(values),
            np.where(significant, 1.0, self.non_significant_alpha),
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
        heatmap_width = max(10.0, n_cols * 0.25)
        fig_width = heatmap_width + 1.0
        fig_height = max(4.5, min(12.0, total_height * 0.30 + 1.5))

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
            edgecolors='white',
            linewidths=0.3,
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
            if cell_id in FLAGGED_CELL_LINES:
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

            if group == 'Cellular Component':
                wrapped_group = group.replace(' ', '\n')
            else:
                wrapped_group = group

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
                    color='white',
                    linewidth=0.8,
                    zorder=5,
                )

        # Add NES colorbar
        sm = ScalarMappable(norm=norm, cmap=self.cmap)
        sm.set_array([])

        cbar_ax = fig.add_axes([
            heatmap_right + 0.02,
            heatmap_bottom,
            0.018,
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
                f"Opaque: FDR < {self.fdr_threshold}  |  "
                f"Faded: FDR ≥ {self.fdr_threshold}  |  "
                "Gray: missing NES  |  "
                "*  Low cell coverage"
            ),
            ha="right",
            va="bottom",
            fontsize=7,
        )

        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    def plot_heatmaps(
        self, gsea_df: pd.DataFrame, filename_prefix: str, title_postfix: str,
    ) -> None:
        """Plot GSEA results.

        Assumes as input the DF generated from the GSEA step, saved as
        `{axis}_shared_axis_hallmark_gsea.csv`

        If influence DF is provided, also plots corresponding mean influence
        scores of the leading genes.
        """
        # prepare matrices
        nes_matrices = self.__prepare_nes_matrices(gsea_df)

        # Same NES color scale for raw and CP10k
        all_nes = pd.concat(
            [
                nes_matrices[mode]["nes"].stack()
                for mode in self.__modes
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

        for mode in self.__modes:
            nes = nes_matrices[mode]["nes"]
            fdr = nes_matrices[mode]["fdr"]

            output_path = self.output_dir / f"{filename_prefix}_{mode}_nes_heatmap.png"

            self.__plot_nes_heatmap(
                nes=nes,
                fdr=fdr,
                title=f"Shared-axis Hallmark GSEA ({mode.upper()}) - {title_postfix}",
                output_path=output_path,
                vmax=vmax,
            )

