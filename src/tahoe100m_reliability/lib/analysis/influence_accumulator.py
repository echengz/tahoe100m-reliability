"""
Provides a vectorized accumulator for the systematic step to
accumulate influence scores.

Handles accumulation by gene and GSEA pathways.
"""
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

from tahoe100m_reliability.lib.analysis.compute_systematic import (
    compute_gene_influences,
    compute_pathway_influences,
)
from tahoe100m_reliability.lib.ingest.intermediate_loaders import load_gene_ids

SPECIFICITY_THRESH = 0.5


class InfluenceAccumulator:

    ACCUM_FIELDS_FLOAT: list[str] = [
        'negative_influence',
        'positive_influence',
        'total_influence',
        'total_influence_abs',
    ]

    ACCUM_FIELDS_INT: list[str] = [
        'num_instances',
        'num_positive_instances',
        'num_negative_instances',
        'categorical_change_towards_shared',
        'categorical_change_towards_specific',
        'categorical_change_neutral',
    ]

    ACCUM_FIELDS_ALL = ACCUM_FIELDS_FLOAT + ACCUM_FIELDS_INT

    def __init__(self, group_ids: dict[str, list[int]]) -> None:
        """Initialize accumulator.

        Takes as input the mapping of the named gene groups to their
            corresponding gene_id (Tahoe-100M-specific) index to be
            used as a mask on the corresponding condition's gene
            expression vector.

        For gene-specific analysis, this is simply a mapping of each
            gene to its gene index.

        For pathway-specific analysis, this is a mapping of the
            named pathway to a list of its lead gene indexes.
        """
        n = len(group_ids)

        self.group_ids = group_ids

        self.__acc = {'group_ids': group_ids}
        self.__acc.update({k: np.zeros(n) for k in self.ACCUM_FIELDS_FLOAT})
        self.__acc.update({k: np.zeros(n, dtype=np.int64) for k in self.ACCUM_FIELDS_INT})

        self.__n_groups = len(group_ids)
        self.__groups = list(group_ids)

    def __setitem__(self, key, item) -> None:
        self.__acc[key] = item

    def __getitem__(self, key) -> Any:
        return self.__acc[key]

    def accumulate(
        self,
        influences: np.ndarray,
        valid: np.ndarray,
        frac_systematic: float,
    ) -> None:
        """Accumulate influence scores."""
        if not np.isfinite(frac_systematic):
            return

        influences = np.asarray(influences, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)

        assert influences.shape == (self.__n_groups,), "Num influences does not match num groups"
        assert valid.shape == (self.__n_groups,), "Num valid indices does not match num groups"

        # get all valid indexes with an influence value
        idx = np.flatnonzero(valid & np.isfinite(influences))
        if len(idx) == 0:
            return

        inf = influences[idx]

        # track change in categorical classification
        # in general most are expected to be neutral,
        # classification only really changes if the original
        # specificity score was close to the threshold
        shared = (
            (frac_systematic < SPECIFICITY_THRESH)
            & (frac_systematic - inf >= SPECIFICITY_THRESH)
        )
        specific = (
            (frac_systematic >= SPECIFICITY_THRESH)
            & (frac_systematic - inf < SPECIFICITY_THRESH)
        )
        neutral = ~(shared | specific)

        # Vectorized accumulation
        # Totals
        self['positive_influence'][idx] += np.maximum(inf, 0)
        self['negative_influence'][idx] += np.minimum(inf, 0)
        self['total_influence'][idx] += inf
        self['total_influence_abs'][idx] += np.abs(inf)

        # Counts
        self['num_instances'][idx] += 1
        self['num_positive_instances'][idx] += (inf > 0).astype(int)
        self['num_negative_instances'][idx] += (inf < 0).astype(int)

        # Categorical changes
        self['categorical_change_towards_shared'][idx] += shared
        self['categorical_change_towards_specific'][idx] += specific
        self['categorical_change_neutral'][idx] += neutral
        del inf

    def get_accumulation(self) -> dict[str, Any]:
        """Get accumulation results by gene group."""
        result = {}

        for i, group_id in enumerate(self.__groups):
            if self['num_instances'][i] == 0:
                continue

            result[group_id] = {
                field: self[field][i].item()
                for field in self.ACCUM_FIELDS_ALL
            }

        return result


class InfluenceAccumulatorHandler:
    """
    Base class to handles influence accumulators by cell line
    """
    def __init__(
        self,
        gene_metadata: pd.DataFrame,
        output_dir: Path,
        cur_time: str,
    ) -> None:
        """Initializer."""
        self.gene_metadata = gene_metadata
        self.output_dir = output_dir
        self.cur_time = cur_time

    def set_up_accumulators_by_mode(self) -> dict:
        """Set up accumulators by mode."""
        return {'raw': {}, 'cp10k': {}}

    def map_gene_ids(
        self, cell_line_id: str, ctrl_gene_expressions_dir: Path
    ) -> tuple[np.array ,pd.DataFrame, dict]:
        """Map gene IDs (integers) to human readable gene names, and map the
        token IDs to position to use for vectorized calculations.
        """
        gene_ids = load_gene_ids(ctrl_gene_expressions_dir, cell_line_id)
        metadata_by_id = self.gene_metadata.set_index('token_id')['gene_symbol']

        token_id_to_position = {
            gene_id: i
            for i, gene_id in enumerate(gene_ids)
        }

        return gene_ids, metadata_by_id, token_id_to_position

    def summarize_group_influence(self, group_id: str, influence: dict) -> dict:
        """Summarize accumulated influence for a gene group."""
        summary = {'group_id': group_id}

        # averages over instances
        num_instances = influence['num_instances']
        num_positive_instances = influence['num_positive_instances']
        num_negative_instances = influence['num_negative_instances']

        summary.update({
            'mean_influence': influence['total_influence'] / num_instances,
            'mean_abs_influence': influence['total_influence_abs'] / num_instances,
            'mean_positive_influence': (
                influence['positive_influence'] / num_positive_instances
                if num_positive_instances else 0
            ),
            'mean_negative_influence': (
                influence['negative_influence'] / num_negative_instances
                if num_negative_instances else 0
            )
        })

        # rest report as is
        summary.update(influence)
        return summary

    def summarize_accumulation(
        self,
        influences: dict,
        extra_fields: dict | None = None,
        n_groups: int | None = None,
        outfile: Path | None = None
    ) -> pd.DataFrame:
        """Summarize the given accumulation of influences."""
        summary = []
        for group_id, influence in influences.items():
            row = self.summarize_group_influence(group_id, influence)
            if extra_fields:
                row.update(extra_fields)

            summary.append(row)

        df = pd.DataFrame(summary)

        # if specified, only keep the largest groups by mean_abs_influence
        if n_groups:
            df = df.nlargest(n_groups, 'mean_abs_influence')

        if outfile is not None:
            df.to_csv(outfile, index=False)

        return df

    def ensure_exists(
        self, mode: str, cell_line_id: str, moa: str, ctrl_gene_expressions_dir: Path
    ) -> None:
        """Check if a given mode/cell line/moa accumulator has been encountered;
        if not, add it.
        """
        pass

    def accumulate_influence(
        self,
        mode: str,
        moa: str,
        cell_line_id: str,
        delta_p,
        delta_avg,
        working_results,
        frac_systematic,
    ) -> None:
        """Accumulate influence"""
        pass

    def finalize_influence(self, mode: str) -> None:
        """Finalize influence for this accumulator handler."""
        pass


class GeneInfluenceAccumulatorHandler(InfluenceAccumulatorHandler):
    """Class to handle gene influence accumulation"""
    def __init__(
        self,
        gene_metadata: pd.DataFrame,
        output_dir: Path,
        cur_time: str,
    ) -> None:
        super().__init__(gene_metadata, output_dir, cur_time)

        # Just use one set of accumulators because the MOA ones can be aggregated
        # to get the cell line and global ones
        self.gene_accumulators = self.set_up_accumulators_by_mode()

    def ensure_exists(
        self, mode: str, cell_line_id: str, moa: str, ctrl_gene_expressions_dir: Path
    ) -> None:
        """Check if a given mode/cell line/moa accumulator has been encountered;
        if not, add it.
        """
        if (cell_line_id, moa) in self.gene_accumulators[mode]:
            return

        # Initialize accumulators for the given cell line, mode, amd moa
        gene_ids, metadata_by_id, token_id_to_position = \
            self.map_gene_ids(cell_line_id, ctrl_gene_expressions_dir)

        # Define gene groups
        gene_groups = {
            metadata_by_id.loc[gene_id]: [position]
            for gene_id, position in token_id_to_position.items()
            if gene_id in metadata_by_id.index
        }

        if len(gene_groups) != len(gene_ids):
            raise ValueError(
                f"Gene metadata mismatch for {cell_line_id}: "
                f"{len(gene_groups)} mapped genes, {len(gene_ids)} vector genes"
            )

        self.gene_accumulators[mode][(cell_line_id, moa)] = InfluenceAccumulator(gene_groups)

    def accumulate_influence(
        self,
        mode: str,
        moa: str,
        cell_line_id: str,
        delta_p,
        delta_avg,
        working_results,
        frac_systematic,
    ) -> None:
        """Accumulate influence."""
        influences, valid = compute_gene_influences(
            delta_p, delta_avg, working_results, frac_systematic
        )

        gene_accumulator = self.gene_accumulators[mode][(cell_line_id, moa)]
        gene_accumulator.accumulate(influences, valid, frac_systematic)

    def finalize_influence(self, mode: str) -> None:
        """Finalize by gene for each cell line and across all cell lines.

        Since there are a lot of genes, only keep n per cell line and m per global.
        """
        n_per_group = 200
        n_per_global = 1000

        output_dir = self.output_dir / 'gene_influences'
        output_dir.mkdir(parents=True, exist_ok=True)

        # to aggregate influence across all cell lines
        global_gene_influences = {}
        cell_line_influences = {}
        top_genes_by_cell_line_moa = []

        for (cell_line_id, moa), accumulator in self.gene_accumulators[mode].items():
            influences = accumulator.get_accumulation()
            if not influences:
                continue

            # summarize influence for this cell line x moa 
            df = self.summarize_accumulation(
                influences,
                extra_fields={'cell_line_id': cell_line_id, 'moa': moa},
                n_groups=n_per_group,
            )
            top_genes_by_cell_line_moa.append(df)

            # aggregate influence globally and by cell line
            if cell_line_id not in cell_line_influences:
                cell_line_influences[cell_line_id] = {}

            # aggregate influence globally and by cell line
            for gene_id, influence in influences.items():
                if gene_id not in cell_line_influences[cell_line_id]:
                    cell_line_influences[cell_line_id][gene_id] = {
                        key: 0
                        for key in influence
                    }

                if gene_id not in global_gene_influences:
                    global_gene_influences[gene_id] = {
                        key: 0
                        for key in influence
                    }
                    # keep track of which cell lines had the given gene
                    global_gene_influences[gene_id]['cell_lines'] = set()

                cell_line_gene = cell_line_influences[cell_line_id][gene_id]
                global_gene = global_gene_influences[gene_id]
                global_gene['cell_lines'].add(cell_line_id)

                for key, value in influence.items():
                    cell_line_gene[key] += value
                    global_gene[key] += value

        # save results
        if top_genes_by_cell_line_moa:
            top_genes_by_cell_line_moa_df = pd.concat(top_genes_by_cell_line_moa, ignore_index=True)
            top_genes_by_cell_line_moa_df.to_csv(
                output_dir / f'systema_{mode}_top{n_per_group}_by_cell_line_moa_{self.cur_time}.csv',
                index=False
            )
            del top_genes_by_cell_line_moa, top_genes_by_cell_line_moa_df

        if cell_line_influences:
            top_genes_by_cell_line = []

            for cell_line_id, gene_influences in cell_line_influences.items():
                df = self.summarize_accumulation(
                    gene_influences,
                    extra_fields={'cell_line_id': cell_line_id},
                    n_groups=n_per_group,
                )
                top_genes_by_cell_line.append(df)

            del cell_line_influences

            top_genes_by_cell_line_df = pd.concat(top_genes_by_cell_line, ignore_index=True)
            top_genes_by_cell_line_df.to_csv(
                output_dir / f'systema_{mode}_top{n_per_group}_by_cell_line_{self.cur_time}.csv',
                index=False
            )
            del top_genes_by_cell_line_df

        if global_gene_influences:
            for influence in global_gene_influences.values():
                influence['cell_lines'] = sorted(influence['cell_lines'])

            self.summarize_accumulation(
                global_gene_influences,
                n_groups=n_per_global,
                outfile=output_dir / f'systema_{mode}_top{n_per_global}_global_{self.cur_time}.csv'
            )
            del global_gene_influences

        # release memory
        self.gene_accumulators[mode].clear()


class GSEAInfluenceAccumulatorHandler(InfluenceAccumulatorHandler):
    """Class to handle GSEA pathway influence accumulation by cell line:
        - global x pathway
        - MOA x pathway
    """
    def __init__(
        self,
        gsea_global_shared_axis_hallmark: Path,
        gsea_moa_shared_axis_hallmark: Path,
        gene_metadata: pd.DataFrame,
        output_dir: Path,
        cur_time: str,
    ) -> None:
        super().__init__(gene_metadata, output_dir, cur_time)

        self.gsea_global_accumulators = self.set_up_accumulators_by_mode()
        self.gsea_moa_accumulators = self.set_up_accumulators_by_mode()

        # load GSEA data
        self.gsea_global_df = (
            pd.read_csv(gsea_global_shared_axis_hallmark)
            if gsea_global_shared_axis_hallmark
            else None
        )

        self.gsea_moa_df = (
            pd.read_csv(gsea_moa_shared_axis_hallmark)
            if gsea_moa_shared_axis_hallmark
            else None
        )

        # GSEA masks, to mask the specific pathway
        self.gsea_global_masks = {'raw': {}, 'cp10k': {}}
        self.gsea_moa_masks = {'raw': {}, 'cp10k': {}}

    def __build_pathway_masks(self, group_ids, n_genes):
        """Build Boolean pathway-by-gene membership matrix."""

        masks = np.zeros(
            (len(group_ids), n_genes),
            dtype=bool,
        )

        for i, indices in enumerate(group_ids.values()):
            masks[i, indices] = True

        return masks

    def __build_pathway_groups(self, gsea_df, symbol_to_id):
        """Map GSEA leading-edge genes to token IDs."""

        group_ids = {}

        for _, row in gsea_df.iterrows():
            lead_genes = (
                row['lead_genes'].split(';')
                if pd.notna(row['lead_genes'])
                else []
            )

            indices = [
                symbol_to_id[gene]
                for gene in lead_genes
                if gene in symbol_to_id
            ]

            if indices:
                group_ids[row['term']] = indices

        return group_ids

    def __initialize_gsea_accumulators(
        self, mode: str, cell_line_id: str, ctrl_gene_expressions_dir: Path,
    ):
        gene_ids, metadata_by_id, token_id_to_position = \
            self.map_gene_ids(cell_line_id, ctrl_gene_expressions_dir)

        symbol_to_position = {
            metadata_by_id.loc[gene_id]: position
            for gene_id, position in token_id_to_position.items()
            if gene_id in metadata_by_id.index
        }

        # Global GSEA accumulator for this cell line and mode
        if self.gsea_global_df is not None:
            global_df = self.gsea_global_df[
                (self.gsea_global_df['cell_line'] == cell_line_id)
                & (self.gsea_global_df['mode'] == mode)
            ]

            global_groups = self.__build_pathway_groups(global_df, symbol_to_position)
            if global_groups:
                self.gsea_global_accumulators[mode][cell_line_id] = \
                    InfluenceAccumulator(global_groups)

                self.gsea_global_masks[mode][cell_line_id] = \
                    self.__build_pathway_masks(global_groups, len(gene_ids))

        # MOA-specific GSEA accumulators for this cell line and mode
        if self.gsea_moa_df is not None:
            moa_df = self.gsea_moa_df[
                (self.gsea_moa_df['cell_line'] == cell_line_id)
                & (self.gsea_moa_df['mode'] == mode)
            ]

            for moa, group_df in moa_df.groupby('moa'):
                moa_groups = self.__build_pathway_groups(group_df, symbol_to_position)
                if moa_groups:
                    self.gsea_moa_accumulators[mode][(cell_line_id, moa)] = \
                        InfluenceAccumulator(moa_groups)

                    self.gsea_moa_masks[mode][(cell_line_id, moa)] = \
                        self.__build_pathway_masks(moa_groups, len(gene_ids))

    def ensure_exists(
        self, mode: str, cell_line_id: str, moa: str, ctrl_gene_expressions_dir: Path
    ) -> None:
        """Check if a given mode/cell line accumulator has been encountered;
        if not, add it.
        """
        if cell_line_id not in self.gsea_global_accumulators[mode]:
            self.__initialize_gsea_accumulators(mode, cell_line_id, ctrl_gene_expressions_dir)

        # assert moa was added to the moa one
        if (cell_line_id, moa) not in self.gsea_moa_accumulators[mode]:
            raise ValueError(f"MOA {moa} not found in GSEA MOA results for {cell_line_id}")

    def accumulate_influence(
        self,
        mode: str,
        moa: str,
        cell_line_id: str,
        delta_p,
        delta_avg,
        working_results,
        frac_systematic,
    ) -> None:
        """Accumulate influence."""
        # GSEA accumulation - global
        gsea_global_accumulator = self.gsea_global_accumulators[mode].get(cell_line_id)
        if gsea_global_accumulator is not None:
            pathway_masks = self.gsea_global_masks[mode][cell_line_id]

            influences, valid = compute_pathway_influences(
                pathway_masks, delta_p, delta_avg, working_results, frac_systematic
            )
            gsea_global_accumulator.accumulate(influences, valid, frac_systematic)

        # GSEA accumulation - MOA
        gsea_moa_accumulator = self.gsea_moa_accumulators[mode].get((cell_line_id, moa))
        if gsea_moa_accumulator is not None:
            pathway_masks = self.gsea_moa_masks[mode][(cell_line_id, moa)]

            influences, valid = compute_pathway_influences(
                pathway_masks, delta_p, delta_avg, working_results, frac_systematic
            )
            gsea_moa_accumulator.accumulate(influences, valid, frac_systematic)

    def finalize_influence(self, mode: str) -> None:
        """Finalize and clean up accumulators for the given cell line and mode for
        GSEA terms. Since not many pathways in the grand scheme of things, keep all
        of them.
        """
        output_dir = self.output_dir / 'gsea_influences'
        output_dir.mkdir(parents=True, exist_ok=True)

        # by GSEA global
        global_rows = []
        for cell_line_id, accumulator in self.gsea_global_accumulators[mode].items():
            influences = accumulator.get_accumulation()
            if not influences:
                continue

            df = self.summarize_accumulation(
                influences,
                extra_fields={'cell_line_id': cell_line_id}
            )
            global_rows.append(df)

        if global_rows:
            pd.concat(global_rows, ignore_index=True).to_csv(
                output_dir / f'systema_{mode}_gsea_global_{self.cur_time}.csv',
                index=False,
            )

        moa_rows = []
        for (cell_line_id, moa), accumulator in self.gsea_moa_accumulators[mode].items():
            influences = accumulator.get_accumulation()
            if not influences:
                continue

            df = self.summarize_accumulation(
                influences,
                extra_fields={'cell_line_id': cell_line_id, 'moa': moa},
            )

            moa_rows.append(df)

        if moa_rows:
            pd.concat(moa_rows, ignore_index=True).to_csv(
                output_dir / f'systema_{mode}_gsea_moa_{self.cur_time}.csv',
                index=False,
            )

        # release memory
        self.gsea_global_accumulators[mode].clear()
        self.gsea_moa_accumulators[mode].clear()
        self.gsea_global_masks[mode].clear()
        self.gsea_moa_masks[mode].clear()
