"""
Accumulator for the systematic step to accumulate influence scores.
"""
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

SPECIFICITY_THRESH = 0.5


class InfluenceAccumulator:

    ACCUM_FIELDS_FLOAT: list[str] = [
        'negative_influence',
        'positive_influence',
        'total_influence',
        'total_influence_abs',
        'delta_p_magnitude_total',
        'delta_p_magnitude_total_abs',
        'delta_p_magnitude_total_pos',
        'delta_p_magnitude_total_neg',
    ]

    ACCUM_FIELDS_INT: list[str] = [
        'num_instances',
        'num_positive_instances',
        'num_negative_instances',
        'categorical_change_towards_shared',
        'categorical_change_towards_specific',
        'categorical_change_neutral',
        'num_upregulated',
        'num_downregulated',
    ]

    ACCUM_FIELDS_ALL = ACCUM_FIELDS_FLOAT + ACCUM_FIELDS_INT

    def __init__(self, gene_ids) -> None:
        """Initialize accumulator."""
        n = len(gene_ids)

        self.__acc = {
            'gene_ids': gene_ids,
            'cell_line_genes': {}
        }
        self.__acc.update({k: np.zeros(n) for k in self.ACCUM_FIELDS_FLOAT})
        self.__acc.update({k: np.zeros(n, dtype=np.int64) for k in self.ACCUM_FIELDS_INT})

    def __setitem__(self, key, item) -> None:
        self.__acc[key] = item

    def __getitem__(self, key) -> Any:
        return self.__acc[key]

    def accumulate(
        self,
        cell_line_id,
        gene_ids,
        delta_p,
        influences,
        valid,
        frac_systematic,
    ) -> None:
        """Accumulate influence scores."""
        idx = np.flatnonzero(valid & np.isfinite(influences))

        if len(idx) == 0:
            return

        inf = influences[idx]
        dp = delta_p[idx]

        # These indices correspond directly to positions in acc
        # Assumes gene_ids has the same ordering in every call.
        assert np.array_equal(gene_ids, self['gene_ids'])

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
        self['positive_influence'][idx] += np.maximum(inf, 0)
        self['negative_influence'][idx] += np.minimum(inf, 0)
        self['total_influence'][idx] += inf
        self['total_influence_abs'][idx] += np.abs(inf)
        self['num_instances'][idx] += 1
        self['num_positive_instances'][idx] += (inf > 0).astype(int)
        self['num_negative_instances'][idx] += (inf < 0).astype(int)

        self['delta_p_magnitude_total'][idx] += dp
        self['delta_p_magnitude_total_abs'][idx] += np.abs(dp)
        self['delta_p_magnitude_total_pos'][idx] += np.maximum(dp, 0)
        self['delta_p_magnitude_total_neg'][idx] += np.minimum(dp, 0)

        self['num_upregulated'][idx] += (dp >= 0)
        self['num_downregulated'][idx] += (dp < 0)

        self['categorical_change_towards_shared'][idx] += shared
        self['categorical_change_towards_specific'][idx] += specific
        self['categorical_change_neutral'][idx] += neutral

        # One set update per perturbation, rather than one per gene
        cell_line_genes = self['cell_line_genes'].setdefault(
            cell_line_id, set()
        )
        cell_line_genes.update(gene_ids[idx])

    def finalize_by_gene(self) -> dict[str, Any]:
        """Finalize accumulation results by gene"""
        gene_ids = self['gene_ids']

        # Build gene -> cell lines mapping
        cell_lines_by_gene = {}

        for cell_line, genes in self['cell_line_genes'].items():
            for gene in genes:
                cell_lines_by_gene.setdefault(gene, set()).add(cell_line)

        result = {}

        for i in np.flatnonzero(self['num_instances']):
            gene = gene_ids[i].item()

            result[gene] = {
                field: self[field][i].item()
                for field in self.ACCUM_FIELDS_ALL
            }

            result[gene]['cell_lines'] = cell_lines_by_gene.get(
                gene, set()
            )

        return result


def summarize_perturbation_influence(gene_id: str, influence):
    num_instances = influence['num_instances']
    num_positive_instances = influence['num_positive_instances']
    num_negative_instances = influence['num_negative_instances']

    return {
        'token_id': gene_id,
        'avg_influence': influence['total_influence'] / num_instances,
        'avg_positive_influence': influence['positive_influence'] / influence['num_positive_instances'] if num_positive_instances else 0,
        'avg_negative_influence': influence['negative_influence'] / influence['num_negative_instances'] if num_negative_instances else 0,
        'negative_influence': influence['negative_influence'],
        'positive_influence': influence['positive_influence'],
        'total_influence': influence['total_influence'],
        'total_influence_abs': influence['total_influence_abs'],
        'mean_abs_influence': influence['total_influence_abs'] / num_instances,
        'num_instances': num_instances,
        'num_positive_instances': num_positive_instances,
        'num_negative_instances': num_negative_instances,
        'num_cell_lines': len(influence['cell_lines']),
        'categorical_change_towards_shared': influence['categorical_change_towards_shared'],
        'categorical_change_towards_specific': influence['categorical_change_towards_specific'],
        'categorical_change_neutral': influence['categorical_change_neutral'],
        'delta_p_magnitude_avg': influence['delta_p_magnitude_total'] / num_instances,
        'delta_p_magnitude_abs_avg': influence['delta_p_magnitude_total_abs'] / num_instances,
        'delta_p_magnitude_pos_avg': influence['delta_p_magnitude_total_pos'] / num_instances,
        'delta_p_magnitude_neg_avg': influence['delta_p_magnitude_total_neg'] / num_instances,
        'num_upregulated': influence['num_upregulated'],
        'num_downregulated': influence['num_downregulated']
    }


class GSEAInfluenceAnnotator:
    """Class to handle adding influence scores to GSEA results."""
    def __init__(self, file: Path, moa_targets: list[str, Any] | None = None) -> None:
        self.__gsea_axis = pd.read_csv(file)

        # assert cell_line_id and normalization mode are in the df
        for required_field in ['cell_line', 'mode']:
            assert required_field in self.__gsea_axis.columns, \
                f"Missing required column {required_field} in gsea_axis data"

        # add MOA targets, if provided, same format as lead_genes
        if moa_targets is not None:
            self.__gsea_axis['moa_target_genes'] = (
                self.__gsea_axis['moa']
                .map(moa_targets)
                .apply(lambda genes: ';'.join(genes) if isinstance(genes, list) else pd.NA)
            )

    def add_influence_score_to_lead_genes(
        self, influence_lookup: dict[str, Any], lead_genes: str
    ) -> str:
        """Add influence score to lead genes. Assumes influence_scores and lead_genes
        are associated with the same context (e.g. cell line or MOA grouping).

        In the GSEA output, the lead_genes column is a ; separated list of genes

        Returns a ; separated list of 
            gene:mean_abs_inf:mean_pos_inf,mean_neg_inf:rank
        """
        updated_genes = []

        for gene in lead_genes.split(';'):
            gene = gene.strip()

            if not gene:
                continue

            influence = influence_lookup.get(gene)

            if influence is None:
                updated_genes.append(f'{gene}:NA:NA:NA:NA')
                continue

            updated_genes.append(
                f'{gene}:'
                f'{influence["mean_abs_influence"]}:'
                f'{influence["avg_positive_influence"]}:'
                f'{influence["avg_negative_influence"]}:'
                f'{influence["influence_rank"]}'
            )

        return ';'.join(updated_genes)

    def update_lead_genes(
        self,
        influence_lookup: dict[str, Any],
        normalization_mode: str,
        cell_line_id: str,
        moa: str | None = None,
    ) -> None:
        mask = self.__gsea_axis['mode'] == normalization_mode
        mask &= self.__gsea_axis['cell_line'] == cell_line_id

        if moa is not None:
            mask &= self.__gsea_axis['moa'] == moa

        for column in ['lead_genes', 'moa_target_genes']:
            if column not in self.__gsea_axis.columns:
                continue

            self.__gsea_axis.loc[mask, column] = (
                self.__gsea_axis.loc[mask, column].apply(
                    lambda genes: self.add_influence_score_to_lead_genes(
                        influence_lookup, genes
                    ) if pd.notna(genes) else genes
                )
            )

    def to_csv(self, outfile: Path) -> None:
        self.__gsea_axis.to_csv(outfile, index=False)
