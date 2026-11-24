"""
Quantify the genes per cell line based on the control parquet files.
"""
import csv
import argparse

from collections import Counter
from pathlib import Path

from tahoe100m_reliability.lib.ingest.intermediate_loaders import (
    load_gene_ids
)
from tahoe100m_reliability.lib.ingest.msigdb_hallmark import (
    load_hallmark_gene_mapping,
    load_hallmark_sets,
)
from tahoe100m_reliability.lib.ingest.tahoe100m_dataset import (
    get_gene_metadata,
    get_organ_by_cell_line,
)


def quantify_genes(ctrl_gene_expressions_dir: Path, outfile: Path) -> None:
    """Quantifies genes per cell line based on the control parquet files."""
    results = []
    organ_by_cell_line = get_organ_by_cell_line()

    # load hallmarks and map to tokens
    gene_to_hallmarks = load_hallmark_gene_mapping()
    gene_metadata = get_gene_metadata(to_pandas=True)

    token_to_hallmarks = (
        gene_metadata.assign(
            gene_symbol=gene_metadata["gene_symbol"].str.upper()
        )
        .assign(
            hallmark_categories=lambda df: (
                df["gene_symbol"].map(gene_to_hallmarks)
                .apply(lambda x: x if isinstance(x, list) else [])
            )
        )
        .set_index("token_id")["hallmark_categories"]
        .to_dict()
    )

    hallmark_terms = list(load_hallmark_sets().keys()) + ['UNASSIGNED']

    for file in ctrl_gene_expressions_dir.glob("CVCL_*_gene_expressions.parquet"):
        cell_line_id = file.stem.removesuffix("_gene_expressions")
        gene_ids = load_gene_ids(ctrl_gene_expressions_dir, cell_line_id)

        # count hallmark genes
        hallmark_counts = Counter()
        for token_id in gene_ids:
            categories = token_to_hallmarks.get(token_id, [])

            if categories:
                hallmark_counts.update(categories)
            else:
                hallmark_counts["UNASSIGNED"] += 1

        num_genes = len(gene_ids)
        result = {
            "cell_line_id": cell_line_id,
            "organ": organ_by_cell_line.get(cell_line_id, "Unknown"),
            "num_genes": num_genes,
        }

        for term in hallmark_terms:
            count = hallmark_counts.get(term, 0)
            result[f"{term}_count"] = count
            result[f"{term}_fraction"] = count / num_genes if num_genes else 0

        results.append(result)

    with outfile.open('w') as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(
        prog='quantify-genes',
        description='Quantifies the number of genes per cell line based on the control parquet files.'
    )

    parser.add_argument(
        '--ctrl-gene-expressions-dir', dest='ctrl_gene_expressions_dir',
        required=True, type=Path,
        help='Path to the directory containing the control gene matrix for all 50 '
            + 'cell lines, each an individual parquet file.'
    )

    parser.add_argument(
        '--outfile', dest='outfile',
        required=True, type=Path,
        help='CSV path to write results to.'
    )

    args = parser.parse_args()

    if not args.ctrl_gene_expressions_dir.is_dir():
        raise FileNotFoundError("Specified control gene expressions directory does not exist")

    quantify_genes(args.ctrl_gene_expressions_dir, args.outfile)


if __name__ == "__main__":
    main()
