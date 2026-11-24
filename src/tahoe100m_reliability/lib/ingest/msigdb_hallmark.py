"""
Help with loading MSigDB hallmark data
"""
import blitzgsea as blitz


def load_hallmark_sets():
    """Load hallmark sets."""
    library = blitz.enrichr.get_library("MSigDB_Hallmark_2020")
    return {
        term: [gene.upper() for gene in genes]
        for term, genes in library.items()
    }


def load_hallmark_gene_mapping() -> dict[str, list[str]]:
    """Map each gene symbol to its Hallmark categories."""
    hallmark_sets = load_hallmark_sets()

    gene_to_hallmarks = {}

    for hallmark, genes in hallmark_sets.items():
        for gene in genes:
            gene_to_hallmarks.setdefault(gene, []).append(hallmark)

    return gene_to_hallmarks
