"""
Functions to help support computing GSEA.
"""
import blitzgsea as blitz
import gseapy as gp
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def gsea_nes(vec, genes_upper, hallmark_sets):
    """Runs preranked GSEA using all genes."""
    series = pd.Series(
        np.asarray(vec, dtype=float),
        index=genes_upper
    )
    series = series[~series.index.duplicated()].sort_values(ascending=False)

    # skip degenerate rankings (near-constant => gseapy Rust backend panics);
    # residual~0 perts correctly -> ns
    if (
        series.shape[0] < 200
        or series.std() < 1e-12
        or (series.value_counts().iloc[0] / series.shape[0]) > 0.5
    ):
        return {}

    try:
        result = gp.prerank(
            rnk=series,
            gene_sets=hallmark_sets,
            min_size=5,
            max_size=500,
            permutation_num=1000,
            threads=4,
            seed=0,
            no_plot=True,
            outdir=None,
            verbose=False,
        ).res2d

        return {
            str(row.Term).replace("HALLMARK_", ""): {
                "nes": float(row['NES']),
                "fdr_qval": float(row['FDR q-val']),
                "lead_genes": row['Lead_genes']
            }
            for _, row in result.iterrows()
        }

    except Exception as exc:
        log.info(f"  WARNING: GSEA failed/skipped: {exc}")
        return {}


# Running GSEA on the specific axis is extremely resource intensive; use a subset
# of cell lines for investigative purposes. Picks the cell line with the most
# samples for each of the 13 tissue types
INVESTIGATIVE_CELL_LINES = [
    'CVCL_0546',  # Bowel
    'CVCL_0480',  # Pancreas
    'CVCL_0459',  # Lung
    'CVCL_1056',  # Kidney
    'CVCL_0293',  # Uterus
    'CVCL_0131',  # CNS/Brain
    'CVCL_0359',  # Bladder
    'CVCL_0371',  # Esophogus
    'CVCL_1381',  # Skin
    'CVCL_0366',  # Liver
    'CVCL_0332',  # Breast
    'CVCL_1094',  # Cervix
    'CVCL_1125',  # PNS
]


def blitzgsea_nes(vec, genes_upper, hallmark_sets, permutations: int = 100):
    """Runs preranked GSEA using all genes (blitzgsea version)."""
    # Construct ranked gene signature
    series = pd.Series(
        np.asarray(vec, dtype=float),
        index=genes_upper
    )
    series = series[~series.index.duplicated()].sort_values(ascending=False)

    # skip degenerate rankings (near-constant => gseapy Rust backend panics);
    # residual~0 perts correctly -> ns
    if (
        series.shape[0] < 200
        or series.std() < 1e-12
        or (series.value_counts().iloc[0] / series.shape[0]) > 0.5
    ):
        return {}

    # blitzGSEA expects a two-column DataFrame
    signature = pd.DataFrame({
        0: series.index,
        1: series.values
    })

    try:
        # Run blitzGSEA
        result = blitz.gsea(
            signature,
            hallmark_sets,
            min_size=5,
            max_size=500,
            permutations=permutations,
            processes=4,
            seed=0,
            verbose=False
        )

        return {
            str(term): {
                "nes": float(row['nes']),
                "fdr_qval": float(row['fdr']),
                "lead_genes": row['leading_edge']
            }
            for term, row in result.iterrows()
        }

    except Exception as exc:
        log.info(f"  WARNING: GSEA failed/skipped: {exc}")
        return {}
