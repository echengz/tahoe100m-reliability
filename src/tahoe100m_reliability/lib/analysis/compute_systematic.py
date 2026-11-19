"""
Functions to support computing vectorized systematic metrics.
"""
import numpy as np


def score_perturbation(
    delta_p,
    delta_avg,
    norm_delta_avg,
    is_rel,
):
    """Score the given perturbation."""
    delta_p = np.asarray(delta_p, dtype=np.float64)
    delta_avg = np.asarray(delta_avg, dtype=np.float64)

    # Calculate shared quantities
    dot_pa = np.dot(delta_p, delta_avg)
    norm_p_sq = np.dot(delta_p, delta_p)
    norm_a_sq = norm_delta_avg ** 2

    norm_delta_p = np.sqrt(norm_p_sq)

    if is_rel and norm_a_sq > 1e-24 and norm_delta_p > 1e-12:
        cos_sim = dot_pa / (norm_delta_p * norm_delta_avg) \
            if norm_delta_p > 1e-12 else np.nan

        frac_systematic = cos_sim ** 2

        norm_delta_specific = np.sqrt(max(
            norm_p_sq - dot_pa**2 / norm_a_sq,
            0.0
        ))

    else:
        cos_sim = np.nan
        frac_systematic = np.nan
        norm_delta_specific = np.nan

    # saved as output
    results = {
        "cos_sim": cos_sim,
        "norm_delta_p": norm_delta_p,
        "norm_delta_avg": norm_delta_avg,
        "norm_delta_specific": norm_delta_specific,
        "frac_systematic_var": frac_systematic,
    }

    # for further processing
    working = {
        "dot_pa": dot_pa,
        "norm_p_sq": norm_p_sq,
        "norm_a_sq": norm_a_sq,
    }

    return results, working


def compute_gene_influences(
    delta_p,
    delta_avg,
    working_results,
    frac_systematic,
):
    """Compute influence for each individual gene."""
    delta_p = np.asarray(delta_p, dtype=np.float64)
    delta_avg = np.asarray(delta_avg, dtype=np.float64)

    dot_pa = working_results["dot_pa"]
    norm_p_sq = working_results["norm_p_sq"]
    norm_a_sq = working_results["norm_a_sq"]

    # Leave-one-gene-out dot products
    dot_without = np.empty_like(delta_p)
    np.multiply(delta_p, delta_avg, out=dot_without)
    np.subtract(dot_pa, dot_without, out=dot_without)

    # Leave-one-gene-out squared norms
    p_without = np.empty_like(delta_p)
    np.square(delta_p, out=p_without)
    np.subtract(norm_p_sq, p_without, out=p_without)

    a_without = np.empty_like(delta_avg)
    np.square(delta_avg, out=a_without)
    np.subtract(norm_a_sq, a_without, out=a_without)

    np.maximum(p_without, 0.0, out=p_without)
    np.maximum(a_without, 0.0, out=a_without)

    valid = (p_without > 1e-24) & (a_without > 1e-24)

    # Compute leave-one-out specificity
    np.multiply(p_without, a_without, out=p_without)

    np.square(dot_without, out=dot_without)
    np.divide(
        dot_without,
        p_without,
        out=dot_without,
        where=valid,
    )

    influences = np.full(delta_p.shape, np.nan)
    influences[valid] = frac_systematic - dot_without[valid]

    return influences, valid


def compute_pathway_influences(
    pathway_masks,
    delta_p,
    delta_avg,
    working_results,
    frac_systematic,
):
    """
    Compute pathway influence for each pathway. Expects pathway
    masks that mask the lead genes as defined by the GSEA analysis
    results.
    """
    pathway_masks = np.asarray(pathway_masks, dtype=bool)
    delta_p = np.asarray(delta_p, dtype=np.float64)
    delta_avg = np.asarray(delta_avg, dtype=np.float64)

    dot_pa = working_results["dot_pa"]
    norm_p_sq = working_results["norm_p_sq"]
    norm_a_sq = working_results["norm_a_sq"]

    # Contributions removed by each pathway
    dot_removed = pathway_masks @ (delta_p * delta_avg)
    p_removed = pathway_masks @ (delta_p ** 2)
    a_removed = pathway_masks @ (delta_avg ** 2)

    # Leave-one-pathway-out quantities
    dot_without = dot_pa - dot_removed
    p_without = np.maximum(norm_p_sq - p_removed, 0.0)
    a_without = np.maximum(norm_a_sq - a_removed, 0.0)

    valid = (p_without > 1e-24) & (a_without > 1e-24)

    # Calculate specificity without each pathway
    denominator = p_without * a_without

    frac_without = np.full(len(pathway_masks), np.nan)

    np.divide(
        dot_without ** 2,
        denominator,
        out=frac_without,
        where=valid,
    )

    influences = frac_systematic - frac_without

    return influences, valid
