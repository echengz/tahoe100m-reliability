"""
Functions to support computing systematic metrics.
"""
import numpy as np


def cosine_similarity(a, b):
    """Cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(dot / (norm_a * norm_b))


def calculate_frac_systematic(delta_p, delta_avg, norm_delta_avg, is_rel):
    """Calculate baseline systematic fraction and associated metrics."""
    norm_delta_p = float(np.linalg.norm(delta_p))

    if is_rel and norm_delta_avg > 1e-12:
        cos_sim = cosine_similarity(delta_p, delta_avg)

        proj_scalar = (
            np.dot(delta_p, delta_avg) / (norm_delta_avg ** 2)
        )

        delta_specific = delta_p - proj_scalar * delta_avg
        norm_delta_specific = float(np.linalg.norm(delta_specific))

        frac_systematic = (
            float(cos_sim ** 2)
            if norm_delta_p > 1e-12
            else np.nan
        )
    else:
        cos_sim = np.nan
        norm_delta_specific = np.nan
        frac_systematic = np.nan

    return (
        cos_sim,
        norm_delta_p,
        norm_delta_specific,
        frac_systematic,
    )


def score_perturbation(
    delta_p,
    delta_avg,
    norm_delta_avg,
    is_rel,
    # gene_ids,
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

        # # Leave-one-gene-out calculations
        # dot_without = np.empty_like(delta_p)
        # np.multiply(delta_p, delta_avg, out=dot_without)
        # np.subtract(dot_pa, dot_without, out=dot_without)

        # p_without = np.empty_like(delta_p)
        # np.square(delta_p, out=p_without)
        # np.subtract(norm_p_sq, p_without, out=p_without)

        # a_without = np.empty_like(delta_avg)
        # np.square(delta_avg, out=a_without)
        # np.subtract(norm_a_sq, a_without, out=a_without)

        # np.maximum(p_without, 0.0, out=p_without)
        # np.maximum(a_without, 0.0, out=a_without)

        # valid = (p_without > 1e-24) & (a_without > 1e-24)
        # np.multiply(p_without, a_without, out=p_without)

        # np.square(dot_without, out=dot_without)
        # np.divide(
        #     dot_without,
        #     p_without,
        #     out=dot_without,
        #     where=valid,
        # )

        # influences = np.full(len(gene_ids), np.nan)
        # influences[valid] = frac_systematic - dot_without[valid]

    else:
        cos_sim = np.nan
        frac_systematic = np.nan
        norm_delta_specific = np.nan

    row = {
        "cos_sim": cos_sim,
        "norm_delta_p": norm_delta_p,
        "norm_delta_avg": norm_delta_avg,
        "norm_delta_specific": norm_delta_specific,
        "frac_systematic_var": frac_systematic,
    }

    return row