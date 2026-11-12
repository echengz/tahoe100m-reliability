
"""
Accumulators for systematic-step influence scores.

InfluenceAccumulator:
    Preserves the original gene-level leave-one-gene-out analysis.

GroupInfluenceAccumulator:
    Accumulates pathway-level and union-of-pathways influence scores.
"""
import numpy as np
from typing import Any


SPECIFICITY_THRESH = 0.5


class InfluenceAccumulator:
    """Accumulate leave-one-gene-out influence scores."""

    ACCUM_FIELDS_FLOAT = [
        "negative_influence",
        "positive_influence",
        "total_influence",
        "total_influence_abs",
        "delta_p_magnitude_total",
        "delta_p_magnitude_total_abs",
        "delta_p_magnitude_total_pos",
        "delta_p_magnitude_total_neg",
    ]

    ACCUM_FIELDS_INT = [
        "num_instances",
        "num_positive_instances",
        "num_negative_instances",
        "categorical_change_towards_shared",
        "categorical_change_towards_specific",
        "categorical_change_neutral",
        "num_upregulated",
        "num_downregulated",
    ]

    ACCUM_FIELDS_ALL = ACCUM_FIELDS_FLOAT + ACCUM_FIELDS_INT

    def __init__(self, gene_ids) -> None:
        gene_ids = np.asarray(gene_ids)
        n = len(gene_ids)

        self.__acc = {
            "gene_ids": gene_ids,
            "cell_line_genes": {},
        }
        self.__acc.update({
            key: np.zeros(n, dtype=np.float64)
            for key in self.ACCUM_FIELDS_FLOAT
        })
        self.__acc.update({
            key: np.zeros(n, dtype=np.int64)
            for key in self.ACCUM_FIELDS_INT
        })

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
        """Accumulate gene-level leave-one-out influence scores."""
        gene_ids = np.asarray(gene_ids)
        delta_p = np.asarray(delta_p)
        influences = np.asarray(influences)
        valid = np.asarray(valid, dtype=bool)

        if not np.array_equal(gene_ids, self["gene_ids"]):
            raise ValueError("gene_ids ordering differs from accumulator ordering")

        idx = np.flatnonzero(valid & np.isfinite(influences))
        if len(idx) == 0:
            return

        inf = influences[idx]
        dp = delta_p[idx]

        shared = (
            (frac_systematic < SPECIFICITY_THRESH)
            & (frac_systematic - inf >= SPECIFICITY_THRESH)
        )
        specific = (
            (frac_systematic >= SPECIFICITY_THRESH)
            & (frac_systematic - inf < SPECIFICITY_THRESH)
        )
        neutral = ~(shared | specific)

        self["positive_influence"][idx] += np.maximum(inf, 0)
        self["negative_influence"][idx] += np.minimum(inf, 0)
        self["total_influence"][idx] += inf
        self["total_influence_abs"][idx] += np.abs(inf)

        self["num_instances"][idx] += 1
        self["num_positive_instances"][idx] += (inf > 0)
        self["num_negative_instances"][idx] += (inf < 0)

        self["delta_p_magnitude_total"][idx] += dp
        self["delta_p_magnitude_total_abs"][idx] += np.abs(dp)
        self["delta_p_magnitude_total_pos"][idx] += np.maximum(dp, 0)
        self["delta_p_magnitude_total_neg"][idx] += np.minimum(dp, 0)

        self["num_upregulated"][idx] += (dp >= 0)
        self["num_downregulated"][idx] += (dp < 0)

        self["categorical_change_towards_shared"][idx] += shared
        self["categorical_change_towards_specific"][idx] += specific
        self["categorical_change_neutral"][idx] += neutral

        cell_line_genes = self["cell_line_genes"].setdefault(
            cell_line_id, set()
        )
        cell_line_genes.update(gene_ids[idx].tolist())

    def finalize_by_gene(self) -> dict[str, Any]:
        """Return accumulated results keyed by gene token ID."""
        cell_lines_by_gene = {}

        for cell_line, genes in self["cell_line_genes"].items():
            for gene in genes:
                cell_lines_by_gene.setdefault(gene, set()).add(cell_line)

        result = {}
        gene_ids = self["gene_ids"]

        for i in np.flatnonzero(self["num_instances"]):
            gene = gene_ids[i].item()

            result[gene] = {
                field: self[field][i].item()
                for field in self.ACCUM_FIELDS_ALL
            }
            result[gene]["cell_lines"] = cell_lines_by_gene.get(gene, set())

        return result


class GroupInfluenceAccumulator:
    """
    Accumulate influence scores for arbitrary gene groups.

    Each row of group_gene_mask represents one group and each column
    represents one gene in the delta vectors.
    """

    ACCUM_FIELDS_FLOAT = InfluenceAccumulator.ACCUM_FIELDS_FLOAT
    ACCUM_FIELDS_INT = InfluenceAccumulator.ACCUM_FIELDS_INT
    ACCUM_FIELDS_ALL = InfluenceAccumulator.ACCUM_FIELDS_ALL

    def __init__(self, group_ids, group_gene_ids):
        self.group_ids = np.asarray(group_ids, dtype=object)

        if self.group_ids.ndim != 1:
            raise ValueError("group_ids must be one-dimensional")

        if len(set(self.group_ids.tolist())) != len(self.group_ids):
            raise ValueError("group_ids must be unique")

        if len(self.group_ids) != len(group_gene_ids):
            raise ValueError("group_ids and group_gene_ids must have equal lengths")

        self.group_gene_ids = {
            group_id: list(genes)
            for group_id, genes in zip(self.group_ids, group_gene_ids)
        }

        n = len(self.group_ids)
        self.acc = {
            key: np.zeros(n, dtype=np.float64)
            for key in self.ACCUM_FIELDS_FLOAT
        }
        self.acc.update({
            key: np.zeros(n, dtype=np.int64)
            for key in self.ACCUM_FIELDS_INT
        })

        self.cell_lines = [set() for _ in range(n)]

    @staticmethod
    def calculate_group_influences(delta_p, delta_avg, group_gene_mask):
        """
        Calculate phi_before - phi_after for all groups simultaneously.

        Vectorized for efficiency
        """
        delta_p = np.asarray(delta_p, dtype=np.float64)
        delta_avg = np.asarray(delta_avg, dtype=np.float64)
        mask = np.asarray(group_gene_mask, dtype=bool)

        if delta_p.ndim != 1 or delta_avg.ndim != 1:
            raise ValueError("delta_p and delta_avg must be one-dimensional")

        if delta_p.shape != delta_avg.shape:
            raise ValueError("delta_p and delta_avg must have equal shapes")

        if mask.ndim != 2 or mask.shape[1] != len(delta_p):
            raise ValueError(
                "group_gene_mask must have shape (n_groups, n_genes)"
            )

        if not (
            np.all(np.isfinite(delta_p))
            and np.all(np.isfinite(delta_avg))
        ):
            return np.full(mask.shape[0], np.nan)

        dot_before = np.dot(delta_p, delta_avg)
        p_sq_before = np.dot(delta_p, delta_p)
        a_sq_before = np.dot(delta_avg, delta_avg)

        denom_before = p_sq_before * a_sq_before
        if denom_before <= 1e-24:
            return np.full(mask.shape[0], np.nan)

        phi_before = dot_before**2 / denom_before

        # Contributions removed by each group.
        removed_dot = mask @ (delta_p * delta_avg)
        removed_p_sq = mask @ (delta_p**2)
        removed_a_sq = mask @ (delta_avg**2)

        dot_after = dot_before - removed_dot
        p_sq_after = np.maximum(p_sq_before - removed_p_sq, 0.0)
        a_sq_after = np.maximum(a_sq_before - removed_a_sq, 0.0)

        denom_after = p_sq_after * a_sq_after
        valid = denom_after > 1e-24

        phi_after = np.full(mask.shape[0], np.nan)
        phi_after[valid] = (
            dot_after[valid] ** 2 / denom_after[valid]
        )

        return phi_before - phi_after

    def accumulate(
        self,
        cell_line_id,
        group_ids,
        delta_p,
        group_gene_mask,
        influences,
        valid,
        frac_systematic,
    ):
        """Accumulate group-level influence scores."""
        group_ids = np.asarray(group_ids, dtype=object)
        delta_p = np.asarray(delta_p, dtype=np.float64)
        mask = np.asarray(group_gene_mask, dtype=bool)
        influences = np.asarray(influences, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        frac_systematic = np.asarray(frac_systematic, dtype=np.float64)

        if not np.array_equal(group_ids, self.group_ids):
            raise ValueError("group_ids differ from accumulator ordering")

        n_groups = len(self.group_ids)

        if mask.shape != (n_groups, len(delta_p)):
            raise ValueError("group_gene_mask has an unexpected shape")

        if not (
            len(influences) == n_groups
            and len(valid) == n_groups
            and len(frac_systematic) == n_groups
        ):
            raise ValueError("Group accumulator inputs have inconsistent lengths")

        valid = (
            valid
            & np.isfinite(influences)
            & np.isfinite(frac_systematic)
        )

        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return

        inf = influences[idx]
        frac = frac_systematic[idx]

        # Mean absolute perturbation magnitude within each removed group.
        group_sizes = mask[idx].sum(axis=1)
        dp_total = mask[idx] @ delta_p
        dp_abs_total = mask[idx] @ np.abs(delta_p)
        dp_pos_total = mask[idx] @ np.maximum(delta_p, 0)
        dp_neg_total = mask[idx] @ np.minimum(delta_p, 0)

        dp_mean = np.divide(
            dp_total,
            group_sizes,
            out=np.zeros_like(dp_total, dtype=float),
            where=group_sizes > 0,
        )
        dp_abs_mean = np.divide(
            dp_abs_total,
            group_sizes,
            out=np.zeros_like(dp_abs_total, dtype=float),
            where=group_sizes > 0,
        )
        dp_pos_mean = np.divide(
            dp_pos_total,
            group_sizes,
            out=np.zeros_like(dp_pos_total, dtype=float),
            where=group_sizes > 0,
        )
        dp_neg_mean = np.divide(
            dp_neg_total,
            group_sizes,
            out=np.zeros_like(dp_neg_total, dtype=float),
            where=group_sizes > 0,
        )

        shared = (frac < SPECIFICITY_THRESH) & (
            frac - inf >= SPECIFICITY_THRESH
        )
        specific = (frac >= SPECIFICITY_THRESH) & (
            frac - inf < SPECIFICITY_THRESH
        )
        neutral = ~(shared | specific)

        self.acc["positive_influence"][idx] += np.maximum(inf, 0)
        self.acc["negative_influence"][idx] += np.minimum(inf, 0)
        self.acc["total_influence"][idx] += inf
        self.acc["total_influence_abs"][idx] += np.abs(inf)

        self.acc["num_instances"][idx] += 1
        self.acc["num_positive_instances"][idx] += (inf > 0)
        self.acc["num_negative_instances"][idx] += (inf < 0)

        self.acc["delta_p_magnitude_total"][idx] += dp_mean
        self.acc["delta_p_magnitude_total_abs"][idx] += dp_abs_mean
        self.acc["delta_p_magnitude_total_pos"][idx] += dp_pos_mean
        self.acc["delta_p_magnitude_total_neg"][idx] += dp_neg_mean

        self.acc["num_upregulated"][idx] += (dp_mean >= 0)
        self.acc["num_downregulated"][idx] += (dp_mean < 0)

        self.acc["categorical_change_towards_shared"][idx] += shared
        self.acc["categorical_change_towards_specific"][idx] += specific
        self.acc["categorical_change_neutral"][idx] += neutral

        for i in idx:
            self.cell_lines[i].add(cell_line_id)

    def finalize_by_group(self):
        """Return accumulated results keyed by group ID."""
        result = {}

        for i, group_id in enumerate(self.group_ids):
            n = int(self.acc["num_instances"][i])
            if n == 0:
                continue

            values = {
                field: self.acc[field][i].item()
                for field in self.ACCUM_FIELDS_ALL
            }
            values["cell_lines"] = self.cell_lines[i]
            values["gene_ids"] = self.group_gene_ids[group_id]
            result[group_id] = values

        return result


def summarize_perturbation_influence(gene_id: str, influence):
    """Summarize one gene's accumulated leave-one-out influence."""
    n = influence["num_instances"]
    n_pos = influence["num_positive_instances"]
    n_neg = influence["num_negative_instances"]

    return {
        "token_id": gene_id,
        "avg_influence": influence["total_influence"] / n,
        "avg_positive_influence": influence["positive_influence"] / n_pos if n_pos else 0,
        "avg_negative_influence": influence["negative_influence"] / n_neg if n_neg else 0,
        "negative_influence": influence["negative_influence"],
        "positive_influence": influence["positive_influence"],
        "total_influence": influence["total_influence"],
        "total_influence_abs": influence["total_influence_abs"],
        "mean_abs_influence": influence["total_influence_abs"] / n,
        "num_instances": n,
        "num_positive_instances": n_pos,
        "num_negative_instances": n_neg,
        "num_cell_lines": len(influence["cell_lines"]),
        "cell_lines": ";".join(sorted(influence["cell_lines"])),
        "categorical_change_towards_shared": influence["categorical_change_towards_shared"],
        "categorical_change_towards_specific": influence["categorical_change_towards_specific"],
        "categorical_change_neutral": influence["categorical_change_neutral"],
        "delta_p_magnitude_avg": influence["delta_p_magnitude_total"] / n,
        "delta_p_magnitude_abs_avg": influence["delta_p_magnitude_total_abs"] / n,
        "delta_p_magnitude_pos_avg": influence["delta_p_magnitude_total_pos"] / n,
        "delta_p_magnitude_neg_avg": influence["delta_p_magnitude_total_neg"] / n,
        "num_upregulated": influence["num_upregulated"],
        "num_downregulated": influence["num_downregulated"],
    }
