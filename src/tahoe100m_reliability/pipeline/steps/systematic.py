"""
Runs the systematic step of the pipeline
"""
import argparse
import logging
import time

from pathlib import Path

from tahoe100m_reliability.lib.tahoe100m_dataset import get_gene_metadata

from .step_helpers.intermediate_loaders import (
    load_reliable_perts_cellular,
    load_systema_deltas,
    RELIABILITY_THRESH,
    SPECIFICITY_THRESH,
)
from .step_helpers.pipeline_step import PipelineStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def cosine_similarity(a, b):
    """Cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def batch_cosine_similarity(deltas, avg_delta):
    """
    Cosine similarity between each row of `deltas` and a single vector.

    Parameters
    ----------
    deltas    : ndarray, shape (N_perts, G)
    avg_delta : ndarray, shape (G,)

    Returns
    -------
    cos_sims : ndarray, shape (N_perts,)
    """
    dots = deltas @ avg_delta
    norms_delta = np.linalg.norm(deltas, axis=1)
    norm_avg = np.linalg.norm(avg_delta)
    denom = norms_delta * norm_avg
    denom[denom < 1e-12] = 1.0
    return dots / denom


def calculate_frac_systematic(
    delta_p, delta_avg, norm_delta_avg, is_rel
):
    norm_delta_p = float(np.linalg.norm(delta_p))

    if is_rel and norm_delta_avg > 1e-12:
        cos_sim = cosine_similarity(
            delta_p,
            delta_avg,
        )

        proj_scalar = (
            np.dot(delta_p, delta_avg)
            / (norm_delta_avg ** 2)
        )

        delta_specific = (
            delta_p
            - proj_scalar * delta_avg
        )

        norm_delta_specific = float(
            np.linalg.norm(delta_specific)
        )

        frac_systematic = (
            float(cos_sim ** 2)
            if norm_delta_p > 1e-12
            else np.nan
        )

    else:
        cos_sim = np.nan
        norm_delta_specific = np.nan
        frac_systematic = np.nan

    return cos_sim, norm_delta_p, norm_delta_specific, frac_systematic


def score_perturbation(
    ds_name,
    condition,
    n_perts,
    n_stim,
    delta_p,
    delta_avg,
    norm_delta_avg,
    is_rel,
    gene_ids,
):
    """"Score a perturbation and calculate gene influence."""
    condition = str(condition)
    n_stim = int(n_stim)

    if n_stim < 4:
        return None, None, None

    # condition format: cell_line_id|drug|dosage
    parts = condition.split("|")

    if ds_name == 'tahoe100m':
        if len(parts) != 3:
            raise ValueError(
                f"Unexpected condition format: {condition}"
            )

        context, drug, dosage = parts
        perturbation = f'{drug}|{dosage}'
    else:
        context, perturbation = parts

    delta_p = delta_p.astype(np.float64)

    cos_sim, norm_delta_p, norm_delta_specific, frac_systematic = \
        calculate_frac_systematic(delta_p, delta_avg, norm_delta_avg, is_rel)

    # Calculate each gene's influence to this result
    gene_results = {}
    influence_summary = None
    valid = None
    influences = None

    if np.isfinite(frac_systematic) and is_rel:
        # Original vector quantities
        dot_pa = np.dot(delta_p, delta_avg)
        norm_p_sq = np.dot(delta_p, delta_p)
        norm_a_sq = np.dot(delta_avg, delta_avg)

        # Leave-one-gene-out quantities, calculated elementwise
        dot_without_g = dot_pa - delta_p * delta_avg
        norm_p_sq_without_g = norm_p_sq - delta_p**2
        norm_a_sq_without_g = norm_a_sq - delta_avg**2

        # Avoid tiny negative values from floating-point rounding
        norm_p_sq_without_g = np.maximum(norm_p_sq_without_g, 0.0)
        norm_a_sq_without_g = np.maximum(norm_a_sq_without_g, 0.0)

        denom = norm_p_sq_without_g * norm_a_sq_without_g

        frac_without_g = np.full(len(gene_ids), np.nan)
        valid = (
            (norm_p_sq_without_g > 1e-24)
            & (norm_a_sq_without_g > 1e-24)
        )

        frac_without_g[valid] = (
            dot_without_g[valid] ** 2 / denom[valid]
        )

        # Influence of removing each gene
        influences = frac_systematic - frac_without_g

    row = {
        "condition": condition,
        "context": context,
        "perturbation": perturbation,
        "n_stim_cells": n_stim,
        "n_perts_in_context": n_perts,
        "cos_sim": cos_sim,
        "norm_delta_p": norm_delta_p,
        "norm_delta_avg": norm_delta_avg,
        "norm_delta_specific": norm_delta_specific,
        "frac_systematic_var": frac_systematic,
    }

    return row, influences, valid


class SystematicStep(PipelineStep):

    def __init__(self, args: argparse.Namespace, regression: bool = False) -> None:
        """Initiailizer."""
        if regression:
            self.__datasets = list(CELLULAR_DATASETS.keys())
        else:
            self.__datasets = ['tahoe100m']

        self.__regression = regression
        self.__reliability_dir = args.reliability_dir
        self.__systema_dir = args.systema_dir
        self.__systema_cp10k_dir = args.systema_cp10k_dir

        self.__output_dir = args.output_dir
        self.__clusters = set([int(x.strip()) for x in args.clusters.split(',')]) if args.clusters else None

    def __summarize_gene_influence_scores(
        cell_line_id,
        all_influence_rows,
        gene_ids,
        delta_p,
        influences,
        valid,
    ) -> None:
        """Summmarize gene influence scores."""
        valid_idx = np.flatnonzero(valid & np.isfinite(influences))

        for i in valid_idx:
            gene_id = gene_ids[i].item()
            influence = float(influences[i])
            delta_p_magnitude = float(delta_p[i])

            if gene_id not in all_influence_rows:
                all_influence_rows[gene_id] = {
                    'negative_influence': 0.0,
                    'positive_influence': 0.0,
                    'total_influence': 0.0,
                    'total_influence_abs': 0.0,
                    'num_instances': 0,
                    'cell_lines': set(),
                    'categorical_change_towards_shared': 0,
                    'categorical_change_towards_specific': 0,
                    'categorical_change_neutral': 0,
                    'delta_p_magnitude_total': 0.0,
                    'delta_p_magnitude_total_abs': 0.0,
                    'delta_p_magnitude_total_pos': 0.0,
                    'delta_p_magnitude_total_neg': 0.0,
                    'num_upregulated': 0,
                    'num_downregulated': 0,
                }

            s = all_influence_rows[gene_id]
            s['positive_influence'] += max(influence, 0)
            s['negative_influence'] += min(influence, 0)
            s['total_influence'] += influence
            s['total_influence_abs'] += abs(influence)
            s['num_instances'] += 1
            s['cell_lines'].add(cell_line_id)

            s['delta_p_magnitude_total'] += delta_p_magnitude
            s['delta_p_magnitude_total_abs'] += abs(delta_p_magnitude)

            if delta_p_magnitude >= 0:
                s['delta_p_magnitude_total_pos'] += delta_p_magnitude
                s['num_upregulated'] += 1
            else:
                s['delta_p_magnitude_total_neg'] += delta_p_magnitude
                s['num_downregulated'] += 1

            frac_without_g = row['frac_systematic_var'] - influence
            if row['frac_systematic_var'] < SPECIFICITY_THRESH and frac_without_g >= SPECIFICITY_THRESH:
                s['categorical_change_towards_shared'] += 1
            elif row['frac_systematic_var'] >= SPECIFICITY_THRESH and frac_without_g < SPECIFICITY_THRESH:
                s['categorical_change_towards_specific'] += 1
            else:
                s['categorical_change_neutral'] += 1

    def compute_systema_metrics(self, ds_name: str, systema_dir: Path) -> pd.DataFrame:
        """
        Compute Systema's systematic variation metric for all (context, perturbation)
        conditions in a cellular context dataset.

        Key properties of this implementation:
          • δ_avg is computed PER CONTEXT (not global)
          • δ_avg uses cells from RELIABLE perts in that context only
          • cos_sim is computed only for reliable perts; unreliable perts get NaN.

        Returns
        -------
        df : DataFrame with columns:
            dataset, condition, context, perturbation, n_stim_cells,
            cos_sim, norm_delta_p, norm_delta_avg, norm_delta_specific,
            frac_systematic_var
        """
        t_start = time.time()
        log.info(f"Computing systema metrics from {systema_dir}")

        # ---- Load reliability (used to filter which perts define δ_avg) ----
        log.info("  [0/3] Loading reliability data for stim data...")
        reliable_conds, all_conds = load_reliable_perts_cellular(
            ds_name, self.__reliability_dir, self.__clusters
        )

        # ---- Load precomputed delta values for each (reliable) cell line ----
        log.info("  [1/3] Loading precomputed delta_p values for reliable data...")
        systema_data = load_systema_deltas(
            ds_name,
            systema_dir,
            all_conds,
            self.__clusters
        )

        # ---- Compute Systema per cell line ----
        log.info("  [2/3] Computing cosine similarities per cell line...")

        all_rows = []
        all_influence_rows = []
        total_conditions = 0
        total_skipped = 0

        for cell_line_id, data in systema_data.items():
            conditions = data["conditions"]
            n_stim_cells = data["n_stim_cells"]
            delta_p_all = data["delta_p"]

            n_perts = len(conditions)

            if n_perts == 0:
                log.info(
                    f"        Cell line '{cell_line_id}': "
                    f"no perturbations, skipping"
                )
                continue

            # Group conditions by context.
            if ds_name == 'tahoe100m':
                contexts = np.asarray([
                    condition.split("|")[0]
                    for condition in conditions
                ])
            else:
                contexts = np.asarray([
                    condition.split("|")[0]
                    for condition in conditions
                ])

            for context_group in np.unique(contexts):
                context_mask = contexts == context_group

                context_conditions = conditions[context_mask]
                context_weights = n_stim_cells[context_mask]
                context_deltas = delta_p_all[context_mask]

                # ---- Step 1: identify reliable perturbations ----
                reliable_mask = np.array(
                    [condition in reliable_conds for condition in context_conditions],
                    dtype=bool,
                )
                n_reliable = reliable_mask.sum()

                # ---- Step 2: δ_avg from reliable perturbations ----
                if n_reliable == 0:
                    delta_avg = np.zeros(
                        delta_p_all.shape[1],
                        dtype=np.float64,
                    )
                    norm_delta_avg = 0.0
                else:
                    reliable_deltas = context_deltas[reliable_mask]
                    reliable_weights = context_weights[reliable_mask]

                    # Cell-weighted average perturbation effect.
                    delta_avg = np.average(
                        reliable_deltas.astype(np.float64),
                        axis=0,
                        weights=reliable_weights,
                    )

                    norm_delta_avg = float(np.linalg.norm(delta_avg))

                # ---- Step 3: score each perturbation ----
                for condition, n_stim, delta_p, is_rel in zip(
                    context_conditions,
                    context_weights,
                    context_deltas,
                    reliable_mask,
                ):

                    # General score
                    row, influences, valid = score_perturbation(
                        ds_name,
                        condition,
                        n_perts,
                        n_stim,
                        delta_p,
                        delta_avg,
                        norm_delta_avg,
                        is_rel,
                        gene_ids,
                    )
                    if row is None:
                        total_skipped += 1
                        continue

                    total_conditions += 1
                    all_rows.append(row)

                    if influences is not None and valid is not None:
                        self.__summarize_gene_influence_scores(
                            cell_line_id,
                            all_influence_rows,
                            gene_ids,
                            delta_p,
                            influences,
                            valid
                        )

        influence_summary = []
        for gene_id, influence in all_influence_rows.items():
            num_instances = influence['num_instances']

            influence_summary.append({
                'token_id': gene_id,
                'avg_influence': influence['total_influence'] / influence['num_instances'],
                'negative_influence': influence['negative_influence'],
                'positive_influence': influence['positive_influence'],
                'total_influence': influence['total_influence'],
                'total_influence_abs': influence['total_influence_abs'],
                'num_instance': num_instances,
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
            })
        del all_influence_rows

        influence_df = pd.DataFrame(influence_summary)
        # map to gene_metadata
        influence_df = influence_df.merge(
            get_gene_metadata(to_pandas=True)[['token_id', 'gene_symbol', 'ensembl_id']],
            on='token_id',
            how='left',
            validate='many_to_one',
        )

        # Keep the top 1000 genes by mean absolute influence
        influence_df = (
            influence_df.assign(
                mean_abs_influence=influence_df['total_influence_abs'] / influence_df['num_instance']
            )
            .nlargest(1000, 'mean_abs_influence')
        )

        gene_influences_dir = self.__output_dir / 'gene_influences'
        gene_influences_dir.mkdir(parents=True, exist_ok=True)

        influence_df.to_csv(
            gene_influences_dir / 'summary_top1000_influential_genes.csv',
            index=False
        )
        del influence_summary

        # Build DataFrames
        df = pd.DataFrame(all_rows)
        del all_rows

        if len(df) == 0:
            raise ValueError(f"  ⚠ No results for Tahoe-100m")

        # ---- Summary statistics ----
        elapsed_total = time.time() - t_start
        log.info(f"\n  ── Summary for Tahoe-100m ──")
        log.info(f"  Conditions computed: {total_conditions}")
        log.info(f"  Skipped (few cells): {total_skipped}")
        log.info(f"  cos_sim — mean: {df.cos_sim.mean():.3f} ± {df.cos_sim.std():.3f}, "
              f"median: {df.cos_sim.median():.3f}")
        log.info(f"  cos_sim — min: {df.cos_sim.min():.3f}, "
              f"max: {df.cos_sim.max():.3f}")
        log.info(f"  frac with cos_sim > 0.5: "
              f"{(df.cos_sim > 0.5).sum()}/{len(df)} "
              f"({100*(df.cos_sim > 0.5).mean():.1f}%)")
        log.info(f"  frac with cos_sim > 0.7: "
              f"{(df.cos_sim > 0.7).sum()}/{len(df)} "
              f"({100*(df.cos_sim > 0.7).mean():.1f}%)")

        # Per-context breakdown
        log.info(f"\n  Per-context cos_sim (median):")
        for ctx, gdf in df.groupby("context"):
            log.info(f"    {ctx:20s}: {gdf.cos_sim.median():.3f} "
                  f"(n={len(gdf)}, n_perts_in_ctx={gdf.n_perts_in_context.iloc[0]})")

        log.info(f"  Total time: {elapsed_total:.1f}s")
        return df

    def merge_with_reliability(
        ds_name: str,
        systema_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Systema cos_sim with pre-computed split-half reliability
        from the cellular context reliability script.

        Merges on the 'condition' column (format: "context|perturbation").
        """
        rel_df = get_combined_split_half_reliability_df(
            self.__reliability_dir, self.__clusters, summary=True, ds_name=ds_name
        )
        log.info(f"  Loaded reliability: {len(rel_df)} conditions")

        # Merge on condition (= "context|perturbation")
        merged = systema_df.merge(
            rel_df,
            on="condition",
            how="inner",
            suffixes=("", "_rel"),
        )

        log.info(f"  Merged: {len(merged)} conditions "
              f"(Systema: {len(systema_df)}, Reliability: {len(rel_df)})")

        if len(merged) < len(systema_df):
            missing = set(systema_df["condition"]) - set(merged["condition"])
            if len(missing) <= 10:
                log.info(f"  Unmatched conditions: {sorted(missing)}")
            else:
                log.info(f"  Unmatched conditions: {len(missing)} "
                      f"(first 5: {sorted(missing)[:5]})")

        # ---- 2D classification ----
        # Use sb_from_median_all_genes as the reliability metric
        rel_col = "sb_from_median_all_genes"
        if rel_col not in merged.columns:
            rel_col = "sb_from_mean_all_genes"

        if rel_col in merged.columns:
            rho = merged[rel_col]
            cos = merged["cos_sim"]

            merged["reliable"] = rho >= RELIABILITY_THRESH
            merged['high_systematic'] = merged['frac_systematic_var'] >= SPECIFICITY_THRESH

            # 2D triage categories
            conditions = [
                (~merged["reliable"]),
                (merged["reliable"]) & (merged["high_systematic"]),
                (merged["reliable"]) & (~merged["high_systematic"]),
            ]
            choices = ["Unreliable", "Shared", "Specific"]
            merged["triage_2d"] = np.select(conditions, choices, default="other")

            # log.info 2D triage summary
            log.info(f"\n  ── 2D Triage Summary ──")
            log.info(f"  Reliability metric: {rel_col}")
            for cat in ["Unreliable", "Shared", "Specific"]:
                n = (merged["triage_2d"] == cat).sum()
                pct = 100 * n / len(merged) if len(merged) > 0 else 0
                log.info(f"    {cat:20s}: {n:>5} ({pct:5.1f}%)")

            # Ceiling on genuine signal
            ceiling_col = "ceiling_from_median_all_genes"
            if ceiling_col not in merged.columns:
                ceiling_col = "ceiling_from_mean_all_genes"
            if ceiling_col in merged.columns:
                genuine = merged[merged["triage_2d"] == "Specific"]
                if len(genuine) > 0:
                    log.info(f"    Genuine signal: mean ceiling = "
                          f"{genuine[ceiling_col].mean():.3f}, "
                          f"mean cos_sim = {genuine['cos_sim'].mean():.3f}")
        else:
            log.info(f"  ⚠ Reliability column '{rel_col}' not found in reliability data")

        return merged

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for the systematic step to the given parser."""
        parser.add_argument(
            '-r', '--reliability-dir', dest='reliability_dir', type=Path, required=True,
            help="Path to the reliability results to serve as input data. Should "
                 + "contain both reliability_summary_tahoe100m_cluster_*.csv files "
                 + "and a systema/ subdirectory containing delta_p values for each "
                 + "cell line across all clusters.",
        )
        parser.add_argument(
            '--systema-dir', dest='systema_dir', type=Path,
            help='Path to systema delta_p scores (not normalized)'
        )
        parser.add_argument(
            '--systema-cp10k-dir', dest='systema_cp10k_dir', type=Path,
            help='Path to systema cp10k delta_p scores (normalized)'
        )

        parser.add_argument(
            '-o', '--output-dir', dest='output_dir', required=True, type=Path,
            help='Output directory to write results to'
        )

        parser.add_argument(
            '--clusters', dest='clusters', required=False, type=str,
            help='Comma-deliminated list of specific cluster(s) to compute the '
                 + 'Systema systematic variation for; if not provided, defaults to '
                 + 'whatever data is available.'
        )

    def run(self) -> None:
        """Run the full pipeline."""
        log.info(f"Systema Systematic Variation + Reliability 2D (Cellular Context)")
        log.info(f"  Clusters:\t{', '.join(self.__clusters) if self.__clusters else 'all'}")
        log.info(f"  Reliability dir:\t{self.__reliability_dir}")
        log.info(f"  Output:\t{self.__output_dir}")

        self.__output_dir.mkdir(parents=True, exist_ok=True)
        cur_time = time.strftime('%Y%m%d-%H%M%S')
        all_summary_rows = {}

        mode_mapping = {
            'systema': self.__systema_dir,
            'systema_cp10k': self.__systema_cp10k_dir
        }

        try:
            for mode, systema_dir in mode_mapping.items():
                if not systema_dir or not systema_dir.is_dir():
                    log.warning(f"{mode} dir not found, skipping mode")
                    continue

                log.info(f"Evaluating {mode}")
                all_summary_rows[mode] = []

                for ds_name in self.__datasets:
                    systema_df = self.compute_systema_metrics(ds_name, systema_dir)

                    # Save Systema-only results
                    systema_path = self.__output_dir / f"{mode}_{ds_name}_{cur_time}.csv"
                    systema_df.to_csv(systema_path, index=False)
                    log.info(f"\n  Saved: {systema_path}")

                    # ---- Merge with reliability ----
                    merged_df = self.merge_with_reliability(ds_name, systema_df)

                    merged_path = self.__output_dir / f"{mode}_cc_reliability_2d_{ds_name}_{cur_time}.csv"
                    merged_df.to_csv(merged_path, index=False)
                    log.info(f"  Saved: {merged_path}")

                    # ---- Cross-dataset summary row ----
                    summary_row = {
                        "n_conditions": len(systema_df),
                        "n_contexts": systema_df["context"].nunique(),
                        "n_perturbations": systema_df["perturbation"].nunique(),
                        "cos_sim_mean": systema_df.cos_sim.mean(),
                        "cos_sim_std": systema_df.cos_sim.std(),
                        "cos_sim_median": systema_df.cos_sim.median(),
                        "frac_cos_gt_0.5": (systema_df.cos_sim > 0.5).mean(),
                        "frac_cos_gt_0.7": (systema_df.cos_sim > 0.7).mean(),
                    }

                    # Add 2D triage counts if available
                    if "triage_2d" in merged_df.columns:
                        for cat in ["Unreliable", "Shared", "Specific"]:
                            n = (merged_df["triage_2d"] == cat).sum()
                            summary_row[f"n_{cat}"] = n
                            summary_row[f"frac_{cat}"] = (
                                n / len(merged_df) if len(merged_df) > 0 else np.nan
                            )

                    all_summary_rows[mode].append(summary_row)

        except Exception as e:
            log.info(f"\n  ✗ FAILED: {e}")
            raise e

        # ---- Cross-dataset summary ----
        for mode, summary_rows in all_summary_rows.items():
            summary_df = pd.DataFrame(summary_rows)

            if self.__regression:
                summary_path = args.output_dir / f"{mode}_cc_summary_regression_{cur_time}.csv"
            else:
                summary_path = args.output_dir / f"{mode}_cc_summary_tahoe100m_{cur_time}.csv"

            summary_df.to_csv(summary_path, index=False)

            log.info(f"\n{'='*70}")
            log.info(f"  Cross-Dataset Summary (Cellular Context), mode {mode}")
            log.info(f"{'='*70}")
            log.info(summary_df.to_string(index=False))
            log.info(f"\n  Saved: {summary_path}")

            # Grand totals
            if "n_Unreliable" in summary_df.columns:
                total = summary_df["n_conditions"].sum()
                total_unmeas = summary_df.get("n_Unreliable", pd.Series([0])).sum()
                total_falsely = summary_df.get("n_Shared", pd.Series([0])).sum()
                total_genuine = summary_df.get("n_Specific", pd.Series([0])).sum()
                total_2d = total_unmeas + total_falsely + total_genuine

                if total_2d > 0:
                    log.info(f"\n  ── Grand Totals (across all datasets) ──")
                    log.info(f"  Total conditions (Systema):  {int(total)}")
                    log.info(f"  Total conditions (merged):   {int(total_2d)}")
                    log.info(f"    Unmeasurable:    {int(total_unmeas):>5} "
                          f"({100*total_unmeas/total_2d:.1f}%)")
                    log.info(f"    Falsely solved:  {int(total_falsely):>5} "
                          f"({100*total_falsely/total_2d:.1f}%)")
                    log.info(f"    Genuine signal:  {int(total_genuine):>5} "
                          f"({100*total_genuine/total_2d:.1f}%)")

                    log.info(f"\n  ★ Effective benchmark size: {int(total_genuine)} conditions "
                          f"({100*total_genuine/total_2d:.1f}% of evaluated)")
