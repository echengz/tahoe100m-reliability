"""
Runs step 1 of the pipeline: reliability.
"""
import argparse
import csv
import duckdb
import json
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import tempfile
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from multiprocessing import shared_memory
from pathlib import Path
from tqdm import tqdm
from typing import Any

from tahoe100m_reliability.lib.analysis.compute_reliability import (
    compute_split_half_reliability,
    cp10k_normalization,
    log_memory_usage,
)

from tahoe100m_reliability.lib.ingest.hg_duckdb_query import batch_duckdb_query
from tahoe100m_reliability.lib.ingest.treatment_set import TreatmentSet
from tahoe100m_reliability.lib.step import BaseReliabilityStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

####################
# GLOBAL CTRL DATA #
####################

_WORKER_CTRL = None
_WORKER_CTRL_GENE_TO_COL = None
_WORKER_CTRL_EXPR_IDX = None
_WORKER_CTRL_MEAN = None  # could be normalized
_WORKER_SHM = None

def init_worker_with_ctrl(
    shm_name,
    shape,
    dtype,
    gene_to_col,
    expr_idx,
    mean_expr,
):
    global _WORKER_CTRL
    global _WORKER_CTRL_GENE_TO_COL
    global _WORKER_CTRL_EXPR_IDX
    global _WORKER_SHM
    global _WORKER_CTRL_MEAN

    _WORKER_SHM = shared_memory.SharedMemory(name=shm_name)

    _WORKER_CTRL = np.ndarray(
        shape,
        dtype=np.dtype(dtype),
        buffer=_WORKER_SHM.buf,
    )

    _WORKER_CTRL_GENE_TO_COL = gene_to_col
    _WORKER_CTRL_EXPR_IDX = expr_idx
    _WORKER_CTRL_MEAN = mean_expr


def load_ctrl_to_shared_memory(
    cell_line_id: str,
    ctrl_gene_expressions_dir: Path,
    expr_k: int,
    remove_genes: set[str] | None,
    normalize_cp10k: bool = False,
    workers: int = 1
):
    """Load control gene expression into shared memory"""
    # Read control data
    ctrl_file = ctrl_gene_expressions_dir / f'{cell_line_id}_gene_expressions.parquet'
    pf = pq.ParquetFile(ctrl_file)

    # load schema to determine dimensions
    schema = pf.schema_arrow
    n_ctrls = pf.metadata.num_rows

    if n_ctrls < 16:
        log.info(
            f"Cell line '{cell_line_id}' only has {n_ctrls} controls, skipping"
        )
        return None

    if remove_genes:
        remove_genes.update({"1", "__index_level_0__"})
    else:
        remove_genes = {"1", "__index_level_0__"}

    # "gene 1" is a start of sequence token, drop, also drop cell ID name
    # and any other genes that were specified to be removed
    columns = [c for c in schema.names if str(c) not in remove_genes]

    # Set dimensions
    n_genes = len(columns)
    dtype = np.dtype(np.float32)
    shape = (n_ctrls, n_genes)

    # Allocate shared memory
    shm = shared_memory.SharedMemory(
        create=True,
        size=int(np.prod(shape)) * dtype.itemsize,
    )

    # Create NumPy view directly over shared memory
    ctrl_dense = np.ndarray(
        shape,
        dtype=dtype,
        buffer=shm.buf,
    )

    # Read parquet in batches and load dense matrix into shared memory
    offset = 0
    for batch in pf.iter_batches(batch_size=10_000, columns=columns):
        n = batch.num_rows

        for j, col in enumerate(columns):
            arr = batch.column(j).to_numpy(zero_copy_only=False)
            arr = np.nan_to_num(arr, nan=0)
            ctrl_dense[offset:offset+n, j] = arr

        offset += n

    assert offset == n_ctrls

    gene_to_col = {
        int(gene): i
        for i, gene in enumerate(columns)
    }

    if normalize_cp10k:
        ctrl_dense = cp10k_normalization(ctrl_dense)

    # Also pre-compute the mean and expr_idx for this control
    # mean will be needed later for calculations requiring delta_p values
    mean_expr = ctrl_dense.mean(axis=0)

    # if specified, subset to top expressed genes
    expr_k = min(expr_k, n_genes) if expr_k >= 0 else n_genes
    if expr_k < n_genes:
        expr_idx = np.argpartition(mean_expr, -expr_k)[-expr_k:]
    else:
        expr_idx = np.arange(n_genes)

    log.info(f"[{cell_line_id}]: {n_ctrls} controls with {len(gene_to_col)} genes")

    return {
        'shm': shm,
        'shape': shape,
        'dtype': dtype,
        'gene_to_col': gene_to_col,
        'expr_idx': expr_idx,
        'mean_expr': mean_expr,
    }


def build_expression_matrix(
    condition: str,
    cell_line_dirs: list[Path],
    fetch_size: int = 10_000
):
    """
    Combine all sparse Parquet files across cell_line_dirs into
    one dense expression matrix.

    Each row of the resulting matrix corresponds to one cell.
    Columns correspond to _WORKER_GENE_TO_COL (target genes).
    """
    n_genes = len(_WORKER_CTRL_GENE_TO_COL)
    parquet_files = []

    for cell_line_dir in cell_line_dirs:
        if not cell_line_dir.is_dir():
            continue

        parquet_files.extend(cell_line_dir.glob("part_*.parquet"))

    # should not happen
    if not parquet_files:
        log.warning(f"Failed to find stimulated cell data parquets for {condition}")
        return None, None

    parquet_files = [str(p) for p in parquet_files]

    # first pass to get number of cells in this condition
    con = duckdb.connect()

    try:
        n_cells = con.execute("""
            SELECT COUNT(*)
            FROM read_parquet(?)
            """,
            [parquet_files],
        ).fetchone()[0]

        # allocate matrix
        matrix = np.zeros((n_cells, n_genes), dtype=np.float32)

        cursor = con.execute(
            """
            SELECT
                genes,
                expressions
            FROM read_parquet(?)
            """,
            [parquet_files]
        )

        # fill in matrix; since input data is not harmonized need to map to the
        # appropriate cell in empty matrix
        row_idx = 0
        dropped_genes = {}

        while rows := cursor.fetchmany(fetch_size):
            for genes, expressions in rows:
                # should align
                assert len(genes) == len(expressions), "Gene and expression lists do not align"

                for gene, expression in zip(genes, expressions):
                    col = _WORKER_CTRL_GENE_TO_COL.get(gene)
                    if col is None:
                        # means gene was not in any controls, so just drop for now
                        # "gene 1" is just a start of sequence token, so ignore:
                        if gene != 1:
                            if gene not in dropped_genes:
                                dropped_genes[gene] = 0

                            dropped_genes[gene] += 1
                        continue

                    matrix[row_idx, col] = expression

                row_idx += 1

        assert row_idx == n_cells
        del rows

    finally:
        con.close()
    return matrix, dropped_genes


def compute_reliability(
    job_num: int,
    condition: str,
    cell_line_dirs: list[Path],
    task_seed: int,
    n_half_grid: list[int],
    n_repeats: int,
    delete_stim_data: bool = True,
    normalize_cp10k: bool = False
):
    """
    Aggregate gene expression matrices for each drug dosage x cell line condition
    and compute reliability against its corresponding control
    """
    stim_dense, dropped_genes = build_expression_matrix(
        condition=condition,
        cell_line_dirs=cell_line_dirs,
    )

    # Remove stim cell line data once done loading it
    if delete_stim_data:
        for cell_line_dir in cell_line_dirs:
            if cell_line_dir.is_dir():
                shutil.rmtree(cell_line_dir)

    if stim_dense is None:
        return None, {}

    n_stim = len(stim_dense)

    # technically not possible but sanity check
    if n_stim == 0:
        log.info(f"[{condition}]: No stimulated cells, skipping")
        del stim_dense
        return None, {}

    # Max n_half: min of half-stim and half-ctrl
    n_half_max = min(n_stim // 2, _WORKER_CTRL.shape[0] // 2)
    if n_half_max < 4:
        log.info(
            f"[{condition}]: Not enough data to evaluate ({n_stim} cells), skipping"
        )
        del stim_dense
        return None, {}

    # fixed N values that are feasible
    n_half_grid_feasible = [
        n for n in n_half_grid if n <= n_half_max
    ]

    compute_results = {
        'n_stim': n_stim
    }

    if normalize_cp10k:
        stim_mean = cp10k_normalization(stim_dense)

    # compute split-half reliability
    compute_results['split_half'] = compute_split_half_reliability(
        condition,
        stim_dense,
        _WORKER_CTRL,
        _WORKER_CTRL_EXPR_IDX,
        n_stim,
        n_half_max,
        n_half_grid_feasible,
        n_repeats,
        task_seed,
    )

    # compute delta_p
    stim_mean = stim_dense.mean(axis=0)
    compute_results['delta_p'] = stim_mean - _WORKER_CTRL_MEAN

    del stim_dense, stim_mean
    return compute_results, dropped_genes


class ReliabilityStep(BaseReliabilityStep):

    def __init__(
        self,
        indexed_treatments_file: Path,
        ctrl_gene_expressions_dir: Path,
        clustered_treatments_csv: Path,
        normalize_cp10k: bool = False,
        clusters: str | None = None,
        target_cell_lines_csv: Path | None = None,
        stim_data_dir: Path | None = None,
        remove_genes: set[str] | None = None,
        cell_line_cooldown: int = 30,
        skip_localization: bool = False,
        **kwargs,
    ) -> None:
        """Initialize reliability step."""
        super().__init__(**kwargs)
        self.indexed_treatments_file = indexed_treatments_file
        self.ctrl_gene_expressions_dir = ctrl_gene_expressions_dir
        self.clustered_treatments_csv = clustered_treatments_csv
        self.normalize_cp10k = normalize_cp10k
        self.clusters = [int(x.strip()) for x in clusters.split(',')] if clusters else None
        self.target_cell_lines_csv = target_cell_lines_csv
        self.stim_data_dir = stim_data_dir
        self.remove_genes = remove_genes
        self.cell_line_cooldown = cell_line_cooldown
        self.skip_localization = skip_localization

        if self.normalize_cp10k:
            self.delta_p_dir = self.output_dir / 'delta_p_cp10k'
        else:
            self.delta_p_dir = self.output_dir / 'delta_p_raw'

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "ReliabilityStep":
        """Create ReliabilityStep from arguments."""
        return cls(**cls.clean_args(args))

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for the reliability step to the given parser."""
        parser.add_argument(
            '-i', '--indexed-treatments-file', dest='indexed_treatments_file',
            required=True, type=Path,
            help='Path to the indexed treatments file, which indexes each sample '
                + 'in Tahoe100m to the parquet files that contain cellular data from '
                + 'that sample.'
        )
        parser.add_argument(
            '-c', '--ctrl-gene-expressions-dir', dest='ctrl_gene_expressions_dir',
            required=True, type=Path,
            help='Path to the directory containing the control gene matrix for all 50 '
                + 'cell lines, each an individual parquet file.'
        )
        parser.add_argument(
            '--clustered-treatments-csv', dest='clustered_treatments_csv', required=True, type=Path,
            help='Path to a CSV file listing all the drug x dosage treatments with their cluster index.'
        )
        parser.add_argument(
            '--normalize-cp10k', dest='normalize_cp10k', action='store_true',
            help='Whether or not to run normalize the gene transcription counts with CP10K'
        )

        parser.add_argument(
            '--clusters', dest='clusters', required=False, type=str,
            help='Comma-deliminated list of specific cluster(s) to compute Δ-reliability index for. '
                + 'Defaults to all if not provided'
        )

        parser.add_argument(
            '--target-cell-lines-csv', dest='target_cell_lines_csv', required=False, type=Path,
            help='Path to a CSV file listing all cell lines to compute the split half  '
                + 'Δ-reliability for. If not provdided, will iterate over all cell lines '
                + 'for a give drug x dosage combination'
        )

        parser.add_argument(
            "--stim-data-dir", dest="stim_data_dir", type=Path, required=False,
            help="If provided, will save downloaded stim data to this directory rather than a temporary one",
        )

        parser.add_argument(
            '--remove-genes', dest='remove_genes', type=str,
            help='Comma-deliminated list of genes (token_id) to remove from reliability calculations'
        )

        parser.add_argument(
            '--cell-line-cooldown', dest='cell_line_cooldown', type= int, required=False, default=30,
            help='Time in seconds to sleep between processing each cell line. Set to 0 to disable.'
        )
        parser.add_argument(
            '--skip-localization', dest='skip_localization', action='store_true',
            help='Skip localization if data already lives under a passed stim-data-dir'
        )
        super().add_cli_args(parser)


    def __aggregate_stim_data(
        self, treatments: list[TreatmentSet], target_cell_lines: set[str], stim_dir: Path
    ):
        """Localize and aggregate stim data from the HuggingFace repository."""
        # get all samples for treatments; should be localized to specific parquet files
        # we know a sample maps to a specific drug/dosage so don't need to query that info
        all_samples = set({})
        parquet_files = set({})

        for treatment in treatments:
            all_samples.update(set(treatment.samples))
            parquet_files.update(set(treatment.parquet_files))

        samples = ', '.join(f"'{s}'" for s in all_samples)
        cell_line_ids = ', '.join(f"'{c}'" for c in target_cell_lines)

        # build query; could include BARCODE_SUB_LIB_ID but for now trying to restrict info
        query = f"""
            SELECT sample, cell_line_id, genes, expressions
            FROM read_parquet(?, filename=true)
            WHERE sample IN ({samples}) AND cell_line_id IN ({cell_line_ids})
        """

        rows_per_chunk = 150_000
        log.info(f"Writing stim data to {stim_dir}")

        def _aggregate_contexts(rows, batch_id):
            """Aggregate all cells and their gene expression for a given sample."""
            log.info(f"Batch ID {batch_id}: Processing {len(rows)} rows in batches of {rows_per_chunk}")
            log_memory_usage(f"Batch ID {batch_id}: Aggregating contexts")

            # part counters for each
            for index in range(0, len(rows), rows_per_chunk):
                chunk = rows[index:index + rows_per_chunk]                
                new_cells = defaultdict(list)

                # organize data by condition (sample/cell line)
                for sample, cell_line_id, genes, expressions in chunk:
                    new_cells[(sample, cell_line_id)].append(
                        (genes, expressions)
                    )

                del chunk

                # generate parquet parts for each sample x cell line
                for (sample, cell_line_id), cells in new_cells.items():
                    cell_path = stim_dir / sample / cell_line_id
                    cell_path.mkdir(parents=True, exist_ok=True)

                    genes = [genes for genes, _ in cells]
                    expressions = [expressions for _, expressions in cells]

                    table = pa.table({
                        'genes': genes,
                        'expressions': expressions
                    })

                    pq.write_table(table, cell_path / f'part_{batch_id:08d}_{index:08d}.parquet')

                del table, genes, expressions, new_cells

            del rows
            log.info(f"Batch ID {batch_id}: Successfully processed rows")

        batch_duckdb_query(
            process_batch=_aggregate_contexts,
            parquet_files=list(parquet_files),
            query=query,
            batch_size=5,
            max_workers=4
        )

    def __set_up_cell_line_jobs(
        self, treatments: list[TreatmentSet], target_cell_lines: set[str], stim_dir: Path,
    ) -> tuple[dict[str, Any], int]:
        """Set up the jobs for the given cell lines."""

        # Group by cell line so only have to load control once
        # Set up jobs; batch by cell_line_id as they all have the same controls, so we
        # can load and reuse the control once for this given cluster
        cell_line_jobs = {}
        num_jobs_total = 0

        for cell_line_id in target_cell_lines:
            jobs = []
            for treatment in treatments:
                # aggregate gene expression matrices for each cell line x drug dosage context
                # and compute reliability against its corresponding control
                condition = f'{cell_line_id}|{treatment.drug}|{treatment.dosage}'

                # get input locations; there can be multiple as there can be multiple samples
                # for this condition
                cell_line_dirs = [
                    stim_dir / sample / cell_line_id
                    for sample in treatment.samples
                ]

                jobs.append((
                    condition,
                    cell_line_dirs,
                    self.seed + num_jobs_total
                ))
                num_jobs_total += 1

            cell_line_jobs[cell_line_id] = jobs

        num_cell_lines = len(cell_line_jobs)
        log.info(f"{num_jobs_total} jobs for {num_cell_lines} cell lines queued; will use {self.workers} workers")
        log_memory_usage()

        return cell_line_jobs, num_jobs_total

    def process_treatment_cluster(
        self,
        cluster: str,
        treatments: list[TreatmentSet],
        target_cell_lines: set[str],
        remove_genes: set[int] | None = None,
    ) -> None:
        """Process and compute reliability results for the given treatment cluster."""
        log.info(f"Processing cluster {cluster} with {len(treatments)} treatments")
        t_start = time.time()

        all_compute_results = {'split_half': []}
        all_dropped_genes = {}
        skipped_conditions = {}

        # temporary directory for this cluster of treatments
        with tempfile.TemporaryDirectory() as tmp_dir:
            stim_dir = Path(tmp_dir) if self.stim_data_dir is None else self.stim_data_dir  # can handle this better

            if not self.skip_localization:
                self.__aggregate_stim_data(treatments, target_cell_lines, stim_dir)

            log.info(f"Computing reliability for each cell line against controls for cluster {cluster}")

            # get cell line jobs
            cell_line_jobs, num_jobs_total = self.__set_up_cell_line_jobs(
                treatments, target_cell_lines, stim_dir
            )
            num_cell_lines = len(cell_line_jobs)
            job_num = 0

            for i, (cell_line_id, jobs) in enumerate(cell_line_jobs.items()):
                # Load control for this cell line
                shared_ctrl = load_ctrl_to_shared_memory(
                    cell_line_id,
                    self.ctrl_gene_expressions_dir,
                    self.expr_k,
                    remove_genes,
                    self.normalize_cp10k,
                    workers=self.workers
                )

                # not enough controls to evaluate this cell line, drop corresponding conditions
                if shared_ctrl is None:
                    for condition, _, _, in jobs:
                        skipped_conditions.add(condition)
                    continue

                delta_p_results = []

                # cool off
                if self.cell_line_cooldown > 0:
                    log.info(f"sleeping {self.cell_line_cooldown} seconds to cool off")
                    time.sleep(self.cell_line_cooldown)

                try:
                    # Execute contexts in parallel
                    with ProcessPoolExecutor(
                        max_workers=self.workers,
                        initializer=init_worker_with_ctrl,
                        initargs=(
                            shared_ctrl['shm'].name,
                            shared_ctrl['shape'],
                            shared_ctrl['dtype'],
                            shared_ctrl['gene_to_col'],
                            shared_ctrl['expr_idx'],
                            shared_ctrl['mean_expr'],
                        )
                    ) as executor:
                        futures = {}
                        for condition, cell_line_dirs, task_seed in jobs:
                            future = executor.submit(
                                compute_reliability,
                                job_num=job_num,
                                condition=condition,
                                cell_line_dirs=cell_line_dirs,
                                task_seed=task_seed,
                                n_half_grid=self.n_half_grid,
                                n_repeats=self.n_repeats,
                                delete_stim_data=self.stim_data_dir is None,
                                normalize_cp10k=self.normalize_cp10k
                            )

                            futures[future] = condition
                            job_num += 1

                        for future in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f'Computing reliability scores for {cell_line_id} ({i + 1}/{num_cell_lines})'
                        ):
                            condition = futures[future]
                            try:
                                compute_results, dropped_genes = future.result()
                                if compute_results is None:
                                    skipped_conditions[condition] = 'not enough data'
                                else:
                                    all_compute_results['split_half'].extend(compute_results['split_half'])

                                    # keep delta_p summary
                                    delta_p_results.append({
                                        'condition': condition,
                                        'n_stim_cells': compute_results['n_stim'],
                                        'delta_p': compute_results['delta_p'].astype(np.float32)
                                    })

                                all_dropped_genes[condition] = dropped_genes
                            except Exception as e:
                                log.error(f"Condition {condition} failed with error: {e}")
                                skipped_conditions[condition] = f'error: {str(e)}'

                    # save delta_p results for downstream analysis
                    # needs to be done across entire cell lines so cannot be done here
                    if delta_p_results:
                        if self.normalize_cp10k:
                            delta_p_filename = f'{cell_line_id}_cp10k_delta_p_cluster{cluster}.npz'
                        else:
                            delta_p_filename = f"{cell_line_id}_delta_p_cluster{cluster}.npz" 

                        self.save_delta_p_results(
                            delta_p_results=delta_p_results,
                            outfile=delta_p_filename,
                            group=cell_line_id,
                        )

                finally:
                    shared_ctrl['shm'].close()
                    shared_ctrl['shm'].unlink()

        # summarize reliability results
        if self.normalize_cp10k:
            dataset_name = f'{dataset_name}_cp10k'
        else:
            dataset_name = f'{dataset_name}_raw'

        self.summarize_split_half_results(
            cluster=cluster,
            all_rows=all_compute_results['split_half'],
            dataset_name=dataset_name,
        )

        # write out dropped genes
        dropped_genes_filepath = self.output_dir / f'dropped_genes_histogram_cluster_{cluster}_{self.cur_time}.json'
        with dropped_genes_filepath.open('w') as fh:
            json.dump(all_dropped_genes, fh, indent=4)

        del all_dropped_genes

        # write out skipped conditions
        if skipped_conditions:
            skipped_conditions_filepath = self.output_dir / f"skipped_conditions_cluster_{cluster}_{self.cur_time}.json"
            with skipped_conditions_filepath.open("w") as fh:
                json.dump({"skipped_conditions": skipped_conditions}, fh, indent=4)

        elapsed_total = time.time() - t_start
        log.info(f"  Total time: {elapsed_total / 60:.1f} min")

        log.info(f"Finished processing cluster {cluster} with {len(treatments)} treatments")
        log_memory_usage()

    def run(self) -> None:
        """Run the full pipeline."""
        remove_genes = set([x.strip() for x in self.remove_genes.split(',')]) \
            if self.remove_genes else None

        # Build target treatment list
        clustered_treatments = TreatmentSet.load_clusters_from_file(
            treatments_file=self.indexed_treatments_file,
            drug_dosages=self.clustered_treatments_csv,
            target_clusters=self.clusters
        )

        target_cell_lines = []
        if self.target_cell_lines_csv:
            with self.target_cell_lines_csv.open('r') as fh:
                reader = csv.DictReader(fh)
                target_cell_lines = set(row['cell_line_id'] for row in reader)
        else:

            # get all possible cell lines from available controls
            for file in self.ctrl_gene_expressions_dir.glob('*.parquet'):
                target_cell_lines.append(file.stem.replace('_gene_expressions', ''))

        log.info(f"Targeting {len(target_cell_lines)} cell lines")

        # Build gene expression matrices for all cell lines in this treatment
        for cluster, treatments in clustered_treatments.items():
            self.process_treatment_cluster(
                cluster=cluster,
                treatments=treatments,
                target_cell_lines=target_cell_lines,
                remove_genes=remove_genes,
            )
