"""
Runs step 0 of the pipeline: preprocessing.
"""
import argparse
import ast
import logging
from pathlib import Path

from tahoe100m_reliability.lib.tahoe100m_dataset import (
    get_cell_line_metadata,
    get_sample_metadata,
    list_tahoe100m_parquet_files,
)
from tahoe100m_reliability.lib.hg_duckdb_query import batch_duckdb_query
from tahoe100m_reliability.lib.treatment_set import TreatmentSet

from .step_helpers.pipeline_step import PipelineStep


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

log.addHandler(handler)


class PreprocessStep(PipelineStep):

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the preprocess step."""
        self.treatment_sets_dir = args.treatment_sets_dir
        self.parquet_shard_start = args.parquet_shard_start

        self.control_gene_expressions_dir  = args.control_gene_expressions_dir

    def get_treatment_sets(self) -> list[TreatmentSet]:
        """Get all known drug x dosage (perturbation) permutations (treatment sets)."""
        log.info("Getting treatment sets...")

        treatment_sets = {}
        sample_metadata = get_sample_metadata()

        for metadata in sample_metadata:
            # drugname_drugconc contains all information; also use as a key
            drugname_drugconc = metadata['drugname_drugconc'].strip()
            sample = metadata['sample']

            if drugname_drugconc not in treatment_sets:
                # Assumes drugname_drugconc is from the sample_metadata and is
                # a list with exactly one tuple of 3 elements. Uses this original
                # string as an unique key for indexing. 
                drug, dosage, dosage_unit = ast.literal_eval(drugname_drugconc)[0]

                treatment_sets[drugname_drugconc] = TreatmentSet(
                    drug=drug.strip(),
                    dosage=dosage,
                    dosage_unit=dosage_unit.strip(),
                )

            treatment_sets[drugname_drugconc].add_sample(sample)

        # TODO - officialy documentation says there are 1,135 + DMSO_TF
        # so investigate why we are getting 1,338 (1,337 + DMSO_TF)
        log.info(f"Found {len(treatment_sets)} unique treatment sets")
        return list(treatment_sets.values())

    def index_treatment_sets(self, treatment_sets: list[TreatmentSet]) -> None:
        """Index the treatment sets, i.e. figure out which parquet
        files it has cell data in. Uses DuckDB for fast SQL processing.
        """
        # list all parquet shards
        parquet_files = list_tahoe100m_parquet_files()
        log.info(f"Total parquet files in Tahoe100m: {len(parquet_files)}")
        log.info(f"Starting at shard: {self.parquet_shard_start}")

        if self.parquet_shard_start > 0:
            parquet_files = parquet_files[self.parquet_shard_start:]

        query = """
            SELECT DISTINCT sample, filename
            FROM read_parquet(?, filename=true)
            WHERE sample IS NOT NULL
        """

        def _save_indexed_treatment_sets(rows, batch_id):
            sample_indexes = defaultdict(set)
            for sample, filename in rows:
                sample_indexes[sample].add(filename)

            # now add parquet information to each treatment set
            for treatment in treatment_sets:
                for sample in treatment.samples:
                    if sample in sample_indexes:
                        treatment.add_parquet_files(sample_indexes[sample])

            # write indexed treatment sets to file - updating as we go
            with (
                self.treatment_sets_dir / 
                f'indexed_treatment_sets_{self.parquet_shard_start:05d}.json'
            ).open('w') as fh:
                json.dump([
                    treatment.model_dump(mode='json')
                    for treatment in treatment_sets
                ], fh, indent=4)

        rows = batch_duckdb_query(
            parquet_files, query, process_batch=_save_indexed_treatment_sets
        )
        log.info("Successfully indexed treatment sets")

    def evaluate_treatment_sets(self, treatment_sets: list[TreatmentSet]):
        """Run some checks on treatment sets."""

        # Ensure each sample maps to an unique drug x dosage (perturbation) combo
        # also count how many samples each unique perturbation has
        samples = {}
        for treatment in treatment_sets:
            for s in treatment.samples:
                if s in samples:
                    assert samples[s]['drug'] == treatment.drug, \
                        f"Sample {s} maps to multiple drugs: {samples[s]['drug']} and {treatment.drug}"
                    assert samples[s]['dosage'] == treatment.dosage, \
                        f"Sample {s} maps to multiple dosages: {samples[s]['dosage']} and {treatment.dosage}"
                else:
                    samples[s] = {
                        'drug': treatment.drug,
                        'dosage': treatment.dosage
                    }

        log.info(f"All {len(samples)} samples are unique")
        log.info(f"Perturbation samples: {len([k for k, v in samples.items() if v['drug'] != 'DMSO_TF'])}")
        log.info(f"Control samples: {len([k for k, v in samples.items() if v['drug'] == 'DMSO_TF'])}")

        # see how many samples per perturbation
        treatment_sets = sorted(
            treatment_sets,
            key=lambda x: len(x.samples),
            reverse=True,
        )

        N = 10
        log.info(f"Perturbations with the most samples: ")
        for treatment in treatment_sets[:N]:
            log.info(f'{treatment.drug}|{treatment.dosage}: {len(treatment.samples)}')

        # list cell line names
        # get all the unique cell lines from the metadata
        cell_line_metadata = get_cell_line_metadata(to_pandas=True)
        log.info(cell_line_metadata.head())

    def combine_indexed_treatment_sets(self) -> list[TreatmentSet]:
        """Combine parts of indexed treatment sets, if needed."""
        treatment_sets = {}
        for file in self.treatment_sets_dir.glob('indexed_treatment_sets_*.json'):
            with open(file, 'r') as fh:
                data = json.load(fh)
                for x in data:
                    key = '_'.join([x['drug'], str(x['dosage']), x['dosage_unit']])
                    if key not in treatment_sets:
                        treatment_sets[key] = TreatmentSet(
                            drug=x['drug'],
                            dosage=x['dosage'],
                            dosage_unit=x['dosage_unit'],
                            samples=x['samples']
                        )
                    treatment_sets[key].add_parquet_files(
                        set(x['parquet_files'])
                    )

            return list(treatment_sets.values())

        # add plate info
        sample_metadata = get_sample_metadata()
        sample_to_plate = {}
        for metadata in sample_metadata:
            sample_to_plate[metadata['sample']] = metadata['plate']

        for treatment in data.values():
            treatment['plates'] = {
                sample: sample_to_plate[sample]
                for sample in treatment['samples']
            }

        with (self.treatment_sets_dir / 'combined_indexed_treatment_sets.json').open('w') as fh:
            json.dump([
                treatment.model_dump(mode='json')
                for treatment in treatment_sets.values()
            ], fh, indent=4)

    def build_control_gene_expressions(self, treatment_sets: list[TreatmentSet]) -> None:
        """Evaluate the control groups and build their expression matrices."""
        log.info("Building control gene expression matrices...")

        # get control group
        control_group = None
        for x in treatment_sets:
            if x.drug == 'DMSO_TF':
                control_group = x
                break

        if not control_group:
            raise ValueError("Could not find control group with drug DMSO_TF")

        log.info("Found control treatment set")

        # get all the unique cell lines from the metadata
        cell_line_metadata = get_cell_line_metadata()

        cell_lines = set({})
        for metadata in cell_line_metadata:
            cell_lines.add(metadata['Cell_ID_Cellosaur'].strip())

        log.info(f"Found {len(cell_lines)} unique cell lines")
        log.info("Building gene expression matrices for each cell line control group")

        def _build_ctrl_gene_expression_matrices(rows, batch_id):

            # identify new cells and group by cell line
            new_cells = defaultdict(list)
            for cell_line_id, barcode_sub_lib_id, genes, expressions in rows:
                cell = pd.Series(expressions, index=genes, name=barcode_sub_lib_id)
                new_cells[cell_line_id].append(cell)

            # add new cells to matrix
            for cell_line_id, cells in new_cells.items():
                filepath = self.control_gene_expressions_dir / f'{cell_line_id}_gene_expressions.parquet'

                matrix = pd.concat(cells, axis=1).T

                # load and concat to an existing matrix if it exists
                if filepath.exists():
                    existing_matrix = pd.read_parquet(filepath)
                    matrix = pd.concat([existing_matrix, matrix], axis=0)

                matrix.to_parquet(filepath)

        # query for control cell lines, and build expression matrices as specified
        # by the function above
        cell_line_batch = ', '.join(f"'{x}'" for x in cell_lines)
        query = f"""
            SELECT cell_line_id, BARCODE_SUB_LIB_ID, genes, expressions
            FROM read_parquet(?, filename=true)
            WHERE drug = 'DMSO_TF' AND cell_line_id IN ({cell_line_batch})
        """

        rows = batch_duckdb_query(
            list(control_group.parquet_files),
            query,
            process_batch=_build_ctrl_gene_expression_matrices,
            batch_size=25,
            start_index=1475
        )

        log.info("Successfully generated all control gene expression matrices")

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for the preprocess step to the given parser."""
        parser.add_argument(
            '--treatment-sets-dir', dest='treatment_sets_dir',
            type=Path, required=True,
            help='Path to the indexed treatment sets directory. Will read and write results here.'

        )

        parser.add_argument(
            '--parquet-shard-start', dest='parquet_shard_start',
            type=int, default=0,
            help='Which parquet shard to start indexing treatments at; for if the process gets interrupted'
        )

        parser.add_argument(
            '--chunk-size', dest='chunk_size',
            type=int, default=10_000,
            help='Chunk size to process dataset with'
        )

        parser.add_argument(
            '--control-gene-expressions-dir', dest='control_gene_expressions_dir',
            type=Path, default=None,
            help='If provided, will build the control gene expression matrices and write to this directory'
        )

    def run(self) -> None:
        """Run the preprocess step."""
        treatment_sets_file = self.treatment_sets_dir / 'combined_indexed_treatment_sets.json'

        if not treatment_sets_file.is_file():
            # generate the treatment sets file
            # 1. Get treatment sets
            treatment_sets = self.get_treatment_sets()

            # 2. Index treatment sets to parquets
            self.index_treatment_sets(
                treatment_sets,
                args.treatment_sets_dir,
                args.parquet_shard_start,
            )

            # 2.1 combine parts
            treatment_sets = self.combine_indexed_treatment_sets()

        else:
            # load from existing file
            treatment_sets = TreatmentSet.load_list_from_file(
                treatments_file=args.treatment_sets_file
            )

        # 3. Evaluate treatment sets to ensure they are valid
        self.evaluate_treatment_sets(treatment_sets)

        # 4. Generate control gene expression matrices
        if self.control_gene_expressions_dir:
            self.build_control_gene_expressions(treatment_sets)
