"""
Base class to handle a step of the pipeline.
"""
import argparse

from pathlib import Path

from tahoe100m_reliability.lib.tahoe100m_dataset import (
    get_drug_metadata,
    get_gene_metadata
)


class PipelineStep:
    """Base pipeline step."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the step."""
        pass

    def run(self) -> None:
        """Run the step."""
        pass

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this pipeline step to the given parser."""
        pass


class BaseSystematicStep(PipelineStep):
    """
    Base step; for pipeline steps that rely on the reliability
    and systematic delta p values.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """Initializer."""
        if not args.systema_raw_dir and not args.systema_cp10k_dir:
            raise ValueError("At least one systema dir must be passed")

        self.mode_mapping = {
            'raw': args.systema_raw_dir,
            'cp10k': args.systema_cp10k_dir
        }

        self.reliability_dir = args.reliability_dir
        self.ctrl_gene_expressions_dir = args.ctrl_gene_expressions_dir
        self.output_dir = args.output_dir

        self.clusters = set(
            [int(x.strip()) for x in args.clusters.split(',')]
        ) if args.clusters else None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gene_metadata = get_gene_metadata(to_pandas=True)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this base systematic step to the given parser."""
        parser.add_argument(
            '-c', '--ctrl-gene-expressions-dir', dest='ctrl_gene_expressions_dir',
            required=True, type=Path,
            help='Path to the directory containing the control gene matrix for all 50 '
                + 'cell lines, each an individual parquet file.'
        )

        parser.add_argument(
            '-r', '--reliability-dir', dest='reliability_dir', type=Path, required=True,
            help="Path to the reliability results to serve as input data. Should "
                 + "contain both reliability_summary_tahoe100m_cluster_*.csv files"
        )

        parser.add_argument(
            '--systema-raw-dir', dest='systema_raw_dir', type=Path,
            help='Path to systema delta_p scores (raw)'
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

    def load_drug_moa_mapping(self) -> dict[str, str]:
        """Load drug-to-fine-MOA annotations from Tahoe100M."""
        drug_metadata = get_drug_metadata(to_pandas=True)
        metadata = drug_metadata[["drug", "moa-fine"]]
        metadata["drug"] = (
            metadata["drug"].astype(str).str.strip()
        )

        # Remove duplicate drug annotations
        metadata = metadata.drop_duplicates(subset=["drug"])
        return metadata.set_index("drug")["moa-fine"].to_dict()
