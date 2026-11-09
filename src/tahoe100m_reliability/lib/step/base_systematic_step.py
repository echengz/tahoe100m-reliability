"""
Base systematic step, to be used by both
Tahoe100M and Regression pipelines.
"""
import argparse

from pathlib import Path

from tahoe100m_reliability.lib.ingest.tahoe100m_dataset import (
    get_drug_metadata,
    get_gene_metadata
)
from .pipeline_step import PipelineStep


class DeltaPMode:
    """Handles what type of delta_p results we have"""

    def __init__(self, reliability_dir: Path | None, delta_p_dir: Path | None) -> None:
        """Initializer. Pair can be missing, but must be provided together otherwise."""
        if (delta_p_dir and not reliability_dir) or (reliability_dir and not delta_p_dir):
            raise FileNotFoundError(
                f"Reliability directory not provided for delta_p results {delta_p_dir}"
            )

        self.reliability_dir = reliability_dir
        self.delta_p_dir = delta_p_dir


class BaseSystematicStep(PipelineStep):
    """
    Base step; for pipeline steps that rely on the reliability
    and systematic delta p values.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """Initializer."""
        if not args.delta_p_raw_dir and not args.delta_p_cp10k_dir:
            raise FileNotFoundError("At least one delta_p dir must be passed")

        self.mode_mapping = {
            'raw': DeltaPMode(args.reliability_raw_dir, args.delta_p_raw_dir),
            'cp10k': DeltaPMode(args.reliability_cp10k_dir, args.delta_p_cp10k_dir)
        }

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
            '--ctrl-gene-expressions-dir', dest='ctrl_gene_expressions_dir',
            required=True, type=Path,
            help='Path to the directory containing the control gene matrix for all 50 '
                + 'cell lines, each an individual parquet file.'
        )

        parser.add_argument(
            '--reliability-raw-dir', dest='reliability_raw_dir', type=Path,
            help="Path to the raw reliability results to serve as input data. Should "
                 + "contain both reliability_summary_tahoe100m_cluster_*.csv files"
        )
        parser.add_argument(
            '--reliability-cp10k-dir', dest='reliability_cp10k_dir', type=Path,
            help="Path to the CP10K reliability results to serve as input data. Should "
                 + "contain both reliability_summary_tahoe100m_cluster_*.csv files"
        )

        parser.add_argument(
            '--delta-p-raw-dir', dest='delta_p_raw_dir', type=Path,
            help='Path to raw delta_p scores'
        )
        parser.add_argument(
            '--delta-p-cp10k-dir', dest='delta_p_cp10k_dir', type=Path,
            help='Path to cp10k delta_p scores (normalized)'
        )

        parser.add_argument(
            '--output-dir', dest='output_dir', required=True, type=Path,
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
        metadata["drug"] = (metadata["drug"].astype(str).str.strip())

        # Remove duplicate drug annotations
        metadata = metadata.drop_duplicates(subset=["drug"])
        return metadata.set_index("drug")["moa-fine"].to_dict()
