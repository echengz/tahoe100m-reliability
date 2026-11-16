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
        """Initializer. If reliability directory exists, so must delta_p directory."""

        if reliability_dir:
            if not reliability_dir.is_dir():
                raise FileNotFoundError(f"Cannot find reliability directory: {reliability_dir}")
            if not delta_p_dir:
                raise FileNotFoundError(f"delta_p_dir not provided when reliability_dir was")
            if not delta_p_dir.is_dir():
                raise FileNotFoundError(f"Cannot find delta_p directory: {delta_p_dir}")

        self.reliability_dir = reliability_dir
        self.delta_p_dir = delta_p_dir


class BaseSystematicStep(PipelineStep):
    """
    Base step; for pipeline steps that rely on the reliability
    and systematic delta p values.
    """
    def __init__(
        self,
        ctrl_gene_expressions_dir: Path,
        output_dir: Path,
        reliability_dir: Path | None = None,
        reliability_cp10k_dir: Path | None = None,
        clusters: str | None = None,
    ) -> None:
        """Initializer."""
        super().__init__()

        if not reliability_dir and not reliability_cp10k_dir:
            raise FileNotFoundError("At least one reliability directory must be passed")

        # we renamed the directories, so try both
        try:
            self.mode_mapping = {
                'raw': DeltaPMode(reliability_dir, reliability_dir / 'delta_p_raw'),
                'cp10k': DeltaPMode(reliability_cp10k_dir, reliability_cp10k_dir / 'delta_p_cp10k')
            }
        except FileNotFoundError:
            self.mode_mapping = {
                'raw': DeltaPMode(reliability_dir, reliability_dir / 'systema'),
                'cp10k': DeltaPMode(reliability_cp10k_dir, reliability_cp10k_dir / 'systema_cp10k')
            }

        self.ctrl_gene_expressions_dir = ctrl_gene_expressions_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clusters = set([int(x.strip()) for x in clusters.split(',')]) if clusters else None

        self.gene_metadata = get_gene_metadata(to_pandas=True)

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "BaseSystematicStep":
        """Create BaseSystematicStep from arguments."""
        return cls(**vars(args))

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
            '--output-dir', dest='output_dir', required=True, type=Path,
            help='Output directory to write results to'
        )

        parser.add_argument(
            '--reliability-dir', dest='reliability_dir', type=Path,
            help="Path to the raw reliability results to serve as input data. Should "
                 + "contain both reliability_summary_*_cluster_*.csv files and delta_p_raw directory."
        )
        parser.add_argument(
            '--reliability-cp10k-dir', dest='reliability_cp10k_dir', type=Path,
            help="Path to the CP10K reliability results to serve as input data. Should "
                 + "contain both reliability_summary_*_cluster_*.csv files and delta_p_cp10k directory"
        )

        parser.add_argument(
            '--clusters', dest='clusters', type=str,
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

    def load_moa_targets(self) -> dict[str, str]:
        """Load MOA targets from Tahoe100M."""
        drug_metadata = get_drug_metadata(to_pandas=True)
        metadata = drug_metadata[["moa-fine", "targets"]]

        # Remove missing values since not all drugs have targets
        metadata = metadata.dropna(subset=['moa-fine', 'targets'])

        # Split comma-separated targets and strip whitespace
        metadata['targets'] = metadata['targets'].str.split(',').apply(
            lambda targets: [target.strip() for target in targets if target.strip()]
        )

        # Group targets by MOA, keeping unique genes
        moa_targets = (
            metadata.groupby('moa-fine')['targets']
            .apply(lambda groups: sorted(set(
                gene for genes in groups for gene in genes
            )))
            .to_dict()
        )

        return moa_targets
