"""
Runs the regression pipeline on the original 12 cellular datasets
to ensure results are still aligned with the original framework.
"""
import argparse
import logging

from pathlib import Path

from tahoe100m_reliability.lib.step.pipeline_step import PipelineStep

from .steps.r_reliability_step import RegressionReliabilityStep
from .steps.r_systematic_step import RegressionSystematicStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class RegressionPipelineRunner(PipelineStep):

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        baseline_reliability_dir: Path,
        baseline_systematic_dir: Path,
    ) -> None:
        super().__init__()
        self.__input_dir = input_dir
        self.__output_dir = output_dir
        self.__baseline_reliability_dir = baseline_reliability_dir
        self.__baseline_systematic_dir = baseline_systematic_dir

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "RegressionPipelineRunner":
        """Create RegressionPipelineRunner from arguments."""
        return cls(**cls.clean_args(args))

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for the regression reliability step to the given parser."""
        parser.add_argument(
            '-i', '--input-dir', dest='input_dir', required=True, type=Path,
            help='Input directory containing h5ad files for the original cellular datasets'
        )
        parser.add_argument(
            '-o', '--output-dir', dest='output_dir', required=True, type=Path,
            help='Output directory to write results to'
        )

        parser.add_argument(
            '--baseline-reliability-dir', dest='baseline_reliability_dir', required=True, type=Path,
            help='Baseline directory containing reliability results to compare to; should contain '
                 + 'reliability_cc_* results from the original framework'
        )

        parser.add_argument(
            '--baseline-systematic-dir', dest='baseline_systematic_dir', required=True, type=Path,
            help='Baseline directory containing systematic results to compare to; should contain '
                 + 'systema_* results from the original framework'
        )

        super().add_cli_args(parser)

    def run(self) -> None:
        """
        Run the full pipeline - only consists of the
        r-reliability and r-systematic steps.
        """
        log.info("RUNNING R-RELIABILITY STEP")
        reliability_dir = self.__output_dir / 'r_reliability'
        RegressionReliabilityStep(
            input_dir=self.__input_dir,
            baseline_dir=self.__baseline_reliability_dir,
            output_dir=reliability_dir,
            # just keep other defaults for regression testing
        ).run()

        log.info("RUNNING R-SYSTEMATIC STEP")
        systematic_dir = self.__output_dir / 'r_systematic'
        RegressionSystematicStep(
            reliability_dir=reliability_dir,
            output_dir=systematic_dir,
            baseline_dir=self.__baseline_systematic_dir
        ).run()
