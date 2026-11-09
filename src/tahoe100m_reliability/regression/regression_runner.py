"""
Runs the regression pipeline on the original 29 datasets to ensure
results are still aligned with the original framework.
"""
import argparse

from tahoe100m_reliability.lib.step.pipeline_step import PipelineStep


class RegressionPipelineRunner(PipelineStep):

    def run(self, args: argparse.Namespace) -> None:
        """Run the full pipeline."""
        pass
