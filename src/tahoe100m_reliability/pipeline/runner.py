"""
Runs the full pipeline
"""
import argparse

from .steps.pipeline_step import PipelineStep

class Tahoe100mPipelineRunner(PipelineStep):

    def run(self, args: argparse.Namespace) -> None:
        """Run the full pipeline."""
        print("not yet implemented")
