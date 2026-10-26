"""
Runs the reliability step of the pipeline
"""
import argparse

from .pipeline_step import PipelineStep

class ReliabilityStep(PipelineStep):

    def run(self, args: argparse.Namespace) -> None:
        """Run the full pipeline."""
        print("not yet implemented")
