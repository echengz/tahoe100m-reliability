"""
Runs the systematic step of the
"""
import argparse

from .pipeline_step import PipelineStep

class SystematicStep(PipelineStep):

    def run(self, args: argparse.Namespace) -> None:
        """Run the full pipeline."""
        print("not yet implemented")
