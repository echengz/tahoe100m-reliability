"""
Runs the full pipeline
"""
import argparse

from tahoe100m_reliability.lib.step.pipeline_step import PipelineStep

class Tahoe100mPipelineRunner(PipelineStep):

    def run(self) -> None:
        """Run the full pipeline."""
        print("not yet implemented")
