"""
Base class to handle a step of the pipeline.
"""
import argparse
import time


class PipelineStep:
    """Base pipeline step."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the step."""

        # time step was initiated; for output files
        self.cur_time = time.strftime('%Y%m%d-%H%M%S')

    def run(self) -> None:
        """Run the step."""
        pass

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this pipeline step to the given parser."""
        pass
