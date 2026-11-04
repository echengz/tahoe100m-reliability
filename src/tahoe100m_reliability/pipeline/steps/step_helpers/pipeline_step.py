"""
Base class to handle a step of the pipeline.
"""
import argparse


class PipelineStep:

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
