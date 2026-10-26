"""
Base class to handle a step of the pipeline.
"""
import argparse


class PipelineStep:

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this pipeline step to the given parser."""
        pass


    def run(self, args: argparse.Namespace) -> None:
        """Run the step."""
        pass
