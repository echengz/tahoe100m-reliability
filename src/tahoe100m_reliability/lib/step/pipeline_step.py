"""
Base class to handle a step of the pipeline.
"""
import argparse
import time


class PipelineStep:
    """Base pipeline step."""

    def __init__(self) -> None:
        """Initialize the step."""

        # time step was initiated; for output files
        self.cur_time = time.strftime('%Y%m%d-%H%M%S')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "PipelineStep":
        """Create PipelineStep from arguments."""
        return cls()

    @classmethod
    def clean_args(cls, args: argparse.Namespace) -> dict:
        """Return cleaned args as a dict."""
        args_dict = vars(args)
        args_dict.pop('command')

        return args_dict

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this pipeline step to the given parser."""
        pass


    def run(self) -> None:
        """Run the step."""
        pass
