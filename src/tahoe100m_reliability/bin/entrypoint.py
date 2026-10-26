"""
Defines the entrypoint for the Tahoe-100M Reliability pipeline.

Allows for running all or specific steps of the pipeline.
"""
import argparse

from tahoe100m_reliability.pipeline.runner import Tahoe100mPipelineRunner
from tahoe100m_reliability.pipeline.steps import (
    PreprocessStep,
    ReliabilityStep,
    SystematicStep,
)


def build_parser() -> argparse.ArgumentParser:
    """Builds subparsers for each step of the pipeline."""
    parser = argparse.ArgumentParser(
        prog='tahoe100m-reliability',
        description='Run the Tahoe-100M Reliability pipeline, either in full or in stages'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
    )

    #################
    # FULL PIPELINE #
    #################
    all_parser = subparsers.add_parser(
        'all',
        help='Run the full pipeline'
    )
    Tahoe100mPipelineRunner.add_cli_args(all_parser)

    ###################
    # PREPROCESS STEP #
    ###################
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Run the preprocessing step of the pipeline'
    )
    PreprocessStep.add_cli_args(preprocess_parser)

    ####################
    # RELIABILITY STEP #
    ####################
    reliability_parser = subparsers.add_parser(
        'reliability',
        help='Run the reliability step of the pipeline'
    )
    ReliabilityStep.add_cli_args(reliability_parser)

    ###################
    # SYSTEMATIC STEP #
    ###################
    systematic_parser = subparsers.add_parser(
        'systematic',
        help='Runs the systematic step of the pipeline'
    )
    SystematicStep.add_cli_args(systematic_parser)

    return parser


def main() -> None:
    """Defines the main entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'all':
        Tahoe100mPipelineRunner().run(args)
    elif args.command == 'preprocess':
        PreprocessStep().run(args)
    elif args.command == 'reliability':
        ReliabilityStep().run(args)
    elif args.command == 'systematic':
        SystematicStep().run(args)
    else:
        raise ValueError(f'Unrecognized command: {args.command}')


if __name__ == "__main__":
    main()
