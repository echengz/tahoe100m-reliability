"""
Defines the entrypoint for the Tahoe-100M Reliability pipeline.

Allows for running all or specific steps of the pipeline.
"""
import argparse

from tahoe100m_reliability.pipeline import (
    Tahoe100mPipelineRunner,
)
from tahoe100m_reliability.pipeline.steps import (
    FiguresStep,
    GseaStep,
    PreprocessStep,
    ReliabilityStep,
    SystematicInfluenceStep,
)

from tahoe100m_reliability.regression import (
    RegressionPipelineRunner,
)
from tahoe100m_reliability.regression.steps import (
    RegressionReliabilityStep,
    RegressionSystematicStep,
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

    # FULL PIPELINE
    all_parser = subparsers.add_parser(
        'all',
        help='Run the full pipeline'
    )
    Tahoe100mPipelineRunner.add_cli_args(all_parser)

    # PREPROCESS STEP
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Run the preprocessing step of the pipeline'
    )
    PreprocessStep.add_cli_args(preprocess_parser)

    # RELIABILITY STEP
    reliability_parser = subparsers.add_parser(
        'reliability',
        help='Run the reliability step of the pipeline'
    )
    ReliabilityStep.add_cli_args(reliability_parser)

    # SYSTEMATIC STEP
    systematic_parser = subparsers.add_parser(
        'systematic',
        help='Runs the systematic step of the pipeline'
    )
    SystematicInfluenceStep.add_cli_args(systematic_parser)

    # GSEA STEP
    gsea_parser = subparsers.add_parser(
        'gsea',
        help='Runs the GSEA step of the pipeline'
    )
    GseaStep.add_cli_args(gsea_parser)

    # FIGURES STEP
    figures_parser = subparsers.add_parser(
        'figures',
        help='Runs the figures step of the pipeline'
    )
    FiguresStep.add_cli_args(figures_parser)

    # REGRESSION PIPELINE
    regression_parser = subparsers.add_parser(
        'regression',
        help='Run the regression pipeline on the original 29 datasets'
    )
    RegressionPipelineRunner.add_cli_args(regression_parser)

    # REGRESSION RELIABILITY
    regression_reliability_parser = subparsers.add_parser(
        'r-reliability',
        help='Run the regression reliability step on the original 29 datasets'
    )
    RegressionReliabilityStep.add_cli_args(regression_reliability_parser)

    # REGRESSION SYSTEMATIC
    regression_systematic_parser = subparsers.add_parser(
        'r-systematic',
        help='Run the regression systematic step on the original 29 datasets'
    )
    RegressionSystematicStep.add_cli_args(regression_systematic_parser)

    return parser


def main() -> None:
    """Defines the main entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    runner_map = {
        'all': Tahoe100mPipelineRunner,
        'preprocess': PreprocessStep,
        'reliability': ReliabilityStep,
        'systematic': SystematicInfluenceStep,
        'gsea': GseaStep,
        'figures': FiguresStep,
        'regression': RegressionPipelineRunner,
        'r-reliability': RegressionReliabilityStep,
        'r-systematic': RegressionSystematicStep,
    }

    runner = runner_map.get(args.command)
    if not runner:
        raise ValueError(f'Unrecognized command: {args.command}')

    runner(args).run()


if __name__ == "__main__":
    main()
