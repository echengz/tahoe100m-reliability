"""
Runs the full Tahoe-100M-specific pipeline.

In this case runs with the minimum required arguments. If you need
to customize a step, run it directly.
"""
import argparse
import logging

from pathlib import Path

from tahoe100m_reliability.lib.step.pipeline_step import PipelineStep

from .steps import (
    FiguresStep,
    GseaStep,
    PreprocessStep,
    ReliabilityStep,
    SystematicInfluenceStep,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Tahoe100mPipelineRunner(PipelineStep):

    def __init__(
        self,
        output_dir: Path,
        preprocess_args: dict,
        reliability_raw_args: dict,
        reliability_cp10k_args: dict,
        gsea_args: dict,
        systematic_args: dict,
        figures_args: dict,
    ) -> None:
        super().__init__()
        self.__output_dir = output_dir

        self.__preprocess_args = preprocess_args
        self.__reliability_raw_args = reliability_raw_args
        self.__reliability_cp10k_args = reliability_cp10k_args
        self.__gsea_args = gsea_args
        self.__systematic_args = systematic_args
        self.__figures_args = figures_args

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Tahoe100mPipelineRunner":
        """
        Create Tahoe100mPipelineRunner from arguments. Group arguments by step.

        For most steps, the arguments of the current step are fed into by the
        results of the prior step.
        """
        args_dict = clean_args(args)

        # shared args
        output_dir = args_dict['output_dir']
        ctrl_gene_expressions_dir = args_dict['control_gene_expressions_dir']

        preprocess_dir = output_dir / 'preprocess'
        preprocess_args = {
            'output_dir': preprocess_dir,
            'treatment_sets_dir': preprocess_dir / 'indexed_treatments',
            'control_gene_expressions_dir': ctrl_gene_expressions_dir,
        }

        reliability_raw_dir = output_dir / 'reliability_raw'
        reliability_raw_args = {
            'output_dir': reliability_raw_dir,
            'normalize_cp10k': False,
            'indexed_treatments_file': preprocess_dir / 'combined_indexed_treatment_sets.json',
            'ctrl_gene_expressions_dir': ctrl_gene_expressions_dir,
            'clustered_treatments_csv': preprocess_dir / 'clustered_treatments.csv',
        }

        reliability_cp10k_dir = output_dir / 'reliability_raw'
        reliability_cp10k_args = {
            'output_dir': reliability_cp10k_dir,
            'normalize_cp10k': True,
            'indexed_treatments_file': preprocess_dir / 'combined_indexed_treatment_sets.json',
            'ctrl_gene_expressions_dir': ctrl_gene_expressions_dir,
            'clustered_treatments_csv': preprocess_dir / 'clustered_treatments.csv',
        }

        gsea_dir = output_dir / 'gsea'
        gsea_args = {
            'output_dir': gsea_dir,
            'ctrl_gene_expressions_dir': ctrl_gene_expressions_dir,
            'reliability_raw_dir': reliability_raw_dir,
            'reliability_cp10k_dir': reliability_cp10k_dir
        }

        systematic_dir = output_dir / 'systematic'
        systematic_args = {
            'output_dir': systematic_dir,
            'ctrl_gene_expressions_dir': ctrl_gene_expressions_dir,
            'reliability_raw_dir': reliability_raw_dir,
            'reliability_cp10k_dir': reliability_cp10k_dir,
            'gsea_global_shared_axis_hallmark': gsea_dir / 'global_shared_axis_hallmark_gsea.csv',
            'gsea_moa_shared_axis_hallmark': gsea_dir / 'moa_shared_axis_hallmark.csv'
        }

        figures_args = {
            'output_dir': output_dir / 'figures',
            'preprocess_dir': reliability_raw_dir,
            'reliability_raw_dir': reliability_raw_dir,
            'reliability_cp10k_dir': reliability_cp10k_dir,
            'gsea_dir': gsea_dir,
            'systematic_dir': systematic_dir,
        }

        return cls(
            output_dir=args_dict['output_dir'],
            preprocess_args=preprocess_args,
            reliability_raw_args=reliability_raw_args,
            reliability_cp10k_args=reliability_cp10k_args,
            gsea_args=gsea_args,
            systematic_args=systematic_args,
            figures_args=figures_args,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for the full Tahoe-100M pipeline to the given parser.

        In this case runs with the minimum required arguments. If you need
        to customize a step, run it directly.
        """
        parser.add_argument(
            '--output-dir', dest='output_dir', required=True, type=Path,
            help='Output directory to write results to'
        )

        parser.add_argument(
            '--control-gene-expressions-dir', dest='control_gene_expressions_dir',
            type=Path, required=True,
            help='Path to the control gene expressions directory'
        )

    def run(self) -> None:
        """Run the full pipeline."""
        log.info("RUNNING PREPROCESS STEP")
        PreprocessStep(**self.__preprocess_args).run()

        log.info("RUNNING RELIABILITY RAW STEP")
        reliability_raw_dir = self.output_dir / 'reliability_raw'
        ReliabilityStep(**self.__reliability_args).run()

        # doing the two reliability steps in parallel would consume an obscene
        # amount of resources, so do sequentially for now
        log.info("RUNNING RELIABILITY CP10K STEP")
        ReliabilityStep(**self.reliability_cp10k_args).run()

        log.info("RUNNING GSEA STEP")
        GseaStep(**self.__gsea_args).run()

        log.info("RUNNING SYSTEMATIC STEP")
        SystematicInfluenceStep(**self.__systematic_args).run()

        log.info("RUNNING FIGURES STEP")
        FiguresStep(**self.__figures_args).run()
