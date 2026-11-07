"""
Step to generate figures.

Most steps generate their own figures if run separately. This step
is meant to regenerate/tweak them as needed assuming the outputs
have been generated
"""
import argparse
import logging
import pandas as pd
import re

from pathlib import Path

from tahoe100m_reliability.lib.tahoe100m_dataset import (
    get_cell_line_metadata,
    get_drug_metadata,
)
from tahoe100m_reliability.lib.treatment_set import TreatmentSet

from .figures.gsea_figures import plot_gsea_figures
from .figures.preprocess_figures import PreprocessFigures
from .figures.reliability_figures import ReliabilityFigures
from .figures.systematic_figures import SystematicFigures
from .step_helpers.intermediate_loaders import load_influence_by_cell_line_moa
from .step_helpers.pipeline_step import PipelineStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class FiguresStep(PipelineStep):

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the step."""
        self.__output_dir = args.output_dir

        self.__preprocess_dir = args.preprocess_dir
        self.__reliability_raw_dir = args.reliability_raw_dir
        self.__reliability_cp10k_dir = args.reliability_cp10k_dir
        self.__systema_dir = args.systema_dir
        self.__gsea_dir = args.gsea_dir

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI args for this pipeline step to the given parser.

        In this case, all the args are just the output directories containing
        results of the other steps. All are optional; if not passed, will
        not generate the figure for that step.
        """
        parser.add_argument(
            '-o', '--output-dir', dest='output_dir', required=True, type=Path,
            help='Output directory to write results to'
        )

        parser.add_argument(
            '--preprocess-dir', dest='preprocess_dir', type=Path,
            help="Path to the preprocess resuls; should contain combined_index_treatment_sets.json"
        )
        parser.add_argument(
            '--reliability-cp10k-dir', dest='reliability_cp10k_dir', type=Path,
            help="Path to the normalized (CP10K) reliability resuls"
        )
        parser.add_argument(
            '--reliability-raw-dir', dest='reliability_raw_dir', type=Path,
            help="Path to the reliability resuls"
        )
        parser.add_argument(
            '--systema-dir', dest='systema_dir', type=Path,
            help='Path to the systematic results (assumes both raw and cp10k are in there)'
        )
        parser.add_argument(
            '--gsea-dir', dest='gsea_dir', type=Path,
            help='Path to GSEA step results'
        )

    def run(self) -> None:
        """Generate figures."""

        # Generate preprocessing figures
        if self.__preprocess_dir:
            log.info("Generating preprocess figures...")
            treatments_file = self.__preprocess_dir / 'combined_indexed_treatment_sets.json'
            treatment_sets = TreatmentSet.load_list_from_file(
                treatments_file=treatments_file
            )

            preprocess_output_dir = self.__output_dir / 'preprocess'
            preprocess_output_dir.mkdir(parents=True, exist_ok=True)

            figures = PreprocessFigures(treatment_sets, preprocess_output_dir)
            figures.generate_figures()

        # generate reliability figures
        rel_map = {
            'raw': self.__reliability_raw_dir,
            'cp10k': self.__reliability_cp10k_dir
        }

        for mode, rel_dir in rel_map.items():
            if rel_dir:
                log.info(f"Generating {mode} reliability figures...")
                reliability_output_dir = self.__output_dir / f'reliability_{mode}'
                reliability_output_dir.mkdir(parents=True, exist_ok=True)

                figures = ReliabilityFigures(rel_dir, reliability_output_dir)
                figures.generate_figures()

        if self.__systema_dir:
            for mode in ['raw', 'cp10k']:
                log.info(f"Generating {mode} systema figures...")
                systema_output_dir = self.__output_dir / f'systema_{mode}'
                systema_output_dir.mkdir(parents=True, exist_ok=True)

                figures = SystematicFigures(self.__systema_dir, systema_output_dir, mode)
                figures.generate_figures()

        # Generate GSEA figures
        if self.__gsea_dir:
            log.info("Generating GSEA figures...")
            gsea_output_dir = self.__output_dir / 'gsea'
            gsea_output_dir.mkdir(parents=True, exist_ok=True)

            metadata = get_cell_line_metadata(to_pandas=True)
            metadata["cellosaurus_id"] = metadata["Cell_ID_Cellosaur"].astype(str).str.strip()
            metadata["organ"] = metadata["Organ"].astype(str).str.strip()
            organ_by_cell_line = metadata.set_index("cellosaurus_id")["organ"].to_dict()

            if self.__systema_dir:
                influence_by_cell_line_moa_dir = self.__systema_dir / 'influence_by_cell_line_moa'
                if influence_by_cell_line_moa_dir.is_dir():
                    influence_df = load_influence_by_cell_line_moa(influence_by_cell_line_moa_dir)

            for axis in ['global', 'moa']:
                df = pd.read_csv(self.__gsea_dir / f"{axis}_shared_axis_hallmark_gsea.csv")

                if axis == 'global':
                    log.info("Plotting global axis")
                    plot_gsea_figures(
                        df,
                        gsea_output_dir,
                        "gsea_global",
                        organ_by_cell_line,
                        "Global",
                        influence_df=influence_df,
                    )
                else:
                    for moa, moa_df in df.groupby("moa"):
                        # Make directory name filesystem-safe
                        safe_moa = re.sub(r"[^A-Za-z0-9._-]+", "_", str(moa).lower())
                        log.info(f"Plotting MOA axis: {moa}")
                        plot_gsea_figures(
                            moa_df,
                            gsea_output_dir,
                            f"gsea_moa_{safe_moa}" ,
                            organ_by_cell_line,
                            moa,
                            influence_df=influence_df,
                        )
