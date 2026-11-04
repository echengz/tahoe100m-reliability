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

from .figures.gsea_figures import plot_gsea
from .figures.preprocess_figures import PreprocessFigures
from .figures.reliability_figures import plot_reliability_figures
from .step_helpers.pipeline_step import PipelineStep

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class FiguresStep(PipelineStep):

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the step."""
        self.__output_dir = args.output_dir

        self.__preprocess_dir = args.preprocess_dir
        self.__reliability_dir = args.reliability_dir
        self.__reliability_normalized_dir = args.reliability_normalized_dir
        self.__systema_raw_dir = args.systema_raw_dir
        self.__systema_cp10k_dir = args.systema_cp10k_dir
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
            '--reliability-normalized-dir', dest='reliability_normalized_dir', type=Path,
            help="Path to the normalized (CP10K) reliability resuls"
        )
        parser.add_argument(
            '--reliability-dir', dest='reliability_dir', type=Path,
            help="Path to the reliability resuls"
        )
        parser.add_argument(
            '--systema-raw-dir', dest='systema_raw_dir', type=Path,
            help='Path to the systematic raw results'
        )
        parser.add_argument(
            '--systema-cp10k-dir', dest='systema_cp10k_dir', type=Path,
            help='Path to systematic CP10k results'
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
            'raw': self.__reliability_dir,
            'cp10k': self.__reliability_normalized_dir
        }

        for mode, rel_dir in rel_map.items():
            if rel_dir:
                log.info("Generating reliability figures...")
                reliability_output_dir = self.__output_dir / f'reliability_{mode}'
                reliability_output_dir.mkdir(parents=True, exist_ok=True)

                plot_reliability_figures(rel_dir, reliability_output_dir)

        # Generate GSEA figures
        if self.__gsea_dir:
            log.info("Generating GSEA figures...")
            gsea_output_dir = self.__output_dir / 'gsea'
            gsea_output_dir.mkdir(parents=True, exist_ok=True)

            metadata = get_cell_line_metadata(to_pandas=True)
            metadata["cellosaurus_id"] = metadata["Cell_ID_Cellosaur"].astype(str).str.strip()
            metadata["organ"] = metadata["Organ"].astype(str).str.strip()
            organ_by_cell_line = metadata.set_index("cellosaurus_id")["organ"].to_dict()

            for axis in ['global', 'moa']:
                df = pd.read_csv(self.__gsea_dir / f"{axis}_shared_axis_hallmark_gsea.csv")

                if axis == 'global':
                    log.info("Plotting global axis")
                    plot_gsea(df, gsea_output_dir / 'global', organ_by_cell_line, "global")
                else:
                    for moa, moa_df in df.groupby("moa"):
                        # Make directory name filesystem-safe
                        safe_moa = re.sub(r"[^A-Za-z0-9._-]+", "_", str(moa))
                        log.info(f"Plotting MOA axis: {moa}")
                        plot_gsea(
                            moa_df, gsea_output_dir / "moa" / safe_moa, organ_by_cell_line, moa
                        )
