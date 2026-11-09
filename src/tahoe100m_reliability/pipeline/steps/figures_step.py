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

from tahoe100m_reliability.lib.ingest.intermediate_loaders import (
    load_influence_by_cell_line_moa,
)
from tahoe100m_reliability.lib.ingest.tahoe100m_dataset import (
    get_cell_line_metadata,
    get_drug_metadata,
)
from tahoe100m_reliability.lib.ingest.treatment_set import TreatmentSet
from tahoe100m_reliability.lib.step.pipeline_step import PipelineStep

from .figures.gsea_figures import plot_gsea_figures
from .figures.gsea_pycirclize import plot_moa_group_circos
from .figures.preprocess_figures import PreprocessFigures
from .figures.reliability_figures import ReliabilityFigures
from .figures.systematic_figures import (
    CombinedSystematicFigures,
    SystematicFigures,
)

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

        # Generate reliability figures
        rel_map = {
            'raw': self.__reliability_raw_dir,
            'cp10k': self.__reliability_cp10k_dir
        }
        reliability_output_dir = self.__output_dir / 'reliability'

        for mode, rel_dir in rel_map.items():
            if rel_dir:
                log.info(f"Generating {mode} reliability figures...")
                reliability_output_dir.mkdir(parents=True, exist_ok=True)

                figures = ReliabilityFigures(rel_dir, reliability_output_dir, mode)
                figures.generate_figures()

        # Generate systematic figures
        if self.__systema_dir:
            systema_output_dir = self.__output_dir / 'systema'
            systema_output_dir.mkdir(parents=True, exist_ok=True)

            for mode in ['raw', 'cp10k']:
                log.info(f"Generating {mode} systema figures...")
                figures = SystematicFigures(self.__systema_dir, systema_output_dir, mode)
                figures.generate_figures()

            # combined results
            combined_figures = CombinedSystematicFigures(self.__systema_dir, systema_output_dir)
            combined_figures.generate_figures()

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
                    # plot_gsea_figures(
                    #     df,
                    #     gsea_output_dir,
                    #     "gsea_global",
                    #     organ_by_cell_line,
                    #     "Global",
                    #     influence_df=influence_df,
                    # )
                    # plot_moa_group_circos(

                    # )

                else:
                    moa_groupings = {
                        "RTK-RAS-MAPK signalling": [
                            "EGFR/ERBB inhibitor",
                            "Multi-TK inhibitor",
                            "Other TK inhibitor",
                            "RAS inhibitor",
                            "RAF inhibitor",
                            "MEK inhibitor",
                            "Other MAPK inhibitor"
                        ]
                    }

                    selected_moas = [
                        "EGFR/ERBB inhibitor",
                        "Multi-TK inhibitor",
                        "Other TK inhibitor",
                        "RAS inhibitor",
                        "RAF inhibitor",
                        "MEK inhibitor",
                        "Other MAPK inhibitor"
                    ]

                    plot_moa_group_circos(
                        gsea_df=df[~df["moa"].isin(['unclear'])].copy(),
                        output_dir=gsea_output_dir,
                        filename_prefix="gsea_pycirclize_by_moa",
                        organ_by_cell_line=organ_by_cell_line,
                        sector_gap=1
                    )

                    # for moa, moa_df in df.groupby("moa"):
                    #     # Make directory name filesystem-safe
                    #     safe_moa = re.sub(r"[^A-Za-z0-9._-]+", "_", str(moa).lower())
                    #     log.info(f"Plotting MOA axis: {moa}")
                    #     plot_gsea_figures(
                    #         moa_df,
                    #         gsea_output_dir,
                    #         f"gsea_moa_{safe_moa}" ,
                    #         organ_by_cell_line,
                    #         moa,
                    #         influence_df=influence_df,
                    #     )
