"""
Step to generate figures. Will only generate figures for
steps whose output data is passed in as input arguments.
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

from .figures.gsea_figure_helpers import (
    GSEAIndividualHeatmaps,
    GSEAInfluenceFigures,
    GSEARadialHeatmaps,
)
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

    def __generate_preprocessing_figures(self):
        # Generate preprocessing figures
        if not self.__preprocess_dir:
            return

        log.info("Generating preprocess figures...")
        treatments_file = self.__preprocess_dir / 'combined_indexed_treatment_sets.json'
        treatment_sets = TreatmentSet.load_list_from_file(
            treatments_file=treatments_file
        )

        preprocess_output_dir = self.__output_dir / 'preprocess'
        preprocess_output_dir.mkdir(parents=True, exist_ok=True)

        figures = PreprocessFigures(treatment_sets, preprocess_output_dir)
        figures.generate_figures()

    def __generate_reliability_figures(self):
        # Generate reliability figures
        rel_map = {
            'raw': self.__reliability_raw_dir,
            'cp10k': self.__reliability_cp10k_dir
        }
        reliability_output_dir = self.__output_dir / 'reliability'

        for mode, rel_dir in rel_map.items():
            if not rel_dir:
                continue

            log.info(f"Generating {mode} reliability figures...")
            reliability_output_dir.mkdir(parents=True, exist_ok=True)

            figures = ReliabilityFigures(rel_dir, reliability_output_dir, mode)
            figures.generate_figures()

    def __generate_systematic_figures(self):
        # Generate systematic figures
        if not self.__systema_dir:
            return

        systema_output_dir = self.__output_dir / 'systema'
        systema_output_dir.mkdir(parents=True, exist_ok=True)

        for mode in ['raw', 'cp10k']:
            log.info(f"Generating {mode} systema figures...")
            figures = SystematicFigures(self.__systema_dir, systema_output_dir, mode)
            figures.generate_figures()

        # combined results
        combined_figures = CombinedSystematicFigures(self.__systema_dir, systema_output_dir)
        combined_figures.generate_figures()

    def __merge_influence_with_gsea(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Merge influence with GSEA results, if influence results exist."""
        if not self.__systema_dir:
            return df, False

        influence_dir = self.__systema_dir / 'gsea_influences'
        influence_df = pd.DataFrame()

        if influence_dir.is_dir():
            files = influence_dir.glob(f"systema_{mode}_gsea_moa_*.csv")
            latest_file = sorted(files)[-1]

            moa_influence_df = pd.read_csv(latest_file)
            moa_influence_df['mode'] = mode
            moa_influence_df = moa_influence_df.rename(
                columns={'group_id': 'term', 'cell_line_id': 'cell_line'}
            )

            influence_df = pd.concat(
                [influence_df, moa_influence_df], ignore_index=True
            )
        else:
            return df, False

        influence_df = df.merge(
            influence_df,
            on=['moa', 'cell_line', 'mode', 'term'],
            how='inner',
            validate='one_to_one'
        )
        influence_df["abs_nes"] = influence_df["nes"].abs()

        return influence_df, True

    def __generate_gsea_figures(self):
        # Generate GSEA figures
        if not self.__gsea_dir:
            return

        log.info("Generating GSEA figures...")
        gsea_output_dir = self.__output_dir / 'gsea'

        metadata = get_cell_line_metadata(to_pandas=True)
        metadata["cellosaurus_id"] = metadata["Cell_ID_Cellosaur"].astype(str).str.strip()
        metadata["organ"] = metadata["Organ"].astype(str).str.strip()
        organ_by_cell_line = metadata.set_index("cellosaurus_id")["organ"].to_dict()

        # Instantiate classes to handle heatmap plotting
        indv_heatmaps = GSEAIndividualHeatmaps(
            output_dir=gsea_output_dir / 'heatmaps_individual',
            organ_by_cell_line=organ_by_cell_line,
        )

        radial_heatmaps = GSEARadialHeatmaps(
            output_dir=gsea_output_dir / 'heatmaps_clustered_moa',
            organ_by_cell_line=organ_by_cell_line,
        )

        # plot everything individually, and also by grouped MOAs
        for axis in ['global', 'moa']:
            # load GSEA results and fix Pperoxisome typo
            df = pd.read_csv(self.__gsea_dir / f"{axis}_shared_axis_hallmark_gsea.csv")
            df.loc[df['term'] == 'Pperoxisome', 'term'] = 'Peroxisome'

            if axis == 'global':
                log.info("Plotting cell line axis")
                indv_heatmaps.plot_heatmaps(
                    gsea_df=df, filename_prefix='gsea_global', title_postfix='Global'
                )
            else:
                log.info("Plotting cell line x MOA axes")

                for moa, moa_df in df.groupby("moa"):
                    # Make name filesystem-safe
                    safe_moa = re.sub(r"[^A-Za-z0-9._-]+", "_", str(moa).lower())
                    log.info(f"Plotting MOA axis: {moa}")

                    indv_heatmaps.plot_heatmaps(
                        gsea_df=moa_df, filename_prefix=f"gsea_moa_{safe_moa}", title_postfix=moa
                    )

                for mode in ['raw', 'cp10k']:
                    # load MOA cluster assignments
                    moa_cluster_assignments = pd.read_csv(
                        self.__gsea_dir / 'moa_clustering' / mode / f'{mode}_moa_cluster_assignments.csv'
                    )

                    moa_clusters = {
                        cluster: sorted(moas)
                        for cluster, moas in moa_cluster_assignments.groupby("cluster")["moa"]
                    }
                    moa_order = []

                    # Plot clustered MOA GSEA results
                    for cluster, moas in moa_clusters.items():
                        radial_heatmaps.plot_moa_radial_heatmaps(
                            gsea_df=df[df["moa"].isin(moas)],
                            mode=mode,
                            filename_prefix=f"gsea_radial_by_moa_cluster{cluster}",
                            title=f'Hallmark GSEA by MOA - Cluster {cluster}'
                        )

                        moa_order.extend(moas)

                    # Plot overall GSEA results
                    radial_heatmaps.plot_moa_radial_heatmaps(
                        gsea_df=df,
                        mode=mode,
                        filename_prefix=f"gsea_radial_all",
                        moa_order=moa_order,
                        title='Hallmark GSEA by MOA - All',
                        moa_clusters=moa_clusters,
                    )

                    # if systematic influence results exist for GSEA, merge them in and plot
                    influence_df, influence_exists = self.__merge_influence_with_gsea(df, mode)

                    if influence_exists:
                        infl_gsea_plots = GSEAInfluenceFigures(
                            output_dir=gsea_output_dir / 'influence_gsea'
                        )
                        infl_gsea_plots.generate_figures(
                            df=influence_df, mode=mode, filename=f"gsea_influence_{mode}.png"
                        )

                        # Identify pathways with largest number of significant NES values and plot those
                        mode_df = influence_df[influence_df["mode"] == mode]

                        significance_fraction = (
                            mode_df.groupby("term")["fdr_qval"]
                            .apply(lambda x: (x < 0.05).mean())
                        )

                        target_pathways = (
                            significance_fraction
                            .sort_values(ascending=False)
                            .head(10)
                            .index
                            .tolist()
                        )

                        infl_gsea_plots.generate_figures(
                            df=influence_df, mode=mode, filename=f"gsea_influence_{mode}_significant.png",
                            target_pathways=target_pathways
                        )

    def run(self) -> None:
        """Generate figures."""

        self.__generate_preprocessing_figures()
        self.__generate_reliability_figures()
        self.__generate_systematic_figures()
        self.__generate_gsea_figures()
