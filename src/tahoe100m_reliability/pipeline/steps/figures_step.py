"""
Step to generate figures. Will only generate figures for
steps whose output data is passed in as input arguments.
"""
import argparse
import logging
import pandas as pd
import re

from pathlib import Path

from tahoe100m_reliability.lib.ingest.tahoe100m_dataset import (
    get_drug_metadata,
    get_organ_by_cell_line,
)
from tahoe100m_reliability.lib.ingest.treatment_set import TreatmentSet
from tahoe100m_reliability.lib.step.pipeline_step import PipelineStep

from .figures.gsea_figures import (
    GSEABarPlots,
    GSEAIndividualHeatmaps,
    GSEAInfluenceFigures,
    GSEARadialHeatmaps,
)
from .figures.preprocess_figures import PreprocessFigures
from .figures.reliability_figures import ReliabilityFigures
from .figures.systematic_figures import (
    CombinedInfluenceSystematicFigures,
    InfluenceSystematicFigures,
    SystematicFigures,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class FiguresStep(PipelineStep):

    def __init__(
        self,
        output_dir: Path | None = None,
        preprocess_dir: Path | None = None,
        reliability_raw_dir: Path | None = None,
        reliability_cp10k_dir: Path | None = None,
        gsea_dir: Path | None = None,
        systematic_dir: Path | None = None,
    ) -> None:
        """Initialize the step."""
        self.__output_dir = output_dir

        self.__preprocess_dir = preprocess_dir
        self.__reliability_raw_dir = reliability_raw_dir
        self.__reliability_cp10k_dir = reliability_cp10k_dir
        self.__gsea_dir = gsea_dir
        self.__systematic_dir = systematic_dir

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "FiguresStep":
        """Create FiguresStep from arguments."""
        return cls(**cls.clean_args(args))

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
            help="Path to the preprocess resuls"
        )
        parser.add_argument(
            '--reliability-raw-dir', dest='reliability_raw_dir', type=Path,
            help="Path to the (raw, unnormalized) reliability resuls"
        )
        parser.add_argument(
            '--reliability-cp10k-dir', dest='reliability_cp10k_dir', type=Path,
            help="Path to the normalized (CP10K) reliability resuls"
        )
        parser.add_argument(
            '--gsea-dir', dest='gsea_dir', type=Path,
            help='Path to GSEA step results'
        )
        parser.add_argument(
            '--systematic-dir', dest='systematic_dir', type=Path,
            help='Path to the systematic results (must contain both raw and cp10k results)'
        )

    def __generate_preprocess_figures(self):
        """Generate preprocess figures."""
        if not self.__preprocess_dir:
            return

        log.info("Generating preprocess figures...")
        treatments_file = self.__preprocess_dir / 'combined_indexed_treatment_sets.json'
        treatment_sets = TreatmentSet.load_list_from_file(
            treatments_file=treatments_file
        )

        preprocess_output_dir = self.__output_dir / 'preprocess'
        preprocess_output_dir.mkdir(parents=True, exist_ok=True)

        clustered_treatments_csv = self.__preprocess_dir / 'clustered_treatments.csv'
        figures = PreprocessFigures(
            treatment_sets, clustered_treatments_csv, preprocess_output_dir
        )
        figures.generate_figures()

    def __generate_reliability_figures(self):
        """Generate reliability figures."""
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
        """Generate systematic figures."""
        if not self.__systematic_dir:
            return

        systema_output_dir = self.__output_dir / 'systematic'
        organ_by_cell_line = get_organ_by_cell_line()

        for mode in ['raw', 'cp10k']:
            log.info(f"Generating {mode} systematic figures...")

            # basic systematic figures
            figures = SystematicFigures(
                self.__systematic_dir,
                systema_output_dir / 'systematic',
                mode,
                organ_by_cell_line,
            )
            figures.generate_figures()

            # influence figures
            inf_figures = InfluenceSystematicFigures(
                self.__systematic_dir,
                systema_output_dir / 'influence',
                mode,
                organ_by_cell_line,
            )
            inf_figures.generate_figures()

        # combined results
        combined_figures = CombinedInfluenceSystematicFigures(
            self.__systematic_dir,
            systema_output_dir / 'influence',
        )
        combined_figures.generate_figures()

    def __combine_gsea_modes(self, axis: str) -> pd.DataFrame:
        """Combine the raw/CP10K results as needed."""
        raw_file = self.__gsea_dir / f"raw_{axis}_shared_axis_hallmark_gsea.csv"
        cp10k_file = self.__gsea_dir / f"cp10k_{axis}_shared_axis_hallmark_gsea.csv"

        df1 = pd.read_csv(raw_file) if raw_file.is_file() else None
        df2 = pd.read_csv(cp10k_file) if cp10k_file.is_file() else None
        dfs = [df for df in [df1, df2] if df is not None]

        # see if there is a combined file
        if not dfs:
            combined_file = self.__gsea_dir / f'{axis}_shared_axis_hallmark_gsea.csv'
            if not combined_file.is_file():
                raise ValueError("Cannot find shared GSEA hallmark results")

            return pd.read_csv(combined_file)

        return pd.concat(dfs, ignore_index=True)

    def __generate_gsea_shared_figures(self):
        """Generate GSEA shared figures."""
        if not self.__gsea_dir:
            return

        log.info("Generating GSEA shared figures...")
        gsea_output_dir = self.__output_dir / 'gsea'
        organ_by_cell_line = get_organ_by_cell_line()

        # Instantiate classes to handle heatmap plotting
        indv_heatmaps = GSEAIndividualHeatmaps(
            output_dir=gsea_output_dir / 'heatmaps_individual',
            organ_by_cell_line=organ_by_cell_line,
        )

        radial_heatmaps = GSEARadialHeatmaps(
            output_dir=gsea_output_dir / 'heatmaps_clustered_moa',
            organ_by_cell_line=organ_by_cell_line,
        )

        # Shared axis
        # Plot everything individually, and also by grouped MOAs
        for axis in ['global', 'moa']:
            # load GSEA results and fix Pperoxisome typo
            df = self.__combine_gsea_modes(axis)
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

    def __generate_gsea_bar_plot_figures(self):
        """Generate GSEA bar plot figures."""
        if not self.__gsea_dir:
            return

        log.info("Generating GSEA bar plot figures...")
        gsea_output_dir = self.__output_dir / 'gsea' / 'bar_plots'

        figures = GSEABarPlots(gsea_output_dir)
        figures.generate_figures(self.__gsea_dir)

    def __merge_influence_with_gsea(self, df: pd.DataFrame, mode: str, axis: str) -> pd.DataFrame:
        """Merge influence with GSEA results, if influence results exist."""
        if not self.__systematic_dir:
            return df, False

        influence_dir = self.__systematic_dir / 'gsea_influences'
        if not influence_dir.is_dir():
            raise FileNotFoundError(
                "Need GSEA influence results from systematic steps to generate these figures"
            )

        influence_df = pd.DataFrame()

        files = influence_dir.glob(f"systema_{mode}_gsea_{axis}_*.csv")
        latest_file = sorted(files)[-1]

        moa_influence_df = pd.read_csv(latest_file)
        moa_influence_df['mode'] = mode
        moa_influence_df = moa_influence_df.rename(
            columns={'group_id': 'term', 'cell_line_id': 'cell_line'}
        )

        influence_df = pd.concat(
            [influence_df, moa_influence_df], ignore_index=True
        )

        merge_on = ['cell_line', 'mode', 'term']
        if axis == 'moa':
            merge_on.append('moa')

        influence_df = df.merge(
            influence_df,
            on=merge_on,
            how='inner',
            validate='one_to_one'
        )
        influence_df["abs_nes"] = influence_df["nes"].abs()

        return influence_df

    def __generate_gsea_influence_figures(self):
        """Generate GSEA x Systematic influence figures."""
        if not self.__gsea_dir or not self.__systematic_dir:
            return

        log.info("Generating GSEA x Systematic influence figures...")
        gsea_influence_output_dir = self.__output_dir / 'gsea' / 'gsea_influence'

        for axis in ['global', 'moa']:
            # ignore global axis for now; does not produce interesting plots
            if axis == 'global':
                continue

            infl_gsea_plots = GSEAInfluenceFigures(
                output_dir=gsea_influence_output_dir, axis=axis
            )

            # load GSEA results and fix Pperoxisome typo
            df = pd.read_csv(self.__gsea_dir / f"{axis}_shared_axis_hallmark_gsea.csv")
            df.loc[df['term'] == 'Pperoxisome', 'term'] = 'Peroxisome'

            for mode in ['raw', 'cp10k']:
                influence_df = self.__merge_influence_with_gsea(df, mode, axis)
                infl_gsea_plots.generate_figures(
                    df=influence_df, mode=mode, file_prefix='gsea_influence'
                )

    def run(self) -> None:
        """Generate figures."""

        self.__generate_preprocess_figures()
        self.__generate_reliability_figures()
        self.__generate_systematic_figures()
        self.__generate_gsea_bar_plot_figures()
        self.__generate_gsea_shared_figures()
        self.__generate_gsea_influence_figures()
