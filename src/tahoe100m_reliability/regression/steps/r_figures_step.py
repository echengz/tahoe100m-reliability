"""
Plots regression-related figures.
"""
import argparse
import logging
import pandas as pd
import re

from pathlib import Path

from tahoe100m_reliability.lib.ingest.intermediate_loaders import (
    get_combined_cluster_split_half_reliability_df,
    get_combined_datasets_split_half_reliability_df,
)
from tahoe100m_reliability.lib.step.pipeline_step import PipelineStep

from .datasets import CELLULAR_DATASETS
from .figures.r_reliability_figures import RegressionReliabilityFigures

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class RegressionFiguresStep(PipelineStep):

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the step."""
        self.__output_dir = args.output_dir

        self.__reliability_raw_dir = args.reliability_raw_dir
        self.__reliability_cp10k_dir = args.reliability_cp10k_dir
        self.__reliability_tahoe100m_raw_dir = args.reliability_tahoe100m_raw_dir
        self.__reliability_tahoe100m_cp10k_dir = args.reliability_tahoe100m_cp10k_dir

        self.__systema_dir = args.systema_dir

        # either all or none must be provided
        reliability_dirs = [
            self.__reliability_raw_dir,
            self.__reliability_cp10k_dir,
            self.__reliability_tahoe100m_raw_dir,
            self.__reliability_tahoe100m_cp10k_dir,
        ]


        if not all(x is None for x in reliability_dirs) and not all(x for x in reliability_dirs):
            raise FileNotFoundError("All reliability directories must be provided if at least one is specified")


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
            '--reliability-raw-dir', dest='reliability_raw_dir', type=Path,
            help="Path to the reliability resuls"
        )
        parser.add_argument(
            '--reliability-cp10k-dir', dest='reliability_cp10k_dir', type=Path,
            help="Path to the normalized (CP10K) reliability resuls"
        )

        parser.add_argument(
            '--reliability-tahoe100m-raw-dir', dest='reliability_tahoe100m_raw_dir', type=Path,
            help="Path to the reliability Tahoe-100M resuls"
        )
        parser.add_argument(
            '--reliability-tahoe100m-cp10k-dir', dest='reliability_tahoe100m_cp10k_dir', type=Path,
            help="Path to the normalized (CP10K) Tahoe-100M reliability resuls"
        )

        parser.add_argument(
            '--systema-dir', dest='systema_dir', type=Path,
            help='Path to the systematic results (assumes both raw and cp10k are in there)'
        )

    def run(self) -> None:
        """Generate figures."""

        # if one is provided, all should have been provided
        if self.__reliability_raw_dir:
            all_dfs = []

            reliability_modes = {
                'raw': self.__reliability_raw_dir,
                'cp10k': self.__reliability_cp10k_dir
            }

            # combine all DFs into one
            for mode, reliability_dir in reliability_modes.items():
                all_dfs.append(
                    get_combined_datasets_split_half_reliability_df(
                        datasets=list(CELLULAR_DATASETS.keys()),
                        reliability_dir=reliability_dir,
                        summary=False,
                        normalization_mode=mode,
                    )
                )

            # add in tahoe100m
            tahoe100m_modes = {
                'raw': self.__reliability_tahoe100m_raw_dir,
                'cp10k': self.__reliability_tahoe100m_cp10k_dir
            }
            for mode, reliability_dir in tahoe100m_modes.items():
                df = get_combined_cluster_split_half_reliability_df(
                    reliability_dir=reliability_dir,
                    clusters=None,
                    summary=False,
                    ds_name='tahoe100m'
                )
                df['dataset'] = 'Tahoe-100M'
                df['normalization'] = mode
                all_dfs.append(df)

            reliability_output_dir = self.__output_dir / 'reliability'
            reliability_output_dir.mkdir(parents=True, exist_ok=True)

            figures = RegressionReliabilityFigures(reliability_output_dir)
            figures.generate_figures(pd.concat(all_dfs, ignore_index=True))

        # Generate systematic figures
        if self.__systema_dir:
            pass
