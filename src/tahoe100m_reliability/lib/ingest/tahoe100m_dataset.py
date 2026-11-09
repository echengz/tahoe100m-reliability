"""
Methods to handle loading the Tahoe100m dataset, namely
associated metadata.
"""
import pandas as pd
from datasets import IterableDataset, load_dataset

TAHOE_100M_DATASET = 'tahoebio/Tahoe-100m'


def __get_dataset(
    dataset_name: str, to_pandas: bool = False
) -> IterableDataset | pd.DataFrame:
    """Load the dataset, either as an IterableDataset or as a pandas dataframe."""
    data = load_dataset(
        TAHOE_100M_DATASET,
        dataset_name,
        streaming=True,
        split='train' 
    )

    if to_pandas:
        return data.to_pandas()

    return data


def get_cell_line_metadata(to_pandas: bool = False) -> IterableDataset | pd.DataFrame:
    return __get_dataset('cell_line_metadata', to_pandas=to_pandas)


def get_drug_metadata(to_pandas: bool = False)  -> IterableDataset | pd.DataFrame:
    return __get_dataset('drug_metadata', to_pandas=to_pandas)


def get_gene_metadata(to_pandas: bool = False)  -> IterableDataset | pd.DataFrame:
    return __get_dataset('gene_metadata', to_pandas=to_pandas)


def get_sample_metadata(to_pandas: bool = False) -> IterableDataset | pd.DataFrame:
    return __get_dataset('sample_metadata', to_pandas=to_pandas)


def list_tahoe100m_parquet_files() -> list[str]:
    files = list_repo_files(repo_id=TAHOE_100M_DATASET, repo_type='dataset')
    return [
        f'hf://datasets/{TAHOE_100M_DATASET}/{f}' for f in files
        if f.startswith('data/') and f.endswith('.parquet')
    ]
