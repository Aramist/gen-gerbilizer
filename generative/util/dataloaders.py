import glob
import os
from os import path
import pathlib
from typing import NewType

import pandas as pd
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


Path = NewType(str)


class FeatherDataset(Dataset):
    def __init__(self,
        feather_path: Path,
        index_path: Path
    ):
        """ Creates a dataset to read audio from a .feather file, taking samples
        from the provided index.
        """
        if not path.exists(index_path):
            raise ValueError(f"Failed to find index with path: {index_path}")
        if not path.exists(feather_path):
            raise ValueError(f"Failed to find dataframe with path {feather_path}")

        self.index = np.load(index_path)
        self.df = pd.read_feather(feather_path)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.df['audio'].iloc[self.index[index]].astype(np.float32)


def create_dataloader(
    feather_dir: Path,
    index_dir: Path,
    batch_size: int = 64,
    num_workers: int = 2
) -> DataLoader:
    """ Creates dataloaders from dataframes located in `feather_dir` as
    .feather files and the corresponding indices, located in `index_dir`
    with the same name, but a .npy extension.
    """
    feather_files = sorted(glob.glob(feather_dir))
    npy_names = [
        pathlib.Path(fn).stem + '.npy'
        for fn in feather_files
    ]
    index_files = [path.join(index_dir, fn) for fn in npy_names]
    datasets = list()
    for df_path, index_path in zip(feather_files, index_files):
        datasets.append(FeatherDataset(df_path, index_path))
    full_ds = ConcatDataset(datasets)

    return DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )