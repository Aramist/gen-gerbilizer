import glob
import os
from os import path
import pathlib
from typing import NewType

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


Path = NewType('Path', str)


class HDFDataset(Dataset):
    def __init__(self,
        hdf_path: Path, *,
        pad_len: int=16384
    ):
        """ Creates a dataset to read audio from a .hdf file, using the index
        embedded in the file
        """
        if not path.exists(hdf_path):
            raise ValueError(f"Failed to find dataset with path {hdf_path}")

        self.dset = h5py.File(hdf_path, 'r')
        self.pad_len = pad_len

    def __len__(self) -> int:
        # Len idx contains the starting indices of every vocalization and
        # the endpoint of the final vocalization, so it has n + 1 elements
        return len(self.dset['len_idx']) - 1

    def __getitem__(self, index: int) -> torch.Tensor:
        start, end = self.dset['len_idx'][index:index+2]
        vox = self.dset['vocalizations'][start:end]
        padded = np.zeros((1, self.pad_len), dtype=np.float32)
        offset = np.random.randint(0, 2**12)
        vox_len_clipped = min(self.pad_len, len(vox) + offset) - offset
        padded[0, offset:min(self.pad_len, len(vox) + offset)] = vox[:vox_len_clipped]
        return padded


def create_dataloader(
    hdf_path: Path,
    batch_size: int = 64
) -> DataLoader:
    """ Creates dataloaders from dataframes located in `hdf_dir` as
    .hdf or .h5 files and the corresponding indices, located in `index_dir`
    with the same name, but a .npy extension.
    """
    hdf_ds = HDFDataset(hdf_path)

    return DataLoader(
        hdf_ds,
        batch_size=batch_size,
        shuffle=True
    )