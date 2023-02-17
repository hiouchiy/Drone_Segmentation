"""Drone Image Shard Descriptor."""

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class DroneDataset(ShardDataset):
    """TinyImageNet shard dataset class."""

    def __init__(self, data_folder: Path, mask_folder: Path, X):
        """Initialize TinyImageNetDataset."""
        self.img_path = data_folder
        self.mask_path = mask_folder
        self.X = X

    def __len__(self) -> int:
        """Return the len of the shard dataset."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Return an item by the index."""
        img_file_path = os.path.join(self.img_path, self.X[index] + '.jpg')
        mask_file_path = os.path.join(self.mask_path, self.X[index] + '.png')
        
        img = self.read_image(img_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.read_image(mask_file_path)

        return img, mask

    def read_image(self, path: Path):
        """Read the image."""
        img = cv2.imread(path)
        return img


class DroneShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(
            self,
            data_folder: str = 'Custum_Dataset/Car/JPEGImages',
            mask_folder: str = 'Custum_Dataset/Car/SegmentationClass',
            **kwargs
    ):
        """Initialize TinyImageNetShardDescriptor."""
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        
        self.df = self.create_df()
        X_trainval, X_test = train_test_split(self.df['id'].values, test_size=0.1, random_state=19)
        self.X_train, self.X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)
        print('Train Size   : ', len(self.X_train))
        print('Val Size     : ', len(self.X_val))
        print('Test Size    : ', len(X_test))

    def create_df(self):
        name = []
        for dirname, _, filenames in os.walk(self.data_folder):
            for filename in filenames:
                name.append(filename.split('.')[0])

        return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

    def get_dataset(self, dataset_type):
        """Return a shard dataset by type."""
        if dataset_type == 'train':
            return DroneDataset(
                data_folder=self.data_folder,
                mask_folder=self.mask_folder,
                X=self.X_train
            )
        else:
            return DroneDataset(
                data_folder=self.data_folder,
                mask_folder=self.mask_folder,
                X=self.X_val
            )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['704', '1056', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['704', '1056', '3']

    @property
    def dataset_description(self) -> str:
        """Return the shard dataset description."""
        return (f'Drone Image Dataset dataset, shard number')

if __name__ == '__main__':
    from openfl.interface.cli import setup_logging

    setup_logging()

    data_folder = 'data'
    rank_worldsize = '1,100'
    enforce_image_hw = '529,622'

    sd = DroneShardDescriptor(
        data_folder="Custum_Dataset/Car/JPEGImages",
        mask_folder="Custum_Dataset/Car/SegmentationClass")
    
    ds = sd.get_dataset('val')
    print(ds[0])
    print(len(ds))
