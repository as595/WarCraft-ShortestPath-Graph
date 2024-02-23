import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs, read_tu_data

import os
import os.path
import numpy as np
import sys

import torch
#import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

class Warcraft12x12(InMemoryDataset):

    """
    Dataset class for graph representation of Warcraft images from Vlastelica+ 20
    """

    images_list = [
                  ['image_data_0.pt', '3d30dca38bd9ec111dc7460be97d9381'],
                  ['image_data_1.pt', 'a6b1f0f9bcf04876d7ca394d39236183'],
                  ['image_data_2.pt', '35d884b080140344723bf7932aabb9f9'],
                  ['image_data_3.pt', 'cfb4805614a454e0942c01d553bff431'],
                  ['image_data_4.pt', '7e64d5252c1f14928583e43245c82cce'],
                ]

    weight_list = [
                  ['weight_data_0.pt', '776bcba4b723f30af189c65cd626c826'],
                  ['weight_data_1.pt', 'a09285c6acf45447ccf34ddc971f3268'],
                  ['weight_data_2.pt', '028a543997240c525193d62cacd1ec48'],
                  ['weight_data_3.pt', '9cea54b132db78715149151d76673021'],
                  ['weight_data_4.pt', 'ae3ebdc13bbc8358e42c987beb2b7afb'],
                ]

    paths_list  = [
                  ['paths_data_0.pt', '7f434470a5611b11adff479a725d0916'],
                  ['paths_data_1.pt', '73f8321b58ae2b2cd4265ac101200805'],
                  ['paths_data_2.pt', '06cadd05456dbcfdcbbb24aa0bd3f8c2'],
                  ['paths_data_3.pt', '77130246f0e0cab2ceca83a13529fa1a'],
                  ['paths_data_4.pt', 'b9c9d6a64a8fd16e5e36c70785729d4c'],
                ]
                
    data_list   = [
                  ['data_0.pt', '7f434470a5611b11adff479a725d0916'],
                  ['data_1.pt', '73f8321b58ae2b2cd4265ac101200805'],
                  ['data_2.pt', '06cadd05456dbcfdcbbb24aa0bd3f8c2'],
                  ['data_3.pt', '77130246f0e0cab2ceca83a13529fa1a'],
                  ['data_4.pt', 'b9c9d6a64a8fd16e5e36c70785729d4c'],

                ]
                
    base_folder = 'warcraft_shortest_path_oneskin/12x12/graphs/'
        
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.base_folder, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.base_folder, 'processed')

    @property
    def raw_file_names(self):
        return [file for file, checksum in self.data_list]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        data_paths = [path for path in self.raw_paths]
        data_list = [torch.load(path) for path in data_paths]
        data_list = [file for list in data_list for file in list]
                
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])


# -----------------------------------------------------------------------------------------------------------------

