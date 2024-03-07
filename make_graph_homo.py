import numpy as np
import pylab as pl
import os, sys
import psutil

import itertools
import functools

import torch
import torchvision.transforms as transforms
import torch_geometric
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import grid

from WarCraft import *

num_cpus = psutil.cpu_count(logical=True)
data_type = 'Warcraft12x12'
data_dir = '/Users/ascaife/SRC/GITHUB/WarCraft-ShortestPath-Graph/data/'
base_folder = 'warcraft_shortest_path_oneskin/12x12'
batch_size = 2000
with_weights = True

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    # specify the image data:
    normalise= transforms.Normalize((0.2411, 0.2741, 0.1212), (0.1595, 0.0650, 0.1601))
    transform = transforms.Compose([transforms.ToTensor(), normalise])
    dataset = locals()[data_type](data_dir, train=True, transform=transform)

    # create a data loader:
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_cpus-1,
                                              persistent_workers=True
                                              )
         
    # iterate over data batches:
    for batch_idx, (image_batch, weight_batch, path_batch) in enumerate(data_loader):
    
        graphs =[]
    
        # loop through batch:
        for i in range(len(image_batch)):
        
            data = Data()
            
            x_img = image_batch[i].reshape(3, image_batch[i].shape[1]*image_batch[i].shape[2]).T.type(torch.float)
            x_wgt = weight_batch[i].flatten().type(torch.float)
            x_pth = path_batch[i].flatten().type(torch.long)
        
            # make image graph:
            n_side = image_batch[i].shape[1]
            edge_index_img, pos_img = grid(height=n_side, width=n_side)
            
            data.x = x_img
            data.edge_index = edge_index_img
            data.edge_attr = torch.ones(size=(edge_index_img.shape[1],1))
            data.pos = pos_img
            if with_weights:
                data.y = torch.stack((x_wgt, x_pth.type(torch.float)))
            else:
                data.y = x_pth
            
            graphs.append(data)
            
        path = data_dir+base_folder+'/graphs/raw/'+'data_{}.pt'.format(batch_idx)
        torch.save(graphs, path)
