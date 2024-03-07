from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch_geometric.nn import SplineConv, GCNConv, max_pool, max_pool_x, voxel_grid
import torch_geometric.transforms as T

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dijkstra import ShortestPath, HammingLoss
from utils import exact_match_accuracy, exact_cost_accuracy
from resnet import GCNResnet18

transform = T.Cartesian(cat=False)

# -----------------------------------------------------------------------------

class Baseline(pl.LightningModule):

    """lightning module to reproduce resnet18 baseline"""

    def __init__(self, out_features, in_channels, lr, l1_regconst, lambda_val, neighbourhood_fn):

        super().__init__()

        #self.encoder = Net(in_channels, out_features)
        self.encoder = GCNResnet18(in_channels, out_features)
        
        self.lr = lr

    def training_step(self, batch, batch_idx):

        data = batch # return torch_geometric DataBatch() object
        
        pred_paths = torch.sigmoid(self.encoder(data))
        
        if len(data.y.shape)==3:
            # data.y contains both true weights and true paths
            true_wgts  = data.y[0,:].view(pred_paths.size())
            true_paths = data.y[1,:].view(pred_paths.size())
            
            accuracy = exact_cost_accuracy(true_paths, pred_paths.round(), true_wgts)
            self.log("exact cost accuracy [train]", accuracy)
        else:
            # data.y only contains true paths
            true_paths = data.y.view(pred_paths.size())
            
        criterion = torch.nn.BCELoss()
        loss = criterion(pred_paths, true_paths.to(dtype=torch.float)).mean()
        self.log("train_loss", loss)
        
        accuracy = exact_match_accuracy(true_paths, pred_paths.round())
        self.log("exact match accuracy [train]", accuracy)

        return loss

    def test_step(self, batch, batch_idx):

        data = batch # return torch_geometric DataBatch() object

        pred_paths = torch.sigmoid(self.encoder(data)).round()
        
        if len(data.y.shape)==3:
            # data.y contains both true weights and true paths
            true_wgts  = data.y[0,:].view(pred_paths.size())
            true_paths = data.y[1,:].view(pred_paths.size())
            
            accuracy = exact_cost_accuracy(true_paths, pred_paths, true_wgts)
            self.log("exact cost accuracy [train]", accuracy)
        else:
            # data.y only contains true paths
            true_paths = data.y.view(pred_paths.size())
        
        accuracy = exact_match_accuracy(true_paths, pred_paths)

        # log the test loss / accuracy:
        #self.log("test_loss", loss, sync_dist=True)
        #self.log('exact match accuracy [test]', accuracy, sync_dist=True)
        self.log("test_loss", loss, sync_dist=True, batch_size=data.x.shape[0])
        self.log('exact match accuracy [test]', accuracy, sync_dist=True, batch_size=data.x.shape[0])
        
        # log some example images:
        true_paths = true_paths.reshape(true_paths.shape[0], int(sqrt(true_paths.shape[1])), int(sqrt(true_paths.shape[1])))
        pred_paths = pred_paths.reshape(pred_paths.shape[0], int(sqrt(pred_paths.shape[1])), int(sqrt(pred_paths.shape[1]))).round()
        
        n_images = 3
        #stacked = torch.stack([true_paths, pred_paths])
        #interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)
        #images = [img.type(torch.float) for img in interleaved[:2*n_images]]
        images = [img.type(torch.float) for img in true_paths[:n_images]]
        captions = [f'True path' for i in range(n_images)]
        self.logger.log_image(
                key='true paths',
                images=images,
                caption=captions)
                
        images = [img.type(torch.float) for img in pred_paths[:n_images]]
        captions = [f'Predicted path' for i in range(n_images)]
        self.logger.log_image(
                key='predicted paths',
                images=images,
                caption=captions)
        
        return

    def configure_optimizers(self):

        # should update this at some point to take optimizer from config file
        optimizer    = torch.optim.Adam(self.parameters(), lr=self.lr)

        # learning rate steps specified in https://arxiv.org/pdf/1912.02175.pdf (A.3.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10], gamma=0.1)

        return [optimizer], [lr_scheduler]

# -----------------------------------------------------------------------------

class Net(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.conv1 = GCNConv(in_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(out_features * 64, 128)
        self.fc2 = torch.nn.Linear(128, out_features)

    def forward(self, data):

        # conv 1
        data.edge_attr = None
        data.x = F.elu(self.conv1(data.x, data.edge_index))

        # max pool 1: 96 --> 48
        cluster = voxel_grid(data.pos, batch=data.batch, size=2)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)
        
        # conv 2
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        
        # max pool 2: 48 --> 24
        cluster = voxel_grid(data.pos, batch=data.batch, size=4)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)
        
        # conv 3
        data.x = F.elu(self.conv3(data.x, data.edge_index))

        # max pool 3: 24 --> 12
        cluster = voxel_grid(data.pos, batch=data.batch, size=8)
        x, _ = max_pool_x(cluster, data.x, data.batch, size=144)

        # reshape
        x = x.view(-1, self.fc1.weight.size(1))

        # FC 1
        x = F.elu(self.fc1(x))

        # FC 2
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # output
        return x


# -----------------------------------------------------------------------------

class Combinatorial(pl.LightningModule):

    """lightning module to reproduce resnet18+dijkstra baseline"""

    def __init__(self, out_features, in_channels, lr, l1_regconst, lambda_val, neighbourhood_fn):

        super().__init__()
        
        self.neighbourhood_fn = neighbourhood_fn
        self.lambda_val = lambda_val
        self.l1_regconst = l1_regconst

        #self.encoder = CombNet(out_features, in_channels)
        self.encoder = CombGCNResNet18(out_features, in_channels)
        self.solver = ShortestPath.apply

        self.lr = lr

    def training_step(self, batch, batch_idx):
        
        data = batch # return torch_geometric DataBatch() object
               
        # get the output from the CNN:
        output = self.encoder(data)
        output = torch.abs(output)
        
        weights = output.reshape(output.shape[0], int(sqrt(output.shape[1])), int(sqrt(output.shape[1]))) # reshape to match the path maps
        assert len(weights.shape) == 3, f"{str(weights.shape)}" # double check dimensions
        
        # pass the predicted weights through the dijkstra algorithm:
        pred_paths = self.solver(weights, self.lambda_val, self.neighbourhood_fn) # only positional arguments allowed (no keywords)
        true_paths = data.y.view(pred_paths.size())
        
        # calculate the Hammingloss
        criterion = HammingLoss()
        loss = criterion(pred_paths, true_paths)
        
        # calculate the regularisation:
        l1reg = self.l1_regconst * torch.mean(output)
        loss += l1reg
        
        # calculate the accuracy:
        accuracy = (torch.abs(pred_paths - true_paths) < 0.5).to(torch.float32).mean()

        last_suggestion = {
            "suggested_weights": weights,
            "suggested_path": pred_paths
        }

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss


    def test_step(self, batch, batch_idx):

        data = batch # return torch_geometric DataBatch() object
               
        # get the output from the CNN:
        output = self.encoder(data)
        output = torch.abs(output)
        
        N = int(sqrt(output.shape[1]))

        weights = output.reshape(output.shape[0], N, N) # reshape to match the path maps
        assert len(weights.shape) == 3, f"{str(weights.shape)}" # double check dimensions
        
        # pass the predicted weights through the dijkstra algorithm:
        pred_paths = self.solver(weights, self.lambda_val, self.neighbourhood_fn) # only positional arguments allowed (no keywords)
        
        # flatten paths for accuracy calculations:
        pred_paths = pred_paths.view(pred_paths.size()[0], -1)
        true_paths = data.y.view(pred_paths.size())
        
        accuracy = exact_match_accuracy(true_paths, pred_paths)
        self.log('exact match accuracy [test]', accuracy)

        # log some example images:
        true_paths = true_paths.reshape(true_paths.shape[0], N, N)
        pred_paths = pred_paths.reshape(pred_paths.shape[0], N, N).round()
        
        n_images = 3
        images = [img.type(torch.float) for img in true_paths[:n_images]]
        captions = [f'True path' for i in range(n_images)]
        self.logger.log_image(
                key='true paths',
                images=images,
                caption=captions)
                
        images = [img.type(torch.float) for img in pred_paths[:n_images]]
        captions = [f'Predicted path' for i in range(n_images)]
        self.logger.log_image(
                key='predicted paths',
                images=images,
                caption=captions)

        return


    def configure_optimizers(self):

        # should update this at some point to take optimizer from config file
        optimizer    = torch.optim.Adam(self.parameters(), lr=self.lr)

        # learning rate steps specified in https://arxiv.org/pdf/1912.02175.pdf (A.3.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)

        return [optimizer], [lr_scheduler]


# -----------------------------------------------------------------------------

class CombNet(nn.Module):

    def __init__(self, out_features, in_channels):
    
        super().__init__()
    
        self.model = Net(in_channels, out_features)

    def forward(self, x):
    
        x = self.model(x)
        
        return x


# -----------------------------------------------------------------------------

class CombGCNResNet18(nn.Module):

    def __init__(self, out_features, in_channels):
    
        super().__init__()
    
        self.resnet_model = GCNResnet18(in_channels, out_features)
        self.model = Net(in_channels, out_features)

    def forward(self, data):
    
        data.x = self.resnet_model.conv1(data.x, data.edge_index)
        data.x = self.resnet_model.bn1(data.x)
        data.x = self.resnet_model.relu(data.x)
        
        # max pool: 96 --> 48
        cluster = voxel_grid(data.pos, batch=data.batch, size=2)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)
        
        data = self.resnet_model.layer1(data)
        
        # avg pool: 48 --> 12
        cluster = voxel_grid(data.pos, batch=data.batch, size=8)
        x, _ = avg_pool_x(cluster, data.x, data.batch, size=144)
        print(x.shape)
        stop
        #x = x.mean(dim=1)
        
        return x


# -----------------------------------------------------------------------------
