import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, GCNConv, max_pool, max_pool_x, avg_pool_x, voxel_grid, DeepGCNLayer

transform = T.Cartesian(cat=False)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

class BasicBlockGCN(nn.Module):
    
    def __init__(self, in_features, hidden_features) -> None:
    
        super().__init__()
        
        self.conv1 = GCNConv(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features, elementwise_affine=True)
        self.act1 = nn.ReLU(inplace=True)

        self.layer1 = DeepGCNLayer(self.conv1, self.norm1, self.act1, block='plain', dropout=0.1)
                                 
        conv = GCNConv(hidden_features, hidden_features)
        norm = nn.LayerNorm(hidden_features, elementwise_affine=True)
        act = nn.ReLU(inplace=True)

        self.layer2 = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1)
        
        
    def forward(self, data):
        
        data.x = self.layer1(data.x, data.edge_index)
        data.x = self.layer2(data.x, data.edge_index)
        
        return data
        
        
 # -------------------------------------------------------------------------------

class GCNResnet18(nn.Module):

    def __init__(self, in_features, out_features, layers=[3,4,6,3]) -> None:
    
        super().__init__()

        self.inplanes = 64
        
        self.conv1 = GCNConv(3, self.inplanes)
        self.bn1 = nn.LayerNorm(self.inplanes, elementwise_affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])
        
        self.fc = nn.Linear(512 * out_features, out_features)


    def _make_layer(self, planes, blocks) -> nn.Sequential:
    
        layers = []
        layers.append(
            BasicBlockGCN(
                self.inplanes, planes
                )
        )
        
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                    BasicBlockGCN(self.inplanes, planes
                    )
            )

        return nn.Sequential(*layers)

    
    def _forward_impl(self, data):
        # See note [TorchScript super()]
        
        data.edge_attr = None
        data.x = self.conv1(data.x, data.edge_index)
        data.x = self.bn1(data.x)
        data.x = self.relu(data.x)
        
        # max pool: 96 --> 48
        cluster = voxel_grid(data.pos, batch=data.batch, size=2)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)
        
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = self.layer4(data)
        
        # avg pool: 48 --> 12
        cluster = voxel_grid(data.pos, batch=data.batch, size=8)
        x, _ = avg_pool_x(cluster, data.x, data.batch, size=144)

        # reshape
        x = x.view(-1, self.fc.weight.size(1))

        # FC 1
        x = self.fc(x)

        return x

    def forward(self, data):
        return self._forward_impl(data)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

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
