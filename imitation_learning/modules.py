import torch as th
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import Batch

device = "cuda" if th.cuda.is_available() else "cpu"


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x: (batch_size, time_steps, *input_size)
        batch_size, time_steps = x.shape[:2]

        # Reshape input to (batch_size * time_steps, *input_size)
        x_reshape = x.contiguous().view(-1, *x.shape[2:])

        # Apply the module to the reshaped input
        y = self.module(x_reshape)

        # Reshape the output back to (batch_size, time_steps, *output_size)
        y = y.contiguous().view(batch_size, time_steps, *y.shape[1:])

        return y


class GCNPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()

        # Define the GCN layers
        self.gcn_conv1 = GCNConv(obs_dims, 128)
        self.gcn_conv2 = GCNConv(128, 128)
        self.gcn_conv3 = GCNConv(128, 128)

        # Define the linear layer
        self.lin = Linear(128, act_dims)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the GCN layers
        x = self.gcn_conv1(x, edge_index).relu()
        x = self.gcn_conv2(x, edge_index).relu()
        x = self.gcn_conv3(x, edge_index).relu()

        # Apply the linear layer
        x = self.lin(x)

        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        return x


class GATPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()

        # Define the GAT layers
        self.gat_conv1 = GATConv(obs_dims, 128)
        self.gat_conv2 = GATConv(128, 128)
        self.gat_conv3 = GATConv(128, 128)

        # Define the linear layer
        self.lin = Linear(128, act_dims)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the GAT layers
        x = self.gat_conv1(x, edge_index).relu()
        x = self.gat_conv2(x, edge_index).relu()
        x = self.gat_conv3(x, edge_index).relu()

        # Apply the linear layer
        x = self.lin(x)

        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        return x


class STGCNLayer(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, time_window):
        super(STGCNLayer, self).__init__()
        self.time_window = time_window
        # Spatial graph convolution layer
        self.spatial_conv = GCNConv(in_channels, spatial_channels)

        # Temporal convolution layer
        self.temporal_conv = nn.Conv2d(
            spatial_channels, out_channels, (1, 3), padding=(0, 1)
        )

        # Additional layer to map to the desired output channels
        self.out_conv = nn.Conv2d(out_channels, out_channels, (1, 1))

        # Batch normalization and activation
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch_size, num_nodes):

        # Apply spatial GCN across all nodes and time steps
        x = self.spatial_conv(x, edge_index)

        # Reshape to [batch_size, num_nodes, time_window, spatial_channels]
        x = x.view(batch_size, self.time_window, num_nodes, -1)
        x = x.permute(
            0, 3, 2, 1
        )  # [batch_size, spatial_channels, num_nodes, time_window]

        # Apply temporal convolution
        x = self.temporal_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        # Additional convolution to map to the output channels
        x = self.out_conv(x).squeeze()

        # Activation
        x = th.tanh(x)

        # Reshape to [batch_size * time_window * num_nodes, spatial_channels]
        x = x.view(-1, x.shape[1])

        return x


class STGCN(nn.Module):
    def __init__(self, node_features, spatial_features, time_window):
        super(STGCN, self).__init__()
        self.time_window = time_window
        self.spatial_features = spatial_features
        self.stgcn_layers = nn.Sequential(
            STGCNLayer(node_features, spatial_features, spatial_features, time_window),
            STGCNLayer(
                spatial_features, spatial_features, spatial_features, time_window
            ),
        )
        self.regressor = nn.Linear(spatial_features, 8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = 128
        num_nodes = 8

        x = self.stgcn_layers[0](x, edge_index, batch_size, num_nodes)
        x = self.stgcn_layers[1](x, edge_index, batch_size, num_nodes)

        x = self.regressor(x)

        x = x.view(batch_size, self.time_window, num_nodes, -1)

        x = x.mean(dim=(1, 3))

        return x
