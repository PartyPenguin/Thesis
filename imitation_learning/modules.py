import torch as th
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, GATConv, RGATConv
from torch_geometric.nn import Linear
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import MeanAggregation


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
        self.gat_conv1 = GATConv(obs_dims, 64, heads=8, edge_dim=1, dropout=0.3)
        self.gat_conv2 = GATConv(64 * 8, 64, heads=8, edge_dim=1, dropout=0.3)
        self.gat_conv3 = GATConv(64 * 8, 64, heads=8, edge_dim=1, dropout=0.3)

        # Define the linear layer
        self.lin = Linear(64 * 8, act_dims)

    def forward(self, x, edge_index, edge_attr, batch):

        # Apply the GAT layers
        x = self.gat_conv1(x, edge_index, edge_attr).relu()
        x = self.gat_conv2(x, edge_index, edge_attr).relu()
        x = self.gat_conv3(x, edge_index, edge_attr).relu()

        # Apply the linear layer
        x = self.lin(x)

        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        return x


class HGATPolicy(nn.Module):
    def __init__(self, hidden_dim, act_dims):
        super().__init__()

        # Define the GAT layers
        self.gat_conv1 = GATConv(
            in_channels=(-1, -1),
            out_channels=64,
            heads=8,
            edge_dim=-1,
            add_self_loops=False,
        )
        self.gat_conv2 = GATConv(
            in_channels=(-1, -1),
            out_channels=64,
            heads=8,
            edge_dim=-1,
            add_self_loops=False,
        )
        self.gat_conv3 = GATConv(
            in_channels=(-1, -1),
            out_channels=64,
            heads=8,
            edge_dim=-1,
            add_self_loops=False,
        )
        # Define the linear layer
        self.lin = Linear(-1, act_dims)
        self.global_pool = MeanAggregation()

    def forward(self, x, edge_index, edge_attr, batch):

        # Apply the GAT layers
        x = self.gat_conv1(x, edge_index, edge_attr).relu()
        x = self.gat_conv2(x, edge_index, edge_attr).relu()
        x = self.gat_conv3(x, edge_index, edge_attr).relu()

        # Apply the linear layer
        x = self.lin(x)

        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)
        # Apply global mean pooling
        x = self.global_pool(x, batch)

        return x


class RGATPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()

        # Define the RGAT layers
        self.rgat_conv1 = RGATConv(
            obs_dims,
            obs_dims,
            num_relations=3,
            heads=8,
            concat=True,
            edge_dim=1,
            dropout=0.5,
        )
        self.rgat_conv2 = RGATConv(
            obs_dims * 8,
            obs_dims,
            num_relations=3,
            heads=8,
            concat=True,
            edge_dim=1,
            dropout=0.5,
        )
        self.rgat_conv3 = RGATConv(
            obs_dims * 8,
            obs_dims,
            num_relations=3,
            heads=8,
            concat=True,
            edge_dim=1,
            dropout=0.5,
        )

        # Define the linear layer
        self.lin = Linear(obs_dims * 8, act_dims)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
            data.batch_dict,
        )

        # Apply the RGAT layers
        x = self.rgat_conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = self.rgat_conv2(x, edge_index, edge_attr=edge_attr).relu()
        x = self.rgat_conv3(x, edge_index, edge_attr=edge_attr).relu()

        # Apply the linear layer
        x = self.lin(x)

        # Apply the tanh activation function because the actions are in the range [-1, 1]
        x = th.tanh(x)

        # Apply global mean pooling
        x = global_mean_pool(x, batch)

        return x
