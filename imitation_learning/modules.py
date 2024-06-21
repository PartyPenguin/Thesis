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
