import h5py
import numpy as np
import torch as th
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx
from sapien.core.pysapien import PinocchioModel
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.envs.sapien_env import BaseEnv
import torch_geometric.transforms as T
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def create_graph(data):
    time_step = data.shape[0]
    nodes = data.shape[1]

    # Initialize the edge index list
    # Create kinematic chain edges between the joints of the robot for each time step
    edge_index = [
        (i + (nodes * j), i + (nodes * j) + 1)
        for j in range(time_step)
        for i in range(nodes - 1)
    ]

    # Create temporal edges between the current time step and the previous time step for each joint
    edge_index.extend(
        [
            (i, i + (nodes * j) + nodes)
            for j in range(time_step - 1)
            for i in range(nodes)
        ]
    )

    edge_index = th.tensor(edge_index).t().contiguous()

    # Edge attributes for the SE3 joint distances between the joints. Only applicable for the kinematic chain edges
    se3_joint_dist = th.linalg.norm(th.diff(data[:, :, 2:5], axis=1), axis=2)
    edge_attr = se3_joint_dist.reshape(-1, 1)

    # Edge attributes for the temporal edges
    temporal_edge_attr = th.tensor(
        np.zeros((nodes * (time_step - 1), edge_attr.shape[1])), dtype=th.float32
    ).to(edge_attr.device)

    # Concatenate the edge attributes
    edge_attr = th.cat([edge_attr, temporal_edge_attr], dim=0)

    data = th.reshape(data, (time_step * nodes, -1))
    # Create the graph
    graph = Data(x=data, edge_index=edge_index, edge_attr=edge_attr)
    graph = T.ToUndirected()(graph)

    return graph


class GeometricManiSkill2Dataset(GeometricDataset):
    def __init__(
        self,
        config,
        root,
        env: BaseEnv,
        transform=None,
        pre_transform=None,
    ):
        self.window_size = config["prepare"]["window_size"]
        super(GeometricManiSkill2Dataset, self).__init__(root, transform, pre_transform)
        self.actions = np.load(
            config["prepare"]["prepared_graph_data_path"] + "act.npy"
        )
        self.observations = np.load(
            config["prepare"]["prepared_graph_data_path"] + "obs.npy"
        )
        self.episode_map = np.load(
            config["prepare"]["prepared_graph_data_path"] + "episode_map.npy"
        )

    def len(self):
        return len(self.observations)

    def get(self, idx):
        # Get the action for the current index and convert it to a PyTorch tensor
        action = th.from_numpy(self.actions[idx]).float()

        # Get the episode number for the current index
        episode = self.episode_map[idx]

        # We want to create a sliding window of observations of size window_size.
        # The window should start at idx-window_size and end at idx.
        # The observations must be from the same episode.

        # Create the window of episode numbers
        episode_window = self.episode_map[max(0, idx - self.window_size + 1) : idx + 1]

        # Create a mask where the episode number matches the current episode
        mask = episode_window == episode

        # Use the mask to select the corresponding observations and convert them to a PyTorch tensor

        obs = th.from_numpy(
            self.observations[max(0, idx - self.window_size + 1) : idx + 1][mask]
        ).float()

        # If the observation tensor is shorter than window_size (because we're at the start of an episode),
        # pad it with zeros at the beginning.
        if obs.shape[0] < self.window_size:
            obs = th.cat(
                [
                    th.zeros(
                        self.window_size - obs.shape[0], obs.shape[1], obs.shape[2]
                    ),
                    obs,
                ],
                dim=0,
            )

        # Return the observation tensor and the action for the current index
        return create_graph(obs), obs, action


class ManiSkill2Dataset(Dataset):
    def __init__(self, config, root, env: BaseEnv, transform=None, pre_transform=None):
        self.config = config
        self.env = env
        self.transform = transform
        self.pre_transform = pre_transform
        self.actions = th.from_numpy(
            np.load(config["prepared_mlp_data_path"] + "act.npy")
        ).float()
        self.observations = th.from_numpy(
            np.load(config["prepared_mlp_data_path"] + "obs.npy")
        ).float()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]
