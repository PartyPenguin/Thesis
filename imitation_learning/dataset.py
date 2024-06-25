import h5py
import numpy as np
import torch as th
from tqdm import tqdm
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Data
from sapien.core.pysapien import PinocchioModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

WINDOW_SIZE = 4


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def standardize(data: np.ndarray, scaler: StandardScaler = None) -> np.ndarray:
    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        joblib.dump(scaler, "standard_scaler.pkl")
    else:
        data = scaler.transform(data)

    return data


def normalize(
    data: np.ndarray, scaler: MinMaxScaler = None, mean=None, std=None
) -> np.ndarray:
    # Clip to remove outliers
    # if mean is None:
    #     mean = np.mean(data, axis=0)
    #     joblib.dump(mean, "mean.pkl")
    # if std is None:
    #     std = np.std(data, axis=0)
    #     joblib.dump(std, "std.pkl")

    # data = np.clip(data, mean - 2 * std, mean + 2 * std)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
        data = scaler.fit_transform(data)
        joblib.dump(scaler, "norm_scaler.pkl")
    else:
        data = scaler.transform(data)

    return data


def create_graph(data):
    time_step = data.shape[0]
    nodes = data.shape[1]

    # Initialize the edge index list
    edge_index = [
        (i + (nodes * j), i + (nodes * j) + 1)
        for i in range(nodes - 1)
        for j in range(time_step)
    ]

    edge_index.extend(
        [
            (i + (nodes * j), i + (nodes * j) + nodes)
            for i in range(nodes)
            for j in range(time_step - 1)
        ]
    )

    data = th.reshape(data, (time_step * nodes, -1))
    # Convert to tensor and make the graph undirected
    edge_index = th.tensor(edge_index, dtype=th.long).t().contiguous()
    edge_index = th.cat([edge_index, edge_index[[1, 0]]], dim=-1)

    # Create the graph
    graph = Data(x=data, edge_index=edge_index)

    return graph


def transform_obs(obs):
    """
    The observations that are returned by the environment need to be transformed into a format that can be used by the graph neural network.
    We want the one dimensional observations to be transformed into a NxQxD format where N is the number of observations,
    Q is the number of joints, and D is the number of dimensions for each joint. This way we get a feature vecotor for each joint of the robot.

    Args:
        obs (numpy.ndarray): The input observations.

    Returns:
        numpy.ndarray: The combined features of joint and context information.

    """

    def plot_histograms(features, feature_names, title):
        sample_size = 10000
        num_features = features.shape[1]
        sample_indices = np.random.choice(features.shape[0], sample_size, replace=False)
        plt.figure(figsize=(20, 15))
        for i in range(num_features):
            plt.subplot(4, 5, i + 1)
            sns.histplot(features[sample_indices, i], kde=True)
            plt.title(f"{feature_names[i]} Distribution")
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

    # Extract joint features from observations
    joint_positions = obs[:, :8]  # Joint property | Joint positions 8 dimensions
    joint_velocities = obs[:, 9:17]  # Joint property | Joint velocities 8 dimensions
    joint_se3_pose = obs[:, 39:95]  # Joint property | Joint SE3 pose 63 dimensions

    # Extract context features from observations
    base_pose = obs[:, 18:25]  # Context | Base pose 7 dimensions
    tcp_pose = obs[:, 25:32]  # Context | TCP pose 7 dimensions
    goal_position = obs[:, 32:39]  # Context | Goal position 7 dimensions
    tcp_to_goal_position = obs[
        :, 102:105
    ]  # Context | TCP to goal position 3 dimensions

    # Plot histograms of the features
    # plot_histograms(
    #     joint_positions, [f"Joint Pos {i}" for i in range(8)], "Joint Positions"
    # )
    # plot_histograms(
    #     joint_velocities, [f"Joint Vel {i}" for i in range(8)], "Joint Velocities"
    # )
    # plot_histograms(base_pose, [f"Base Pose {i}" for i in range(7)], "Base Pose")
    # plot_histograms(tcp_pose, [f"TCP Pose {i}" for i in range(7)], "TCP Pose")
    # plot_histograms(goal_position, [f"Goal Pos {i}" for i in range(7)], "Goal Position")
    # plot_histograms(
    #     tcp_to_goal_position,
    #     [f"TCP to Goal Pos {i}" for i in range(3)],
    #     "TCP to Goal Position",
    # )

    # Stack joint positions and velocities along a new axis to create joint features
    # This transforms the observations to a NxQxD format with N being the number of observations,
    # Q being the number of joints, and D being the number of dimensions for each joint.
    joint_features = np.concatenate(
        (
            joint_positions[..., None],
            joint_velocities[..., None],
            joint_se3_pose.reshape(joint_se3_pose.shape[0], 8, -1),
        ),
        axis=2,
    )

    # Concatenate all context features into a single array
    context_info = np.hstack([base_pose, tcp_pose, goal_position, tcp_to_goal_position])

    # Repeat the context information for each joint and reshape to match the shape of joint_features
    context_features = np.repeat(
        context_info[:, np.newaxis, :], joint_features.shape[1], axis=1
    )

    # Concatenate joint features and context features along the last axis
    combined_features = np.concatenate([joint_features, context_features], axis=2)

    return combined_features


class GeometricManiSkill2Dataset(GeometricDataset):
    def __init__(
        self,
        dataset_file: str,
        root,
        load_count=-1,
        transform=None,
        pre_transform=None,
    ):
        super(GeometricManiSkill2Dataset, self).__init__(root, transform, pre_transform)
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        from mani_skill2.utils.io_utils import load_json

        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.episode_steps = [
            episode["elapsed_steps"] for episode in self.json_data["episodes"]
        ]

        self.observations = []
        self.actions = []
        self.episode_map = []
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
            self.episode_map.append(
                np.full(len(trajectory["obs"]) - 1, eps["episode_id"])
            )

        self.observations = normalize(np.vstack(self.observations))
        self.observations = transform_obs(self.observations)

        self.actions = np.vstack(self.actions)
        self.episode_map = np.hstack(self.episode_map)

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
        episode_window = self.episode_map[max(0, idx - WINDOW_SIZE + 1) : idx + 1]

        # Create a mask where the episode number matches the current episode
        mask = episode_window == episode

        # Use the mask to select the corresponding observations and convert them to a PyTorch tensor

        obs = th.from_numpy(
            self.observations[max(0, idx - WINDOW_SIZE + 1) : idx + 1][mask]
        ).float()

        # If the observation tensor is shorter than window_size (because we're at the start of an episode),
        # pad it with zeros at the beginning.
        if obs.shape[0] < WINDOW_SIZE:
            obs = th.cat(
                [
                    th.zeros(WINDOW_SIZE - obs.shape[0], obs.shape[1], obs.shape[2]),
                    obs,
                ],
                dim=0,
            )

        # Return the observation tensor and the action for the current index
        return obs, action

    def close(self):
        self.data.close()
