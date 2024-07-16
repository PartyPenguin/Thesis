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

WINDOW_SIZE = 4


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


def fourier_encode(data: th.Tensor) -> th.Tensor:
    scales = th.tensor([-1, 0, 1, 2, 3, 4, 5, 6], dtype=data.dtype, device=data.device)
    scaled_data = data.reshape(-1, 1) / (3.0**scales).reshape(1, -1)

    sin_features = th.sin(scaled_data)
    cos_features = th.cos(scaled_data)

    features = th.cat([sin_features, cos_features], dim=1)
    return features.reshape(data.shape[0], -1)


def draw_hetero_graph(data):
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx
    from pyvis.network import Network

    graph = to_networkx(data, to_undirected=False)

    # Create a pyvis network
    net = Network(notebook=True)

    # Define colors for nodes and edges
    node_type_colors = {
        "joint": "#4599C3",
        "tcp": "#ED8546",
        "goal": "#70B349",
        "object": "#8B4D9E",
    }

    edge_type_colors = {
        ("joint", "joint_connects_joint", "joint"): "#8B4D9E",
        ("joint", "joint_follows_joint", "joint"): "#DFB825",
        ("tcp", "tcp_to_joint", "joint"): "#70B349",
        ("tcp", "tcp_follows_tcp", "tcp"): "#DB5C64",
        ("goal", "goal_to_tcp", "tcp"): "#4599C3",
        ("goal", "goal_follows_goal", "goal"): "#ED8546",
        ("object", "object_to_tcp", "tcp"): "#DB5C64",
        ("object", "object_follows_object", "object"): "#8B4D9E",
        ("object", "object_to_goal", "goal"): "#70B349",
    }

    # Add nodes to the pyvis network
    for node, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        color = node_type_colors.get(
            node_type, "#000000"
        )  # default to black if type not found
        net.add_node(node, label=f"{node_type[:1].upper()}{node}", color=color)

    # Add edges to the pyvis network
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        color = edge_type_colors.get(
            edge_type, "#000000"
        )  # default to black if type not found
        net.add_edge(from_node, to_node, color=color)

    net.show_buttons(filter_=["physics"])
    # Show the network
    net.show("hetero_graph.html")


def create_heterogeneous_graph(data: th.Tensor):
    """
    Create a heterogeneous graph from the data. The graph should contain the following node types:
    - Joint nodes
    - Object nodes
    """
    time_step = data.shape[0]
    nodes = data.shape[1]
    graph = HeteroData()

    # ================== Create the joint nodes ==================

    # Create the joint nodes (time_step * nodes, 16)
    joint_nodes = th.reshape(data, (time_step * nodes, -1))
    graph["joint"].x = fourier_encode(joint_nodes)

    # Create the TCP node (time_step, 7)
    # tcp_node = th.reshape(data[:, 0, 16:23], (time_step, -1))
    # graph["tcp"].x = fourier_encode(tcp_node)

    # Create the goal node (time_step, 3)
    # goal_node = th.reshape(data[:, 0, 23:26], (time_step, -1))
    # graph["goal"].x = fourier_encode(goal_node)

    # # Create the object node (time_step, 7)
    # object_node = th.reshape(data[:, 0, 29:36], (time_step, -1))
    # graph["object"].x = fourier_encode(object_node)

    # ================== Create the edges ==================

    # Joint Edges
    # ----------------

    # Create the kinematic chain edges between the joints of the robot for each time step
    link_edge_index = [
        (i + (nodes * j), i + (nodes * j) + 1)
        for j in range(time_step)
        for i in range(nodes - 1)
    ]

    # Create the temporal edges between the current time step and the previous time step for each joint
    temporal_link_edge_index = [
        (i, i + (nodes * j) + nodes) for j in range(time_step - 1) for i in range(nodes)
    ]

    # Edge attributes

    # Edge attributes for the SE3 joint distances between the joints. Only applicable for the kinematic chain edges
    se3_joint_dist = th.linalg.norm(th.diff(data[:, :, 2:5], axis=1), axis=2)
    se3_joint_dist_reshape = se3_joint_dist.reshape(-1, 1)

    graph["joint", "joint_connects_joint", "joint"].edge_index = (
        th.tensor(link_edge_index).t().contiguous()
    )
    graph["joint", "joint_follows_joint", "joint"].edge_index = (
        (th.tensor(temporal_link_edge_index).t().contiguous())
        if time_step > 1
        else None
    )

    graph["joint", "joint_connects_joint", "joint"].edge_attr = fourier_encode(
        se3_joint_dist_reshape
    )

    # # TCP Edges
    # # ----------------

    # # Create the edges between the joints and the TCP at each time step
    # tcp_edge_index = [
    #     (j, i + (nodes * j)) for j in range(time_step) for i in range(nodes)
    # ]

    # # Create the temporal edges between the TCP and the previous TCP
    # temporal_tcp_edge_index = [(0, i + 1) for i in range(time_step - 1)]

    # # Edge attributes
    # # Edge attributes for the SE3 joint distances between the joints and the TCP. Only applicable for the tcp edges
    # se3_tcp_dist = th.linalg.norm(data[:, :, 2:5] - data[:, :, 16:19], axis=2)
    # se3_tcp_dist_reshape = se3_tcp_dist.reshape(-1, 1)

    # graph["tcp", "tcp_to_joint", "joint"].edge_index = (
    #     th.tensor(tcp_edge_index).t().contiguous()
    # )
    # graph["tcp", "tcp_follows_tcp", "tcp"].edge_index = (
    #     (th.tensor(temporal_tcp_edge_index).t().contiguous()) if time_step > 1 else None
    # )

    # graph["tcp", "tcp_to_joint", "joint"].edge_attr = fourier_encode(
    #     se3_tcp_dist_reshape
    # )

    # Goal Edges
    # ----------------

    # Create the edges between the nodes and the goal at each time step
    goal_edge_index = [
        (j, i + (nodes * j)) for j in range(time_step) for i in range(nodes)
    ]

    # Create the temporal edges between the goal and the previous goals
    temporal_goal_edge_index = [(0, i + 1) for i in range(time_step - 1)]

    # Edge attributes
    tcp_to_goal_dist = (
        th.linalg.norm(data[:, 0, 26:29], axis=1).unsqueeze(1).repeat(1, nodes)
    )
    tcp_to_goal_dist_reshape = tcp_to_goal_dist.reshape(-1, 1)

    # graph["goal", "goal_to_tcp", "tcp"].edge_index = (
    #     th.tensor(goal_edge_index).t().contiguous()
    # )

    # graph["goal", "goal_to_joint", "joint"].edge_index = (
    #     th.tensor(goal_edge_index).t().contiguous()
    # )

    # graph["goal", "goal_follows_goal", "goal"].edge_index = (
    #     (th.tensor(temporal_goal_edge_index).t().contiguous())
    #     if time_step > 1
    #     else None
    # )

    # graph["goal", "goal_to_joint", "joint"].edge_attr = fourier_encode(
    #     tcp_to_goal_dist_reshape
    # )
    # Object Edges
    # ----------------

    # Create the edges between the joints and the object at each time step
    object_edge_index = [
        (j, i + (nodes * j)) for j in range(time_step) for i in range(nodes)
    ]

    # Create the temporal edges between the object and the previous objects
    temporal_object_edge_index = [(0, i + 1) for i in range(time_step - 1)]

    # Create the edges between the obkect and the goal at each time step
    object_goal_edge_index = [(i, i) for i in range(time_step)]

    # Edge attributes
    tcp_to_obj_dist = (
        th.linalg.norm(data[:, 0, 36:39], axis=1).unsqueeze(1).repeat(1, nodes)
    )
    tcp_to_obj_dist_reshape = tcp_to_obj_dist.reshape(-1, 1)

    obj_to_goal_dist = th.linalg.norm(data[:, 0, 39:42], axis=1)
    obj_to_goal_dist_reshape = obj_to_goal_dist.reshape(-1, 1)

    # graph["object", "object_to_joint", "joint"].edge_index = (
    #     th.tensor(object_edge_index).t().contiguous()
    # )
    # graph["object", "object_to_goal", "goal"].edge_index = (
    #     th.tensor(object_goal_edge_index).t().contiguous()
    # )
    # graph["object", "object_follows_object", "object"].edge_index = (
    #     (th.tensor(temporal_object_edge_index).t().contiguous())
    #     if time_step > 1
    #     else None
    # )

    # graph["object", "object_to_joint", "joint"].edge_attr = fourier_encode(
    #     tcp_to_obj_dist_reshape
    # )
    # graph["object", "object_to_goal", "goal"].edge_attr = fourier_encode(
    #     obj_to_goal_dist_reshape
    # )

    graph = T.ToUndirected()(graph)

    # draw_hetero_graph(graph)

    return graph


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


def transform_obs(obs, pinocchio_model: PinocchioModel):
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

    def compute_joint_se3_pose(joint_positions, base_pose):
        joint_se3_pose = []
        for i in range(joint_positions.shape[0]):
            pinocchio_model.compute_forward_kinematics(joint_positions[i])

            pose = np.asarray(
                [
                    vectorize_pose(pinocchio_model.get_link_pose(j))
                    for j in range(joint_positions.shape[1] - 1)
                ]
            ).flatten()
            pose[:3] = pose[:3] - base_pose[i, :3]
            joint_se3_pose.append(pose)

        return np.array(joint_se3_pose)

    # Extract joint features from observations
    joint_positions = obs[:, :9]  # Joint property | Joint positions 8 dimensions
    joint_velocities = obs[:, 9:17]  # Joint property | Joint velocities 8 dimensions

    # Extract context features from observations
    base_pose = obs[:, 18:25]  # Context | Base pose 7 dimensions
    tcp_pose = obs[:, 25:32]  # Context | TCP pose 7 dimensions
    goal_position = obs[:, 32:35]  # Context | Goal position 7 dimensions
    tcp_to_goal_position = obs[:, 35:38]  # Context | TCP to goal position 3 dimensions
    obj_pose = obs[:, 38:45]  # Context | Object pose 7 dimensions
    tcp_to_obj_pos = obs[:, 45:48]  # Context | TCP to object position 3 dimensions
    obj_to_goal_pos = obs[:, 48:51]  # Context | Object to goal position 3 dimensions
    # peg_pose = obs[:, 33:40]  # Context | Peg pose 7 dimensions
    # peg_half_size = obs[:, 40:43]  # Context | Peg half size 3 dimensions
    # box_hole_pose = obs[:, 43:50]  # Context | Box hole pose 7 dimensions
    # box_hole_radius = obs[:, 50:51]  # Context | Box hole radius 1 dimension

    joint_se3_pose = compute_joint_se3_pose(joint_positions, base_pose)

    joint_se3_reshape = joint_se3_pose.reshape(joint_se3_pose.shape[0], 8, -1)

    # Stack joint positions and velocities along a new axis to create joint features
    # This transforms the observations to a NxQxD format with N being the number of observations,
    # Q being the number of joints, and D being the number of dimensions for each joint.
    joint_features = np.concatenate(
        (joint_positions[:, :-1, None], joint_velocities[..., None], joint_se3_reshape),
        axis=2,
    )

    # Adjust base_pose offset
    # tcp_pose[:, :3] = tcp_pose[:, :3] - base_pose[:, :3]
    # goal_position[:, :3] = goal_position - base_pose[:, :3]
    # obj_pose[:, :3] = obj_pose[:, :3] - base_pose[:, :3]

    # Concatenate all context features into a single array
    context_info = np.hstack(
        [
            # base_pose,
            tcp_pose,
            goal_position,
            tcp_to_goal_position,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_goal_pos,
        ]
    )

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
        config,
        root,
        env: BaseEnv,
        load_count=-1,
        transform=None,
        pre_transform=None,
    ):
        super(GeometricManiSkill2Dataset, self).__init__(root, transform, pre_transform)
        self.actions = np.load(config["prepared_graph_data_path"] + "act.npy")
        self.observations = np.load(config["prepared_graph_data_path"] + "obs.npy")
        self.episode_map = np.load(
            config["prepared_graph_data_path"] + "episode_map.npy"
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
        return create_graph(obs), obs, action


class ManiSkill2Dataset(Dataset):
    def __init__(self, config, root, env: BaseEnv, transform=None, pre_transform=None):
        self.config = config
        self.env = env
        self.transform = transform
        self.pre_transform = pre_transform
        self.actions = np.load(config["prepared_mlp_data_path"] + "act.npy")
        self.observations = np.load(config["prepared_mlp_data_path"] + "obs.npy")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]
