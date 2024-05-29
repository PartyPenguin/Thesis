# Import required packages
import argparse
import os.path as osp
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch as th
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, RGCNConv, GATConv, SAGEConv, Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import to_hetero
from torch_geometric.nn import MeanAggregation
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mani_skill2.envs

import pytorch_kinematics as pk

from mani_skill2.utils.wrappers import RecordEpisode

device = "cuda" if th.cuda.is_available() else "cpu"


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def build_graph(obs, action):
    q_pos = obs[:8]  # Joint positions 8 dimensions
    q_vel = obs[9:17]  # Joint velocities 8 dimensions
    base_pose = obs[18:25]  # Base pose 7 dimensions
    tcp_pose = obs[25:32]  # TCP pose 7 dimensions
    goal_pos = obs[32:35]  # Goal position 3 dimensions
    tcp_to_goal_pos = obs[35:38]  # TCP to goal position 3 dimensions

    # Build a graph based on the observations. Each joint of the robot is a node in the graph.
    # The edges are the connections between the joints. The features of the nodes are the joint positions
    # and velocities as well as the base pose, TCP pose, goal position, and TCP to goal position.

    x = []
    for i in range(q_pos.shape[0]):
        x.append(
            th.cat(
                [
                    q_pos[i].unsqueeze(0),
                    q_vel[i].unsqueeze(0),
                    goal_pos,
                    tcp_to_goal_pos,
                    base_pose,
                    tcp_pose,
                ],
                dim=0,
            )
        )
    x = th.stack(x, dim=0)
    edge_index = []
    for i in range(x.shape[0] - 1):
        edge_index.append([i, i + 1])

    # Add Skip connections between joint 2 and join 7 and joint 3 and joint 6
    edge_index.append([1, 6])
    edge_index.append([2, 5])
    edge_index = th.tensor(edge_index, dtype=th.long).t().contiguous()
    # Make graph undirected
    edge_index = th.cat([edge_index, edge_index[[1, 0]]], dim=-1)

    data = Data(x=x, y=action.unsqueeze(0), edge_index=edge_index)
    # import networkx as nx
    # import matplotlib.pyplot as plt

    # g = to_networkx(data)
    # nx.draw(g)
    # plt.show()
    return data


class GeometricManiSkill2Dataset(GeometricDataset):
    def __init__(
        self, dataset_file: str, root, load_count=-1, transform=None, pre_transform=None
    ):
        super(GeometricManiSkill2Dataset, self).__init__(root, transform, pre_transform)
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        import h5py
        from mani_skill2.utils.io_utils import load_json

        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.total_frames = 0
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
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)

    def len(self):
        return len(self.observations)

    def get(self, idx):
        obs = th.from_numpy(self.observations[idx]).float()
        action = th.from_numpy(self.actions[idx]).float()
        return build_graph(obs, action)

    def close(self):
        self.data.close()


class GCNPolicy(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super().__init__()
        self.conv1 = GCNConv(obs_dims, 1024)
        self.conv2 = GCNConv(1024, 1024)
        self.lin = Linear(1024, act_dims)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = self.lin(x)
        x = x.tanh()
        x = global_mean_pool(x, data.batch)

        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "Simple script demonstrating how to train an agent with imitation learning (behavior cloning) using ManiSkill2 environmnets and demonstrations"
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-d", "--demos", type=str, help="path to demonstration dataset .h5py file"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/bc_state",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--steps", type=int, help="number of training steps", default=30000
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args


def load_data(path):
    dataset = GeometricManiSkill2Dataset(path, root="")
    dataloader = GeometricDataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    dataset.close()
    return dataloader, dataset


def main():
    args = parse_args()
    env_id = args.env_id
    demo_path = args.demos
    log_dir = args.log_dir
    iterations = args.steps

    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)

    ckpt_dir = osp.join(log_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    obs_mode = "state"
    control_mode = "pd_joint_delta_pos"
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="cameras",
    )
    if args.eval:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "checkpoints/ckpt_latest.pt")
        # Load the saved model
        policy = th.load(model_path)
    else:
        assert (
            demo_path is not None
        ), "Need to provide a demonstration dataset via --demos"
        dataloader, dataset = load_data(demo_path)
        data: Data = dataset[0]
        # create our policy
        policy = GCNPolicy(data.x.shape[1], data.y.shape[1])
    # move model to gpu if possible
    policy = policy.to(device)
    print(policy)

    loss_fn = nn.HuberLoss()

    # a short save function to save our model
    def save_model(policy, path):
        th.save(policy, path)

    def train_step(policy, data, optim, loss_fn):
        optim.zero_grad()
        # move data to appropriate device first
        data = data.to(device)

        pred_actions = policy(data)

        # compute loss and optimize
        # L1 regularization
        l1_lambda = 0.0001
        l1_norm = sum(p.abs().sum() for p in policy.parameters())

        # L2 regularization
        l2_lambda = 0.0001
        l2_norm = sum(p.pow(2.0).sum() for p in policy.parameters())

        # compute loss and optimize
        loss = loss_fn(data.y, pred_actions)
        # loss = loss_fn(pred_ee_pos, ee_pos)
        loss.backward()
        optim.step()
        return loss.item()

    def evaluate_policy(env, policy, num_episodes=10):
        obs, _ = env.reset()
        successes = []
        i = 0
        pbar = tqdm(total=num_episodes, leave=False)
        while i < num_episodes:
            graph = build_graph(th.from_numpy(obs).float(), th.zeros(8))
            # move to appropriate device and unsqueeze to add a batch dimension
            obs_device = graph.to(device)
            with th.no_grad():
                action = policy(obs_device).squeeze().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes.append(info["success"])
                i += 1
                obs, _ = env.reset(seed=i)
                pbar.update(1)
        success_rate = np.mean(successes)
        return success_rate

    if not args.eval:
        writer = SummaryWriter(log_dir)
        optim = th.optim.Adam(policy.parameters(), lr=1e-3)
        best_epoch_loss = np.inf
        epoch = 0
        steps = 0
        dagger_iter = 0
        pbar = tqdm(dataloader, total=iterations)
        # RecordEpisode wrapper auto records a new video once an episode is completed
        env = RecordEpisode(
            env, output_dir=osp.join(log_dir, "videos"), info_on_video=True
        )
        while steps < iterations:
            epoch_loss = 0
            for batch in dataloader:
                steps += 1
                # Add noise to the actions to make the policy more robust
                batch.x += th.randn_like(batch.x) * 0.05
                loss_val = train_step(policy, batch, optim, loss_fn)

                # track the loss and print it
                writer.add_scalar("train/mse_loss", loss_val, steps)
                epoch_loss += loss_val
                pbar.set_postfix(dict(loss=loss_val))
                pbar.update(1)

                # periodically save the policy
                if steps % 2000 == 0:
                    save_model(policy, osp.join(ckpt_dir, f"ckpt_{steps}.pt"))
                if steps >= iterations:
                    break

            epoch_loss = epoch_loss / len(dataloader)

            # save a new model if the average MSE loss in an epoch has improved
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))
            if epoch % 100 == 0:
                print("Evaluating")
                success_rate = evaluate_policy(env, policy)
                writer.add_scalar("test/success_rate", success_rate, epoch)
            writer.add_scalar("train/mse_loss_epoch", epoch_loss, epoch)
            epoch += 1
        save_model(policy, osp.join(ckpt_dir, "ckpt_latest.pt"))

    # run a final evaluation
    success_rate = evaluate_policy(env, policy)
    print(f"Final Success Rate {success_rate}")


if __name__ == "__main__":
    main()
