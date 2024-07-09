# %%
# # Standard library imports
import argparse
import os.path as osp
from collections import deque
from pathlib import Path

# Related third party imports
import gymnasium as gym
import h5py
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv1d
from tqdm import tqdm
import joblib


# Local application/library specific imports
import mani_skill2.envs
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.wrappers import RecordEpisode
from modules import GCNPolicy, GATPolicy, RGATPolicy
from dataset import GeometricManiSkill2Dataset
from dataset import transform_obs
from dataset import create_graph, create_heterogeneous_graph
from dataset import normalize, standardize

# Torch geometric imports
import torch.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch
from torch_geometric.nn import to_hetero

# Pytorch kinematics
import pytorch_kinematics as pk

device = "cuda" if th.cuda.is_available() else "cpu"

# Config
DOF = 8  # 8 degrees of freedom for the robot
WINDOW_SIZE = 4  # Number of observations to use for each training step
DEFAULT_Q_POS = (
    th.tensor(
        [0.0, -np.pi / 4, 0, -np.pi * 3 / 4, 0, np.pi * 2 / 4, np.pi / 4, 0.04, 0.04]
    )
    .to(device)
    .float()
)

config = {
    "obs_mode": "state",
    "control_mode": "pd_joint_delta_pos",
    "render_mode": "cameras",
    "num_steps": 100000,
    "batch_size": 128,
    "num_workers": 4,
    "lr": 1e-3,
    "seed": 42,
    "log_dir": "logs/with_l2reg",
    "env_id": "PickCube-v0",
    "demo_path": "imitation_learning/datasets/PickCube/trajectory.state.pd_joint_delta_pos.h5",
    "iterations": 30000,
    "eval": False,
    "loss_fn": nn.MSELoss(),
}


def load_data(path, env=None):
    dataset = GeometricManiSkill2Dataset(path, root="", env=env)
    dataloader = GeometricDataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    dataset.close_h5()
    return dataloader, dataset


def main():

    if config["seed"] is not None:
        th.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    ckpt_dir = osp.join(config["log_dir"], "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    control_mode = "pd_joint_delta_pos"
    env: BaseEnv = gym.make(
        id=config["env_id"],
        obs_mode=config["obs_mode"],
        control_mode=config["control_mode"],
        render_mode="cameras",
    )
    pinocchio_model = env.agent.robot.create_pinocchio_model()

    dataloader, dataset = load_data(config["demo_path"], env=env)
    tmp_graph, obs, actions = dataset[0]
    tmp_graph = Batch.from_data_list([tmp_graph])
    # create our policy
    policy = GATPolicy(obs.shape[2], actions.shape[0])
    # policy = to_hetero(policy, tmp_graph.metadata(), aggr="mean")

    # with th.no_grad():
    #     x = policy(
    #         tmp_graph.x_dict,
    #         tmp_graph.edge_index_dict,
    #         tmp_graph.edge_attr_dict,
    #         tmp_graph.batch_dict,
    #     )

    # move model to gpu if possible
    policy = policy.to(device)
    print(policy)

    loss_fn = config["loss_fn"]

    # a short save function to save our model
    def save_model(policy, path):
        th.save(policy, path)

    def compute_nullspace_proj(q_pos_batch: th.Tensor, q_delta_batch: th.Tensor):
        def compute_jacobian(q_batch: th.Tensor) -> th.Tensor:
            J_batch = []
            q_batch_numpy = q_batch.cpu().numpy()
            for q in q_batch_numpy:
                J_batch.append(
                    pinocchio_model.compute_single_link_local_jacobian(q, 12)
                )
            J_batch = np.array(J_batch)
            J_batch = th.tensor(J_batch, device=device, dtype=th.double)
            return J_batch

        # Append a column of ones to q_batch for homogeneous coordinates
        q_batch = (
            th.cat(
                [
                    q_pos_batch + q_delta_batch,
                    th.ones_like(q_delta_batch[:, 0]).unsqueeze(-1),
                ],
                dim=-1,
            )
            .double()
            .to(device)
        )

        q_delta_batch = th.cat(
            [
                q_delta_batch,
                th.ones_like(q_delta_batch[:, 0]).unsqueeze(-1).double().to(device),
            ],
            dim=-1,
        )

        # Detach q_batch and convert to numpy for the Pinocchio function
        q_batch_detached = q_batch.detach()
        J_batch = compute_jacobian(q_batch_detached)

        # Ensure J_batch is a tensor and requires gradient
        J_batch = J_batch.requires_grad_()

        # Compute the nullspace of the Jacobian
        eye_batch = th.eye(J_batch.shape[2], device=device).repeat(
            J_batch.shape[0], 1, 1
        )
        nullspace_batch = eye_batch - th.bmm(th.pinverse(J_batch), J_batch)

        # Project the joint positions into the nullspace
        nullspace_projection = th.bmm(nullspace_batch, q_delta_batch.unsqueeze(2))

        return nullspace_projection

    def train_step(policy, data, optim, loss_fn):
        optim.zero_grad()
        policy.train()

        graph, obs, actions = data

        graph = graph.to(device)
        obs = obs.to(device)
        actions = actions.to(device)

        # create batched graph
        # graph_list = (
        #     [create_heterogeneous_graph(obs[i]) for i in range(obs.shape[0])]
        #     if obs.shape[0] != 1
        #     else [create_heterogeneous_graph(obs.squeeze(0))]
        # )
        # graph = Batch.from_data_list(graph_list).to(device)

        pred_actions = policy(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        q_pos = obs[:, -1, :, 0]
        q_pos = q_pos.float()
        # Compute the norm of the nullspace projection as a regularization term
        nullspace_proj = compute_nullspace_proj(q_pos, pred_actions).float()

        # Calculate the norm of the nullspace projection
        nullspace_norm = th.norm(nullspace_proj, dim=1)

        default_pos_error = th.abs((DEFAULT_Q_POS[:-1] - q_pos)).float()

        # compute loss and optimize
        loss = (
            loss_fn(actions, pred_actions)
            + 0.0001 * nullspace_norm.mean()
            + 0.0005 * (nullspace_proj.squeeze()[:, :-1] @ default_pos_error.T).mean()
        )
        loss.backward()
        optim.step()
        return loss.item()

    def evaluate_policy(env, policy, num_episodes=10):
        obs_list = deque(maxlen=WINDOW_SIZE)
        # Fill obs_list with zeros
        for _ in range(WINDOW_SIZE):
            obs_list.append(np.zeros_like(env.reset()[0]))
        obs_list.append(env.reset()[0])
        successes = []
        i = 0
        pbar = tqdm(total=num_episodes, leave=False)
        while i < num_episodes:
            obs = np.array(obs_list)
            obs = transform_obs(np.array(obs_list), pinocchio_model=pinocchio_model)
            obs = th.tensor(obs, device=device).float().unsqueeze(0)
            # create batched graph
            graph_list = (
                [create_graph(obs[i]) for i in range(obs.shape[0])]
                if obs.shape[0] != 1
                else [create_graph(obs.squeeze(0))]
            )
            graph = Batch.from_data_list(graph_list).to(device)
            with th.no_grad():
                action = (
                    policy(
                        graph.x,
                        graph.edge_index,
                        graph.edge_attr,
                        graph.batch,
                    )
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            if terminated or truncated:
                successes.append(info["success"])
                i += 1
                obs_list.append(env.reset(seed=i)[0])
                pbar.update(1)
        success_rate = np.mean(successes)
        return success_rate

    writer = SummaryWriter(config["log_dir"])
    optim = th.optim.Adam(policy.parameters(), lr=config["lr"])
    best_epoch_loss = np.inf
    epoch = 0
    steps = 0
    pbar = tqdm(dataloader, total=config["iterations"], leave=False)
    # RecordEpisode wrapper auto records a new video once an episode is completed
    env = RecordEpisode(
        env, output_dir=osp.join(config["log_dir"], "videos"), info_on_video=True
    )
    while steps < config["iterations"]:
        epoch_loss = 0
        for batch in dataloader:
            steps += 1
            # Add noise to the actions to make the policy more robust
            # batch.x += th.randn_like(batch.x) * 0.05
            loss_val = train_step(policy, batch, optim, loss_fn)

            # track the loss and print it
            writer.add_scalar("train/mse_loss", loss_val, steps)
            epoch_loss += loss_val
            pbar.set_postfix(dict(loss=loss_val))
            pbar.update(1)

            # periodically save the policy
            if steps % 2000 == 0:
                save_model(policy, osp.join(ckpt_dir, f"ckpt_{steps}.pt"))
            if steps >= config["iterations"]:
                break
        epoch_loss = epoch_loss / len(dataloader)

        # save a new model if the average MSE loss in an epoch has improved
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))
        if epoch % 5 == 0:
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
