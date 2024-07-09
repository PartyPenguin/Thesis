# main.py

# Standard library imports
import os.path as osp
from collections import deque
from pathlib import Path

# Related third-party imports
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dvclive import Live

# Local application/library-specific imports
import mani_skill2.envs
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.wrappers import RecordEpisode
from modules import GATPolicy
from util import load_data, compute_nullspace_proj, evaluate_policy

# Torch geometric imports
from torch_geometric.data import Batch

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


def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)


def save_model(policy, path):
    th.save(policy, path)


def train_step(policy, data, optim, loss_fn, env, device):
    optim.zero_grad()
    policy.train()

    graph, obs, actions = data

    graph = graph.to(device)
    obs = obs.to(device)
    actions = actions.to(device)

    pred_actions = policy(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

    q_pos = obs[:, -1, :, 0]
    q_pos = q_pos.float()

    nullspace_proj = compute_nullspace_proj(
        q_pos, pred_actions, env=env, device=device
    ).float()
    nullspace_norm = th.norm(nullspace_proj, dim=1)
    default_pos_error = th.abs((DEFAULT_Q_POS[:-1] - q_pos)).float()

    loss = (
        loss_fn(actions, pred_actions)
        + 0.0001 * nullspace_norm.mean()
        + 0.0005 * (nullspace_proj.squeeze()[:, :-1] @ default_pos_error.T).mean()
    )
    loss.backward()
    optim.step()
    return loss.item()


def main():

    if config["seed"] is not None:
        set_seed(config["seed"])

    ckpt_dir = osp.join(config["log_dir"], "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    env: BaseEnv = gym.make(
        id=config["env_id"],
        obs_mode=config["obs_mode"],
        control_mode=config["control_mode"],
        render_mode=config["render_mode"],
    )

    dataloader, dataset = load_data(config["demo_path"], env=env, config=config)
    tmp_graph, obs, actions = dataset[0]
    tmp_graph = Batch.from_data_list([tmp_graph])
    policy = GATPolicy(obs.shape[2], actions.shape[0]).to(device)

    loss_fn = config["loss_fn"]

    writer = SummaryWriter(config["log_dir"])
    optim = th.optim.Adam(policy.parameters(), lr=config["lr"])
    best_epoch_loss = np.inf
    epoch = 0
    steps = 0
    pbar = tqdm(dataloader, total=config["iterations"], leave=False)
    env = RecordEpisode(
        env, output_dir=osp.join(config["log_dir"], "videos"), info_on_video=True
    )

    while steps < config["iterations"]:
        epoch_loss = 0
        for batch in dataloader:
            steps += 1
            loss_val = train_step(policy, batch, optim, loss_fn, env, device)
            writer.add_scalar("train/mse_loss", loss_val, steps)
            epoch_loss += loss_val
            pbar.set_postfix(dict(loss=loss_val))
            pbar.update(1)
            if steps % 2000 == 0:
                save_model(policy, osp.join(ckpt_dir, f"ckpt_{steps}.pt"))
            if steps >= config["iterations"]:
                break

        epoch_loss = epoch_loss / len(dataloader)
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))

        if epoch % 5 == 0:
            success_rate = evaluate_policy(env, policy)
            writer.add_scalar("test/success_rate", success_rate, epoch)

        writer.add_scalar("train/mse_loss_epoch", epoch_loss, epoch)
        epoch += 1

    save_model(policy, osp.join(ckpt_dir, "ckpt_latest.pt"))
    success_rate = evaluate_policy(env, policy)
    print(f"Final Success Rate {success_rate}")


if __name__ == "__main__":
    main()
