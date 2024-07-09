# main.py

# Standard library imports
import os.path as osp
from collections import deque
from pathlib import Path
import yaml

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
from src.utils.util import load_data, compute_nullspace_proj, evaluate_policy

# Torch geometric imports
from torch_geometric.data import Batch

device = "cuda" if th.cuda.is_available() else "cpu"

# Load config from params.yaml
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

DOF = 8  # 8 degrees of freedom for the robot
WINDOW_SIZE = 4  # Number of observations to use for each training step
DEFAULT_Q_POS = (
    th.tensor(
        [0.0, -np.pi / 4, 0, -np.pi * 3 / 4, 0, np.pi * 2 / 4, np.pi / 4, 0.04, 0.04]
    )
    .to(device)
    .float()
)


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

    if config["train"]["seed"] is not None:
        set_seed(config["train"]["seed"])

    ckpt_dir = osp.join(config["train"]["log_dir"], "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    env: BaseEnv = gym.make(
        id=config["train"]["env_id"],
        obs_mode=config["train"]["obs_mode"],
        control_mode=config["train"]["control_mode"],
        render_mode=config["train"]["render_mode"],
    )

    dataloader, dataset = load_data(
        config["train"]["demo_path"], env=env, config=config
    )
    tmp_graph, obs, actions = dataset[0]
    tmp_graph = Batch.from_data_list([tmp_graph])
    policy = GATPolicy(obs.shape[2] * 12, actions.shape[0]).to(device)

    loss_fn = nn.MSELoss()

    writer = SummaryWriter(config["train"]["log_dir"])
    optim = th.optim.Adam(policy.parameters(), lr=config["train"]["lr"])
    best_epoch_loss = np.inf
    epoch = 0
    steps = 0
    pbar = tqdm(dataloader, total=config["train"]["iterations"], leave=False)
    env = RecordEpisode(
        env,
        output_dir=osp.join(config["train"]["log_dir"], "videos"),
        info_on_video=True,
    )
    with Live() as live:

        while steps < config["train"]["iterations"]:

            epoch_loss = 0
            for batch in dataloader:
                steps += 1
                loss_val = train_step(policy, batch, optim, loss_fn, env, device)
                writer.add_scalar("train/mse_loss", loss_val, steps)
                live.log_metric("train/loss", loss_val)
                epoch_loss += loss_val
                pbar.set_postfix(dict(loss=loss_val))
                pbar.update(1)
                if steps % 2000 == 0:
                    save_model(policy, osp.join(ckpt_dir, f"ckpt_{steps}.pt"))
                if steps >= config["train"]["iterations"]:
                    break

            epoch_loss = epoch_loss / len(dataloader)
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))

            if epoch % 5 == 0:
                success_rate = evaluate_policy(env, policy)
                writer.add_scalar("test/success_rate", success_rate, epoch)
                live.log_metric("test/success_rate", success_rate)

            writer.add_scalar("train/mse_loss_epoch", epoch_loss, epoch)
            live.log_metric("train/mse_loss_epoch", epoch_loss)
            epoch += 1
            live.next_step()

        save_model(policy, osp.join(ckpt_dir, "ckpt_latest.pt"))
        success_rate = evaluate_policy(env, policy)
        print(f"Final Success Rate {success_rate}")


if __name__ == "__main__":
    main()
