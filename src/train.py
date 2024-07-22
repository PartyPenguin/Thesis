# main.py

# Standard library imports
import os.path as osp
from collections import deque
from pathlib import Path
import yaml
import json
import wandb

# Related third-party imports
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Local application/library-specific imports
import mani_skill2.envs
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.wrappers import RecordEpisode
from src.modules import GATPolicy, GraphSAGEPolicy
from src.utils.util import (
    load_data,
    compute_nullspace_proj,
    evaluate_policy,
    compute_fk,
)

# Torch geometric imports
from torch_geometric.data import Batch

device = "cuda" if th.cuda.is_available() else "cpu"

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

    pred_actions = policy(graph)

    q_pos = obs[:, -1, :, 0]
    q_pos = q_pos.float()

    base_pose = obs[:, -1, 0, 9:16]

    # nullspace_proj = compute_nullspace_proj(
    #     q_pos, pred_actions, env=env, device=device
    # ).float()

    # ef_pos, ef_rot = compute_fk(q_pos + pred_actions, env=env, device=device)
    # ef_pos_true, ef_rot_true = compute_fk(q_pos + actions, env=env, device=device)
    # ef_pos = ef_pos + base_pose[:, :3]
    # ef_pos_true = ef_pos_true + base_pose[:, :3]

    # ef_rot = R.from_quat(ef_rot.detach().cpu().numpy())
    # ef_rot_true = R.from_quat(ef_rot_true.detach().cpu().numpy())
    # rel_rot = ef_rot.inv() * ef_rot_true
    # angle = th.as_tensor(rel_rot.magnitude()).to(device).float()

    # nullspace_norm = th.norm(nullspace_proj, dim=1)
    # default_pos_error = th.abs((DEFAULT_Q_POS[:-1] - q_pos)).float()

    loss = (
        loss_fn(actions, pred_actions)
        # + 0.0001 * nullspace_norm.mean()
        # + 0.0005 * (nullspace_proj.squeeze()[:, :-1] @ default_pos_error.T).mean()
        # + 0.0005 * th.norm(ef_pos - ef_pos_true, dim=1).mean()
        # + 0.0005 * angle.mean()
    )
    loss.backward()
    optim.step()
    return loss.item()


def train(config: dict):

    if config["train"]["seed"] is not None:
        set_seed(config["train"]["seed"])

    ckpt_dir = osp.join(config["train"]["log_dir"], "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    env: BaseEnv = gym.make(
        id=config["env"]["env_id"],
        obs_mode=config["env"]["obs_mode"],
        control_mode=config["env"]["control_mode"],
        render_mode=config["env"]["render_mode"],
    )

    dataloader, dataset = load_data(env=env, config=config)
    tmp_graph, obs, actions = dataset[0]
    tmp_graph = Batch.from_data_list([tmp_graph])
    policy = GATPolicy(obs.shape[2], actions.shape[0]).to(device)

    loss_fn = nn.MSELoss()

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

    while steps < config["train"]["iterations"]:
        wandb.watch(policy, criterion=loss_fn, log="all", log_freq=10)
        epoch_loss = 0
        for batch in dataloader:
            steps += 1
            loss_val = train_step(policy, batch, optim, loss_fn, env, device)
            epoch_loss += loss_val
            pbar.set_postfix(dict(loss=loss_val))
            pbar.update(1)

            if steps % 2000 == 0:
                wandb.unwatch()
                save_model(policy, osp.join(ckpt_dir, f"ckpt_{steps}.pt"))
            if steps >= config["train"]["iterations"]:
                break

        epoch_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch_loss": epoch_loss}, step=steps)
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            wandb.unwatch()
            save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))
            wandb.save(osp.join(ckpt_dir, "ckpt_best.pt"))

        if epoch % 5 == 0:
            success_rate = evaluate_policy(env, policy, config)
            wandb.log({"success_rate": success_rate}, step=steps)

        epoch += 1
    wandb.unwatch()
    save_model(policy, osp.join(ckpt_dir, "ckpt_latest.pt"))
