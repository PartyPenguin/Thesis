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
from modules import GCNPolicy
from dataset import GeometricManiSkill2Dataset
from dataset import transform_obs
from dataset import create_graph
from dataset import normalize

# Torch geometric imports
import torch.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch

# Pytorch kinematics
import pytorch_kinematics as pk

device = "cuda" if th.cuda.is_available() else "cpu"

# Config
DOF = 8  # 8 degrees of freedom for the robot
WINDOW_SIZE = 4  # Number of observations to use for each training step


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
    env: BaseEnv = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="cameras",
    )
    pinocchio_model = env.agent.robot.create_pinocchio_model()
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
        obs, actions = dataset[0]
        # create our policy
        policy = GCNPolicy(obs.shape[2], actions.shape[0])

    # move model to gpu if possible
    policy = policy.to(device)
    print(policy)

    loss_fn = nn.HuberLoss()

    # a short save function to save our model
    def save_model(policy, path):
        th.save(policy, path)

    def compute_nullspace_norm(q_pos_batch: th.Tensor, q_delta_batch: th.Tensor):
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

        # Calculate the norm of the nullspace projection
        nullspace_norm = th.norm(nullspace_projection, dim=1)

        return nullspace_norm

    def train_step(policy, data, optim, loss_fn):
        optim.zero_grad()
        # move data to appropriate device first
        obs, actions = data
        obs = obs.to(device)
        actions = actions.to(device)

        # create batched graph
        graph_list = (
            [create_graph(obs[i]) for i in range(obs.shape[0])]
            if obs.shape[0] != 1
            else [create_graph(obs.squeeze(0))]
        )
        graph = Batch.from_data_list(graph_list).to(device)

        pred_actions = policy(graph)

        # compute loss and optimize
        # L1 regularization
        l1_lambda = 0.0001
        l1_norm = sum(p.abs().sum() for p in policy.parameters())

        # L2 regularization
        l2_lambda = 0.00005
        l2_norm = sum(p.pow(2.0).sum() for p in policy.parameters())

        q_pos = obs[:, -1, :, 0]
        # Compute the norm of the nullspace projection as a regularization term
        nullspace_reg = compute_nullspace_norm(q_pos, pred_actions)

        # compute loss and optimize
        loss = loss_fn(actions, pred_actions)  # + 0.001 * nullspace_reg.mean()
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
            obs = transform_obs(np.array(obs_list))
            # Normalize data
            scaler = joblib.load("scaler.pkl")
            mean = joblib.load("mean.pkl")
            std = joblib.load("std.pkl")
            obs = normalize(data=obs, scaler=scaler, mean=mean, std=std)

            obs = th.tensor(obs, device=device).float()
            # move to appropriate device and unsqueeze to add a batch dimension
            obs_device = obs.to(device)
            graph = create_graph(obs_device).to(device)
            with th.no_grad():
                action = policy(graph).squeeze().detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            if terminated or truncated:
                successes.append(info["success"])
                i += 1
                obs_list.append(env.reset(seed=i)[0])
                pbar.update(1)
        success_rate = np.mean(successes)
        return success_rate

    if not args.eval:
        writer = SummaryWriter(log_dir)
        optim = th.optim.Adam(policy.parameters(), lr=1e-2)
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
                if steps >= iterations:
                    break

            epoch_loss = epoch_loss / len(dataloader)

            # save a new model if the average MSE loss in an epoch has improved
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                save_model(policy, osp.join(ckpt_dir, "ckpt_best.pt"))
            if epoch % 10 == 0:
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
