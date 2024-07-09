from src.dataset import GeometricManiSkill2Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
import torch as th
from sapien.core.pysapien import PinocchioModel
import numpy as np
from tqdm import tqdm
from collections import deque
from src.dataset import transform_obs
from src.dataset import create_graph
from src.dataset import WINDOW_SIZE
from mani_skill2.envs.sapien_env import BaseEnv
from torch_geometric.data import Batch


def load_data(path, env, config):
    """
    Load data from a given path and create a data loader.

    Args:
        path (str): The path to the data.
        env: The environment object.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        tuple: A tuple containing the data loader and the dataset object.
    """
    dataset = GeometricManiSkill2Dataset(path, root="", env=env)
    dataloader = GeometricDataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    dataset.close_h5()
    return dataloader, dataset


def compute_nullspace_proj(
    q_pos_batch: th.Tensor,
    q_delta_batch: th.Tensor,
    env: BaseEnv,
    device: str,
) -> th.Tensor:
    """
    Compute the nullspace projection of joint positions.

    Args:
        q_pos_batch (torch.Tensor): Batch of initial joint positions.
        q_delta_batch (torch.Tensor): Batch of joint position changes.
        env (BaseEnv): Environment object.
        device (str): Device to perform computations on.

    Returns:
        torch.Tensor: Nullspace projection of joint positions.
    """

    def compute_jacobian(q_batch: th.Tensor) -> th.Tensor:
        """
        Compute the Jacobian matrix for a batch of joint positions.

        Args:
            q_batch (torch.Tensor): Batch of joint positions.

        Returns:
            torch.Tensor: Jacobian matrix.
        """
        pinocchio_model = env.agent.robot.create_pinocchio_model()
        J_batch = []
        q_batch_numpy = q_batch.cpu().numpy()
        for q in q_batch_numpy:
            J_batch.append(pinocchio_model.compute_single_link_local_jacobian(q, 12))
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
    eye_batch = th.eye(J_batch.shape[2], device=device).repeat(J_batch.shape[0], 1, 1)
    nullspace_batch = eye_batch - th.bmm(th.pinverse(J_batch), J_batch)

    # Project the joint positions into the nullspace
    nullspace_projection = th.bmm(nullspace_batch, q_delta_batch.unsqueeze(2))

    return nullspace_projection


def evaluate_policy(env, policy, num_episodes=10, device="cuda"):
    """
    Evaluate the performance of a policy in a given environment.

    Args:
        env (gym.Env): The environment to evaluate the policy in.
        policy (callable): The policy function to evaluate.
        num_episodes (int, optional): The number of episodes to run the evaluation for. Defaults to 10.
        device (str, optional): The device to use for computation. Defaults to "cuda".

    Returns:
        float: The success rate of the policy, defined as the proportion of successful episodes.
    """
    pinocchio_model = env.agent.robot.create_pinocchio_model()
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