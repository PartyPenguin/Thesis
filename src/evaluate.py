import gymnasium as gym
import mani_skill2.envs
import torch as th
import numpy as np
from src.modules import GATPolicy
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.envs.sapien_env import BaseEnv
from collections import deque
from src.prepare import base_transform_obs
from src.dataset import create_graph
import json
from torch_geometric.data import Batch
from typing import List, Tuple
import yaml
from pathlib import Path

# Load config from params.yaml
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

WINDOW_SIZE = config["window_size"]


def initialize_environment() -> BaseEnv:
    env: BaseEnv = gym.make(
        id=config["train"]["env_id"],
        obs_mode=config["train"]["obs_mode"],
        control_mode=config["train"]["control_mode"],
        render_mode=config["evaluate"]["render_mode"],
    )
    env.reset(seed=0)
    return env


def load_model(model_path: str, device: str) -> th.nn.Module:
    return th.load(model_path).to(device)


def initialize_obs_list(env: BaseEnv, window_size: int) -> deque:
    obs_list = deque(maxlen=WINDOW_SIZE)
    # Fill obs_list with zeros
    for _ in range(WINDOW_SIZE):
        obs_list.append(np.zeros_like(env.reset()[0]))
    obs_list.append(env.reset()[0])
    return obs_list


def prepare_observation(obs_list: deque, env: BaseEnv, device: str) -> th.Tensor:
    obs, shape = base_transform_obs(np.array(obs_list), env=env)
    obs = th.tensor(obs, device=device).float().reshape(shape).unsqueeze(0)
    return obs


def create_graph_batch(obs: th.Tensor, device: str) -> Batch:
    graph_list = (
        [create_graph(obs[i]) for i in range(obs.shape[0])]
        if obs.shape[0] != 1
        else [create_graph(obs.squeeze(0))]
    )
    graph = Batch.from_data_list(graph_list).to(device)
    return graph


def main():
    log_dir = "logs/eval"
    model_path = "logs/output/checkpoints/ckpt_best.pt"
    device = "cuda" if th.cuda.is_available() else "cpu"
    num_runs = 100

    env = initialize_environment()
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    policy = load_model(model_path, device)

    from src.utils.util import evaluate_policy

    success_rate = evaluate_policy(env, policy, num_episodes=100, render=True)

    print("Success rate", success_rate)

    # Save success rate to file
    Path("logs/output/eval").mkdir(parents=True, exist_ok=True)
    with open("logs/output/eval/success_rate.json", "w") as f:
        json.dump({"success_rate": success_rate}, f)

    env.close()


if __name__ == "__main__":
    main()
