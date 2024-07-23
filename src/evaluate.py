import gymnasium as gym
import torch as th
from mani_skill2.envs.sapien_env import BaseEnv
import json
from pathlib import Path
import os
import wandb
from src.train import GATPolicy

device = "cuda" if th.cuda.is_available() else "cpu"


def initialize_environment(config: dict) -> BaseEnv:
    env: BaseEnv = gym.make(
        id=config["env"]["env_id"],
        obs_mode=config["env"]["obs_mode"],
        control_mode=config["env"]["control_mode"],
        render_mode=config["env"]["render_mode"],
    )
    env.reset(seed=0)
    return env


def load_model(model_path: str, device: str) -> th.nn.Module:
    return th.load(model_path).to(device)


def evaluate(config: dict):
    model_path = os.path.join(config["train"]["log_dir"], "checkpoints/ckpt_best.pth")

    env = initialize_environment(config)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    checkpoint = th.load(model_path)
    obs_dim = checkpoint["input_dim"]
    act_dim = checkpoint["output_dim"]

    policy = GATPolicy(obs_dim, act_dim).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])

    from src.utils.util import evaluate_policy

    success_rate = evaluate_policy(env, policy, config, num_episodes=100, render=True)

    print("Success rate", success_rate)
    wandb.log({"final_success_rate": success_rate})

    env.close()
