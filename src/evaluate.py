import gymnasium as gym
import torch as th
from mani_skill2.envs.sapien_env import BaseEnv
import json
from pathlib import Path
import os
import wandb

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
    model_path = os.path.join(config["train"]["log_dir"], "checkpoints/ckpt_best.pt")

    env = initialize_environment(config)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    policy = load_model(model_path, device)

    from src.utils.util import evaluate_policy

    success_rate = evaluate_policy(env, policy, config, num_episodes=100, render=True)

    print("Success rate", success_rate)
    wandb.log({"final_success_rate": success_rate})

    env.close()
