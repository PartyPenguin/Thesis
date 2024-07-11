import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as th
import gymnasium as gym
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.sapien_utils import vectorize_pose
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import yaml
from mani_skill2.utils.io_utils import load_json

from pathlib import Path


def load_h5_data(data):
    out = {}
    for k, v in data.items():
        if isinstance(v, h5py.Dataset):
            out[k] = v[:]
        else:
            out[k] = load_h5_data(v)
    return out


def standardize(data: np.ndarray, scaler: StandardScaler = None) -> np.ndarray:
    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        joblib.dump(scaler, "standard_scaler.pkl")
    else:
        data = scaler.transform(data)
    return data


def normalize(data: np.ndarray, scaler: MinMaxScaler = None) -> np.ndarray:
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
        data = scaler.fit_transform(data)
        joblib.dump(scaler, "norm_scaler.pkl")
    else:
        data = scaler.transform(data)
    return data


def fourier_encode(data: np.ndarray) -> np.ndarray:
    scales = np.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=data.dtype)
    scaled_data = data.reshape(-1, 1) / (3.0**scales).reshape(1, -1)
    sin_features = np.sin(scaled_data)
    cos_features = np.cos(scaled_data)
    features = np.concatenate([sin_features, cos_features], axis=1)
    return features.reshape(data.shape[0], -1)


TRANSFORMATIONS = {
    "standardize": standardize,
    "normalize": normalize,
    "fourier": fourier_encode,
}


def load_raw_data(config):
    dataset_file = config["raw_data_path"] + config["data_file"]
    data = h5py.File(dataset_file, "r")
    json_path = dataset_file.replace(".h5", ".json")
    json_data = load_json(json_path)
    episodes = json_data["episodes"]

    observations, actions, episode_map = [], [], []
    load_count = (
        len(episodes)
        if config["prepare"]["load_count"] == -1
        else config["prepare"]["load_count"]
    )
    for eps_id in tqdm(range(load_count)):
        eps = episodes[eps_id]
        trajectory = load_h5_data(data[f"traj_{eps['episode_id']}"])
        observations.append(trajectory["obs"][:-1])
        actions.append(trajectory["actions"])
        episode_map.append(np.full(len(trajectory["obs"]) - 1, eps["episode_id"]))

    observations = np.vstack(observations)
    actions = np.vstack(actions)
    episode_map = np.hstack(episode_map)

    return observations, actions, episode_map


def base_transform_obs(obs, env):
    def compute_joint_se3_pose(joint_positions, base_pose):
        pinocchio_model = env.unwrapped.agent.robot.create_pinocchio_model()
        joint_se3_pose = []
        for i in range(joint_positions.shape[0]):
            pinocchio_model.compute_forward_kinematics(joint_positions[i])
            pose = np.asarray(
                [
                    vectorize_pose(pinocchio_model.get_link_pose(j))
                    for j in range(joint_positions.shape[1] - 1)
                ]
            ).flatten()
            pose[:3] -= base_pose[i, :3]
            joint_se3_pose.append(pose)
        return np.array(joint_se3_pose)

    joint_positions = obs[:, :9]
    joint_velocities = obs[:, 9:17]
    base_pose = obs[:, 18:25]
    tcp_pose = obs[:, 25:32]
    goal_position = obs[:, 32:35]
    tcp_to_goal_position = obs[:, 35:38]
    obj_pose = obs[:, 38:45]
    tcp_to_obj_pos = obs[:, 45:48]
    obj_to_goal_pos = obs[:, 48:51]

    joint_se3_pose = compute_joint_se3_pose(joint_positions, base_pose)
    joint_se3_reshape = joint_se3_pose.reshape(joint_se3_pose.shape[0], 8, -1)

    joint_features = np.concatenate(
        (joint_positions[:, :-1, None], joint_velocities[..., None], joint_se3_reshape),
        axis=2,
    )

    context_info = np.hstack(
        [
            tcp_pose,
            goal_position,
            tcp_to_goal_position,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_goal_pos,
        ]
    )

    context_features = np.repeat(
        context_info[:, np.newaxis, :], joint_features.shape[1], axis=1
    )
    combined_features = np.concatenate([joint_features, context_features], axis=2)
    original_shape = combined_features.shape
    combined_features = combined_features.reshape(-1, combined_features.shape[-1])

    return combined_features, original_shape


def apply_transformations(array, config):
    for transformation in config["prepare"]["transformations"]:
        t_type = transformation["type"]
        transform_func = TRANSFORMATIONS[t_type]
        array = transform_func(array)
    return array


def prepare(config):
    env: BaseEnv = gym.make(
        id=config["env_id"],
        obs_mode=config["obs_mode"],
        control_mode=config["control_mode"],
        render_mode=config["render_mode"],
    )
    obs, act, episode_map = load_raw_data(config)
    obs, obs_shape = base_transform_obs(obs, env)
    obs = apply_transformations(obs, config).reshape(obs_shape)

    # Create a directory to save the prepared data
    Path(config["prepared_data_path"]).mkdir(parents=True, exist_ok=True)

    np.save(config["prepared_data_path"] + "obs.npy", obs)
    np.save(config["prepared_data_path"] + "act.npy", act)
    np.save(config["prepared_data_path"] + "episode_map.npy", episode_map)


with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    prepare(config)
