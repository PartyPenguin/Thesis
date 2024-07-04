import gymnasium as gym
import mani_skill2.envs
import torch as th
import numpy as np
from main import GCNPolicy
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.envs.sapien_env import BaseEnv
from collections import deque
from main import transform_obs
from main import WINDOW_SIZE
from dataset import create_graph, create_heterogeneous_graph
import joblib
from dataset import normalize, standardize
from torch_geometric.data import Batch

log_dir = "logs/eval"
env: BaseEnv = gym.make(
    "PickCube-v0",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="human",
)
# env = RecordEpisode(env, output_dir=osp.join(log_dir, "videos"), info_on_video=True)
env.reset(seed=0)
device = "cuda" if th.cuda.is_available() else "cpu"
model_path = "logs/with_reg/checkpoints/ckpt_best.pt"
print("Observation space", env.observation_space)
print("Action space", env.action_space)

pinocchio_model = env.agent.robot.create_pinocchio_model()
obs_list = deque(maxlen=WINDOW_SIZE)
# Fill obs_list with zeros
for _ in range(WINDOW_SIZE):
    obs_list.append(np.zeros_like(env.reset()[0]))
obs_list.append(env.reset()[0])
obs, _ = env.reset(seed=np.random.randint(1000))
obs_list.append(obs)
tmp_obs = th.tensor(
    transform_obs(np.array(obs_list), pinocchio_model=pinocchio_model)[0]
)
tmp_graph = create_heterogeneous_graph(tmp_obs.unsqueeze(0))


terminated, truncated = False, False
policy = th.load(model_path).to(device)

with th.no_grad():
    x = policy(
        tmp_graph.x_dict,
        tmp_graph.edge_index_dict,
        tmp_graph.edge_attr_dict,
        tmp_graph.batch_dict,
    )

step = 0
score = 0
num_runs = 100
run = 0
run_score = 0
while run < num_runs:
    # Normalize data
    scaler = joblib.load("norm_scaler.pkl")
    # mean = joblib.load("mean.pkl")
    # std = joblib.load("std.pkl")
    # obs = normalize(data=np.array(obs_list), scaler=scaler)
    obs = th.as_tensor(
        transform_obs(np.array(obs_list), pinocchio_model=pinocchio_model)
    )
    obs_device = obs.to(device).float()
    graph_list = (
        [create_heterogeneous_graph(obs[i]) for i in range(obs.shape[0])]
        if obs.shape[0] != 1
        else [create_heterogeneous_graph(obs.squeeze(0))]
    )
    graph = Batch.from_data_list(graph_list).to(device)
    action = (
        policy(
            graph.x_dict,
            graph.edge_index_dict,
            graph.edge_attr_dict,
            graph.batch_dict,
        )
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    obs, reward, terminated, truncated, info = env.step(action)
    run_score += reward
    obs_list.append(obs)
    env.render()  # a display is required to render
    step += 1
    if step > 200:
        avg_score = run_score / step
        print("Run", run, "Score", avg_score)
        score += avg_score
        # env.update_task()
        env.reset()
        obs_list.clear()
        for _ in range(WINDOW_SIZE):
            obs_list.append(np.zeros_like(env.get_obs()))
        obs_list.append(env.get_obs())
        step = 0
        run += 1
        run_score = 0

print("Average score", score / num_runs)

env.close()
