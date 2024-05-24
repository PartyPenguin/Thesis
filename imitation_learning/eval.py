import gymnasium as gym
import mani_skill2.envs
from envs.my_env import MyEnv
import torch as th
import numpy as np
from main import GCNPolicy
from main import build_hetero_graph
from mani_skill2.utils.wrappers import RecordEpisode
import os.path as osp

log_dir = "logs/eval"
env = gym.make(
    "MyEnv-v0",
    obs_mode="state",
    control_mode="pd_joint_vel",
    render_mode="human",
)
# env = RecordEpisode(env, output_dir=osp.join(log_dir, "videos"), info_on_video=True)
device = "cuda" if th.cuda.is_available() else "cpu"
model_path = "logs/bc_state/checkpoints/ckpt_best.pt"
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=np.random.randint(1000))


terminated, truncated = False, False
policy = th.load(model_path).to(device)

step = 0
while step < 2000:
    graph = build_hetero_graph(th.from_numpy(obs).float(), th.zeros(8)).to(device)
    action = policy(graph).squeeze().detach().cpu().numpy()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # a display is required to render
    step += 1
env.close()
