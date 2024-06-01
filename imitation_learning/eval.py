import gymnasium as gym
import mani_skill2.envs
import torch as th
import numpy as np
from main import GCNPolicy
from main import build_graph
from mani_skill2.utils.wrappers import RecordEpisode
import os.path as osp
from collections import deque
from main import transform_obs
from main import WINDOW_SIZE

log_dir = "logs/eval"
env = gym.make(
    "MyEnv-v0",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="human",
)
# env = RecordEpisode(env, output_dir=osp.join(log_dir, "videos"), info_on_video=True)
device = "cuda" if th.cuda.is_available() else "cpu"
model_path = "logs/bc_state/checkpoints/ckpt_best.pt"
print("Observation space", env.observation_space)
print("Action space", env.action_space)


obs_list = deque(maxlen=WINDOW_SIZE)
# Fill obs_list with zeros
for _ in range(WINDOW_SIZE):
    obs_list.append(np.zeros_like(env.reset()[0]))
obs_list.append(env.reset()[0])
obs, _ = env.reset(seed=np.random.randint(1000))
obs_list.append(obs)

terminated, truncated = False, False
policy = th.load(model_path).to(device)

step = 0
while step < 2000:
    obs = th.as_tensor(transform_obs(np.array(obs_list)))
    obs_device = obs.to(device).unsqueeze(0).float()
    action = policy(obs_device).squeeze().detach().cpu().numpy()
    obs, reward, terminated, truncated, info = env.step(action)
    obs_list.append(obs)
    env.render()  # a display is required to render
    step += 1
env.close()
