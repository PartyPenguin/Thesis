# Standard library imports
import copy

# Third-party imports
import numpy as np
import torch as th
from tqdm import tqdm

# Local application imports
import mani_skill2.envs
from motion_planner import MotionPlanner
from modules import GCNPolicy
from mani_skill2.utils.wrappers import RecordEpisode

# Initialize the device based on CUDA availability.
device = "cuda" if th.cuda.is_available() else "cpu"


class AggregateRecordEpisode(RecordEpisode):
    """Custom environment wrapper to aggregate and record episodes."""

    def step(self, action, true_action):
        obs, rew, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1

        # Save trajectory data if enabled
        if self.save_trajectory:
            state = self.env.unwrapped.get_state()
            data = {
                "s": state,
                "o": copy.deepcopy(obs),
                "a": true_action,
                "r": rew,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
            self._episode_data.append(data)
            self._episode_info["elapsed_steps"] += 1
            self._episode_info["info"] = info

        return obs, rew, terminated, truncated, info


import h5py


def merge_hdf5_files(file_a_path, file_b_path):
    with h5py.File(file_a_path, "a") as file_a, h5py.File(file_b_path, "r") as file_b:
        # Extract numerical parts of keys from file A and find the maximum value
        keys_file_a = list(file_a.keys())
        max_index = -1
        if keys_file_a:
            # Assuming keys are in the format 'traj_#'
            numbers = [
                int(key.split("_")[-1])
                for key in keys_file_a
                if key.startswith("traj_")
            ]
            if numbers:
                max_index = max(numbers)

        # Copy datasets from file B to file A with new keys
        keys_file_b = list(file_b.keys())
        keys_file_b.sort(
            key=lambda x: int(x.split("_")[-1])
        )  # Sort by numerical part to maintain order

        for key in keys_file_b:
            new_index = max_index + 1
            new_key = f"traj_{new_index}"
            file_b.copy(key, file_a, name=new_key)
            max_index = new_index  # Update the index for the next dataset

        print("Merge complete. File A updated with data from File B.")


import json


def merge_json_files_into_file_a(file_path_a, file_path_b):
    # Load JSON data from both files
    with open(file_path_a, "r") as file_a:
        data_a = json.load(file_a)

    with open(file_path_b, "r") as file_b:
        data_b = json.load(file_b)

    # Determine the last episode_id from the first JSON file
    if data_a["episodes"]:
        last_episode_id_a = max(episode["episode_id"] for episode in data_a["episodes"])
    else:
        last_episode_id_a = -1  # Start from -1 so the first new ID is 0

    # Increment episode_ids in the second file
    increment = last_episode_id_a + 1
    for episode in data_b["episodes"]:
        episode["episode_id"] += increment

    # Append episodes from file B to file A
    data_a["episodes"].extend(data_b["episodes"])

    # Save the updated data back to file A
    with open(file_path_a, "w") as file_a:
        json.dump(data_a, file_a, indent=4)

    print("File A updated with data from File B.")


# def aggregate_dataset(env, policy, iter, max_trajectories=500):
#     """Aggregate dataset by simulating environment with policy."""
#     num_trajectories = 0
#     mp = MotionPlanner(env=env)
#     recorder_env = AggregateRecordEpisode(
#         env=env,
#         output_dir="",
#         trajectory_name="trajectories_" + str(iter),
#         save_video=False,
#     )

#     with tqdm(total=max_trajectories) as progress:
#         while num_trajectories < max_trajectories:
#             obs, reset_info = recorder_env.reset()
#             steps = 0
#             while steps < 100:
#                 plan = mp.move_to_pose_with_screw(recorder_env.goal_site.pose)
#                 if plan["status"] != "success":
#                     break

#                 graph = build_graph(th.from_numpy(obs).float(), th.zeros(8)).to(device)
#                 action = policy(graph).squeeze().detach().cpu().numpy()

#                 delta_qpos = (
#                     plan["position"][0] - recorder_env.agent.robot.get_qpos()[:-2]
#                 )
#                 true_action = np.hstack([delta_qpos, 0.1])
#                 true_action = (
#                     true_action / 0.1
#                 )  # Normalize to action space range [-1,1]

#                 obs, reward, done, truncated, info = recorder_env.step(
#                     action, true_action
#                 )
#                 steps += 1
#             num_trajectories += 1
#             progress.update(1)
#         recorder_env._h5_file.close()

#         merge_hdf5_files(f"trajectories.h5", f"trajectories_{iter}.h5")
#         merge_json_files_into_file_a("trajectories.json", f"trajectories_{iter}.json")
