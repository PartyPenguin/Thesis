import numpy as np
from scipy.stats import special_ortho_group
from sapien.core import Pose
from collections import OrderedDict
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.registration import register_env
from scipy.spatial.transform import Rotation as R
from mani_skill2.envs.pick_and_place.base_env import StationaryManipulationEnv


@register_env("MyEnv-v0", max_episode_steps=200)
class MyEnv(StationaryManipulationEnv):
    goal_thresh = 0.025

    def __init__(self, **kwargs):
        super().__init__(robot_init_qpos_noise=0.3, **kwargs)

    def _load_actors(self):
        # self._add_ground(render=self.bg_name is None)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_task(self):
        def random_quaternion():
            u1, u2, u3 = np.random.uniform(0, 1, 3)

            q = np.array(
                [
                    np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                    np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                    np.sqrt(u1) * np.sin(2 * np.pi * u3),
                    np.sqrt(u1) * np.cos(2 * np.pi * u3),
                ]
            )

            return q

        # Generate a random quaternion
        random_quaternion = random_quaternion()

        goal_x = self._episode_rng.uniform(0.2, 0.6)
        goal_y = self._episode_rng.uniform(-0.2, 0.2)
        goal_z = self._episode_rng.uniform(0, 0.5)
        self.goal_pos = np.hstack([goal_x, goal_y, goal_z])

        self.goal_site.set_pose(Pose(self.goal_pos, np.roll(random_quaternion, 1)))

    def _initialize_agent(self):
        super()._initialize_agent()
        if self.robot_uid == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            qpos = np.array(
                [0.0, -np.pi / 4, 0, -np.pi * 3 / 4, 0, np.pi * 2 / 4, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
        self.agent.robot.set_pose(Pose([0, 0, 0]))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pose=vectorize_pose(self.goal_site.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
            )
        return obs

    def check_target_reached(self):
        gripper_pos = self.tcp.pose.p
        dist = np.linalg.norm(gripper_pos - self.goal_pos)
        return dist < self.goal_thresh

    def evaluate(self, **kwargs) -> dict:
        is_target_reached = self.check_target_reached()
        return dict(
            is_target_reached=is_target_reached,
            success=is_target_reached,
        )

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_goal_pos = self.goal_pos - self.tcp.pose.p
        tcp_to_goal_dist = np.linalg.norm(tcp_to_goal_pos)

        reaching_reward = 1 - np.tanh(5.0 * tcp_to_goal_dist)
        reward += reaching_reward

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 5.0

    def render_human(self):
        self.goal_site.unhide_visual()
        ret = super().render_human()
        self.goal_site.hide_visual()
        return ret

    def render_cameras(self):
        self.goal_site.unhide_visual()
        ret = super().render_cameras()
        self.goal_site.hide_visual()
        return ret
