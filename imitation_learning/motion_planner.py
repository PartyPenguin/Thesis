import pymp
import numpy as np
import gymnasium as gym


class MotionPlanner:
    def __init__(self, env):
        self.planner = None
        self.env = env.unwrapped

        link_names = [link.get_name() for link in self.env.agent.robot.get_links()]
        joint_names = [
            joint.get_name() for joint in self.env.agent.robot.get_active_joints()
        ]

        self.planner = pymp.Planner(
            urdf="assets/descriptions/panda_v2.urdf",
            srdf="assets/descriptions/panda_v2.srdf",
            user_joint_names=joint_names,
            ee_link_name="panda_hand_tcp",
            base_pose=self.env.agent.robot.pose,
            joint_vel_limits=0.5,
            joint_acc_limits=0.5,
            timestep=self.env.control_timestep,
        )

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)
        for i in range(100):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def move_to_pose_with_RRTConnect(self, pose):
        result = self.planner.plan_qpos_to_pose(
            pose, self.env.robot.get_qpos(), time_step=self.env.timestep
        )
        if result["status"] != "Success":
            return -1
        return result

    def move_to_pose_with_screw(self, pose):
        result = self.planner.plan_screw(pose, self.env.agent.robot.get_qpos())
        return result
