import random
import time
from typing import Dict, Optional

import pybullet as p
import pybullet_data
import numpy as np

from gesture_recognition import Gesture, GestureRecognizer


class PandaArmSimulation:
    def __init__(
            self,
            gesture_semantics: Dict[Gesture, str],
            recognizer: GestureRecognizer,
            time_limit_seconds=60
    ):
        self.recognizer = recognizer
        self.gesture_semantics = gesture_semantics
        self.time_limit_seconds = time_limit_seconds
        self.start_time_seconds = -1

        self.goal_position = np.array([0.6, 0.0, 0.2])  # Target position for end-effector

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.ee_index = 11  # end effector link index

        for j in range(7):
            p.resetJointState(self.robot, j, 0.0)

        print("Simulation started.")
        self.display_state()

    def start(self):
        self.start_time_seconds = time.time()

    def display_state(self):
        ee_pos = p.getLinkState(self.robot, self.ee_index)[4]
        print(f"\n--- CURRENT EE POSITION ---\nEnd-Effector: {np.round(ee_pos, 3)}")
        print(f"Goal Position: {np.round(self.goal_position, 3)}\n")

    def apply_command(self, command: str):
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_index)[4])
        delta = 0.02
        if command == "point_up":
            ee_pos[2] += delta
        elif command == "point_down":
            ee_pos[2] -= delta
        elif command == "point_left":
            ee_pos[1] += delta
        elif command == "point_right":
            ee_pos[1] -= delta
        elif command == "point_forward":
            ee_pos[0] += delta
        elif command == "point_backward":
            ee_pos[0] -= delta

        joint_poses = p.calculateInverseKinematics(self.robot, self.ee_index, ee_pos.tolist())
        for i in range(7):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_poses[i], force=200)
        for _ in range(10):
            p.stepSimulation()
            time.sleep(1 / 240)

    def update(self, current_time: float, image: bytes) -> Optional[bool]:
        if current_time > self.time_limit_seconds + self.start_time_seconds:
            print("Time expired.")
            return False

        recognized_gestures = self.recognizer.recognize(image, list(self.gesture_semantics.keys()))

        for gesture in recognized_gestures:
            command = self.gesture_semantics.get(gesture)
            if command:
                print(f"Recognized gesture: {gesture}, executing command: {command}")
                self.apply_command(command)

        self.display_state()

        ee_pos = np.array(p.getLinkState(self.robot, self.ee_index)[4])
        if np.linalg.norm(ee_pos - self.goal_position) < 0.05:
            print("Success! Goal position reached.")
            return True

        return None

    def close(self):
        p.disconnect()
