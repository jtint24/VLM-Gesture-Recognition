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

        # Random initial state
        self.state = {
            "position_x": round(random.choice([x / 100 for x in range(-6, 7, 2)]), 2),
            "gripper_open": random.choice([True, False])
        }

        # Random goal state, must be different from initial state
        self.goal = {
            "position_x": round(random.choice([x / 100 for x in range(-6, 7, 2)]), 2),
            "gripper_open": random.choice([True, False])
        }
        while self.goal == self.state:
            self.goal = {
                "position_x": round(random.choice([x / 100 for x in range(-6, 7, 2)]), 2),
                "gripper_open": random.choice([True, False])
            }

        # PyBullet setup
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.ee_index = 11

        for j in range(7):
            p.resetJointState(self.robot, j, 0.0)

        # Move to random initial state
        ee_pos = [0.6 + self.state["position_x"], 0, 0.4]
        joint_poses = p.calculateInverseKinematics(self.robot, self.ee_index, ee_pos)

        for i in range(7):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_poses[i], force=200)

        gripper_val = 0.04 if self.state["gripper_open"] else 0.0
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, gripper_val, force=200)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, gripper_val, force=200)

        for _ in range(30):
            p.stepSimulation()
            time.sleep(1 / 240)

        print("Simulation started.")
        self.display_state()

    def start(self):
        self.start_time_seconds = time.time()

    def display_state(self):
        print(f"\n--- CURRENT STATE ---")
        print(f"X Position Offset: {round(self.state['position_x'], 3)}")
        print(f"Gripper: {'Open' if self.state['gripper_open'] else 'Closed'}")

        print(f"\n--- GOAL STATE ---")
        print(f"X Position Offset: {round(self.goal['position_x'], 3)}")
        print(f"Gripper: {'Open' if self.goal['gripper_open'] else 'Closed'}\n")

    def apply_command(self, command: str):
        delta = 0.02
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_index)[4])
        if command == "x_forward":
            self.state["position_x"] += delta
        elif command == "x_backward":
            self.state["position_x"] -= delta
        elif command == "gripper_close":
            self.state["gripper_open"] = False
        elif command == "gripper_open":
            self.state["gripper_open"] = True
        """
        if command == "thumbs_up":
            self.state["position_x"] += delta
        elif command == "thumbs_down":
            self.state["position_x"] -= delta
        elif command == "ok_sign":
            self.state["gripper_open"] = True
        elif command == "peace":
            self.state["gripper_open"] = False
        """           
        ee_pos[0] = 0.6 + self.state["position_x"]
        joint_poses = p.calculateInverseKinematics(self.robot, self.ee_index, ee_pos.tolist())

        for i in range(7):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_poses[i], force=200)

        gripper_val = 0.04 if self.state["gripper_open"] else 0.0
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, gripper_val, force=200)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, gripper_val, force=200)

        for _ in range(10):
            p.stepSimulation()
            time.sleep(1 / 240)

    def update(self, current_time: float, image: bytes) -> Optional[bool]:
        if current_time > self.time_limit_seconds + self.start_time_seconds:
            print("Time expired.")
            return False

        recognized_gestures = self.recognizer.recognize(image)

        for gesture in recognized_gestures:
            command = self.gesture_semantics.get(gesture)
            if command:
                print(f"Recognized gesture: {gesture}, executing command: {command}")
                self.apply_command(command)

        self.display_state()

        if (
            abs(self.state["position_x"] - self.goal["position_x"]) < 0.01 and
            self.state["gripper_open"] == self.goal["gripper_open"]
        ):
            print("Success! Goal state reached.")
            return True

        return None


    def close(self):
        p.disconnect()
