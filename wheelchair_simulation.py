import time
import pybullet as p
import pybullet_data
from gesture_recognition import Gesture, GestureRecognizer

class wheelchair_sim:
    def __init__(self, gesture_semantics, recognizer, time_limit_seconds=60):
        self.recognizer = recognizer
        self.gesture_semantics = gesture_semantics
        self.time_limit_seconds = time_limit_seconds
        self.start_time_seconds = -1

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        plane_id = p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("wheelchair.urdf", [0, 0, 0.1])
        self.target_position = [0, 0, 0.1]  # initial position

        print("Wheelchair simulation started.")

    def start(self):
        self.start_time_seconds = time.time()

    def apply_command(self, command: str):
        step_size = 0.1
        if command == "forward":
            self.target_position[1] += step_size
        elif command == "backward":
            self.target_position[1] -= step_size
        elif command == "left":
            self.target_position[0] -= step_size
        elif command == "right":
            self.target_position[0] += step_size

        p.resetBasePositionAndOrientation(self.robot_id, self.target_position, [0, 0, 0, 1])

    def update(self, current_time: float, image: bytes):
        if current_time > self.time_limit_seconds + self.start_time_seconds:
            print("Time expired.")
            return False

        recognized_gestures = self.recognizer.recognize(image, list(self.gesture_semantics.keys()))

        for gesture in recognized_gestures:
            command = self.gesture_semantics.get(gesture)
            if command:
                print(f"Recognized gesture: {gesture.name}, executing command: {command}")
                self.apply_command(command)

        return None