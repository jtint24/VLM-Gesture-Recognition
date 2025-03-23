import random
import time
from typing import Dict, Optional

import pybullet as p

from gesture_recognition import Gesture, GestureRecognizer


class ControlsSimulation:
    def __init__(
            self,
            gesture_semantics: Dict[Gesture, str],
            recognizer: GestureRecognizer,
            time_limit_seconds = 60,
            start_state = None,
            end_state = None
    ):
        self.recognizer = recognizer
        self.gesture_semantics = gesture_semantics
        self.time_limit_seconds = time_limit_seconds
        self.start_time_seconds = -1

        self.state = start_state or {
            "thermostat": random.randint(65, 70),
            "lights_on": random.choice([True, False])
        }

        self.goal = end_state or {
            "thermostat": random.randint(65, 70),
            "lights_on": random.choice([True, False])
        }

        print("Simulation started.")
        self.display_state()

    def start(self):
        self.start_time_seconds = time.time()

    def display_state(self):
        print(f"\n--- CURRENT STATE ---")
        print(f"Thermostat: {self.state['thermostat']}°F")
        print(f"Lights: {'On' if self.state['lights_on'] else 'Off'}")

        print(f"\n--- GOAL STATE ---")
        print(f"Thermostat: {self.goal['thermostat']}°F")
        print(f"Lights: {'On' if self.goal['lights_on'] else 'Off'}\n")

    def apply_command(self, command: str):
        if command == "thermostat_up":
            self.state["thermostat"] += 1
        elif command == "thermostat_down":
            self.state["thermostat"] -= 1
        elif command == "lights_on":
            self.state["lights_on"] = True
        elif command == "lights_off":
            self.state["lights_on"] = False

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

        if self.state == self.goal:
            print("Success! Goal state reached.")
            return True

        return None

