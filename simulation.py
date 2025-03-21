import time
import pybullet as p
import random
from gesture_model import GestureInterpreter
from gesture_recognition import GestureRecognition


def initialize_simulation():
    p.connect(p.GUI)

    # wheelchair_id = p.loadURDF("wheelchair.urdf", basePosition=[0, 0, 0])
    # arm_id = p.loadURDF("robot_arm.urdf", basePosition=[0, 0, 0])

    return wheelchair_id, arm_id


# Task: Navigate robotic wheelchair
def navigate_wheelchair(gesture, wheelchair_id):
    if gesture == "move_forward":
        p.setJointMotorControl2(wheelchair_id, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=1.0)
    elif gesture == "move_backward":
        p.setJointMotorControl2(wheelchair_id, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=-1.0)
    elif gesture == "turn_left":
        p.setJointMotorControl2(wheelchair_id, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=1.0)
    elif gesture == "turn_right":
        p.setJointMotorControl2(wheelchair_id, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=-1.0)


# Task: Robotic arm for object retrieval
def move_robotic_arm(gesture, arm_id):
    if gesture == "move_left":
        p.setJointMotorControl2(arm_id, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=-1.0)
    elif gesture == "move_right":
        p.setJointMotorControl2(arm_id, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=1.0)
    elif gesture == "grasp":
        p.setJointMotorControl2(arm_id, jointIndex=1, controlMode=p.POSITION_CONTROL, targetPosition=0.5)
    elif gesture == "release":
        p.setJointMotorControl2(arm_id, jointIndex=1, controlMode=p.POSITION_CONTROL, targetPosition=0.0)


# Task: lights, thermostat, etc.
def control_environment(gesture):
    if gesture == "turn_on_light":
        print("Turning on the light...")
    elif gesture == "turn_off_light":
        print("Turning off the light...")
    elif gesture == "increase_temp":
        print("Increasing temperature...")
    elif gesture == "decrease_temp":
        print("Decreasing temperature...")


def main():
    wheelchair_id, arm_id = initialize_simulation()

    gesture_recognition = GestureRecognition()
    gesture_interpreter = GestureInterpreter()

    while True:
        captured_gesture = gesture_recognition.recognize_gesture()
        interpreted_action = gesture_interpreter.interpret(captured_gesture)

        if interpreted_action in ["move_forward", "move_backward", "turn_left", "turn_right"]:
            navigate_wheelchair(interpreted_action, wheelchair_id)
        elif interpreted_action in ["move_left", "move_right", "grasp", "release"]:
            move_robotic_arm(interpreted_action, arm_id)
        elif interpreted_action in ["turn_on_light", "turn_off_light", "increase_temp", "decrease_temp"]:
            control_environment(interpreted_action)

        time.sleep(1)


if __name__ == "__main__":
    main()
