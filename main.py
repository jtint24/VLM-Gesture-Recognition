import io
import time
import PIL.Image as Image

from gesture_recognition import Gesture, GestureRecognizer, capture_image
from model import Model
from robo_arm_sim import PandaArmSimulation  

if __name__ == "__main__":
    model = Model("llava")
    recognizer = GestureRecognizer(model)
    gestures = [
        Gesture("two fingers", "Show two fingers to move the robot forward."),
        Gesture("three fingers", "Show three fingers to move the robot backward."),
        Gesture("open hand", "Show open hand to open the gripper."),
        Gesture("closed fist", "Show closed fist to close the gripper."),
    ]

    gesture_semantics = {
        gestures[0]: "two_fingers",
        gestures[1]: "three_fingers",
        gestures[2]: "grip_open",
        gestures[3]: "grip_close",
    }


    simulation = PandaArmSimulation(
        gesture_semantics=gesture_semantics,
        recognizer=recognizer,
        time_limit_seconds=60
    )

    simulation.start()
    while True:
        image = capture_image()
        if image is None:
            break

        try:
            image_file = Image.open(io.BytesIO(image))
            image_file.save("frame.jpg")
        except Exception as e:
            print(f"ERROR: {e}")

        result = simulation.update(time.time(), image)

        if result is True:
            print(f"Task completed successfully! Time: {time.time() - simulation.start_time_seconds:.2f} seconds")
            break
        elif result is False:
            print("Task failed.")
            break
