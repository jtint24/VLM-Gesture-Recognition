import io
import time
import PIL.Image as Image

from gesture_recognition import Gesture, GestureRecognizer, capture_image
from model import Model
from robo_arm_sim import PandaArmSimulation  # Make sure this matches the name in your simulation.py

if __name__ == "__main__":
    model = Model("llava")
    recognizer = GestureRecognizer(model)

    gestures = [
        Gesture("point up", "Pointing upward with one finger."),
        Gesture("point down", "Pointing downward."),
        Gesture("point left", "Pointing to the left."),
        Gesture("point right", "Pointing to the right."),
        Gesture("point forward", "Pointing toward the camera."),
        Gesture("point backward", "Pointing away from the camera."),
    ]

    gesture_semantics = {
        gestures[0]: "point_up",
        gestures[1]: "point_down",
        gestures[2]: "point_left",
        gestures[3]: "point_right",
        gestures[4]: "point_forward",
        gestures[5]: "point_backward",
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
