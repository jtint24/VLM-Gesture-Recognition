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
        Gesture("smile", "face smiling."),
        Gesture("frown", "face frowning."),
        Gesture("wink", "one eye is closed"),
        Gesture("eyes open", "eyes are both open"),
    ]
    gesture_semantics = {
        gestures[0]: "smile",
        gestures[1]: "frown",
        gestures[2]: "wink",
        gestures[3]: "eyes_open",
    }
    """
    gestures = [
        Gesture("thumbs_up", "One hand with thumb raised straight up, other fingers closed. Palm facing sideways. Thumb not facing downwards"),
        Gesture("thumbs_down", "One hand with thumb pointing straight down, other fingers closed. Palm facing sideways. Thumb not facing updawards"),
        Gesture("ok_sign", "One hand with thumb and index finger forming a circle, other three fingers pointing up."),
        Gesture("peace", "Hand showing index and middle fingers raised in a V-shape, other fingers folded. Palm facing outward.")
    ]

    gesture_semantics = {
        gestures[0]: "thumbs_up",
        gestures[1]: "thumbs_down",
        gestures[2]: "ok_sign",
        gestures[3]: "peace",
    }
"""

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
