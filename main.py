import io
import time

from gesture_recognition import Gesture, GestureRecognizer, capture_image
from model import Model

import PIL.Image as Image

from simulation import ControlsSimulation

"""if __name__ == "__main__":
    model = Model("llava")
    recognizer = GestureRecognizer(model)

    gestures = [
        Gesture("thumbs-up", "Raising your hand with your thumb pointed upwards."),
        Gesture("head nod", "Moving the head up and down to signal agreement."),
        Gesture("wave", "A hand movement used to greet or say goodbye."),
    ]
    
    gestures = [
        Gesture("smile", "face smiling."),
        Gesture("frown", "face frowning."),
        Gesture("wink", "one eye is closed"),
        Gesture("eyes open", "eyes are both open"),
    ]
    
    hand_gestures = [
        Gesture("open palm", "palm is open"),
        Gesture("closed fist", "fist is closed"),
        Gesture("one", "one finger is raised"),
        Gesture("peace", "two fingers are raised")
    ]

    print("capping image...")
    image = capture_image()

    image_file = Image.open(io.BytesIO(image))
    image_file.save("frame.jpg")

    if image is not None:
        print("recognizing gestures...")
        recognized_gestures = recognizer.recognize(image, gestures)

        if recognized_gestures:
            print("Recognized gestures:")
            for g in recognized_gestures:
                print(f"- {g.name}")
        else:
            print("no gestures recognized")"""


if __name__ == "__main__":
    model = Model("llava")
    recognizer = GestureRecognizer(model)

    gestures = [
        Gesture("open palm", "palm is open"),
        Gesture("closed fist", "fist is closed"),
        Gesture("one", "one finger is raised"),
        Gesture("peace", "two fingers are raised")
    ]

    gesture_semantics = {
        gestures[0]: "thermostat_up", #open
        gestures[1]: "thermostat_down",  #closed
        gestures[2]: "lights_on",  #one
        gestures[3]: "lights_off", #two
    }

    simulation = ControlsSimulation(
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
            print(f"Task completed successfully! time: {time.time() - simulation.start_time_seconds}")
            break
        elif result is False:
            print("Task failed")
            break
