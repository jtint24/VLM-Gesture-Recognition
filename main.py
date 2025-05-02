import io
import time

from gesture_recognition import Gesture, GestureRecognizer, capture_image, PromptGestureRecognizer, \
    EmbeddingGestureRecognizer
from model import Model, OllamaModel, OClipModel

import PIL.Image as Image

from oclip.src import app
from wheelchair_simulation import wheelchair_sim
#from simulation import ControlsSimulation

"""if __name__ == "__main__":
    model = Model("llava")
    recognizer = GestureRecognizer(model)

    gestures = [
        Gesture("thumbs-up", "Raising your hand with your thumb pointed upwards."),
        Gesture("head nod", "Moving the head up and down to signal agreement."),
        Gesture("wave", "A hand movement used to greet or say goodbye."),
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
    # app.startup()

    model = OClipModel("hf-hub:apple/MobileCLIP-B-OpenCLIP")

    '''
    gestures = [
        Gesture("smile", "face smiling."),
        Gesture("frown", "face frowning."),
        Gesture("wink", "one eye is closed"),
        Gesture("eyes open", "eyes are both open"),
    ]
    '''
    
    gestures = [
        Gesture("one", "One finger is raised"),
        Gesture("peace", "Two fingers (V shape) are raised"),
        Gesture("thumbs up", "Thumb is pointed upwards"),
        Gesture("thumbs down", "Thumb is pointed downwards"),
    ]


    recognizer = EmbeddingGestureRecognizer(model, gestures, 1)


    '''
    gesture_semantics = {
        gestures[0]: "thermostat_up",
        gestures[1]: "thermostat_down",
        gestures[2]: "lights_on",
        gestures[3]: "lights_off",
    }
    '''
    
    gesture_semantics = {
        gestures[0]: "forward",
        gestures[1]: "backward",
        gestures[2]: "left",
        gestures[3]: "right",
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