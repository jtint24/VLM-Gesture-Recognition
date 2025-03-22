import io

from gesture_recognition import Gesture, GestureRecognizer, capture_image
from model import Model

import PIL.Image as Image

if __name__ == "__main__":
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
            print("no gestures recognized")
