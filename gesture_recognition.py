from typing import Optional, NamedTuple, List

import cv2
import time
import io

from model import Model


class Gesture(NamedTuple):
    name: str
    description: str


class GestureRecognizer:
    def __init__(self, model: Model):
        self.model = model

    def recognize(self, image_bytes: bytes, gestures: List[Gesture]) -> List[Gesture]:
        """
        Given an image and a list of known gestures, return which ones are being performed.
        """
        recognized = []
        for gesture in gestures:
            prompt = f"Is the following gesture being performed in the image?\n\n{gesture.name}: {gesture.description}\n\nRespond only with YES or NO."

            response = self.model(prompt, [image_bytes], max_tokens=2)

            print(prompt,response)

            if "yes" in response.lower():
                recognized.append(gesture)

        return recognized


def capture_image() -> Optional[bytes]:
    """
    Captures a single image from webcam, converts it to bytes, and runs gesture recognition.
    """
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # Warm up the camera

    if not cap.isOpened():
        print("Error: Could not Open Webcam!")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to Capture Image!!")
        return None

    # resize and encode to memory buffer
    resized_frame = cv2.resize(frame, (180, 120))
    success, encoded_image = cv2.imencode(".jpg", resized_frame)
    if not success:
        print("Error: Failed To Encode Image!!!!!!!!")
        return None

    image_bytes = io.BytesIO(encoded_image).getvalue()
    return image_bytes