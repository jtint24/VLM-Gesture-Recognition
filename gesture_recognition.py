from typing import Optional, NamedTuple

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

    def recognize(self, image_bytes: bytes, gestures: list[Gesture]) -> list[Gesture]:
        """
        Given an image and a list of known gestures, return which ones are being performed.
        """

        gesture_list_text = "\n".join(
            f"{n+1}. {gesture.name}: {gesture.description}" for n, gesture in enumerate(gestures)
        )

        prompt = f"For each of these gestures, list if it's present in the image. Respond with the gesture's name, then YES or NO: \n{gesture_list_text}"

        response = self.model(prompt, [image_bytes], max_tokens=500)

        response_lines = response.split("\n")
        print(prompt, "\n", response)
        recognized = []

        for gesture in gestures:
            for line in response_lines:
                if gesture.name in line.lower() and "yes" in line.lower():
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
    resized_frame = cv2.resize(frame, (360, 240))
    success, encoded_image = cv2.imencode(".jpg", resized_frame)
    if not success:
        print("Error: Failed To Encode Image!!!!!!!!")
        return None

    image_bytes = io.BytesIO(encoded_image).getvalue()
    return image_bytes