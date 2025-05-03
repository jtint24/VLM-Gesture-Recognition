from typing import Optional, NamedTuple, List

import cv2
import time
import io

import numpy as np

from model import Model


class Gesture(NamedTuple):
    name: str
    description: str


class GestureRecognizer:
    def __init__(self, model: Model, gestures: List[Gesture]):
        raise NotImplementedError

    def recognize(self, image_bytes: bytes) -> List[Gesture]:
        pass

class PromptGestureRecognizer(GestureRecognizer):
    def __init__(self, model: Model, gestures: List[Gesture]):
        self.model = model
        self.gestures = gestures

    def recognize(self, image_bytes: bytes) -> List[Gesture]:
        """
        Given an image and a list of known gestures, return which ones are being performed.
        """
        recognized = []
        for gesture in self.gestures:
            prompt = f"Is the following gesture being performed in the image?\n\n{gesture.name}: {gesture.description}\n\nRespond only with YES or NO."

            response = self.model(prompt, [image_bytes], max_tokens=2)

            # print(prompt,response)

            if "yes" in response.lower():
                recognized.append(gesture)

        return recognized



class EmbeddingGestureRecognizer(GestureRecognizer):
    def __init__(self, model: Model, gestures: List[Gesture], threshold: float):
        self.model = model
        self.threshold = threshold
        self.gestures = gestures
        self.gesture_embeddings = {
            gesture: self.model.embed(text=gesture.description)
            for gesture in self.gestures
        }

    def recognize(self, image_bytes: bytes) -> List[Gesture]:
        """
        Given an image and a list of known gestures, return which ones are being performed.
        """
        recognized = []

        image_embedding = self.model.embed(image=image_bytes)

        img_emb_norm = image_embedding / np.linalg.norm(image_embedding)

        for gesture, gesture_embedding in self.gesture_embeddings.items():
            gesture_emb_norm = gesture_embedding / np.linalg.norm(gesture_embedding)
            
            # Calculate cosine similarity explicitly
            similarity = np.dot(img_emb_norm, gesture_emb_norm)

            # Threshold now means similarity score between -1 and 1
            if similarity >= self.threshold:
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