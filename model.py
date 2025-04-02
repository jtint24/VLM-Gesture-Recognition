import base64
import json
from json import JSONDecoder
from typing import List, Dict, Union, IO, Optional
from oclip.src import app

import requests
import torch
from ollama import chat, embed

"""
model.py

A simple wrapper for Ollama models
"""

class Model:
    def __init__(self, name: str):
        raise NotImplementedError

    def __call__(self, message: str, images: List[bytes], max_tokens: int = -1):
        pass


class OClipModel(Model):
    def __init__(self, name: str):
        self.name = name

    def embed(self, text: Optional[str] = None, image: Optional[bytes] = None) -> torch.Tensor:
        if image is not None:
            # Process image embedding
            files = [
                ('image', ('image', image, 'application,octet')),
                ('data', ('data', json.dumps([{"model": self.name}]), 'application/json'))
            ]
            response = requests.post("http://localhost:11435/api/embed", files=files, timeout=20)
            response.raise_for_status()
            embeddings = response.json()['embeddings']
            return torch.tensor(embeddings[0])

        elif text is not None:
            # Process text embedding
            response = requests.post(
                "http://localhost:11435/api/embed",
                json={"model": self.name, "input": [text]}
            )
            response.raise_for_status()
            embeddings = response.json()['embeddings']
            return torch.tensor(embeddings[0])

        else:
            raise ValueError("Either text or image must be provided for embedding.")


class OllamaModel(Model):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, message: str, images: List[bytes], max_tokens: int = -1):
        encoded_images = [base64.b64encode(image_data).decode("ascii") for image_data in images]
        response = chat(model=self.name, messages=[
            {
                "role": "user",
                "content": message,
                "images": encoded_images
            },
        ], options={"num_predict": max_tokens})
        return response["message"]["content"]

    def respond(self, messages: List[Dict[str, str]], max_tokens: int = -1):
        response = chat(model=self.name, messages=messages, options={"num_predict": max_tokens})
        return response["message"]["content"]

    def embed(self, message: str):
        return embed(model=self.name, input=message)
