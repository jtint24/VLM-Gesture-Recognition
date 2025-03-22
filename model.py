import base64
from json import JSONDecoder
from typing import List, Dict, Union, IO

import requests
from ollama import chat, embed

"""
model.py

A simple wrapper for Ollama models
"""

class Model:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, message: str, images: Union[List[bytes], List[str]], max_tokens: int = -1):
        encoded_images = [base64.b64encode(image_file).decode("ascii") for image_file in images]
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
