import ollama


class GestureInterpreter:
    def __init__(self):
        self.model = ollama.load("Llava")

    def interpret(self, gesture):
        # interpret the gesture using the VLM; the model will generate semantically rich actions based on the gesture description.
        gesture_description = self.get_gesture_description(gesture)
        input_text = f"Interpret this gesture: {gesture_description}"

        response = self.model.generate(input_text)

        return response.strip()

    def get_gesture_description(self, gesture):
        # Return a description for the gesture
        gesture_descriptions = {
            "move_forward": "A forward movement gesture",
            "move_backward": "A backward movement gesture",
            "turn_left": "A left turn gesture",
            "turn_right": "A right turn gesture",
            "move_left": "Move the arm to the left",
            "move_right": "Move the arm to the right",
            "grasp": "Grasp an object",
            "release": "Release an object",
            "turn_on_light": "Turn on the light",
            "turn_off_light": "Turn off the light",
            "increase_temp": "Increase temperature",
            "decrease_temp": "Decrease temperature",
        }
        return gesture_descriptions.get(gesture, "Unknown gesture")

