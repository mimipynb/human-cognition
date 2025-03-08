"""

    main.py

    Main file to run actions for this module:
    - Setup the inferencing models for chatting inferences
    - Returns the inferencing pipeline for streaming conversational bots in the form of:
        - Decorator
        - Individual functions

"""

import joblib
from enum import Enum
from pathlib import Path

from src.classifier import Classifier
from src.mbti import setup_mbti_chain
from src.dialogue import setup_dialogue, intent_map

root_path = Path(__file__).parent
output_path = root_path / "output"

if not output_path.exists():
    output_path.mkdir(exist_ok=True)

default_file_name = {
    'dialog_emote': output_path / 'dialogue_intensity.joblib',
    'mbti_chain': output_path / 'mbti_chain.joblib'
}

class FeatureLabel(Enum):
    emotion = ['positive', 'negative', 'neutral']
    intent = intent_map[1:]

def setup_cognitive_state():
    """ Setup the inferencing models for chatting inferences. """

    try:
        setup_dialogue(default_file_name['dialog_emote'])
        setup_mbti_chain(default_file_name['mbti_chain'])
        print("Finished setup!")
    except Exception as e:
        print(e)

def load_cognitive_state():
    """ Load all inference tools for agent. """

    try:
        if not any(output_path.iterdir()):
            raise FileExistsError

        models = {}
        for file_path in output_path.iterdir():
            if file_path.suffix in ('joblib', 'pkl'):
                model = joblib.load(file_path)
                models[file_path.stem] = model

        return models
    except Exception as e:
        print(e)

class CogniBee:
    """ For Streaming chat inferences."""

    __slots__ = ('dialog_emote', 'mbti_chain', 'clf')

    def __init__(self):
        for label, load_path in default_file_name.items():
            setattr(self, label, joblib.load(load_path))
            print(f"Added {label} to Cognitive State!")

        self.clf = Classifier()

    def feature_message(self, func):
        """Decorator: Adds message features before calling the original function."""

        def wrapper(chat_session, role, content):
            func(chat_session, role, content)
            # Extract features and store them inside the message
            message = chat_session._messages[-1]
            message.inference = {
                'emotion': self.clf.fetch(content, FeatureLabel.emotion.value, multi_label=True),
                'intent': self.clf.fetch(content, FeatureLabel.intent.value, multi_label=True)
            }
            message.inference['emote_label'] = max(message.inference['emotion'], key=message.inference['emotion'].get)
            message.inference['intent_label'] = max(message.inference['intent'], key=message.inference['intent'].get)

            print(f"Finished basic inferencing for message from {message.role} >> Emote: {message.inference['emote_label']} >> Intent: {message.inference['intent_label']}")

        return wrapper

    def feature_dialogue(self, chat_history):
        """ Takes in history and returns features relevant to history. Still debating on this. """
        """
        If inherits from promptless then:
            # user_dialogue
            embedding = self.model.encode(chat_history)
            self.mbti_chain.fetch(embedding)

        """
        pass