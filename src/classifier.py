"""
    classifier.py

"""


from dataclasses import dataclass
from transformers import pipeline

model_card = "facebook/bart-large-mnli"

@dataclass
class Classifier:
    pipe: pipeline = pipeline("zero-shot-classification", model=model_card)

    def fetch(self, inputs, labels, multi_label=True):
        output = self.pipe(inputs, labels, multi_label=multi_label)

        if set(output.keys()) != set('labels', 'scores', 'sequence'):
            raise ValueError

        return dict(zip(output['labels'], output['scores']))

