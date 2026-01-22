"""
toxicity_guardrail.py
=====================

Implements a toxicity guardrail model server for the Banking Agent Demo application. This module defines a custom MLRun Model for:

- Detecting toxic language in user input using the 'toxicity' evaluation module.
- Enforcing input safety by classifying whether the input text is below a configurable toxicity threshold.

This component is intended to be used as part of a serving graph to ensure that user queries do not contain harmful or inappropriate content before further processing by downstream models or agents.
"""

import evaluate
import mlrun


class ToxicityClassifierModelServer(mlrun.serving.Model):
    """
    MLRun Model for toxicity detection.

    Uses the 'toxicity' evaluation module to check if input text contains toxic language.

    :param context: MLRun context.
    :param name: Name of the function.
    :param threshold: Toxicity threshold (default 0.7).
    """

    def __init__(self, name: str, threshold: float = 0.7, **kwargs):
        # Initialize the base server:
        super().__init__(name, **kwargs)

        # Store the threshold of toxicity:
        self.threshold = threshold

    def load(self):
        self.model = evaluate.load("toxicity", module_type="measurement")

    def predict(self, inputs: dict) -> str:
        """
        Predicts whether the input content is below the toxicity threshold.

        :param inputs: A dictionary containing an "inputs" key, which is a list of dictionaries with a "content" key.

        :returns: A list containing a boolean indicating if the predicted toxicity is below the threshold.
        """

        return {"response" :[
            self.model.compute(predictions=[i["content"] for i in inputs["inputs"]])[
                "toxicity"
            ][0]
            < self.threshold
        ]}
