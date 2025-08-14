"""
v2_model_server.py
==================

Implements a custom MLRun V2ModelServer for churn prediction in the Banking Agent Demo application.

This module defines the `ChurnModel` class, which loads a pickled scikit-learn model and serves it via MLRun's V2ModelServer interface. The server receives input samples, computes churn probabilities using the loaded model, and returns the probability of the positive (churn) class for each input.

Intended for use as a backend model server in the serving graph, enabling real-time churn prediction as part of the banking agent workflow.
"""

import warnings
from typing import List

import mlrun
import numpy as np
from cloudpickle import load
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")


class ChurnModel(mlrun.serving.V2ModelServer):
    """
    MLRun V2ModelServer for churn prediction.

    Loads a pickled scikit-learn model and predicts churn likelihood for input samples.
    The model should output class probabilities; this server returns the probability
    of the positive (churn) class for each input.

    :param context: MLRun context.
    :param name: Name of the function.
    """

    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict_proba(feats)
        # Only interested in churn likelihood
        return [i[1] for i in result.tolist()]
