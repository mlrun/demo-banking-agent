import warnings
from typing import List
import typing

import mlrun
import numpy as np
from cloudpickle import load
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")

class ChurnModel(mlrun.serving.Model):
    def __init__(
        self,
        *args,
        artifact_uri: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args, artifact_uri=artifact_uri, **kwargs
        )
        self.artifact_uri = artifact_uri

    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict_proba(feats)
        # Only interested in churn likelihood
        body['results'] = [i[1] for i in result.tolist()]
        return body


    def get_model(self, suffix=""):
        """get the model file(s) and metadata from model store

        the method returns a path to the model file and the extra data (dict of dataitem objects)
        it also loads the model metadata into the self.model_spec attribute, allowing direct access
        to all the model metadata attributes.

        get_model is usually used in the model .load() method to init the model
        Examples
        --------
        ::

            def load(self):
                model_file, extra_data = self.get_model(suffix=".pkl")
                self.model = load(open(model_file, "rb"))
                categories = extra_data["categories"].as_df()

        Parameters
        ----------
        suffix : str
            optional, model file suffix (when the model_path is a directory)

        Returns
        -------
        str
            (local) model file
        dict
            extra dataitems dictionary

        """
        if self.artifact_uri:
            model_file, self.model_spec, extra_dataitems = mlrun.artifacts.get_model(
                self.artifact_uri, suffix
            )
            if self.model_spec and self.model_spec.parameters:
                for key, value in self.model_spec.parameters.items():
                    self._params[key] = value
            return model_file, extra_dataitems
        return None, None