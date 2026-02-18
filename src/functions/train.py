"""
train.py
========

Implements the training pipeline for the Banking Agent Demo application using MLRun and scikit-learn. This module provides a handler function to train a RandomForestClassifier on provided training and test datasets, apply MLRun tracking and analysis features, and log the trained model as an artifact for downstream serving and inference.

Key functionalities include:

- Splitting input DataFrames into features and labels
- Initializing and training a RandomForestClassifier with configurable hyperparameters
- Integrating MLRun for experiment tracking, model analysis, and artifact management
- Logging the trained model artifact and returning its URI for use in deployment pipelines

This module is intended to be used as part of an MLRun pipeline for automated model training, evaluation, and deployment in the banking agent workflow.
"""

import mlrun
import pandas as pd
from mlrun.frameworks.sklearn import apply_mlrun
from sklearn import ensemble


@mlrun.handler(outputs=["model_uri"])
def train_model(
    context: mlrun.MLClientCtx,
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    bootstrap: bool,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_split: int,
    n_estimators: int,
    model_name: str,
):
    """
    Train a RandomForestClassifier and log the model with MLRun.

    Splits the input train and test DataFrames into features and labels, initializes a RandomForestClassifier
    with the provided hyperparameters, wraps the model with MLRun features for tracking and analysis,
    trains the model, and logs the trained model as an artifact in the MLRun project.

    :param context: MLRun context object.
    :param train: Training dataset as a pandas DataFrame.
    :param test: Test dataset as a pandas DataFrame.
    :param label_column: Name of the column containing the target label.
    :param bootstrap: Whether bootstrap samples are used when building trees.
    :param max_depth: The maximum depth of the tree.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
    :param min_samples_split: The minimum number of samples required to split an internal node.
    :param n_estimators: The number of trees in the forest.
    :param model_name: Name to use for logging the model artifact.

    :returns: The URI of the logged model artifact.
    """
    # X, y split
    X_train = train.drop(label_column, axis=1)
    y_train = train[label_column]
    X_test = test.drop(label_column, axis=1)
    y_test = test[label_column]

    # Pick an ideal ML model
    model = ensemble.RandomForestClassifier(
        bootstrap=bootstrap,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
    )

    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    apply_mlrun(model, model_name=model_name, x_test=X_test, y_test=y_test)

    # Train our model
    model.fit(X_train, y_train)
    
    # Log model artifact URI for serving
    import time
    time.sleep(5)  # Ensure model is fully registered before retrieval
    project = context.get_project_object()
    model_artifact = project.list_artifacts(
        name=model_name, iter=1, tag="latest"
    ).to_objects()[0]
    return model_artifact.uri
