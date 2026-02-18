"""
validate.py
===========

Implements MLRun handler functions for validating tabular data and models using Deepchecks. This module provides utilities for:

- Creating Deepchecks Dataset objects from pandas DataFrames
- Running data integrity checks on datasets
- Validating train-test splits for consistency and quality
- Evaluating trained models using Deepchecks suites

Each handler logs results and HTML reports to the MLRun context, and can be configured to allow or disallow validation failures.
"""

import cloudpickle
import mlrun
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import (
    data_integrity,
    model_evaluation,
    train_test_validation,
)


def create_deepchecks_dataset(df: pd.DataFrame, label_column: str) -> Dataset:
    """
    Creates a Deepchecks Dataset object from a pandas DataFrame, specifying the label column
    and automatically detecting categorical features.

    :param df: The input pandas DataFrame containing the data.
    :param label_column: The name of the column to be used as the label.

    :returns: A Deepchecks Dataset object with the specified label and detected categorical features.
    """
    categorical_columns = list(
        set(df.select_dtypes("object").columns) - set([label_column])
    )
    return Dataset(df=df, label=label_column, cat_features=categorical_columns)


@mlrun.handler()
def validate_data_integrity(
    context: mlrun.MLClientCtx,
    data: pd.DataFrame,
    label_column: str,
    allow_validation_failure: bool = False,
):
    """
    MLRun handler for validating the integrity of a pandas DataFrame using Deepchecks.

    Creates a Deepchecks dataset from the provided DataFrame and label column, runs the data integrity suite,
    logs the results and report to the MLRun context, and asserts that the suite passes unless validation failures are allowed.

    :param context: MLRun context for logging results and artifacts.
    :param data: The pandas DataFrame to validate.
    :param label_column: The name of the label column in the DataFrame.
    :param allow_validation_failure: If True, allows the validation to fail without raising an assertion error (default: False).
    """
    # Create deepchecks dataset with column metadata
    dataset = create_deepchecks_dataset(df=data, label_column=label_column)

    # Run suite
    data_integrity_suite = data_integrity()
    suite_result = data_integrity_suite.run(dataset)

    # Export results
    passed_suite = suite_result.passed()
    suite_report = suite_result.save_as_html()

    context.log_result("passed_suite", passed_suite)
    context.log_artifact("suite_report", local_path=suite_report)

    assert allow_validation_failure or passed_suite == True, (
        "Data integrity validation failed"
    )


@mlrun.handler()
def validate_train_test_split(
    context: mlrun.MLClientCtx,
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    allow_validation_failure: bool = False,
):
    """
    MLRun handler for validating the train-test split using Deepchecks.

    Creates Deepchecks datasets from the provided train and test DataFrames, runs the train-test validation suite,
    logs the results and report to the MLRun context, and asserts that the suite passes unless validation failures are allowed.

    :param context: MLRun context for logging results and artifacts.
    :param train: The training dataset as a pandas DataFrame.
    :param test: The test dataset as a pandas DataFrame.
    :param label_column: The name of the label column in the datasets.
    :param allow_validation_failure: If True, allows the validation to fail without raising an assertion error (default: False).
    """
    # Create deepchecks dataset with column metadata
    train_dataset = create_deepchecks_dataset(df=train, label_column=label_column)
    test_dataset = create_deepchecks_dataset(df=test, label_column=label_column)

    # Run suite
    train_test_validation_suite = train_test_validation()
    suite_result = train_test_validation_suite.run(
        train_dataset=train_dataset, test_dataset=test_dataset
    )

    # Export results
    passed_suite = suite_result.passed()
    suite_report = suite_result.save_as_html()

    context.log_result("passed_suite", passed_suite)
    context.log_artifact("suite_report", local_path=suite_report)

    assert allow_validation_failure or passed_suite == True, (
        "Train test split validation failed"
    )


@mlrun.handler()
def validate_model(
    context: mlrun.MLClientCtx,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_path: str,
    label_column: str,
    allow_validation_failure: bool = False,
):
    """
    MLRun handler for validating a trained model using Deepchecks.

    Loads a model from the specified path, prepares Deepchecks datasets for the training and test data,
    runs the model evaluation suite, logs the results and report to the MLRun context, and asserts that
    the suite passes unless validation failures are allowed.

    :param context: MLRun context for logging results and artifacts.
    :param train: The training dataset as a pandas DataFrame.
    :param test: The test dataset as a pandas DataFrame.
    :param model_path: Path to the trained model artifact.
    :param label_column: Name of the label column in the datasets.
    :param allow_validation_failure: If True, allows the validation to fail without raising an assertion error (default: False).
    """
    # Load model
    model_file, *_ = mlrun.artifacts.get_model(model_path)
    with open(model_file, "rb") as f:
        model = cloudpickle.load(f)

    # Create deepchecks dataset with column metadata
    train_dataset = create_deepchecks_dataset(df=train, label_column=label_column)
    test_dataset = create_deepchecks_dataset(df=test, label_column=label_column)

    # Run suite
    model_evaluation_suite = model_evaluation()
    suite_result = model_evaluation_suite.run(
        train_dataset=train_dataset, test_dataset=test_dataset, model=model
    )

    # Export results
    passed_suite = suite_result.passed()
    suite_report = suite_result.save_as_html()

    context.log_result("passed_suite", passed_suite)
    context.log_artifact("suite_report", local_path=suite_report)

    assert allow_validation_failure or passed_suite == True, "Model validation failed"
