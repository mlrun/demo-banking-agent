import mlrun
from kfp import dsl


@dsl.pipeline(name="Churn Training Pipeline")
def pipeline(
    source_url: str,
    label_column: str,
    allow_validation_failure: bool,
    test_size: float,
    model_name: str,
    sentiment_model: str,
    text_column: str,
    sentiment_column: str,
    ordinal_columns: list,
    drop_columns: list,
):
    """
    Orchestrates the end-to-end workflow for training and deploying a machine learning model.

    This pipeline performs the following steps:
    - Ingests data from the specified source URL.
    - Computes sentiment analysis on the text column using the provided sentiment model.
    - Validates data integrity.
    - Processes the data, including splitting into train and test sets, encoding ordinal features, dropping specified columns, and adding sentiment.
    - Validates the train/test split.
    - Trains a model with hyperparameter optimization.
    - Validates the trained model.
    - Deploys the trained model to a serving endpoint.

    :param source_url: The URL or path to the source data.
    :param label_column: The name of the column containing the target labels.
    :param allow_validation_failure: Whether to allow validation steps to fail without stopping the pipeline.
    :param test_size: The proportion of the dataset to include in the test split.
    :param model_name: The name to assign to the trained model.
    :param sentiment_model: The name or path of the sentiment analysis model to use.
    :param text_column: The name of the column containing text data for sentiment analysis.
    :param sentiment_column: The name of the column to store sentiment analysis results.
    :param ordinal_columns: List of column names to treat as ordinal features.
    :param drop_columns: List of column names to drop from the dataset.
    """
    # Get our project object
    project = mlrun.get_current_project()

    # Ingest data
    ingest = project.run_function(
        "data",
        handler="get_data",
        inputs={"data": source_url},
        outputs=["data"],
    )

    # Compute sentiment data
    sentiment_fn = project.get_function("data")
    sentiment_fn.with_requests(cpu=2)
    sentiment_fn.with_limits(cpu=6)
    sentiment = project.run_function(
        sentiment_fn,
        handler="sentiment_analysis",
        inputs={"data": ingest.outputs["data"]},
        params={
            "sentiment_model": sentiment_model,
            "text_column": text_column,
        },
        outputs=["data_w_sentiment"],
    )

    # Validate data integrity
    validate_data_integrity = project.run_function(
        "validate",
        handler="validate_data_integrity",
        inputs={"data": sentiment.outputs["data_w_sentiment"]},
        params={
            "label_column": label_column,
            "allow_validation_failure": allow_validation_failure,
        },
        outputs=["passed_suite"],
    )

    # Process data
    process = project.run_function(
        "data",
        handler="process_data",
        inputs={"data": sentiment.outputs["data_w_sentiment"]},
        params={
            "label_column": label_column,
            "test_size": test_size,
            "ordinal_columns": ordinal_columns,
            "drop_columns": drop_columns,
            "sentiment_column": sentiment_column,
        },
        outputs=["train", "test"],
    ).after(validate_data_integrity)

    # Validate train test split
    validate_train_test_split = project.run_function(
        "validate",
        handler="validate_train_test_split",
        inputs={"train": process.outputs["train"], "test": process.outputs["test"]},
        params={
            "label_column": label_column,
            "allow_validation_failure": allow_validation_failure,
        },
        outputs=["passed_suite"],
    )

    train = project.run_function(
        "train",
        inputs={
            "train": process.outputs["train"],
            "test": process.outputs["test"],
        },
        params={"label_column": label_column, "model_name": model_name},
        hyperparams={
            "bootstrap": [True, False],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        },
        selector="max.accuracy",
        hyper_param_options=mlrun.model.HyperParamOptions(
            strategy="random", max_iterations=5
        ),
        outputs=["model", "model_uri"],
    ).after(validate_train_test_split)

    validate_model = project.run_function(
        "validate",
        handler="validate_model",
        inputs={
            "train": process.outputs["train"],
            "test": process.outputs["test"],
        },
        params={
            "model_path": train.outputs["model_uri"],
            "label_column": label_column,
            "allow_validation_failure": allow_validation_failure,
        },
        outputs=["passed_suite"],
    )

    # Deploy model to endpoint
    serving_fn = project.get_function("serving")

    graph = serving_fn.set_topology("flow", engine="async")
    model_runner_step = mlrun.serving.ModelRunnerStep()

    model_runner_step.add_model(
        model_class="ChurnModel",
        artifact_uri=train.outputs["model_uri"],
        endpoint_name="banking-topic-guardrail",
        execution_mechanism="naive",
        model_name=model_name,
    )
    graph.to(model_runner_step).respond()
    serving_fn.set_tracking()
    func_path = serving_fn.save()
    project.set_function(name="serving", func=func_path)
    
    deploy = project.deploy_function("serving"
    ).after(validate_model)
