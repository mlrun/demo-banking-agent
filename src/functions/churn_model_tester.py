"""
churn_model_tester.py
==================

Implements a utility function for testing MLRun model serving functions. This module provides a tester that:

- Sends test data to a deployed model serving function endpoint
- Compares model predictions to expected labels
- Logs test statistics, including accuracy and latency metrics
- Optionally raises errors on mismatches or failed requests

Intended for validating model deployments in the Banking Agent Demo or similar MLRun-based applications.
"""

import json
from datetime import datetime

import numpy as np
import requests
from mlrun.artifacts import ChartArtifact
from mlrun.datastore import DataItem


def model_server_tester(
    context,
    table: DataItem,
    serving_fn_name: str,
    label_column: str = "label",
    model: str = "",
    match_err: bool = False,
    rows: int = 20,
):
    """Test a model serving function by sending test data and validating predictions.

    Sends each row of the provided test dataset to the specified model serving function,
    compares the predicted output to the expected label, and logs test statistics.

    :param table:         DataItem containing the test dataset (csv/parquet)
    :param serving_fn_name: name of the deployed serving function to test
    :param label_column:  name of the label column in the dataset
    :param model:         name of the model to test (used in the request URL)
    :param match_err:     if True, raises an error if any prediction does not match the label
    :param rows:          number of rows to sample from the test set for testing
    """
    project = context.get_project_object()
    serving_fn = project.get_function(serving_fn_name, ignore_cache=True)
    addr = serving_fn.get_url()

    table = table.as_df()

    y_list = table.pop(label_column).values.tolist()
    context.logger.info(f"testing with dataset against {addr}, model: {model}")
    if rows and rows < table.shape[0]:
        table = table.sample(rows)

    count = err_count = match = 0
    times = []
    for x, y in zip(table.values, y_list):
        count += 1
        event_data = json.dumps({"inputs": [x.tolist()]})
        had_err = False
        try:
            start = datetime.now()
            resp = requests.put(f"{addr}/v2/models/{model}/infer", json=event_data)
            if not resp.ok:
                context.logger.error(f"bad function resp!!\n{resp.text}")
                err_count += 1
                continue
            times.append((datetime.now() - start).microseconds)

        except OSError as err:
            context.logger.error(f"error in request, data:{event_data}, error: {err}")
            err_count += 1
            continue

        resp_data = resp.json()
        print(resp_data)
        y_resp = resp_data["outputs"][0]
        if y == y_resp:
            match += 1

    context.log_result("total_tests", count)
    context.log_result("errors", err_count)
    context.log_result("match", match)
    if count - err_count > 0:
        times_arr = np.array(times)
        context.log_result("avg_latency", int(np.mean(times_arr)))
        context.log_result("min_latency", int(np.amin(times_arr)))
        context.log_result("max_latency", int(np.amax(times_arr)))

        chart = ChartArtifact("latency", header=["Test", "Latency (microsec)"])
        for i in range(len(times)):
            chart.add_row([i + 1, int(times[i])])
        context.log_artifact(chart)

    context.logger.info(
        f"run {count} tests, {err_count} errors and {match} match expected value"
    )

    if err_count:
        raise ValueError(f"failed on {err_count} tests of {count}")

    if match_err and match != count:
        raise ValueError(f"only {match} results match out of {count}")
