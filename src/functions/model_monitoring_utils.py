import os

import mlrun
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    DatastoreProfilePostgreSQL,
    DatastoreProfileV3io,
)


def enable_model_monitoring(
    project: mlrun.projects.MlrunProject = None,
    tsdb_profile_name: str = "tsdb-profile",
    stream_profile_name: str = "stream-profile",
    base_period: int = 10,
    wait_for_deployment: bool = False,
    deploy_histogram_data_drift_app: bool = True,
) -> mlrun.projects.MlrunProject:
    """
    Enables model monitoring for an MLRun project by configuring and registering the required datastore profiles,
    setting model monitoring credentials, and enabling the model monitoring feature.

    This function sets up the necessary TSDB and stream profiles for model monitoring, handling both Community Edition (CE)
    and non-CE modes. It registers the profiles with the project, sets the credentials, and enables model monitoring with
    the specified configuration.

    :param project: The MLRun project to enable model monitoring on.
    :param tsdb_profile_name: Name for the TSDB datastore profile (default: "tsdb-profile").
    :param stream_profile_name: Name for the stream datastore profile (default: "stream-profile").
    :param base_period: The base period (in seconds) for model monitoring scheduling (default: 10).
    :param wait_for_deployment: Whether to wait for the model monitoring deployment to complete (default: False).
    :param deploy_histogram_data_drift_app: Whether to deploy the histogram data drift application (default: True).

    :returns: The updated MLRun project with model monitoring enabled.
    """
    # Setting model monitoring creds
    tsdb_profile = DatastoreProfileV3io(name=tsdb_profile_name)
    stream_profile = DatastoreProfileV3io(
        name=stream_profile_name, v3io_access_key=mlrun.mlconf.get_v3io_access_key()
    )

    if mlrun.mlconf.is_ce_mode():
        mlrun_namespace = os.environ.get("MLRUN_NAMESPACE", "mlrun")
        tsdb_profile = DatastoreProfilePostgreSQL(
            name=tsdb_profile_name,
            user="root",
            password="taosdata",
            host=f"timescaledb.{mlrun_namespace}.svc.cluster.local",
            port="5432", #the last one was 6041
            database="mlrun",
        )

        stream_profile = DatastoreProfileKafkaSource(
            name=stream_profile_name,
            brokers=f"kafka-stream.{mlrun_namespace}.svc.cluster.local:9092",
            topics=[],
        )

    project.register_datastore_profile(stream_profile)
    project.register_datastore_profile(tsdb_profile)

    project.set_model_monitoring_credentials(
        replace_creds=True,
        tsdb_profile_name=tsdb_profile.name,
        stream_profile_name=stream_profile.name,
    )

    try:
        project.enable_model_monitoring(
            base_period=base_period,
            wait_for_deployment=wait_for_deployment,
            deploy_histogram_data_drift_app=deploy_histogram_data_drift_app,
        )
    except mlrun.errors.MLRunConflictError:
        print("Model monitoring is already enabled for this project.")
    return project
