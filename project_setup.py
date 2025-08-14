import os

import mlrun
from pathlib import Path


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source")
    secrets_file = project.get_param("secrets_file")
    base_image = project.get_param("base_image", "mlrun/mlrun")
    requirements_file = project.get_param("requirements_file", "requirements.txt")
    force_build = project.get_param("force_build", False)

    # Set project git/archive source and enable pulling latest code at runtime
    if source:
        print(f"Project Source: {source}")
        project.set_source(project.get_param("source"), pull_at_runtime=True)
        
        if ".zip" in source:
            print(f"Exporting project as zip archive to {source}...")
            project.export(source)

    # Create project secrets and also load secrets in local environment
    if secrets_file and os.path.exists(secrets_file):
        project.set_secrets(file_path=secrets_file)
        mlrun.set_env_from_file(secrets_file)

    # Set default project docker image - functions that do not specify image will use this
    if base_image and requirements_file and force_build:
        requirements = Path(requirements_file).read_text().split()
        commands = [
            f'pip install {" ".join(requirements)}'
        ]
        project.build_image(
            base_image=base_image,
            commands=commands,
            set_as_default=True,
            overwrite_build_params=True,
            with_mlrun=False,
        )

    # MLRun Functions
    project.set_function(
        name="data",
        func="src/functions/data.py",
        kind="job",
    )
    project.set_function(
        name="train",
        func="src/functions/train.py",
        kind="job",
        handler="train_model",
        image=base_image,
    )
    project.set_function(
        name="validate", func="src/functions/validate.py", kind="job"
    )
    project.set_function(
        name="serving",
        func="src/functions/v2_model_server.py",
        kind="serving",
        image=base_image
    )
    project.set_function(
        name="model-server-tester",
        func="src/functions/v2_model_tester.py",
        kind="job",
        handler="model_server_tester",
    )

    # MLRun Workflows
    project.set_workflow("main", "src/workflows/train_and_deploy_workflow.py", image="mlrun/mlrun-kfp:1.8.0")

    # Save and return the project:
    project.save()
    return project
