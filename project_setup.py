import os
import tarfile
from pathlib import Path

import mlrun


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source", default=None)
    build_image = project.get_param("build_image", default=False)

    # Adding secrets to the projects:
    assert (os.environ.get("OPENAI_API_KEY", None) is not None) and (
        os.environ.get("OPENAI_BASE_URL", None) is not None
    ), (
        "\
    Missing OpenAI credentials, make sure they are set as environment variables."
    )

    project.set_secrets(
        {
            "OPENAI_API_KEY": mlrun.get_secret_or_env("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": mlrun.get_secret_or_env("OPENAI_BASE_URL"),
        }
    )

    # Set project git/archive source and enable pulling latest code at runtime
    if not source:
        print("Setting Source for the demo:")
        make_archive("../banking_agent", "gztar", "./", exclude=["project.yaml"])
        # Logging as artifact
        proj_artifact = project.log_artifact(
            "project_source", local_path="../banking_agent.tar.gz", upload=True
        )
        os.remove("../banking_agent.tar.gz")
        project.set_source(source=proj_artifact.target_path, pull_at_runtime=False)
        print(f"Project Source: {source}")
        source = proj_artifact.target_path

    project.set_source(source, pull_at_runtime=False)
    project.set_default_image(f".mlrun-project-image-{project.name}")

    # Set default project docker image - functions that do not specify image will use this
    if build_image:
        print("Building default image for the demo:")
        requirements = Path("requirements-churn.txt").read_text().split()
        project.build_image(
            image=project.default_image,
            base_image="mlrun/mlrun",
            set_as_default=True,
            overwrite_build_params=True,
            with_mlrun=False,
            # requirements_file="requirements-churn.txt"
            commands = [
                "apt-get update && apt-get install -y curl build-essential && "
                "curl https://sh.rustup.rs -sSf | bash -s -- -y && "
                ". $HOME/.cargo/env && "
                "rustup default nightly && "
                "pip install --upgrade pip && "
                "pip install --index-url https://download.pytorch.org/whl/cpu "
                "--extra-index-url https://pypi.org/simple "
                f"{' '.join(requirements)}"
            ],
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
    )
    project.set_function(name="validate", func="src/functions/validate.py", kind="job")
    project.set_function(
        name="serving",
        func="src/functions/v2_model_server.py",
        kind="serving",
    )
    project.set_function(
        name="model-server-tester",
        func="src/functions/v2_model_tester.py",
        kind="job",
        handler="model_server_tester",
    )

    # MLRun Workflows
    project.set_workflow(
        "main",
        "src/workflows/train_and_deploy_workflow.py",
        image="mlrun/mlrun-kfp",
    )

    # Save and return the project:
    project.save()
    return project


def make_archive(base_name, format="gztar", root_dir=".", exclude=None):
    """
    Create a tar.gz archive with exclusions.

    Args:
        base_name (str): Output file name (without extension).
        format (str): Archive format ("gztar", "tar").
        root_dir (str): Root directory to archive.
        exclude (list): Filenames (or directory names) to exclude.
    """
    exclude = set(exclude or [])
    suffix = ".tar.gz" if format == "gztar" else ".tar"
    archive_name = base_name + suffix

    mode = "w:gz" if format == "gztar" else "w"
    with tarfile.open(archive_name, mode) as tar:
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f in exclude:
                    continue
                path = os.path.join(root, f)
                arcname = os.path.relpath(path, root_dir)
                tar.add(path, arcname=arcname)
    return archive_name
