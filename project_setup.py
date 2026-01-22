import os
import tarfile
from pathlib import Path
import fnmatch

import mlrun

def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source", default=None)
    build_image = project.get_param("build_image", default=False)
    set_functions = project.get_param("set_functions", default=False)

    project.set_secrets(
        {
            "OPENAI_API_KEY": mlrun.get_secret_or_env("OPENAI_API_KEY", default=''),
            "OPENAI_BASE_URL": mlrun.get_secret_or_env("OPENAI_BASE_URL", default=''),
        }
    )

    openai_available = mlrun.get_secret_or_env("OPENAI_API_KEY", default=False)
    if openai_available:
        register_openai_profile(project=project)
    # Set project git/archive source and enable pulling latest code at runtime
    if not source:
        print("Setting Source for the demo:")
        make_archive("../banking_agent", "gztar", "./",
                     exclude=[  ".venv", "venv", "node_modules",
                                "__pycache__", "*.pyc", "*.pyo", ".DS_Store",
                                ".git", ".pytest_cache", ".mypy_cache",
                                "project.yaml", "images", ".venv", "venv" ])        # Logging as artifact
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
        requirements = Path("requirements.txt").read_text().split()
        project.build_image(
            image=project.default_image,
            base_image="mlrun/mlrun",
            set_as_default=True,
            overwrite_build_params=True,
            with_mlrun=False,
            # requirements_file="requirements-churn.txt"
            commands = [
                "pip install -U --index-url https://download.pytorch.org/whl/cpu "
                "--extra-index-url https://pypi.org/simple "
                f"{' '.join(requirements)}"
            ],
        )
    if set_functions:
        print("setting functions")
        # MLRun Functions
        project.set_function(
            name="data",
            func="src/functions/data.py",
            kind="job",
        ).save()
        project.set_function(
            name="train",
            func="src/functions/train.py",
            kind="job",
            handler="train_model",
        ).save()
        project.set_function(
            name="validate",
            func="src/functions/validate.py",
            kind="job"
        ).save()

        project.set_function(
            name="serving",
            func="src/functions/churn_model.py",
            kind="serving",
        ).save()
        project.set_function(
            name="model-server-tester",
            func="src/functions/churn_model_tester.py",
            kind="job",
            handler="model_server_tester",
        ).save()

        if not openai_available:
            project.set_function(
                name="banking-topic-guardrail",
                func="src/functions/banking_topic_guardrail.py",
                kind="serving",
        ).save()

        else:
            project.set_function(
                mlrun.new_function(
                name="banking-topic-guardrail",
                kind="serving",
                )
            ).save()

        project.set_function(
            name="toxicity-guardrail",
            func="src/functions/toxicity_guardrail.py",
            kind="serving",
            ).save()

        project.set_function(
            name="banking-agent",
            func="src/functions/agent_graph.py",
            kind="serving",
            image=project.default_image,
        ).save()

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

    exclude can contain:
      - directory names: ".venv", "venv", "images"
      - file names: "project.yaml"
      - relative paths from root_dir: "data/bigfile.bin"
      - glob patterns: "__pycache__", "*.pyc", ".venv/*"
    """
    exclude = exclude or []
    suffix = ".tar.gz" if format == "gztar" else ".tar"
    archive_name = base_name + suffix

    mode = "w:gz" if format == "gztar" else "w"

    def is_excluded(relpath: str) -> bool:
        # normalize to forward slashes so patterns work on Windows too
        rel = relpath.replace(os.sep, "/")
        base = os.path.basename(rel)

        for pat in exclude:
            pat_norm = pat.replace(os.sep, "/").rstrip("/")

            # Match:
            # - exact basename (e.g. "project.yaml", ".venv")
            # - exact relpath (e.g. "foo/bar.txt")
            # - glob on basename or relpath (e.g. "*.pyc", "__pycache__", ".venv/*")
            if base == pat_norm or rel == pat_norm:
                return True
            if fnmatch.fnmatch(base, pat_norm) or fnmatch.fnmatch(rel, pat_norm) or fnmatch.fnmatch(rel, pat_norm + "/*"):
                return True

            # Also handle "exclude top-level folder name anywhere in path"
            # e.g. pat=".venv" should exclude ".venv/..." not just a file named ".venv"
            if rel.startswith(pat_norm + "/"):
                return True

        return False

    with tarfile.open(archive_name, mode) as tar:
        for root, dirs, files in os.walk(root_dir):
            rel_root = os.path.relpath(root, root_dir)
            rel_root = "" if rel_root == "." else rel_root

            # PRUNE directories so os.walk doesn't even enter them
            dirs[:] = [d for d in dirs if not is_excluded(os.path.join(rel_root, d))]

            for f in files:
                rel_file = os.path.join(rel_root, f)
                if is_excluded(rel_file):
                    continue

                path = os.path.join(root, f)
                arcname = rel_file.replace(os.sep, "/")  # stable paths in tar
                tar.add(path, arcname=arcname)

    return archive_name


def register_openai_profile(project):
    open_ai_profile = mlrun.datastore.datastore_profile.OpenAIProfile(
                name="openai_profile",
                api_key=os.environ.get("OPENAI_API_KEY"),
                organization=os.environ.get("OPENAI_ORG_ID"),
                project=os.environ.get("OPENAI_PROJECT_ID"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                timeout=os.environ.get("OPENAI_TIMEOUT"),
                max_retries=os.environ.get("OPENAI_MAX_RETRIES"),
            )
    project.register_datastore_profile(open_ai_profile)
    mlrun.datastore.datastore_profile.register_temporary_client_datastore_profile(open_ai_profile)
    project.save()