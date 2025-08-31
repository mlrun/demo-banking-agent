import os
import shutil
import tarfile
import mlrun


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source", default=None)
    build_image = project.get_param("build_image", default=False)
    default_image = project.get_param("default_image", default=None)

    # Adding secrets to the projects:
    assert (os.environ.get("OPENAI_API_KEY", None) is not None) and (os.environ.get("OPENAI_BASE_URL", None) is not None), "\
    Missing OpenAI credentials, make sure they are set as environment variables."

    project.set_secrets({"OPENAI_API_KEY": mlrun.get_secret_or_env("OPENAI_API_KEY"),
                         "OPENAI_BASE_URL": mlrun.get_secret_or_env("OPENAI_BASE_URL")})

    # Set project git/archive source and enable pulling latest code at runtime
    if not source:
        print("Setting Source for the demo:")
        make_archive("../banking_agent", "gztar", "./", exclude=["project.yaml"])
        # Logging as artifact
        proj_artifact = project.log_artifact('project_source', local_path='../banking_agent.tar.gz', upload=True)
        os.remove('../banking_agent.tar.gz')
        project.set_source(source=proj_artifact.target_path, pull_at_runtime=False)
        print(f"Project Source: {source}")
        source = proj_artifact.target_path

    project.set_source(source, pull_at_runtime=False)

    if default_image:
        project.set_default_image(default_image)

    # Set default project docker image - functions that do not specify image will use this
    if build_image:    
        print("Building default image for the demo:")
        project.build_image(
            image=default_image,
            base_image='mlrun/mlrun-kfp',
            set_as_default=True,
            overwrite_build_params=True,
            requirements=['PyGithub==1.59.0',
                          'deepchecks==0.18.1',
                          'pandera==0.20.3',
                          'transformers==4.48.1',
                          'datasets==3.2.0',
                          'torch==1.13.1'])

    # MLRun Functions
    project.set_function(
        name="data",
        func="src/functions/data.py",
        kind="job",
        image=project.default_image
    )
    project.set_function(
        name="train",
        func="src/functions/train.py",
        kind="job",
        handler="train_model",
        image=project.default_image
    )
    project.set_function(
        name="validate", func="src/functions/validate.py", kind="job", image=project.default_image
    )
    project.set_function(
        name="serving",
        func="src/functions/v2_model_server.py",
        kind="serving",
        image=project.default_image
    )
    project.set_function(
        name="model-server-tester",
        func="src/functions/v2_model_tester.py",
        kind="job",
        handler="model_server_tester",
        image=project.default_image
    )

    # MLRun Workflows
    project.set_workflow("main", "src/workflows/train_and_deploy_workflow.py", image=project.default_image)

    # Save and return the project:
    project.save()
    return project


def make_archive(base_name, format="gztar", root_dir=".", exclude=None):
    """
    Create a tar.gz archive like shutil.make_archive but with exclusions.
    
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
