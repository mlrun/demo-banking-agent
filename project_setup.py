import os
import shutil
import mlrun
from pathlib import Path


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source", None)
    force_build = project.get_param("force_build", False)

    # Adding secrets to the projects:
    assert (os.environ.get("OPENAI_API_KEY", None) is not None) and (os.environ.get("OPENAI_BASE_URL", None) is not None), "\
    Missing OpenAI credentials, make sure they are set as environment variables."

    project.set_secrets({"OPENAI_API_KEY": mlrun.get_secret_or_env("OPENAI_API_KEY"),
                         "OPENAI_BASE_URL": mlrun.get_secret_or_env("OPENAI_BASE_URL")})


    # Set project git/archive source and enable pulling latest code at runtime
    if source is None and not project.default_image:
        shutil.make_archive('./banking_agent', 'zip', "./")
        # Logging as artifact
        proj_artifact = project.log_artifact('project_zip', local_path='./banking_agent.zip', upload=True)
        os.remove('./banking_agent.zip')
        project.set_source(source=proj_artifact.target_path, pull_at_runtime=False)
        print(f"Project Source: {source}")
        source = proj_artifact.target_path

    # Set default project docker image - functions that do not specify image will use this
    if force_build:    
        project.set_source(source, pull_at_runtime=False)    
        project.build_image(
            base_image='mlrun/mlrun-kfp',
            set_as_default=True,
            overwrite_build_params=True,
            with_mlrun=False,
            requirements=['PyGithub==1.59.0',
                          'deepchecks==0.18.1',
                          'pandera==0.20.3',
                          'transformers==4.48.1',
                          'datasets==3.2.0',
                          'torch==1.13.1'],)

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
        image="mlrun/mlrun",
    )
    project.set_function(
        name="validate", func="src/functions/validate.py", kind="job"
    )
    project.set_function(
        name="serving",
        func="src/functions/v2_model_server.py",
        kind="serving",
        image="mlrun/mlrun"
    )
    project.set_function(
        name="model-server-tester",
        func="src/functions/v2_model_tester.py",
        kind="job",
        handler="model_server_tester",
    )

    # MLRun Workflows
    project.set_workflow("main", "src/workflows/train_and_deploy_workflow.py", image=project.default_image)

    # Save and return the project:
    project.save()
    return project
