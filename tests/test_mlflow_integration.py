"""Integration test: needs the stack incl. MLflow. Run with: uv run pytest -m integration"""

import os
import uuid

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv

pytestmark = pytest.mark.integration

load_dotenv(find_dotenv(usecwd=True))

import mlflow  # noqa: E402

from murray.io.registry import configure  # noqa: E402
from murray.io.s3 import list_keys  # noqa: E402


class AddOne(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: pd.DataFrame, params=None):
        return model_input["level_m"] + 1.0


@pytest.fixture
def client():
    client = configure()
    try:
        client.search_experiments(max_results=1)
    except Exception as exc:
        pytest.skip(f"MLflow not reachable: {exc}")
    return client


def test_track_register_alias_load(client, tmp_path, monkeypatch):
    # Prove artifact proxying: the client must not need S3 credentials.
    for var in ("S3_ACCESS_KEY", "S3_SECRET_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)

    suffix = uuid.uuid4().hex[:8]
    mlflow.set_experiment(f"pytest-{suffix}")
    model_name = f"pytest-model-{suffix}"

    note = tmp_path / "note.txt"
    note.write_text("hello barmah")
    with mlflow.start_run() as run:
        mlflow.log_param("horizon_days", 7)
        mlflow.log_metric("mae_m", 0.05)
        mlflow.log_artifact(str(note))
        info = mlflow.pyfunc.log_model(name="model", python_model=AddOne(),
                                       registered_model_name=model_name)

    try:
        version = info.registered_model_version
        client.set_registered_model_alias(model_name, "champion", version)
        loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
        assert loaded.predict(pd.DataFrame({"level_m": [2.0]})).tolist() == [3.0]

        assert client.get_run(run.info.run_id).data.metrics["mae_m"] == 0.05
        assert run.info.artifact_uri.startswith("mlflow-artifacts:")  # proxied, not s3://
    finally:
        client.delete_registered_model(model_name)
        client.delete_experiment(run.info.experiment_id)


def test_artifacts_land_in_lakehouse_bucket(client):
    """The server stores proxied artifacts under s3://lakehouse/mlflow/."""
    if "S3_ACCESS_KEY" not in os.environ:
        pytest.skip("needs S3 credentials to inspect the bucket")
    assert any(k.startswith("mlflow/") for k in list_keys("mlflow/"))
