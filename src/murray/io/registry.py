"""MLflow tracking + model registry access.

Artifacts are proxied by the MLflow server (--serve-artifacts), so this side only
needs MLFLOW_TRACKING_URI — no S3 credentials. Inside Docker use http://mlflow:5000.
"""

import os

import mlflow
from mlflow import MlflowClient

DEFAULT_TRACKING_URI = "http://localhost:5050"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def configure() -> MlflowClient:
    """Point the mlflow module at our server and return a client for registry calls."""
    # The server advertises presigned S3 downloads/uploads, but those URLs point at
    # http://seaweedfs:8333, which only resolves inside Docker. Force every artifact
    # transfer through the tracking server instead (respecting an explicit override).
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence an AI-assistant log hint

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)
