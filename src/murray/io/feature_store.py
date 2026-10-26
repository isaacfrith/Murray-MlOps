"""Feast feature store access.

Feast stores data-source settings inside its registry file (in S3), so secrets must NOT go
into IcebergSource(catalog_properties=...). Instead, PyIceberg reads the catalog config for
the catalog named "murray" from PYICEBERG_CATALOG__MURRAY__* environment variables, which
this module derives from the project's own variables. Same for the S3 registry (boto3).
"""

import os
from pathlib import Path

from feast import FeatureStore

from murray.io.catalog import CATALOG_NAME, catalog_uri

# Laptop: <repo>/features. Docker: mounted at the path in FEAST_REPO_PATH.
DEFAULT_REPO_PATH = Path(__file__).resolve().parents[3] / "features"


def export_feast_env() -> None:
    """Translate project env vars into what PyIceberg/boto3 inside Feast expect."""
    prefix = f"PYICEBERG_CATALOG__{CATALOG_NAME.upper()}__"
    s3_endpoint = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
    env = {
        prefix + "TYPE": "sql",
        prefix + "URI": catalog_uri(),
        prefix + "WAREHOUSE": os.environ.get("ICEBERG_WAREHOUSE", "s3://lakehouse/warehouse"),
        prefix + "S3__ENDPOINT": s3_endpoint,
        prefix + "S3__ACCESS_KEY_ID": os.environ["S3_ACCESS_KEY"],
        prefix + "S3__SECRET_ACCESS_KEY": os.environ["S3_SECRET_KEY"],
        prefix + "S3__REGION": os.environ.get("S3_REGION", "us-east-1"),
        # Feast's S3 registry store (plain boto3)
        "FEAST_S3_ENDPOINT_URL": s3_endpoint,
        "AWS_ACCESS_KEY_ID": os.environ["S3_ACCESS_KEY"],
        "AWS_SECRET_ACCESS_KEY": os.environ["S3_SECRET_KEY"],
        "AWS_DEFAULT_REGION": os.environ.get("S3_REGION", "us-east-1"),
        # feature_store.yaml interpolates these
        "VALKEY_HOST": os.environ.get("VALKEY_HOST", "localhost"),
        "VALKEY_PORT": os.environ.get("VALKEY_PORT", "6379"),
    }
    os.environ.update(env)

    # PyIceberg snapshots PYICEBERG_* env vars once, at import time (murray.io.catalog has
    # usually imported it already). Re-read them so Feast's load_catalog("murray") sees them.
    import pyiceberg.catalog
    from pyiceberg.utils.config import Config

    pyiceberg.catalog._ENV_CONFIG = Config()


def get_store(repo_path: str | Path | None = None) -> FeatureStore:
    export_feast_env()
    path = repo_path or os.environ.get("FEAST_REPO_PATH") or DEFAULT_REPO_PATH
    return FeatureStore(repo_path=str(path))
