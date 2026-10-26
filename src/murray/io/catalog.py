"""Iceberg catalog access. The only place that knows how to reach the catalog.

Config comes from environment variables, so the same code works from the laptop
(localhost) and later from inside Docker/Airflow (postgres, seaweedfs):

    ICEBERG_DB_PASSWORD   required
    S3_ACCESS_KEY         required
    S3_SECRET_KEY         required
    POSTGRES_HOST         default localhost
    POSTGRES_PORT         default 5432
    S3_ENDPOINT           default http://localhost:8333
    S3_REGION             default us-east-1
    ICEBERG_WAREHOUSE     default s3://lakehouse/warehouse

This module does not load .env itself; the caller does (load_dotenv() in a
notebook, or `uv run --env-file .env ...`).
"""

import os

from pyiceberg.catalog.sql import SqlCatalog
from sqlalchemy import URL

CATALOG_NAME = "murray"
CATALOG_DB = "iceberg_catalog"  # database and role share this name (see infra/postgres/init)


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; load .env or export it")
    return value


def catalog_uri() -> str:
    url = URL.create(
        "postgresql+psycopg2",  # driver installed by pyiceberg[sql-postgres]
        username=CATALOG_DB,
        password=_require("ICEBERG_DB_PASSWORD"),  # URL.create escapes special characters
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=CATALOG_DB,
    )
    return url.render_as_string(hide_password=False)


def get_catalog(
    uri: str | None = None,
    warehouse: str | None = None,
    s3_endpoint: str | None = None,
) -> SqlCatalog:
    """Return the project's Iceberg SQL catalog. Arguments override the environment."""
    return SqlCatalog(
        CATALOG_NAME,
        uri=uri or catalog_uri(),
        warehouse=warehouse or os.environ.get("ICEBERG_WAREHOUSE", "s3://lakehouse/warehouse"),
        **{
            "s3.endpoint": s3_endpoint or os.environ.get("S3_ENDPOINT", "http://localhost:8333"),
            "s3.access-key-id": _require("S3_ACCESS_KEY"),
            "s3.secret-access-key": _require("S3_SECRET_KEY"),
            "s3.region": os.environ.get("S3_REGION", "us-east-1"),
        },
    )
