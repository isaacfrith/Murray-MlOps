"""Smoke test: can Airflow tasks reach every service, using the murray package?

Each task runs in /opt/murray/.venv (built from uv.lock), not Airflow's own Python.
Functions passed to @task.external_python must be self-contained: imports go inside.
Trigger it manually from the UI (it has no schedule).
"""

from datetime import datetime

from airflow.sdk import dag, task

MURRAY_PYTHON = "/opt/murray/.venv/bin/python"


@dag(
    dag_id="smoke_stack",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["smoke"],
    doc_md=__doc__,
)
def smoke_stack():
    @task.external_python(python=MURRAY_PYTHON)
    def iceberg() -> str:
        import uuid

        import pyarrow as pa

        from murray.io.catalog import get_catalog

        catalog = get_catalog()
        catalog.create_namespace_if_not_exists("smoke")
        name = f"smoke.t_{uuid.uuid4().hex[:8]}"
        table = catalog.create_table(name, schema=pa.schema([("v", pa.float64())]))
        table.append(pa.table({"v": [1.0, 2.0]}))
        rows = table.scan().to_arrow().num_rows
        catalog.purge_table(name)
        return f"iceberg ok: wrote and read {rows} rows"

    @task.external_python(python=MURRAY_PYTHON)
    def mlflow_tracking() -> str:
        import mlflow

        from murray.io.registry import configure

        configure()
        mlflow.set_experiment("smoke")
        with mlflow.start_run(run_name="airflow-smoke") as run:
            mlflow.log_metric("ok", 1.0)
        return f"mlflow ok: run {run.info.run_id} at {mlflow.get_tracking_uri()}"

    @task.external_python(python=MURRAY_PYTHON)
    def valkey() -> str:
        import os

        import redis

        client = redis.Redis(
            host=os.environ["VALKEY_HOST"],
            port=int(os.environ["VALKEY_PORT"]),
            password=os.environ["VALKEY_PASSWORD"],
        )
        return f"valkey ok: ping={client.ping()}"

    @task
    def report(*results: str) -> None:
        for line in results:
            print(line)

    report(iceberg(), mlflow_tracking(), valkey())


smoke_stack()
