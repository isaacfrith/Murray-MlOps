"""Inference pipeline DAG: Feast online features + @champion → forecast + wakeboarding rating.

Runs whenever fresh features are online (BARMAH_FEATURES asset, set by features_barmah after
materialising to Valkey). Writes predictions.barmah_forecast (Iceberg, for monitoring) and
site/forecast.json (the dashboard). Publishing the site is a separate, deliberate step:
`make publish` commits forecast.json and pushes, and GitHub Pages redeploys.
"""

from datetime import datetime, timedelta

from airflow.sdk import Asset, dag, task

MURRAY_PYTHON = "/opt/murray/.venv/bin/python"
BARMAH_FEATURES = Asset("iceberg://lakehouse/features/barmah_daily")


@dag(
    dag_id="inference_barmah",
    schedule=[BARMAH_FEATURES],
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["inference"],
    doc_md=__doc__,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5),
                  "execution_timeout": timedelta(minutes=15)},
)
def inference_barmah():
    @task.external_python(python=MURRAY_PYTHON)
    def forecast() -> dict:
        import logging

        from murray.pipelines.inference_pipeline import run

        logging.basicConfig(level=logging.INFO, force=True)
        return run()

    forecast()


inference_barmah()
