"""Inference pipeline DAG: Feast online features + @champion → forecast + wakeboarding rating.

Runs whenever fresh features are online (BARMAH_FEATURES asset, set by features_barmah after
materialising to Valkey). Writes predictions.barmah_forecast (Iceberg, for monitoring) and
site/*.json, then publishes the JSON to the `site-data` branch on GitHub, which redeploys
GitHub Pages. Publishing only happens after a successful forecast, and is skipped (not failed)
until GITHUB_PUBLISH_TOKEN is set.
"""

from datetime import datetime, timedelta

from airflow.sdk import Asset, chain, dag, task

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

    @task.external_python(python=MURRAY_PYTHON)
    def publish() -> str:
        import logging
        import os

        from murray.io.github_publish import publish

        logging.basicConfig(level=logging.INFO, force=True)
        return publish(os.environ["SITE_DIR"])

    chain(forecast(), publish())


inference_barmah()
