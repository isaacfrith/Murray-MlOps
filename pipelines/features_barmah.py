"""Feature pipeline DAG: raw.gauge_daily → features.barmah_daily → Valkey.

Runs whenever raw_gauges_daily (or the backfill) updates the raw table: it is scheduled on
the RAW_GAUGE_DAILY asset, not on a clock, so it never runs on half-ingested data.
FTI rule: this DAG does not call the ingest or training pipelines; it only reads raw data
and writes features.
"""

from datetime import datetime, timedelta

from airflow.sdk import Asset, chain, dag, task

MURRAY_PYTHON = "/opt/murray/.venv/bin/python"
RAW_GAUGE_DAILY = Asset("iceberg://lakehouse/raw/gauge_daily")
RAW_WEATHER_DAILY = Asset("iceberg://lakehouse/raw/weather_daily")
BARMAH_FEATURES = Asset("iceberg://lakehouse/features/barmah_daily")


@dag(
    dag_id="features_barmah",
    # Either raw source updating triggers a rebuild (OR, not AND): a weather outage must
    # not stop river features from refreshing.
    schedule=(RAW_GAUGE_DAILY | RAW_WEATHER_DAILY),
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["features", "feast"],
    doc_md=__doc__,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5),
                  "execution_timeout": timedelta(minutes=30)},
)
def features_barmah():
    @task.external_python(python=MURRAY_PYTHON, outlets=[BARMAH_FEATURES])
    def build_features() -> int:
        import logging

        from murray.pipelines.feature_pipeline import build_features

        logging.basicConfig(level=logging.INFO, force=True)
        return build_features()

    @task.external_python(python=MURRAY_PYTHON)
    def feast_apply() -> list:
        from murray.pipelines.feature_pipeline import apply_definitions

        return apply_definitions()

    @task.external_python(python=MURRAY_PYTHON)
    def materialize() -> str:
        from murray.pipelines.feature_pipeline import materialize

        return materialize()

    chain(build_features(), feast_apply(), materialize())


features_barmah()
