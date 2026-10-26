"""Raw ingest DAGs: river gauge daily means → Iceberg raw.gauge_daily (append-only).

- raw_gauges_backfill: trigger once by hand. Full history since 1970, one gauge at a time.
- raw_gauges_daily:    10:00 Melbourne time. Re-fetches the last 14 completed days so
                       agency revisions of provisional values are captured.

Source: state water agencies (VIC for Barmah, NSW for Yarrawonga) via MDBA's gauge-getter.
Daily means change once a day, so a daily schedule is enough (not 6-hourly).
"""

from datetime import datetime, timedelta

import pendulum
from airflow.sdk import chain, dag, task

MURRAY_PYTHON = "/opt/murray/.venv/bin/python"
GAUGES = ["409215", "409025"]  # Barmah, Yarrawonga Weir downstream
MELBOURNE = pendulum.timezone("Australia/Melbourne")


@dag(
    dag_id="raw_gauges_backfill",
    schedule=None,
    start_date=datetime(2026, 10, 1, tzinfo=MELBOURNE),
    catchup=False,
    max_active_runs=1,
    tags=["raw", "ingest"],
    doc_md=__doc__,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10),
                  "execution_timeout": timedelta(hours=2)},
)
def raw_gauges_backfill():
    @task.external_python(python=MURRAY_PYTHON)
    def backfill_gauge(site_id: str) -> int:
        import logging

        from murray.pipelines.raw_ingest import backfill

        logging.basicConfig(level=logging.INFO, force=True)  # gauge-getter configures logging on import
        return backfill(site_id, var="L")

    # Sequential, not parallel: be gentle with the agencies' free services.
    chain(*[backfill_gauge.override(task_id=f"backfill_{g}")(g) for g in GAUGES])


@dag(
    dag_id="raw_gauges_daily",
    schedule="0 10 * * *",
    start_date=datetime(2026, 10, 1, tzinfo=MELBOURNE),
    catchup=False,
    max_active_runs=1,
    tags=["raw", "ingest"],
    doc_md=__doc__,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=15),
                  "execution_timeout": timedelta(minutes=30)},
)
def raw_gauges_daily():
    @task.external_python(python=MURRAY_PYTHON)
    def ingest_recent_gauge(site_id: str) -> int:
        import logging

        from murray.pipelines.raw_ingest import ingest_recent

        logging.basicConfig(level=logging.INFO, force=True)  # gauge-getter configures logging on import
        return ingest_recent(site_id, days=14, var="L")

    chain(*[ingest_recent_gauge.override(task_id=f"ingest_{g}")(g) for g in GAUGES])


raw_gauges_backfill()
raw_gauges_daily()
