"""Monitoring DAG: live forecasts vs actuals; retrain early if the champion has degraded.

Runs when new actuals land (the raw gauge asset). Normal retraining is weekly
(training_barmah); this adds a data-driven trigger: if live error over the last 30 days is more
than 1.5x the champion's backtest error (with >= 5 matured forecasts), training runs now.
Training still has to pass the promotion gate, so a retrain never forces a worse model in.
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Asset, dag, task

MURRAY_PYTHON = "/opt/murray/.venv/bin/python"
RAW_GAUGE_DAILY = Asset("iceberg://lakehouse/raw/gauge_daily")


@dag(
    dag_id="monitoring_barmah",
    schedule=[RAW_GAUGE_DAILY],
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["monitoring", "mlflow"],
    doc_md=__doc__,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5),
                  "execution_timeout": timedelta(minutes=15)},
)
def monitoring_barmah():
    @task.external_python(python=MURRAY_PYTHON)
    def monitor() -> dict:
        import logging

        from murray.pipelines.monitoring_pipeline import run

        logging.basicConfig(level=logging.INFO, force=True)
        return run()

    @task.short_circuit
    def degraded(result: dict) -> bool:
        return bool(result["degraded"])

    retrain = TriggerDagRunOperator(task_id="retrain_now", trigger_dag_id="training_barmah",
                                    conf={"reason": "monitoring: live error above threshold"})

    degraded(monitor()) >> retrain


monitoring_barmah()
