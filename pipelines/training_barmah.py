"""Training pipeline DAG: Feast historical features → LightGBM → MLflow → maybe @champion.

Weekly (Sunday 03:00 Melbourne) and on demand. FTI rule: reads only the feature store and
raw labels, writes only to the model registry. Promotion happens inside the run and only if
the candidate beats both the persistence baseline and the current champion on the same test
period; otherwise the new version is registered but not promoted.
"""

from datetime import datetime, timedelta

import pendulum
from airflow.sdk import chain, dag, task

MURRAY_PYTHON = "/opt/murray/.venv/bin/python"
MELBOURNE = pendulum.timezone("Australia/Melbourne")


@dag(
    dag_id="training_barmah",
    schedule="0 3 * * 0",
    start_date=datetime(2026, 10, 1, tzinfo=MELBOURNE),
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlflow"],
    doc_md=__doc__,
    default_args={"retries": 0, "execution_timeout": timedelta(hours=1)},
)
def training_barmah():
    @task.external_python(python=MURRAY_PYTHON)
    def train_and_maybe_promote(feature_set: str) -> dict:
        import logging

        from murray.pipelines.training_pipeline import run

        logging.basicConfig(level=logging.INFO, force=True)
        result = run(run_name=f"airflow-weekly-{feature_set}", feature_set=feature_set)
        return {k: result[k] for k in ("version", "promoted", "reason", "run_id")}

    # Each candidate competes against whatever is @champion at that moment.
    chain(*[train_and_maybe_promote.override(task_id=f"train_{fs}")(fs)
            for fs in ("river_only", "river_weather")])


training_barmah()
