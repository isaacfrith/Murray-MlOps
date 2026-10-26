"""Monitoring pipeline: live forecasts vs actual Barmah levels → verdict, MLflow, site/monitoring.json.

FTI-friendly: reads the predictions table (written by inference) and raw actuals; writes only
metrics and a JSON summary. If the champion has degraded, the DAG triggers retraining.
The Timestral second opinion is scored the same way but never triggers anything: it isn't
trained on our data, so retraining can't fix it.
"""

import datetime as dt
import json
import logging
import os
from pathlib import Path

import mlflow

from murray.io.catalog import get_catalog
from murray.io.prediction_tables import barmah_forecast_table
from murray.io.raw_tables import gauge_daily_table
from murray.io.registry import configure
from murray.pipelines.training_pipeline import CHAMPION, MODEL_NAME
from murray.transforms.features import BARMAH, clean_levels, daily_wide
from murray.transforms.labels import HORIZONS
from murray.transforms.monitoring import (
    latest_forecasts,
    match_actuals,
    summarize,
    with_model_name,
)
from murray.transforms.raw import latest

log = logging.getLogger(__name__)
EXPERIMENT = "barmah-monitoring"


def run(site_dir: str | Path | None = None) -> dict:
    client = configure()
    catalog = get_catalog()

    all_preds = with_model_name(barmah_forecast_table(catalog).scan().to_pandas(), MODEL_NAME)
    wide = daily_wide(clean_levels(latest(gauge_daily_table(catalog).scan().to_pandas())))
    actual = wide[BARMAH].dropna()
    actual.index = actual.index.date
    preds = all_preds[all_preds["model_name"] == MODEL_NAME]
    matched = match_actuals(latest_forecasts(preds), actual)

    champ = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    champ_metrics = client.get_run(champ.run_id).data.metrics
    backtest = {h: champ_metrics[f"test_mae_h{h}"] for h in HORIZONS}
    summary = summarize(matched, backtest)
    second = _second_opinion_summary(client, all_preds, actual)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"monitor-{dt.date.today().isoformat()}"):
        mlflow.set_tags({"champion_version": champ.version, "degraded": summary["degraded"]})
        mlflow.log_metric("matured_forecasts", summary["matured"])
        for h in summary["horizons"]:
            if h["live_mae_m"] is not None:
                mlflow.log_metrics({f"live_mae_h{h['horizon']}": h["live_mae_m"],
                                    f"live_ratio_h{h['horizon']}": h["ratio"],
                                    f"matured_h{h['horizon']}": h["n"]})
        if second:
            mlflow.log_metric("timestral_matured_forecasts", second["matured"])
            for h in second["horizons"]:
                if h["live_mae_m"] is not None:
                    mlflow.log_metric(f"timestral_live_mae_h{h['horizon']}", h["live_mae_m"])

    first_due = (preds["target_date"].min().isoformat() if len(preds) else None)
    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "champion_version": champ.version, "first_target_date": first_due,
        **summary,
        "recent": [{"target_date": r.target_date.isoformat(), "horizon": int(r.horizon),
                    "predicted_m": round(float(r.level_m), 3), "actual_m": round(float(r.actual_m), 3),
                    "persistence_m": round(float(r.persistence_m), 3)}
                   for r in matched.tail(7 * 60).itertuples()],
        "second_opinion": second,
    }
    from murray.pipelines.inference_pipeline import DEFAULT_SITE_DIR

    site = Path(site_dir or os.environ.get("SITE_DIR") or DEFAULT_SITE_DIR)
    (site / "monitoring.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    log.info("monitoring: %d matured forecasts, degraded=%s", summary["matured"], summary["degraded"])
    return {"matured": summary["matured"], "degraded": summary["degraded"],
            "champion_version": champ.version}


def _second_opinion_summary(client: mlflow.MlflowClient, preds, actual) -> dict | None:
    """Live accuracy of the Timestral @live model, against its own backtest. None if absent."""
    from murray.models.timestral_model import ALIAS, MODEL_NAME as TS_NAME

    ts = preds[preds["model_name"] == TS_NAME]
    try:
        mv = client.get_model_version_by_alias(TS_NAME, ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    m = client.get_run(mv.run_id).data.metrics
    summary = summarize(match_actuals(latest_forecasts(ts), actual),
                        {h: m[f"test_mae_h{h}"] for h in HORIZONS})
    summary.pop("degraded")  # informational only: nothing to retrain
    return {"model_name": TS_NAME, "model_version": mv.version,
            "first_target_date": ts["target_date"].min().isoformat() if len(ts) else None,
            **summary}
