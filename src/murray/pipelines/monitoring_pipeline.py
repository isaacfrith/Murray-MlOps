"""Monitoring pipeline: live forecasts vs actual Barmah levels → verdict, MLflow, site/monitoring.json.

FTI-friendly: reads the predictions table (written by inference) and raw actuals; writes only
metrics and a JSON summary. If the champion has degraded, the DAG triggers retraining.
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
from murray.transforms.monitoring import latest_forecasts, match_actuals, summarize
from murray.transforms.raw import latest

log = logging.getLogger(__name__)
EXPERIMENT = "barmah-monitoring"


def run(site_dir: str | Path | None = None) -> dict:
    client = configure()
    catalog = get_catalog()

    preds = barmah_forecast_table(catalog).scan().to_pandas()
    wide = daily_wide(clean_levels(latest(gauge_daily_table(catalog).scan().to_pandas())))
    actual = wide[BARMAH].dropna()
    actual.index = actual.index.date
    matched = match_actuals(latest_forecasts(preds), actual)

    champ = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    champ_metrics = client.get_run(champ.run_id).data.metrics
    backtest = {h: champ_metrics[f"test_mae_h{h}"] for h in HORIZONS}
    summary = summarize(matched, backtest)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"monitor-{dt.date.today().isoformat()}"):
        mlflow.set_tags({"champion_version": champ.version, "degraded": summary["degraded"]})
        mlflow.log_metric("matured_forecasts", summary["matured"])
        for h in summary["horizons"]:
            if h["live_mae_m"] is not None:
                mlflow.log_metrics({f"live_mae_h{h['horizon']}": h["live_mae_m"],
                                    f"live_ratio_h{h['horizon']}": h["ratio"],
                                    f"matured_h{h['horizon']}": h["n"]})

    first_due = (preds["target_date"].min().isoformat() if len(preds) else None)
    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "champion_version": champ.version, "first_target_date": first_due,
        **summary,
        "recent": [{"target_date": r.target_date.isoformat(), "horizon": int(r.horizon),
                    "predicted_m": round(float(r.level_m), 3), "actual_m": round(float(r.actual_m), 3),
                    "persistence_m": round(float(r.persistence_m), 3)}
                   for r in matched.tail(7 * 60).itertuples()],
    }
    from murray.pipelines.inference_pipeline import DEFAULT_SITE_DIR

    site = Path(site_dir or os.environ.get("SITE_DIR") or DEFAULT_SITE_DIR)
    (site / "monitoring.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    log.info("monitoring: %d matured forecasts, degraded=%s", summary["matured"], summary["degraded"])
    return {"matured": summary["matured"], "degraded": summary["degraded"],
            "champion_version": champ.version}
