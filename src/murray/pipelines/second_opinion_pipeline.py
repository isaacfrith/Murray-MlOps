"""Second opinion: Timestral @live forecasts the same days as today's main forecast.

Runs as its own process after inference (LightGBM and PyTorch each bundle an OpenMP runtime;
both running in one process aborts on macOS). Reads the issued site/forecast.json for the issue
(observed day D, wind, time), the daily level history from the feature store's offline table,
and the model from the registry. Adds "second_opinion" to forecast.json and appends rows to
predictions.barmah_forecast. Never fails the main forecast: problems are reported on the page.
"""

import datetime as dt
import json
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd

from murray.io.catalog import get_catalog
from murray.io.feature_tables import daily_levels
from murray.io.prediction_tables import append_forecast, barmah_forecast_table, forecast_rows
from murray.io.registry import configure
from murray.models.timestral_model import ALIAS, MODEL_NAME, is_calibrated, second_opinion_days
from murray.rules.wakeboarding import load_rule

log = logging.getLogger(__name__)
DEFAULT_SITE_DIR = Path(__file__).resolve().parents[3] / "site"


def _forecast(main: dict) -> dict:
    client = configure()
    obs_date = dt.date.fromisoformat(main["observed_date"])
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    history = daily_levels(get_catalog()).loc[:pd.Timestamp(obs_date)]  # never past day D
    if history.index[-1].date() != obs_date:
        raise ValueError(f"offline history ends {history.index[-1].date()}, expected {obs_date}")
    quantiles = model.predict(history.reset_index(drop=True))

    coverage = client.get_run(mv.run_id).data.metrics.get("coverage80_mean")
    calibrated = is_calibrated(coverage)
    wind = {dt.date.fromisoformat(d["date"]): d["wind_kmh"] for d in main["days"]
            if d["wind_kmh"] is not None}
    days = second_opinion_days(quantiles, obs_date, main["observed_level_m"], wind,
                               load_rule(), calibrated)
    append_forecast(barmah_forecast_table(get_catalog()),
                    forecast_rows(dt.datetime.fromisoformat(main["issued_at"]), MODEL_NAME,
                                  mv.version, obs_date, days))
    log.info("second opinion (Timestral v%s) from %s: %s", mv.version, obs_date,
             ", ".join(f"{d['level_m']:.2f}m [{d['q10_m']:.2f}–{d['q90_m']:.2f}]" for d in days))
    return {"status": "ok", "model_name": MODEL_NAME, "model_version": mv.version,
            "variant": mv.tags.get("variant", ""), "coverage80": coverage,
            "calibrated": calibrated, "days": days}


def run(site_dir: str | Path | None = None) -> dict:
    site = Path(site_dir or os.environ.get("SITE_DIR") or DEFAULT_SITE_DIR)
    path = site / "forecast.json"
    main = json.loads(path.read_text())
    if main["status"] != "ok":
        second = {"status": "unavailable", "reason": "no main forecast to compare with"}
    else:
        try:
            second = _forecast(main)
        except Exception as e:  # torch missing, registry down, stale history...
            log.warning("Timestral second opinion unavailable: %s", e, exc_info=True)
            second = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}
    main["second_opinion"] = second
    path.write_text(json.dumps(main, indent=2) + "\n")
    return {"status": second["status"], "model_version": second.get("model_version")}
