"""Inference pipeline: Feast ONLINE features + @champion → 7-day forecast → wakeboarding rating.

Outputs: an append-only Iceberg row per forecast day (for monitoring) and site/forecast.json
(for the static dashboard). FTI rule: reads only the feature store and the model registry.
"""

import datetime as dt
import json
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import pandas as pd

from murray.io.catalog import get_catalog
from murray.io.feature_store import get_store
from murray.io.prediction_tables import append_forecast, barmah_forecast_table
from murray.io.registry import configure
from murray.io.weather import fetch_forecast
from murray.pipelines.training_pipeline import CHAMPION, MODEL_NAME
from murray.rules.wakeboarding import NO_FORECAST, load_rule, rate_day
from murray.transforms.features import BARMAH, FEATURE_COLUMNS
from murray.transforms.labels import HORIZONS
from murray.transforms.weather_features import WEATHER_FEATURE_COLUMNS

log = logging.getLogger(__name__)

LOCAL_TZ = ZoneInfo("Australia/Melbourne")
VIEW_OF = {**{c: "barmah_daily" for c in FEATURE_COLUMNS},
           **{c: "barmah_weather" for c in WEATHER_FEATURE_COLUMNS}}
DEFAULT_SITE_DIR = Path(__file__).resolve().parents[3] / "site"
HISTORY_DAYS = 30


def _online_features(columns: list[str]) -> tuple[pd.DataFrame, dt.date | None]:
    """Latest feature row for Barmah from Valkey, and the observed day D it describes."""
    resp = get_store().get_online_features(
        features=[f"{VIEW_OF[c]}:{c}" for c in columns], entity_rows=[{"site_id": BARMAH}],
    ).to_dict(include_event_timestamps=True)
    row = pd.DataFrame({c: [resp[c][0]] for c in columns}, dtype="float64")
    ts = resp.get("barmah_level_m__ts", [0])[0]
    if not ts or pd.isna(row.loc[0, "barmah_level_m"]):
        return row, None  # missing or older than the feature view TTL
    known_at = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).astimezone(LOCAL_TZ)
    return row, known_at.date() - dt.timedelta(days=1)  # row D is known on D+1 morning


def _recent_history(obs_date: dt.date) -> list[dict]:
    """Last HISTORY_DAYS observed Barmah levels, for the dashboard chart."""
    table = get_catalog().load_table("features.barmah_daily")
    start = obs_date - dt.timedelta(days=HISTORY_DAYS - 1)
    df = table.scan(row_filter=f"obs_date >= '{start.isoformat()}'",
                    selected_fields=("obs_date", "barmah_level_m")).to_pandas()
    df = df.sort_values("obs_date")
    return [{"date": d.isoformat(), "level_m": round(float(v), 3)}
            for d, v in zip(df["obs_date"], df["barmah_level_m"])]


def run(now: dt.datetime | None = None, site_dir: str | Path | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    client = configure()
    version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
    columns = model.metadata.get_input_schema().input_names()
    rule = load_rule()

    features, obs_date = _online_features(columns)
    out = {"site": "Murray River at Barmah (409215)", "issued_at": now.isoformat(),
           "model_version": version, "rule_version": rule["version"]}

    if obs_date is None:
        out.update(status=NO_FORECAST, reason="river data is missing or stale (older than 2 days)",
                   days=[], history=[])
        log.warning("no forecast: stale or missing online features")
    else:
        pred = model.predict(features).iloc[0]
        current = float(features.loc[0, "barmah_level_m"])
        wind = fetch_forecast().set_index("obs_date")["wind_speed_10m_max"]
        days, prev = [], current
        for h in HORIZONS:
            target = obs_date + dt.timedelta(days=h)
            level = float(pred[f"h{h}"])
            w = float(wind.get(target)) if target in wind.index else None
            r = rate_day(level, level - prev, w, rule)
            days.append({"date": target.isoformat(), "horizon": h, "level_m": round(level, 3),
                         "wind_kmh": w, "rating": r.rating, "reasons": r.reasons})
            prev = level
        out.update(status="ok", observed_date=obs_date.isoformat(),
                   observed_level_m=round(current, 3), days=days,
                   history=_recent_history(obs_date))

        rows = pd.DataFrame([{"issued_at": pd.Timestamp(now), "model_version": str(version),
                              "obs_date": obs_date, "target_date": dt.date.fromisoformat(d["date"]),
                              "horizon": d["horizon"], "level_m": d["level_m"],
                              "wind_kmh": d["wind_kmh"], "rating": d["rating"]} for d in days])
        append_forecast(barmah_forecast_table(get_catalog()), rows)
        log.info("forecast from %s with v%s: %s", obs_date, version,
                 ", ".join(f"{d['date']} {d['level_m']:.2f}m {d['rating']}" for d in days))

    site = Path(site_dir or os.environ.get("SITE_DIR") or DEFAULT_SITE_DIR)
    site.mkdir(parents=True, exist_ok=True)
    (site / "forecast.json").write_text(json.dumps(out, indent=2) + "\n")
    return {"status": out["status"], "observed_date": out.get("observed_date"),
            "model_version": version}
