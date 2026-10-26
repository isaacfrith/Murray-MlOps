"""Live monitoring: compare issued forecasts with what Barmah actually did (pure functions).

A forecast "matures" once its target day's daily mean has been ingested. Live accuracy is
compared with the champion's backtest accuracy: a live error well above the backtest means the
model has degraded (or live data differs from the corrected history it was tested on).
"""

import pandas as pd

from murray.transforms.labels import HORIZONS

WINDOW_DAYS = 30        # judge on the most recent month of matured forecasts
MIN_MATURED = 5         # per horizon, before any verdict
DEGRADED_RATIO = 1.5    # live MAE > 1.5 × backtest MAE → degraded


def with_model_name(preds: pd.DataFrame, default: str) -> pd.DataFrame:
    """Rows written before the model_name column existed (schema evolution) are the default model."""
    p = preds.copy()
    if "model_name" not in p:
        p["model_name"] = None
    p["model_name"] = p["model_name"].fillna(default)
    return p


def latest_forecasts(preds: pd.DataFrame) -> pd.DataFrame:
    """Inference can run several times from the same observed day; keep the last issue (per model)."""
    keys = ["model_name", "obs_date", "horizon"] if "model_name" in preds else ["obs_date", "horizon"]
    return (preds.sort_values("issued_at")
            .drop_duplicates(subset=keys, keep="last")
            .reset_index(drop=True))


def match_actuals(forecasts: pd.DataFrame, actual: pd.Series) -> pd.DataFrame:
    """Attach actual level on target_date, and the persistence forecast (level on obs_date).

    `actual` is indexed by date (cleaned daily mean levels). Unmatured rows are dropped.
    """
    f = forecasts.copy()
    f["actual_m"] = f["target_date"].map(actual)
    f["persistence_m"] = f["obs_date"].map(actual)
    f = f.dropna(subset=["actual_m", "level_m"])
    f["error_m"] = f["level_m"] - f["actual_m"]
    f["persistence_error_m"] = f["persistence_m"] - f["actual_m"]
    return f.sort_values(["target_date", "horizon"]).reset_index(drop=True)


def summarize(matched: pd.DataFrame, backtest_mae_m: dict[int, float],
              window_days: int = WINDOW_DAYS) -> dict:
    """Per-horizon live MAE over the last `window_days` of target dates, with a verdict."""
    if not matched.empty:
        cutoff = pd.Timestamp(matched["target_date"].max()) - pd.Timedelta(days=window_days - 1)
        recent = matched[pd.to_datetime(matched["target_date"]) >= cutoff]
    else:
        recent = matched
    horizons, degraded = [], False
    for h in HORIZONS:
        rows = recent[recent["horizon"] == h] if not recent.empty else recent
        n = len(rows)
        live = float(rows["error_m"].abs().mean()) if n else None
        pers = float(rows["persistence_error_m"].abs().mean()) if n else None
        back = backtest_mae_m.get(h)
        ratio = live / back if (live is not None and back) else None
        if n < MIN_MATURED:
            status = "waiting"
        elif ratio is not None and ratio > DEGRADED_RATIO:
            status, degraded = "degraded", True
        else:
            status = "ok"
        horizons.append({"horizon": h, "n": n, "live_mae_m": live, "persistence_mae_m": pers,
                         "backtest_mae_m": back, "ratio": ratio, "status": status})
    return {"window_days": window_days, "matured": int(len(matched)), "degraded": degraded,
            "horizons": horizons}
