"""Explanations for the Barmah forecaster (pure functions over model + data, no I/O).

Uses LightGBM's exact TreeSHAP (pred_contrib=True). SHAP values are additive, so related
features can be summed into human-sized groups without breaking the maths:

    forecast level = today's level + base + Σ group contributions
"""

import datetime as dt

import numpy as np
import pandas as pd

from murray.models.evaluate import persistence_forecast, score_table
from murray.models.forecaster import BarmahForecaster
from murray.transforms.labels import HORIZONS

GROUP_ORDER = ["Current level", "Barmah trend", "Upstream: Yarrawonga", "Season", "Weather"]
EFFECT_FEATURE, EFFECT_HORIZON = "yarra_change_7d", 4  # ~4-day travel time
BACKTEST_HORIZONS = (1, 4, 7)


def group_of(feature: str) -> str:
    if feature in ("barmah_level_m", "barmah_mean_7d"):
        return "Current level"
    if feature.startswith("barmah_"):
        return "Barmah trend"
    if feature.startswith("yarra_"):
        return "Upstream: Yarrawonga"
    if feature.startswith("doy_"):
        return "Season"
    return "Weather"


def _cm(x) -> float:
    return round(float(x) * 100, 2)


def grouped(contrib: pd.DataFrame) -> pd.DataFrame:
    """Sum feature contributions into groups (columns in GROUP_ORDER, only those present)."""
    feats = [c for c in contrib.columns if c != "_base"]
    g = contrib[feats].T.groupby([group_of(f) for f in feats]).sum().T
    return g[[name for name in GROUP_ORDER if name in g.columns]]


def global_importance(model: BarmahForecaster, test: pd.DataFrame) -> dict:
    """Mean |SHAP| (cm) per group and per feature, for each horizon."""
    out = {}
    for h, c in model.contributions(test).items():
        feats = c.drop(columns="_base")
        out[str(h)] = {
            "groups": {k: _cm(v) for k, v in grouped(c).abs().mean().items()},
            "features": {k: _cm(v) for k, v in feats.abs().mean().sort_values(ascending=False).items()},
        }
    return out


def effect_data(model: BarmahForecaster, test: pd.DataFrame, feature: str = EFFECT_FEATURE,
                horizon: int = EFFECT_HORIZON) -> dict:
    c = model.contributions(test)[horizon]
    pts = pd.DataFrame({"x": test[feature], "y": c[feature]}).dropna()
    return {"feature": feature, "horizon": horizon,
            "points": [[round(float(x) * 100, 1), _cm(y)] for x, y in zip(pts.x, pts.y)]}


def backtest_series(model: BarmahForecaster, test: pd.DataFrame) -> dict:
    pred = model.predict(None, test)
    base = persistence_forecast(test)
    out = {}
    for h in BACKTEST_HORIZONS:
        target_dates = pd.to_datetime(test["obs_date"]) + pd.Timedelta(days=h)
        ok = test[f"target_h{h}"].notna()
        out[str(h)] = {
            "dates": [d.date().isoformat() for d in target_dates[ok]],
            "actual": [round(float(v), 3) for v in test.loc[ok, f"target_h{h}"]],
            "predicted": [round(float(v), 3) for v in pred.loc[ok, f"h{h}"]],
            "persistence": [round(float(v), 3) for v in base.loc[ok, f"h{h}"]],
        }
    return out


def model_card(model: BarmahForecaster, test: pd.DataFrame, *, version: str | None = None,
               feature_set: str | None = None) -> dict:
    """Everything the model page shows about a trained model (global view)."""
    table = score_table(model.predict(None, test), test)
    hcols = [f"h{h}" for h in HORIZONS]
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": version, "feature_set": feature_set,
        "n_features": len(model.feature_columns),
        "test_start": str(test["obs_date"].min()), "test_end": str(test["obs_date"].max()),
        "test_rows": len(test),
        "scores_cm": [{"regime": r["regime"], "forecaster": r["forecaster"], "n": int(r["n"]),
                       "mae": [_cm(r[c]) for c in hcols]} for _, r in table.iterrows()],
        "importance": global_importance(model, test),
        "effect": effect_data(model, test),
        "backtest": backtest_series(model, test),
    }


# --- Local explanation (today's forecast) ----------------------------------------------

def _caption(group: str, contrib_cm: float, row: pd.Series, horizon: int, target: str) -> str:
    direction = "up" if contrib_cm > 0 else "down"
    size = f"{abs(contrib_cm):.1f} cm"
    if group == "Upstream: Yarrawonga":
        ch = row.get("yarra_change_7d")
        if ch is not None and not np.isnan(ch) and abs(ch) >= 0.02:
            verb = "risen" if ch > 0 else "fallen"
            return (f"Yarrawonga has {verb} {abs(ch) * 100:.0f} cm over the past week; that water "
                    f"takes ~4 days to reach Barmah, pushing the {target} forecast {direction} {size}.")
        return f"Steady flows upstream at Yarrawonga push the {target} forecast {direction} {size}."
    if group == "Barmah trend":
        ch = row.get("barmah_change_3d")
        if ch is not None and not np.isnan(ch) and abs(ch) >= 0.01:
            verb = "rising" if ch > 0 else "falling"
            return (f"Barmah has been {verb} ({ch * 100:+.0f} cm in 3 days); the model expects "
                    f"some of that to continue, {direction} {size} by {target}.")
        return f"Barmah's recent pattern pushes the {target} forecast {direction} {size}."
    if group == "Current level":
        return f"Today's level itself pushes the {target} forecast {direction} {size}."
    if group == "Season":
        return f"The time of year pushes the {target} forecast {direction} {size}."
    return f"Recent weather pushes the {target} forecast {direction} {size}."


def local_explanation(model: BarmahForecaster, row: pd.DataFrame, obs_date: dt.date) -> list[dict]:
    """Per forecast day: the waterfall (cm), feature detail and 1–2 plain-language captions."""
    current = float(row["barmah_level_m"].iloc[0])
    days = []
    for h, c in model.contributions(row).items():
        target_date = obs_date + dt.timedelta(days=h)
        target = target_date.strftime("%a %d %b")
        g = grouped(c).iloc[0]
        base = float(c["_base"].iloc[0])
        top = g.abs().sort_values(ascending=False).index[:2]
        days.append({
            "date": target_date.isoformat(), "horizon": h,
            "current_m": round(current, 3), "base_cm": _cm(base),
            "groups_cm": {k: _cm(v) for k, v in g.items()},
            "features_cm": {k: _cm(v) for k, v in c.drop(columns="_base").iloc[0]
                            .sort_values(key=abs, ascending=False).items()},
            "forecast_m": round(current + base + float(g.sum()), 3),
            "inputs": {k: (None if pd.isna(row[k].iloc[0]) else round(float(row[k].iloc[0]), 4))
                       for k in ("barmah_level_m", "barmah_change_3d", "yarra_change_7d")
                       if k in row.columns},
            "captions": [_caption(k, _cm(g[k]), row.iloc[0], h, target)
                         for k in top if abs(g[k]) >= 0.0005],
        })
    return days
