"""Timestral (Mistral's time-series foundation model) as a zero-shot Barmah forecaster.

Zero-shot: nothing is trained on our data. For each forecast origin D the model reads the last
CONTEXT_DAYS of Barmah's daily mean level (optionally Yarrawonga as a past-only covariate) up to
and including D, and returns 21 quantiles for D+1..D+7.

The pure helpers (contexts, probability in a range) don't import torch; the forecaster and the
MLflow wrapper need the optional `murray[timestral]` extra.
"""

import datetime as dt

import numpy as np
import pandas as pd

import mlflow

from murray.rules.wakeboarding import rate_day
from murray.transforms.labels import HORIZONS

MODEL_NAME = "barmah-level-timestral"   # MLflow registered model
ALIAS = "live"                          # the variant that runs daily (never "champion")
EXPERIMENT = "barmah-level-timestral"
MODEL_ID = "mistralai/Timestral-28M"
WEIGHTS_REVISION = "49b26409ac87b5935539a5d8ffe7f397b720db47"
CODE_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
CONTEXT_DAYS = 4096
QUANTILE_LEVELS = (0.01, *[round(0.05 * i, 2) for i in range(1, 20)], 0.99)  # 21 levels
VARIANTS = {"barmah_only": False, "with_yarrawonga": True}
# "Chance of a good level" is shown only if the backtest's 80% range held the actual level
# 70–90% of the time: outside that, the stated probabilities can't be trusted.
CALIBRATED_COVERAGE = (0.70, 0.90)


def qcol(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def contexts(series: np.ndarray, end_positions: list[int], length: int = CONTEXT_DAYS) -> np.ndarray:
    """(B, length) windows ending AT each position (inclusive), left-padded with NaN.

    Only data up to the origin is included, so a backtest can't see the future.
    """
    out = np.full((len(end_positions), length), np.nan, dtype="float32")
    for i, end in enumerate(end_positions):
        start = max(0, end + 1 - length)
        window = series[start:end + 1]
        out[i, length - len(window):] = window
    return out


def prob_in_range(quantiles: dict[float, float], lo: float, hi: float) -> float:
    """P(lo <= level <= hi) from a quantile forecast, by linear interpolation of the CDF."""
    levels = np.array(sorted(quantiles))
    values = np.maximum.accumulate(np.array([quantiles[q] for q in levels]))  # monotone

    def cdf(x: float) -> float:
        return float(np.interp(x, values, levels, left=0.0, right=1.0))

    return max(0.0, cdf(hi) - cdf(lo))


def is_calibrated(coverage80: float | None) -> bool:
    lo, hi = CALIBRATED_COVERAGE
    return coverage80 is not None and lo <= coverage80 <= hi


def second_opinion_days(quantiles: pd.DataFrame, obs_date: dt.date, current_m: float,
                        wind: dict[dt.date, float], rule: dict, calibrated: bool) -> list[dict]:
    """Dashboard days from a quantile forecast (rows = horizon, columns q01..q99).

    Rated on the median, exactly like the main forecast. p_good_level = chance the level lands
    in the rule's good range (level only: wind is rated separately), or None if uncalibrated.
    """
    lv = rule["level_m"]
    days, prev = [], current_m
    for h in HORIZONS:
        row = quantiles.loc[h]
        target = obs_date + dt.timedelta(days=h)
        median = float(row[qcol(0.5)])
        w = wind.get(target)
        r = rate_day(median, median - prev, w, rule)
        p_good = (prob_in_range({q: float(row[qcol(q)]) for q in QUANTILE_LEVELS},
                                lv["good_min"], lv["good_max"]) if calibrated else None)
        days.append({"date": target.isoformat(), "horizon": h, "level_m": round(median, 3),
                     "q10_m": round(float(row[qcol(0.1)]), 3),
                     "q90_m": round(float(row[qcol(0.9)]), 3),
                     "p_good_level": None if p_good is None else round(p_good, 2),
                     "wind_kmh": w, "rating": r.rating, "reasons": r.reasons})
        prev = median
    return days


def load_forecaster(weights_dir: str):
    from timestral import TimestralForecaster

    return TimestralForecaster.from_pretrained(weights_dir, device="cpu", local_files_only=True)


def forecast_batch(forecaster, barmah_ctx: np.ndarray, yarra_ctx: np.ndarray | None) -> dict:
    """Quantile forecasts for a batch of origins: {q: array (B, 7)} for every quantile level."""
    kwargs = {}
    if yarra_ctx is not None:
        kwargs["past_covariates"] = yarra_ctx[:, None, :]  # (B, 1 covariate, T)
    fc = forecaster.forecast(barmah_ctx, prediction_length=max(HORIZONS), **kwargs)
    shape = (barmah_ctx.shape[0], max(HORIZONS))
    return {q: np.asarray(fc.quantile(q), dtype="float64").reshape(shape) for q in QUANTILE_LEVELS}


class TimestralForecasterModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc. Input: daily history (columns barmah_level_m, yarra_level_m) ending on the
    last observed day. Output: one row per horizon, columns q01..q99 (levels in metres)."""

    def __init__(self, use_yarrawonga: bool):
        self.use_yarrawonga = use_yarrawonga
        self.context_days = CONTEXT_DAYS
        self._forecaster = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_forecaster"] = None
        return state

    def load_context(self, context):
        self._forecaster = load_forecaster(context.artifacts["weights"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        b = model_input["barmah_level_m"].to_numpy("float32")
        end = [len(b) - 1]
        y = (contexts(model_input["yarra_level_m"].to_numpy("float32"), end, self.context_days)
             if self.use_yarrawonga else None)
        qs = forecast_batch(self._forecaster, contexts(b, end, self.context_days), y)
        return pd.DataFrame({qcol(q): qs[q][0] for q in QUANTILE_LEVELS},
                            index=pd.Index(list(HORIZONS), name="horizon"))
