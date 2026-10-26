"""Evaluate Timestral zero-shot on the same test period as LightGBM, register the better variant.

Same rules as the LightGBM backtest: one forecast per test day D (rolling origin), using only
data up to D; scored on D+1..D+7 against the same targets, regimes and persistence baseline.
History comes from the feature store's offline table (features.barmah_daily).
"""

import datetime as dt
import logging

import mlflow
import numpy as np
import pandas as pd

from murray.io.catalog import get_catalog
from murray.io.feature_tables import daily_levels
from murray.io.registry import configure
from murray.models.evaluate import mae_by_horizon, persistence_forecast, score_table
from murray.models.timestral_model import (
    ALIAS,
    CODE_COMMIT,
    CONTEXT_DAYS,
    EXPERIMENT,
    MODEL_ID,
    MODEL_NAME,
    QUANTILE_LEVELS,
    VARIANTS,
    WEIGHTS_REVISION,
    TimestralForecasterModel,
    contexts,
    forecast_batch,
    load_forecaster,
)
from murray.pipelines.training_pipeline import TEST_START, VAL_START, load_training_frame
from murray.transforms.labels import HORIZONS, time_split

log = logging.getLogger(__name__)

BATCH = 64


def backtest(forecaster, history: pd.DataFrame, test: pd.DataFrame, use_yarra: bool) -> dict:
    """Quantile forecasts for every test origin: {q: DataFrame(index=test.index, h1..h7)}."""
    pos = history.index.get_indexer(pd.to_datetime(test["obs_date"]))
    b = history["barmah_level_m"].to_numpy("float32")
    y = history["yarra_level_m"].to_numpy("float32")
    parts = {q: [] for q in QUANTILE_LEVELS}
    for i in range(0, len(pos), BATCH):
        ends = list(pos[i:i + BATCH])
        qs = forecast_batch(forecaster, contexts(b, ends), contexts(y, ends) if use_yarra else None)
        for q in QUANTILE_LEVELS:
            parts[q].append(qs[q])
    cols = [f"h{h}" for h in HORIZONS]
    return {q: pd.DataFrame(np.vstack(parts[q]), index=test.index, columns=cols)
            for q in QUANTILE_LEVELS}


def coverage(q10: pd.DataFrame, q90: pd.DataFrame, test: pd.DataFrame) -> dict[int, float]:
    """Share of test days where the actual level fell inside the 10–90% range (ideal: 0.80)."""
    out = {}
    for h in HORIZONS:
        t = test[f"target_h{h}"]
        ok = t.notna()
        out[h] = float(((t[ok] >= q10.loc[ok, f"h{h}"]) & (t[ok] <= q90.loc[ok, f"h{h}"])).mean())
    return out


def _card(variant: str, qs: dict, test: pd.DataFrame, cov: dict) -> dict:
    median = qs[0.5]
    table = score_table(median, test)
    backtest_series = {}
    for h in (1, 4, 7):
        ok = test[f"target_h{h}"].notna()
        dates = (pd.to_datetime(test.loc[ok, "obs_date"]) + pd.Timedelta(days=h)).dt.date
        backtest_series[str(h)] = {
            "dates": [d.isoformat() for d in dates],
            "median": [round(float(v), 3) for v in median.loc[ok, f"h{h}"]],
            "q10": [round(float(v), 3) for v in qs[0.1].loc[ok, f"h{h}"]],
            "q90": [round(float(v), 3) for v in qs[0.9].loc[ok, f"h{h}"]],
        }
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "variant": variant, "model_id": MODEL_ID, "context_days": CONTEXT_DAYS,
        "test_start": str(test["obs_date"].min()), "test_end": str(test["obs_date"].max()),
        "scores_cm": [{"regime": r["regime"], "forecaster": r["forecaster"], "n": int(r["n"]),
                       "mae": [round(float(r[f"h{h}"]) * 100, 2) for h in HORIZONS]}
                      for _, r in table.iterrows()],
        "coverage_80": [round(cov[h], 3) for h in HORIZONS],
        "backtest": backtest_series,
    }


def run() -> dict:
    from huggingface_hub import snapshot_download

    configure()
    mlflow.set_experiment(EXPERIMENT)
    weights = snapshot_download(MODEL_ID, revision=WEIGHTS_REVISION)
    forecaster = load_forecaster(weights)
    history = daily_levels(get_catalog())
    test = time_split(load_training_frame(), VAL_START, TEST_START)["test"]

    results = {}
    for variant, use_yarra in VARIANTS.items():
        with mlflow.start_run(run_name=f"timestral-28m-{variant}") as run:
            qs = backtest(forecaster, history, test, use_yarra)
            mae = mae_by_horizon(qs[0.5], test)
            base = mae_by_horizon(persistence_forecast(test), test)
            cov = coverage(qs[0.1], qs[0.9], test)
            mlflow.log_params({"model_id": MODEL_ID, "weights_revision": WEIGHTS_REVISION,
                               "code_commit": CODE_COMMIT, "variant": variant,
                               "context_days": CONTEXT_DAYS, "zero_shot": True,
                               "test_start": TEST_START, "test_rows": len(test)})
            for h in HORIZONS:
                mlflow.log_metrics({f"test_mae_h{h}": mae[h], f"persistence_mae_h{h}": base[h],
                                    f"coverage80_h{h}": cov[h], f"skill_h{h}": 1 - mae[h] / base[h]})
            mean = float(np.mean(list(mae.values())))
            mlflow.log_metrics({"test_mae_mean": mean,
                                "persistence_mae_mean": float(np.mean(list(base.values()))),
                                "coverage80_mean": float(np.mean(list(cov.values())))})
            card = _card(variant, qs, test, cov)
            mlflow.log_dict(card, "explain/model_card.json")
            results[variant] = {"run_id": run.info.run_id, "mae_mean": mean, "card": card,
                                "use_yarra": use_yarra}
            log.info("timestral %s: mean MAE %.2f cm, 80%% coverage %.0f%%", variant, mean * 100,
                     100 * float(np.mean(list(cov.values()))))

    best = min(results, key=lambda v: results[v]["mae_mean"])
    version = register(results[best]["run_id"], best, results[best]["use_yarra"])
    return {v: {"mae_mean_cm": round(r["mae_mean"] * 100, 2),
                "coverage80": r["card"]["coverage_80"]} for v, r in results.items()} | {
        "live_variant": best, "version": version}


def register(run_id: str, variant: str, use_yarra: bool) -> str:
    """Log the model INTO MLflow (weights included: self-contained, no Hugging Face at load)."""
    import tempfile

    from huggingface_hub import snapshot_download

    client = configure()
    history = daily_levels(get_catalog())
    with tempfile.TemporaryDirectory() as tmp:
        # local_dir gives real files; the HF cache uses symlinks, which MLflow rejects.
        # LICENSE must travel with the weights (Modified MIT attribution requirement).
        weights = snapshot_download(MODEL_ID, revision=WEIGHTS_REVISION,
                                    local_dir=f"{tmp}/timestral-28m",
                                    allow_patterns=["config.json", "model.safetensors", "LICENSE"])
        with mlflow.start_run(run_id=run_id):
            info = mlflow.pyfunc.log_model(
                name="model", python_model=TimestralForecasterModel(use_yarra),
                artifacts={"weights": weights}, registered_model_name=MODEL_NAME,
                input_example=history.tail(30).reset_index(drop=True),
            )
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    client.set_model_version_tag(MODEL_NAME, version, "variant", variant)
    log.info("registered %s v%s (%s) as @%s", MODEL_NAME, version, variant, ALIAS)
    return version
