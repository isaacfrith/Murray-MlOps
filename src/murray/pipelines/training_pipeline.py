"""Training pipeline: Feast (point-in-time) features + labels → LightGBM → MLflow → @champion.

Promotion rule (both must hold, on the SAME test period):
  1. the candidate beats the persistence baseline (mean MAE over h1..h7), and
  2. the candidate beats the current @champion re-scored on that test period.
Otherwise the version is registered but @champion does not move.
"""

import datetime as dt
import logging
import os
import tempfile

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from murray.io.catalog import get_catalog
from murray.io.feature_store import get_store
from murray.io.raw_tables import gauge_daily_table
from murray.io.registry import configure
from murray.models.evaluate import mae_by_horizon, persistence_forecast, score_table
from murray.models.explain import model_card
from murray.models.forecaster import (
    DEFAULT_PARAMS,
    BarmahForecaster,
    best_iterations,
    feature_importance,
    train,
)
from murray.transforms.features import (
    BARMAH,
    FEATURE_COLUMNS,
    clean_levels,
    daily_wide,
    feature_timestamp,
)
from murray.transforms.labels import HORIZONS, make_labels, time_split
from murray.transforms.raw import latest
from murray.transforms.weather_features import WEATHER_FEATURE_COLUMNS

log = logging.getLogger(__name__)

EXPERIMENT = "barmah-level-forecast"
MODEL_NAME = "barmah-level-forecast"
CHAMPION = "champion"
VAL_START = dt.date(2017, 1, 1)
TEST_START = dt.date(2022, 1, 1)
FEATURE_SETS = {
    "river_only": list(FEATURE_COLUMNS),
    "river_weather": list(FEATURE_COLUMNS) + list(WEATHER_FEATURE_COLUMNS),
}
# Always fetch everything, so any champion (whatever its feature set) can be re-scored.
FEATURE_REFS = ([f"barmah_daily:{c}" for c in FEATURE_COLUMNS]
                + [f"barmah_weather:{c}" for c in WEATHER_FEATURE_COLUMNS])


def load_training_frame() -> pd.DataFrame:
    """One row per observed day D: point-in-time features (via Feast) + targets D+1..D+7."""
    raw = gauge_daily_table(get_catalog()).scan().to_pandas()
    labels = make_labels(daily_wide(clean_levels(latest(raw))))

    # Ask Feast: "what were the features at the moment each forecast would have been made?"
    entity_df = pd.DataFrame({
        "site_id": BARMAH,
        "obs_date": labels["obs_date"],
        "event_timestamp": feature_timestamp(labels["obs_date"]),
    })
    features = get_store().get_historical_features(
        entity_df=entity_df, features=FEATURE_REFS
    ).to_df()
    frame = features.merge(labels, on="obs_date", how="inner")
    frame = frame[frame["barmah_level_m"].notna()].sort_values("obs_date").reset_index(drop=True)
    log.info("training frame: %d rows, %s → %s", len(frame),
             frame["obs_date"].min(), frame["obs_date"].max())
    return frame


def _mean(d: dict[int, float]) -> float:
    return float(pd.Series(d).mean())


def _champion_mae(test: pd.DataFrame) -> tuple[str | None, float | None]:
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except mlflow.exceptions.MlflowException:
        return None, None
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
        inputs = model.metadata.get_input_schema().input_names()  # the champion's own features
        return version.version, _mean(mae_by_horizon(model.predict(test[inputs]), test))
    except Exception as exc:  # e.g. feature schema changed since that version
        log.warning("could not re-score champion v%s (%s); treating as no champion",
                    version.version, exc)
        return version.version, None


def run(params: dict | None = None, promote: bool = True, run_name: str | None = None,
        feature_set: str = "river_only") -> dict:
    cols = FEATURE_SETS[feature_set]
    client = configure()
    mlflow.set_experiment(EXPERIMENT)

    frame = load_training_frame()
    split = time_split(frame, VAL_START, TEST_START)
    tr, va, te = split["train"], split["val"], split["test"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            **{f"lgb_{k}": v for k, v in {**DEFAULT_PARAMS, **(params or {})}.items()},
            "feature_set": feature_set, "n_features": len(cols),
            "val_start": VAL_START, "test_start": TEST_START, "embargo_days": max(HORIZONS),
            "train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
            "train_first_date": tr["obs_date"].min(), "test_last_date": te["obs_date"].max(),
        })

        boosters = train(tr, va, params, feature_columns=cols)
        mlflow.log_params(best_iterations(boosters))
        model = BarmahForecaster(boosters, feature_columns=cols)
        pred = model.predict(None, te)

        model_mae = mae_by_horizon(pred, te)
        base_mae = mae_by_horizon(persistence_forecast(te), te)
        for h in HORIZONS:
            mlflow.log_metrics({f"test_mae_h{h}": model_mae[h], f"persistence_mae_h{h}": base_mae[h],
                                f"skill_h{h}": 1 - model_mae[h] / base_mae[h]})
        mean_mae, mean_base = _mean(model_mae), _mean(base_mae)
        mlflow.log_metrics({"test_mae_mean": mean_mae, "persistence_mae_mean": mean_base,
                            "skill_mean": 1 - mean_mae / mean_base})

        table = score_table(pred, te)
        for _, r in table[table.forecaster == "model"].iterrows():
            mlflow.log_metric(f"test_mae_mean_{r['regime']}",
                              float(r[[f"h{h}" for h in HORIZONS]].mean()))
        with tempfile.TemporaryDirectory() as tmp:
            table.to_csv(os.path.join(tmp, "score_table.csv"), index=False)
            feature_importance(boosters).to_csv(os.path.join(tmp, "feature_importance.csv"))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        example = te[cols].head(5)
        info = mlflow.pyfunc.log_model(
            name="model", python_model=model, registered_model_name=MODEL_NAME,
            input_example=example, signature=infer_signature(example, model.predict(None, example)),
        )
        version = info.registered_model_version
        # Model card (global SHAP, effect plot, backtest) for every run, promoted or not.
        mlflow.log_dict(model_card(model, te, version=str(version), feature_set=feature_set),
                        "explain/model_card.json")

        # --- Promotion -----------------------------------------------------------------
        champ_version, champ_mae = _champion_mae(te)
        beats_baseline = mean_mae < mean_base
        beats_champion = champ_mae is None or mean_mae < champ_mae
        promoted = promote and beats_baseline and beats_champion
        reason = (f"candidate {mean_mae:.4f} m vs persistence {mean_base:.4f} m"
                  + (f" vs champion v{champ_version} {champ_mae:.4f} m" if champ_mae else
                     " (no comparable champion)"))
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        client.set_model_version_tag(MODEL_NAME, version, "promotion",
                                     ("promoted: " if promoted else "not promoted: ") + reason)
        mlflow.set_tags({"promoted": promoted, "promotion_reason": reason})
        log.info("v%s %s — %s", version, "PROMOTED" if promoted else "kept champion", reason)

        return {"run_id": run.info.run_id, "version": version, "promoted": promoted,
                "test_mae": model_mae, "persistence_mae": base_mae, "reason": reason}
