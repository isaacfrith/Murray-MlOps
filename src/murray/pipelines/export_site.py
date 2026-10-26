"""Export what the static model page needs from the (private) MLflow registry.

site/models.json = registry history (every version, its scores and promotion reason) + the
current champion's model card. Runs at the end of every inference so the page always
describes the model that made today's forecast.
"""

import datetime as dt
import json
import logging
import os
import tempfile
from pathlib import Path

import mlflow

from murray.models.explain import model_card
from murray.pipelines.training_pipeline import (
    CHAMPION,
    MODEL_NAME,
    TEST_START,
    VAL_START,
    load_training_frame,
)
from murray.transforms.labels import HORIZONS, time_split

log = logging.getLogger(__name__)
CARD_PATH = "explain/model_card.json"


def champion_card(client: mlflow.MlflowClient) -> dict:
    """The champion's model card; built and attached to its run if it predates cards."""
    version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = client.download_artifacts(version.run_id, CARD_PATH, tmp)
            return json.loads(Path(path).read_text())
    except Exception:
        log.info("champion v%s has no model card; building it (no retraining)", version.version)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}").unwrap_python_model()
    test = time_split(load_training_frame(), VAL_START, TEST_START)["test"]
    feature_set = client.get_run(version.run_id).data.params.get("feature_set", "river_only")
    card = model_card(model, test, version=version.version, feature_set=feature_set)
    client.log_dict(version.run_id, card, CARD_PATH)
    return card


def registry_history(client: mlflow.MlflowClient) -> list[dict]:
    champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    rows = []
    for v in client.search_model_versions(f"name = '{MODEL_NAME}'"):
        run = client.get_run(v.run_id)
        m, p = run.data.metrics, run.data.params
        rows.append({
            "version": v.version, "champion": v.version == champion,
            "created": dt.datetime.fromtimestamp(v.creation_timestamp / 1000,
                                                 tz=dt.timezone.utc).isoformat(),
            "run_name": run.info.run_name,
            "feature_set": p.get("feature_set", p.get("features", "river_only")),
            "mae_mean_cm": round(m["test_mae_mean"] * 100, 2),
            "persistence_mean_cm": round(m["persistence_mae_mean"] * 100, 2),
            "mae_cm": [round(m[f"test_mae_h{h}"] * 100, 2) for h in HORIZONS],
            "promotion": v.tags.get("promotion", ""),
        })
    return sorted(rows, key=lambda r: int(r["version"]), reverse=True)


def export_models_json(client: mlflow.MlflowClient, site_dir: str | Path | None = None) -> Path:
    from murray.pipelines.inference_pipeline import DEFAULT_SITE_DIR

    site = Path(site_dir or os.environ.get("SITE_DIR") or DEFAULT_SITE_DIR)
    out = {"exported_at": dt.datetime.now(dt.timezone.utc).isoformat(), "model_name": MODEL_NAME,
           "history": registry_history(client), "champion_card": champion_card(client)}
    path = site / "models.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    log.info("wrote %s (%.0f KB)", path, path.stat().st_size / 1024)
    return path
