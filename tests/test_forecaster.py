import pickle

import numpy as np
import pandas as pd

from murray.models.forecaster import BarmahForecaster, train
from murray.transforms.features import FEATURE_COLUMNS
from murray.transforms.labels import HORIZONS


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["barmah_level_m"] = 2.5 + 0.1 * rng.normal(size=n)
    for h in HORIZONS:  # future level = current + a learnable push from upstream change
        df[f"target_h{h}"] = df["barmah_level_m"] + 0.05 * h * df["yarra_change_3d"]
    return df


def test_forecaster_learns_and_survives_pickling():
    boosters = train(synthetic(600, 0), synthetic(200, 1), num_boost_round=200)
    model = pickle.loads(pickle.dumps(BarmahForecaster(boosters)))  # what MLflow does
    test = synthetic(200, 2)
    pred = model.predict(None, test)
    assert list(pred.columns) == [f"h{h}" for h in HORIZONS]
    model_mae = (pred["h7"] - test["target_h7"]).abs().mean()
    persistence_mae = (test["barmah_level_m"] - test["target_h7"]).abs().mean()
    assert model_mae < 0.5 * persistence_mae
