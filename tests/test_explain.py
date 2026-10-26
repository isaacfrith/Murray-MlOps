import datetime as dt

import numpy as np
import pandas as pd

from murray.models.explain import grouped, group_of, local_explanation, model_card
from murray.models.forecaster import BarmahForecaster, train
from tests.test_forecaster import synthetic


def fitted():
    return BarmahForecaster(train(synthetic(600, 0), synthetic(200, 1), num_boost_round=150))


def test_groups_cover_every_feature():
    assert group_of("barmah_level_m") == "Current level"
    assert group_of("barmah_change_3d") == "Barmah trend"
    assert group_of("yarra_level_lag4") == "Upstream: Yarrawonga"
    assert group_of("doy_sin") == "Season"
    assert group_of("rain_albury_7d") == "Weather"


def test_shap_adds_up_to_the_prediction():
    model, test = fitted(), synthetic(50, 2)
    pred = model.predict(None, test)
    for h, c in model.contributions(test).items():
        delta = pred[f"h{h}"] - test["barmah_level_m"]
        assert np.allclose(c.sum(axis=1), delta, atol=1e-9)        # features + base
        assert np.allclose(grouped(c).sum(axis=1) + c["_base"], delta, atol=1e-9)  # groups too


def test_local_explanation_waterfall_reaches_the_forecast():
    model, test = fitted(), synthetic(1, 3)
    days = local_explanation(model, test, dt.date(2026, 10, 25))
    pred = model.predict(None, test).iloc[0]
    assert [d["date"] for d in days][:2] == ["2026-10-26", "2026-10-27"]
    for d in days:
        assert abs(d["forecast_m"] - pred[f"h{d['horizon']}"]) < 1e-3
        assert 1 <= len(d["captions"]) <= 2


def test_model_card_has_every_section():
    model, test = fitted(), synthetic(120, 4)
    test["obs_date"] = pd.date_range("2024-01-01", periods=len(test)).date
    card = model_card(model, test, version="9", feature_set="river_only")
    assert {"scores_cm", "importance", "effect", "backtest"} <= set(card)
    assert set(card["backtest"]) == {"1", "4", "7"}
