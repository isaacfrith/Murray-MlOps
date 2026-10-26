import importlib.util

import numpy as np
import pytest

from murray.models.timestral_model import QUANTILE_LEVELS, contexts, prob_in_range


def test_contexts_end_at_origin_and_pad_left():
    s = np.arange(10, dtype="float32")
    c = contexts(s, [3, 9], length=5)
    assert np.isnan(c[0, 0]) and c[0, -1] == 3      # padded, ends AT the origin
    assert c[1].tolist() == [5, 6, 7, 8, 9]
    assert np.nanmax(c[0]) == 3                       # nothing after the origin leaks in


def test_quantile_grid_is_21_levels():
    assert len(QUANTILE_LEVELS) == 21 and QUANTILE_LEVELS[0] == 0.01 and QUANTILE_LEVELS[-1] == 0.99


def test_prob_in_range_from_quantiles():
    q = {lvl: 2.0 + lvl for lvl in QUANTILE_LEVELS}   # level 2.0..3.0, uniform-ish
    assert prob_in_range(q, 0.0, 9.0) == pytest.approx(1.0)   # covers every outcome
    assert prob_in_range(q, 2.25, 2.75) == pytest.approx(0.5, abs=1e-9)
    assert prob_in_range(q, 3.5, 4.0) == 0.0


@pytest.mark.skipif(importlib.util.find_spec("timestral") is None, reason="needs murray[timestral]")
def test_real_model_forecast_shape():
    from huggingface_hub import snapshot_download

    from murray.models.timestral_model import (
        MODEL_ID, WEIGHTS_REVISION, forecast_batch, load_forecaster)

    try:
        path = snapshot_download(MODEL_ID, revision=WEIGHTS_REVISION, local_files_only=True)
    except Exception:
        pytest.skip("weights not downloaded")
    f = load_forecaster(path)
    b = contexts(np.linspace(2.0, 2.5, 500).astype("float32"), [499], length=512)
    qs = forecast_batch(f, b, None)
    assert qs[0.5].shape == (1, 7) and np.all(qs[0.1] <= qs[0.9])


def _quantile_frame(median_by_h: dict[int, float], spread: float = 0.2):
    import pandas as pd

    from murray.models.timestral_model import qcol

    return pd.DataFrame({qcol(q): [m + spread * (q - 0.5) for m in median_by_h.values()]
                         for q in QUANTILE_LEVELS}, index=list(median_by_h))


def test_second_opinion_days_rates_the_median_and_gates_probability():
    import datetime as dt

    from murray.models.timestral_model import second_opinion_days
    from murray.rules.wakeboarding import load_rule

    q = _quantile_frame({h: 2.65 for h in range(1, 8)})
    d0 = dt.date(2026, 10, 25)
    wind = {d0 + dt.timedelta(days=1): 10.0}
    days = second_opinion_days(q, d0, 2.65, wind, load_rule(), calibrated=True)
    assert [d["date"] for d in days][:2] == ["2026-10-26", "2026-10-27"]
    assert days[0]["rating"] == "good" and days[0]["wind_kmh"] == 10.0
    assert days[0]["q10_m"] < days[0]["level_m"] < days[0]["q90_m"]
    assert days[0]["p_good_level"] == pytest.approx(1.0)   # whole range inside 2.30–3.00

    hidden = second_opinion_days(q, d0, 2.65, wind, load_rule(), calibrated=False)
    assert all(d["p_good_level"] is None for d in hidden)


def test_calibration_gate():
    from murray.models.timestral_model import is_calibrated

    assert is_calibrated(0.82) and not is_calibrated(0.6) and not is_calibrated(None)
