import datetime as dt

import pandas as pd

from murray.transforms.monitoring import latest_forecasts, match_actuals, summarize

D = dt.date


def preds(obs, horizon, level, issued="2026-10-26 00:00"):
    return {"obs_date": obs, "horizon": horizon, "target_date": obs + dt.timedelta(days=horizon),
            "level_m": level, "issued_at": pd.Timestamp(issued, tz="UTC"), "model_version": "1",
            "wind_kmh": 10.0, "rating": "good"}


def test_latest_issue_wins_per_obs_date_and_horizon():
    p = pd.DataFrame([preds(D(2026, 10, 1), 1, 2.0, "2026-10-02 00:00"),
                      preds(D(2026, 10, 1), 1, 2.1, "2026-10-02 05:00")])
    assert latest_forecasts(p)["level_m"].tolist() == [2.1]


def test_match_and_summarize_flags_degradation():
    actual = pd.Series({D(2026, 10, d): 2.0 for d in range(1, 30)})
    rows = [preds(D(2026, 10, d), 1, 2.10) for d in range(1, 20)]   # 10 cm off, every day
    rows.append(preds(D(2026, 10, 29), 1, 2.0))                      # matures beyond data → dropped
    m = match_actuals(latest_forecasts(pd.DataFrame(rows)), actual)
    assert len(m) == 19 and abs(m["error_m"].abs().mean() - 0.10) < 1e-9
    s = summarize(m, backtest_mae_m={1: 0.011})
    h1 = s["horizons"][0]
    assert h1["status"] == "degraded" and s["degraded"]
    assert all(h["status"] == "waiting" for h in s["horizons"][1:])  # no data for h2..h7


def test_waiting_until_enough_matured():
    actual = pd.Series({D(2026, 10, d): 2.0 for d in range(1, 5)})
    m = match_actuals(pd.DataFrame([preds(D(2026, 10, 1), 1, 2.5)]), actual)
    s = summarize(m, backtest_mae_m={1: 0.011})
    assert s["horizons"][0]["status"] == "waiting" and not s["degraded"]


def test_old_rows_default_to_lightgbm_and_models_dedupe_separately():
    from murray.transforms.monitoring import with_model_name

    p = pd.DataFrame({"issued_at": pd.to_datetime(["2026-10-26 01:00", "2026-10-26 02:00",
                                                   "2026-10-26 02:00"], utc=True),
                      "model_name": [None, None, "barmah-level-timestral"],
                      "obs_date": [dt.date(2026, 10, 25)] * 3, "horizon": [1, 1, 1],
                      "level_m": [2.1, 2.2, 2.3]})
    out = latest_forecasts(with_model_name(p, "barmah-level-forecast"))
    assert sorted(zip(out["model_name"], out["level_m"])) == [
        ("barmah-level-forecast", 2.2), ("barmah-level-timestral", 2.3)]
