import datetime as dt

import numpy as np
import pandas as pd

from murray.models.evaluate import mae_by_horizon, persistence_forecast, score_table
from murray.transforms.features import BARMAH, YARRAWONGA
from murray.transforms.labels import make_labels, target_columns, time_split


def wide(values, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D", name="obs_date")
    return pd.DataFrame({BARMAH: values, YARRAWONGA: 1.0}, index=idx)


def test_labels_are_future_values_of_barmah():
    lab = make_labels(wide([float(i) for i in range(10)])).set_index("obs_date")
    row = lab.loc[dt.date(2026, 1, 1)]
    assert [row[c] for c in target_columns()] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert np.isnan(lab.loc[dt.date(2026, 1, 8), "target_h3"])  # beyond the data


def test_time_split_has_embargo_and_no_overlap():
    df = pd.DataFrame({"obs_date": pd.date_range("2020-01-01", "2020-12-31").date})
    s = time_split(df, val_start=dt.date(2020, 6, 1), test_start=dt.date(2020, 10, 1))
    assert s["train"]["obs_date"].max() == dt.date(2020, 5, 24)  # 7-day embargo before val
    assert s["val"]["obs_date"].min() == dt.date(2020, 6, 1)
    assert s["val"]["obs_date"].max() == dt.date(2020, 9, 23)
    assert s["test"]["obs_date"].min() == dt.date(2020, 10, 1)


def test_persistence_and_mae():
    df = pd.DataFrame({"barmah_level_m": [2.0, 2.0], "yarra_change_7d": [0.0, 0.5],
                       **{f"target_h{h}": [2.0 + 0.1 * h, 2.0] for h in range(1, 8)}})
    base = persistence_forecast(df)
    assert np.isclose(mae_by_horizon(base, df)[1], 0.05)  # (0.1 + 0) / 2
    table = score_table(base, df)
    assert set(table["regime"]) == {"all", "steady", "release_change"}
