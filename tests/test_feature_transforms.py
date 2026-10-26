import datetime as dt

import numpy as np
import pandas as pd

from murray.transforms.features import (
    BARMAH,
    FEATURE_COLUMNS,
    YARRAWONGA,
    barmah_features,
    clean_levels,
    daily_wide,
    feature_timestamp,
)

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 26, tzinfo=UTC)


def raw_rows(site, start, values, quality=3):
    return pd.DataFrame({
        "site_id": site, "variable": "level_m",
        "obs_date": [start + dt.timedelta(days=i) for i in range(len(values))],
        "value": values, "quality_code": quality, "source": "X",
    })


def wide_from(barmah, yarra, start=dt.date(2026, 1, 1)):
    raw = pd.concat([raw_rows(BARMAH, start, barmah), raw_rows(YARRAWONGA, start, yarra)])
    return daily_wide(clean_levels(raw))


def test_clean_levels_blanks_no_data_placeholders():
    raw = pd.concat([raw_rows(BARMAH, dt.date(2026, 1, 1), [0.0], quality=255),
                     raw_rows(BARMAH, dt.date(2026, 1, 2), [2.5], quality=2),
                     raw_rows(BARMAH, dt.date(2026, 1, 3), [0.0], quality=2)])
    assert clean_levels(raw)["value"].isna().tolist() == [True, False, True]


def test_daily_wide_inserts_missing_days():
    raw = pd.concat([raw_rows(BARMAH, dt.date(2026, 1, 1), [2.0]),
                     raw_rows(BARMAH, dt.date(2026, 1, 3), [2.2])])
    wide = daily_wide(clean_levels(raw))
    assert len(wide) == 3 and np.isnan(wide.loc["2026-01-02", BARMAH])


def test_lags_and_changes_only_look_backwards():
    barmah = [2.0 + 0.1 * i for i in range(15)]
    yarra = [1.0 + 0.01 * i for i in range(15)]
    f = barmah_features(wide_from(barmah, yarra), NOW).set_index("obs_date")
    row = f.loc[dt.date(2026, 1, 11)]  # day index 10
    assert np.isclose(row["barmah_level_m"], 3.0)
    assert np.isclose(row["barmah_level_lag1"], 2.9)
    assert np.isclose(row["barmah_level_lag7"], 2.3)
    assert np.isclose(row["barmah_change_3d"], 0.3)
    assert np.isclose(row["yarra_level_lag10"], 1.0)
    assert list(f.columns[: len(["site_id"])]) == ["site_id"]
    assert set(FEATURE_COLUMNS) <= set(f.columns)


def test_no_future_leakage():
    """Changing a value on day D+1 must not change any feature of day D."""
    base = [2.0 + 0.1 * i for i in range(15)]
    bumped = base.copy()
    bumped[11] += 5.0  # 12 Jan
    yarra = [1.0] * 15
    f1 = barmah_features(wide_from(base, yarra), NOW).set_index("obs_date")
    f2 = barmah_features(wide_from(bumped, yarra), NOW).set_index("obs_date")
    day = dt.date(2026, 1, 11)
    pd.testing.assert_series_equal(f1.loc[day, FEATURE_COLUMNS], f2.loc[day, FEATURE_COLUMNS])


def test_rows_without_barmah_reading_are_dropped():
    barmah = [2.0, np.nan, 2.2]
    f = barmah_features(wide_from(barmah, [1.0, 1.0, 1.0]), NOW)
    assert f["obs_date"].tolist() == [dt.date(2026, 1, 1), dt.date(2026, 1, 3)]


def test_feature_timestamp_is_next_morning_melbourne_dst_aware():
    ts = feature_timestamp(pd.Series([dt.date(2026, 1, 15), dt.date(2026, 7, 15)]))
    # Summer (AEDT, UTC+11): 16 Jan 10:00 local = 15 Jan 23:00 UTC
    assert ts[0] == pd.Timestamp("2026-01-15 23:00", tz="UTC")
    # Winter (AEST, UTC+10): 16 Jul 10:00 local = 16 Jul 00:00 UTC
    assert ts[1] == pd.Timestamp("2026-07-16 00:00", tz="UTC")
