"""Barmah forecast features from cleaned daily gauge levels (pure functions, no I/O).

One feature row per day D, describing what is known once D's daily means are published.
Every feature uses data from day D or earlier only; targets (D+1..D+7) are built in the
training pipeline, never here.

Point-in-time rule: row D gets event_timestamp = D+1 at 10:00 Australia/Melbourne, when the
daily ingest (raw_gauges_daily) fetches D's completed daily mean. Feast's point-in-time join
then only hands a forecast made at time t the rows with event_timestamp <= t.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BARMAH = "409215"
YARRAWONGA = "409025"
LOCAL_TZ = ZoneInfo("Australia/Melbourne")
AVAILABLE_AT_LOCAL = dt.time(10, 0)  # matches raw_gauges_daily's schedule

# Agency quality codes meaning "no data" (the value is a 0.00 placeholder).
NO_DATA_QUALITY = {160, 201, 255}

BARMAH_LAGS = range(1, 8)  # level 1..7 days before D
YARRA_LAGS = range(1, 11)  # travel time Yarrawonga→Barmah peaks ~4 days, plateau to ~10


def clean_levels(latest_raw: pd.DataFrame) -> pd.DataFrame:
    """Blank out placeholder readings: no-data quality codes and non-positive levels."""
    out = latest_raw.copy()
    bad = out["quality_code"].isin(NO_DATA_QUALITY) | (out["value"] <= 0)
    out.loc[bad, "value"] = np.nan
    return out


def daily_wide(clean: pd.DataFrame, variable: str = "level_m") -> pd.DataFrame:
    """One row per calendar day, one column per site. Missing days become NaN rows."""
    wide = clean[clean["variable"] == variable].pivot(
        index="obs_date", columns="site_id", values="value"
    )
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index), name="obs_date")
    return wide.asfreq("D")


def feature_timestamp(obs_dates: pd.Series) -> pd.Series:
    """When row D becomes known: D+1 at 10:00 Melbourne time, expressed in UTC."""
    local = (pd.to_datetime(obs_dates) + pd.Timedelta(days=1)
             + pd.Timedelta(hours=AVAILABLE_AT_LOCAL.hour, minutes=AVAILABLE_AT_LOCAL.minute))
    return local.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward",
                                ambiguous=False).dt.tz_convert("UTC")


def barmah_features(wide: pd.DataFrame, computed_at: dt.datetime) -> pd.DataFrame:
    """Feature rows for the Barmah forecast (entity site_id = 409215)."""
    b = wide[BARMAH]
    y = wide[YARRAWONGA]
    f = pd.DataFrame(index=wide.index)

    f["barmah_level_m"] = b
    for k in BARMAH_LAGS:
        f[f"barmah_level_lag{k}"] = b.shift(k)
    for k in (1, 3, 7):
        f[f"barmah_change_{k}d"] = b - b.shift(k)
    f["barmah_mean_7d"] = b.rolling(7, min_periods=5).mean()
    f["barmah_std_7d"] = b.rolling(7, min_periods=5).std()

    f["yarra_level_m"] = y
    for k in YARRA_LAGS:
        f[f"yarra_level_lag{k}"] = y.shift(k)
    for k in (1, 3, 7):
        f[f"yarra_change_{k}d"] = y - y.shift(k)

    doy = f.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    f = f.reset_index()
    f.insert(0, "site_id", BARMAH)
    f["obs_date"] = f["obs_date"].dt.date
    f["event_timestamp"] = feature_timestamp(f["obs_date"])
    f["created_timestamp"] = pd.Timestamp(computed_at).tz_convert("UTC")
    # A row with no Barmah reading that day carries no usable "current state".
    return f[f["barmah_level_m"].notna()].reset_index(drop=True)


FEATURE_COLUMNS = (
    ["barmah_level_m"]
    + [f"barmah_level_lag{k}" for k in BARMAH_LAGS]
    + [f"barmah_change_{k}d" for k in (1, 3, 7)]
    + ["barmah_mean_7d", "barmah_std_7d", "yarra_level_m"]
    + [f"yarra_level_lag{k}" for k in YARRA_LAGS]
    + [f"yarra_change_{k}d" for k in (1, 3, 7)]
    + ["doy_sin", "doy_cos"]
)
