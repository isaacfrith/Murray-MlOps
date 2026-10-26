"""Weather features for the Barmah forecast (pure functions, no I/O).

PAST weather only (up to and including day D). Observed FUTURE rain must never be a training
feature: live, nobody knows tomorrow's rain, so a model trained on it would be cheating.
Same availability rule as the river features: row D is known at D+1 10:00 Melbourne.
"""

import datetime as dt

import pandas as pd

from murray.transforms.features import BARMAH, feature_timestamp
from murray.transforms.raw import KEY  # noqa: F401  (documents the raw-table convention)

RAIN_WINDOWS = (1, 3, 7, 14, 30)
RAIN_LOCATIONS = ("barmah", "yarrawonga", "wangaratta", "albury")

WEATHER_FEATURE_COLUMNS = (
    [f"rain_{loc}_{w}d" for loc in RAIN_LOCATIONS for w in RAIN_WINDOWS]
    + ["tmax_barmah_mean_7d", "et0_barmah_sum_7d"]
)


def latest_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """Most recently ingested row per (location, obs_date)."""
    return (raw.sort_values("ingested_at")
            .drop_duplicates(subset=["location", "obs_date"], keep="last"))


def weather_features(latest: pd.DataFrame, computed_at: dt.datetime) -> pd.DataFrame:
    def wide(var: str) -> pd.DataFrame:
        w = latest.pivot(index="obs_date", columns="location", values=var)
        w.index = pd.DatetimeIndex(pd.to_datetime(w.index), name="obs_date")
        return w.asfreq("D")

    rain, tmax, et0 = wide("precipitation_sum"), wide("temperature_2m_max"), \
        wide("et0_fao_evapotranspiration")
    f = pd.DataFrame(index=rain.index)
    for loc in RAIN_LOCATIONS:
        for w in RAIN_WINDOWS:  # trailing window ending on D (inclusive): past only
            f[f"rain_{loc}_{w}d"] = rain[loc].rolling(w, min_periods=w).sum()
    f["tmax_barmah_mean_7d"] = tmax["barmah"].rolling(7, min_periods=5).mean()
    f["et0_barmah_sum_7d"] = et0["barmah"].rolling(7, min_periods=7).sum()

    f = f.reset_index()
    f.insert(0, "site_id", BARMAH)
    f["obs_date"] = f["obs_date"].dt.date
    f["event_timestamp"] = feature_timestamp(f["obs_date"])
    f["created_timestamp"] = pd.Timestamp(computed_at).tz_convert("UTC")
    return f
