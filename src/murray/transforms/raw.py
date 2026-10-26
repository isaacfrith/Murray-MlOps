"""Pure functions over the append-only raw gauge table (no I/O).

The raw table keeps every value ever ingested, stamped with `ingested_at`. Agencies revise
provisional readings, so the same (site_id, variable, obs_date) can appear several times.
"""

import datetime as dt

import pandas as pd

KEY = ["site_id", "variable", "obs_date"]


def stamp(readings: pd.DataFrame, ingested_at: dt.datetime) -> pd.DataFrame:
    """Drop empty readings and add the ingestion timestamp (must be timezone-aware)."""
    if ingested_at.tzinfo is None:
        raise ValueError("ingested_at must be timezone-aware (use UTC)")
    out = readings.dropna(subset=["value"]).copy()
    out["ingested_at"] = pd.Timestamp(ingested_at).tz_convert("UTC")
    return out.reset_index(drop=True)


def only_completed_days(readings: pd.DataFrame, today_local: dt.date) -> pd.DataFrame:
    """Today's daily mean is still accumulating; keep days strictly before today."""
    return readings[readings["obs_date"] < today_local].reset_index(drop=True)


def latest(raw: pd.DataFrame) -> pd.DataFrame:
    """The most recently ingested value per (site_id, variable, obs_date)."""
    return (
        raw.sort_values("ingested_at")
        .drop_duplicates(subset=KEY, keep="last")
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def as_of(raw: pd.DataFrame, known_at: dt.datetime) -> pd.DataFrame:
    """What we knew at `known_at`: latest values among rows ingested at or before it.

    This is the point-in-time view features must use, so a backtest never sees a
    revision that arrived after the forecast was made.
    """
    return latest(raw[raw["ingested_at"] <= pd.Timestamp(known_at)])
