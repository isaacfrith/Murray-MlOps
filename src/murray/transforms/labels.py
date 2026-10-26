"""Training targets and time splits (pure functions, no I/O).

Horizon h means "the daily mean level h days after the last observed day D". The forecast is
issued at D+1 10:00 Melbourne (when row D becomes known), so h=1 is the day the forecast is
issued, h=2 is tomorrow, ... h=7 is six days after that.

Targets use the latest (corrected) values: they are the ground truth we are scored against.
"""

import datetime as dt

import pandas as pd

from murray.transforms.features import BARMAH

HORIZONS = range(1, 8)


def target_columns() -> list[str]:
    return [f"target_h{h}" for h in HORIZONS]


def make_labels(wide: pd.DataFrame) -> pd.DataFrame:
    """obs_date, then target_h1..h7 = Barmah level on D+h (NaN where that day is missing)."""
    b = wide[BARMAH]
    labels = pd.DataFrame({f"target_h{h}": b.shift(-h) for h in HORIZONS}, index=wide.index)
    labels = labels.reset_index()
    labels["obs_date"] = labels["obs_date"].dt.date
    return labels


def time_split(
    df: pd.DataFrame, val_start: dt.date, test_start: dt.date, embargo_days: int = max(HORIZONS)
) -> dict[str, pd.DataFrame]:
    """Train < val_start <= val < test_start <= test, by obs_date.

    The embargo drops the last `embargo_days` rows before each boundary: their targets reach
    up to 7 days ahead, i.e. into the next period, which would leak it into the earlier one.
    """
    d = pd.to_datetime(df["obs_date"])
    gap = pd.Timedelta(days=embargo_days)
    return {
        "train": df[d < pd.Timestamp(val_start) - gap],
        "val": df[(d >= pd.Timestamp(val_start)) & (d < pd.Timestamp(test_start) - gap)],
        "test": df[d >= pd.Timestamp(test_start)],
    }
