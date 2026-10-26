import datetime as dt

import pandas as pd
import pytest

from murray.io.gauges import year_chunks
from murray.transforms.raw import as_of, latest, only_completed_days, stamp

UTC = dt.timezone.utc


def readings(values, start=dt.date(2026, 10, 1)):
    return pd.DataFrame({
        "site_id": "409215",
        "variable": "level_m",
        "obs_date": [start + dt.timedelta(days=i) for i in range(len(values))],
        "value": values,
        "quality_code": 3,
        "source": "VIC",
    })


def test_year_chunks_split_on_calendar_years():
    assert year_chunks(dt.date(2024, 11, 5), dt.date(2026, 2, 1)) == [
        (dt.date(2024, 11, 5), dt.date(2024, 12, 31)),
        (dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
        (dt.date(2026, 1, 1), dt.date(2026, 2, 1)),
    ]
    assert year_chunks(dt.date(2026, 1, 2), dt.date(2026, 1, 1)) == []


def test_stamp_drops_missing_and_requires_aware_time():
    out = stamp(readings([2.7, None, 2.8]), dt.datetime(2026, 10, 5, tzinfo=UTC))
    assert out["value"].tolist() == [2.7, 2.8]
    assert str(out["ingested_at"].dt.tz) == "UTC"
    with pytest.raises(ValueError):
        stamp(readings([2.7]), dt.datetime(2026, 10, 5))


def test_only_completed_days_excludes_today():
    out = only_completed_days(readings([2.7, 2.8, 2.9]), today_local=dt.date(2026, 10, 3))
    assert out["obs_date"].max() == dt.date(2026, 10, 2)


def test_latest_and_as_of_handle_revisions():
    first = stamp(readings([2.70, 2.80]), dt.datetime(2026, 10, 3, tzinfo=UTC))
    revised = stamp(readings([2.75]), dt.datetime(2026, 10, 10, tzinfo=UTC))  # 1 Oct revised
    raw = pd.concat([first, revised], ignore_index=True)

    assert latest(raw)["value"].tolist() == [2.75, 2.80]
    # Before the revision arrived, we only knew 2.70 for 1 Oct:
    known = as_of(raw, dt.datetime(2026, 10, 5, tzinfo=UTC))
    assert known["value"].tolist() == [2.70, 2.80]
