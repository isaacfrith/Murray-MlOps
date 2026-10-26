import datetime as dt

import pandas as pd

from murray.transforms.weather_features import (
    RAIN_LOCATIONS,
    WEATHER_FEATURE_COLUMNS,
    latest_weather,
    weather_features,
)

UTC = dt.timezone.utc


def raw(rain_barmah, start=dt.date(2026, 1, 1), ingested=dt.datetime(2026, 2, 1, tzinfo=UTC)):
    rows = []
    for loc in RAIN_LOCATIONS:
        for i, r in enumerate(rain_barmah):
            rows.append({"location": loc, "obs_date": start + dt.timedelta(days=i),
                         "precipitation_sum": r if loc == "barmah" else 0.0,
                         "temperature_2m_max": 25.0, "et0_fao_evapotranspiration": 5.0,
                         "wind_speed_10m_max": 10.0, "ingested_at": pd.Timestamp(ingested)})
    return pd.DataFrame(rows)


def test_rain_windows_are_trailing_and_exclude_the_future():
    rain = [0.0] * 40
    rain[35] = 10.0  # rain on 5 Feb
    f = weather_features(latest_weather(raw(rain)), dt.datetime.now(UTC)).set_index("obs_date")
    assert f.loc[dt.date(2026, 2, 4), "rain_barmah_7d"] == 0.0   # day before: not seen
    assert f.loc[dt.date(2026, 2, 5), "rain_barmah_1d"] == 10.0
    assert f.loc[dt.date(2026, 2, 9), "rain_barmah_3d"] == 0.0   # window moved past it
    assert f.loc[dt.date(2026, 2, 9), "rain_barmah_7d"] == 10.0
    assert set(WEATHER_FEATURE_COLUMNS) <= set(f.columns)


def test_latest_weather_takes_newest_ingest():
    old = raw([1.0] * 3)
    new = raw([2.0] * 3, ingested=dt.datetime(2026, 3, 1, tzinfo=UTC))
    lw = latest_weather(pd.concat([old, new]))
    assert set(lw[lw.location == "barmah"]["precipitation_sum"]) == {2.0}
