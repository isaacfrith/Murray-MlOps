"""Daily weather from Open-Meteo's historical (ERA5-based) archive API.

The archive reaches from 1940 to yesterday, so the same endpoint serves the backfill and the
daily refresh. Free for non-commercial use; attribution required (CC BY 4.0, link to
open-meteo.com) — see README.
"""

import datetime as dt
import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "murray-forecast/0.1 (hobby MLOps project)"

# Rain upstream matters more than rain at Barmah: water takes days to arrive.
LOCATIONS = {
    "barmah": (-36.02, 144.96),
    "yarrawonga": (-36.01, 146.00),
    "wangaratta": (-36.36, 146.33),   # Ovens River, joins the Murray above Yarrawonga
    "albury": (-36.08, 146.92),       # upper Murray, below Hume Dam
}
DAILY_VARS = [
    "precipitation_sum",            # mm
    "temperature_2m_max",           # °C
    "et0_fao_evapotranspiration",   # mm (evaporation demand)
    "wind_speed_10m_max",           # km/h (for the wakeboarding rule, not the level model)
]


def decade_chunks(start: dt.date, end: dt.date, years: int = 10) -> list[tuple[dt.date, dt.date]]:
    chunks, cursor = [], start
    while cursor <= end:
        chunk_end = min(dt.date(cursor.year + years - 1, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return chunks


def _get(params: dict, attempts: int = 4, base_delay_s: float = 10.0) -> list[dict]:
    for attempt in range(1, attempts + 1):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=60,
                            headers={"User-Agent": USER_AGENT})
        if resp.status_code == 200:
            body = resp.json()
            return body if isinstance(body, list) else [body]
        if resp.status_code not in (429, 500, 502, 503, 504) or attempt == attempts:
            raise requests.HTTPError(f"Open-Meteo {resp.status_code}: {resp.text[:200]}")
        delay = base_delay_s * 2 ** (attempt - 1)
        log.warning("Open-Meteo %s, retry %d in %.0fs", resp.status_code, attempt, delay)
        time.sleep(delay)


def fetch_daily(start: dt.date, end: dt.date, pause_s: float = 1.0) -> pd.DataFrame:
    """All LOCATIONS, start..end inclusive. Columns: location, obs_date, <DAILY_VARS>."""
    names = list(LOCATIONS)
    frames = []
    for i, (cs, ce) in enumerate(decade_chunks(start, end)):
        responses = _get({
            "latitude": ",".join(str(LOCATIONS[n][0]) for n in names),
            "longitude": ",".join(str(LOCATIONS[n][1]) for n in names),
            "start_date": cs.isoformat(), "end_date": ce.isoformat(),
            "daily": ",".join(DAILY_VARS), "timezone": "Australia/Melbourne",
        })
        for name, body in zip(names, responses):  # responses come back in request order
            df = pd.DataFrame(body["daily"]).rename(columns={"time": "obs_date"})
            df.insert(0, "location", name)
            frames.append(df)
        log.info("weather %s..%s: %d locations", cs, ce, len(responses))
        if i:
            time.sleep(pause_s)
    out = pd.concat(frames, ignore_index=True)
    out["obs_date"] = pd.to_datetime(out["obs_date"]).dt.date
    return out
