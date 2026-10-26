"""Raw ingest pipeline: state agencies (via gauge-getter) → Iceberg raw.gauge_daily.

Plain functions so they run the same from Airflow, a notebook or a test.
"""

import datetime as dt
import logging
from zoneinfo import ZoneInfo

import pandas as pd

from murray.io.catalog import get_catalog
from murray.io.gauges import GAUGES, VARIABLES, fetch_daily
from murray.io.raw_tables import (
    append_gauge_daily,
    append_rows,
    gauge_daily_table,
    weather_daily_table,
)
from murray.io.weather import fetch_daily as fetch_weather
from murray.transforms.raw import only_completed_days, stamp

log = logging.getLogger(__name__)

LOCAL_TZ = ZoneInfo("Australia/Melbourne")  # Barmah/Yarrawonga agency day boundaries
HISTORY_START = dt.date(1970, 1, 1)


def ingest(site_id: str, start: dt.date, end: dt.date, var: str = "L",
           now: dt.datetime | None = None) -> int:
    """Fetch start..end (inclusive) for one gauge and append it. Returns rows written."""
    now = now or dt.datetime.now(dt.timezone.utc)
    today_local = now.astimezone(LOCAL_TZ).date()
    readings = fetch_daily(site_id, start, min(end, today_local), var)
    rows = stamp(only_completed_days(readings, today_local), ingested_at=now)
    written = append_gauge_daily(gauge_daily_table(get_catalog()), rows)
    log.info("%s %s: fetched %d, wrote %d rows", site_id, var, len(readings), written)
    return written


def ingest_recent(site_id: str, days: int = 14, var: str = "L") -> int:
    """Daily run: re-fetch the last `days` completed days. Overlap is deliberate:
    it captures the agencies' revisions of provisional values (append-only keeps both)."""
    today_local = dt.datetime.now(LOCAL_TZ).date()
    return ingest(site_id, today_local - dt.timedelta(days=days), today_local, var)


def backfill(site_id: str, var: str = "L", force: bool = False) -> int:
    """One-off: full history. Refuses to run twice unless force=True (it would
    duplicate the whole history under a new ingested_at)."""
    table = gauge_daily_table(get_catalog())
    existing = table.scan(
        row_filter=f"site_id == '{site_id}' and variable == '{VARIABLES[var]}'",
        selected_fields=("site_id",),
        limit=1,
    ).to_arrow().num_rows
    if existing and not force:
        log.info("%s already has history; skipping backfill (force=True to override)", site_id)
        return 0
    return ingest(site_id, HISTORY_START, dt.date.today(), var)


def ingest_weather(start: dt.date, end: dt.date, now: dt.datetime | None = None) -> int:
    """Weather for all locations, start..end (only completed local days), append-only."""
    now = now or dt.datetime.now(dt.timezone.utc)
    today_local = now.astimezone(LOCAL_TZ).date()
    rows = only_completed_days(fetch_weather(start, min(end, today_local)), today_local)
    rows = rows.copy()
    rows["ingested_at"] = pd.Timestamp(now).tz_convert("UTC")
    written = append_rows(weather_daily_table(get_catalog()), rows)
    log.info("weather: wrote %d rows", written)
    return written


def weather_backfill(force: bool = False) -> int:
    table = weather_daily_table(get_catalog())
    if table.scan(selected_fields=("location",), limit=1).to_arrow().num_rows and not force:
        log.info("weather already has history; skipping backfill")
        return 0
    return ingest_weather(HISTORY_START, dt.date.today())


def weather_ingest_recent(days: int = 14) -> int:
    today_local = dt.datetime.now(LOCAL_TZ).date()
    return ingest_weather(today_local - dt.timedelta(days=days), today_local)


__all__ = ["GAUGES", "backfill", "ingest", "ingest_recent", "weather_backfill",
           "weather_ingest_recent"]
