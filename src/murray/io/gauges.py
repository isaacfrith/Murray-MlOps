"""River gauge readings via MDBA's gauge-getter (the state water agencies behind MDBA River Data).

Barmah (409215) is served by the Victorian water portal, Yarrawonga (409025) by WaterNSW.
gauge-getter knows each gauge's state, endpoint and variable codes; this module adds what it
lacks for a scheduled pipeline:

- one state request per calendar-year chunk (long ranges return gaps or time out)
- retry with backoff on throttling/timeouts (WaterNSW: "Too many requests in queue")
- a pause between requests, one gauge at a time (these are free public services)
- no BoM fallback: gauge-getter's fallback path is broken; an empty answer stays empty

Values are agency-computed DAILY MEANS keyed by the agency's local calendar day.
"""

import datetime as dt
import logging
import time

import pandas as pd
import requests
from mdba_gauge_getter import gauge_getter as gg

log = logging.getLogger(__name__)

GAUGES = {
    "409215": "Murray River at Barmah",
    "409025": "Murray River at Yarrawonga Weir (downstream)",
}
# gauge-getter variable code → our column-friendly variable name
VARIABLES = {"L": "level_m", "F": "flow_mld"}
# Hydstra "datasource" per state, as used by gauge-getter's own gauge_pull()
DATASOURCE = {"NSW": "CP", "VIC": "PUBLISH"}

RETRYABLE_STATUS = ("408", "429", "500", "502", "503", "504")


def state_for(site_id: str) -> str:
    if gg.gauges is None:  # gauge-getter's gauge→state lookup table, loaded lazily
        gg.init()
    states = gg.get_states_for_gauge(site_id)
    if len(states) != 1:
        raise ValueError(f"gauge {site_id}: expected one owning state, got {states}")
    (state,) = states
    if state not in DATASOURCE:
        raise ValueError(f"gauge {site_id}: state {state} not supported (have {list(DATASOURCE)})")
    return state


def year_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Split [start, end] (inclusive) into calendar-year pieces."""
    if start > end:
        return []
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(dt.date(cursor.year, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return chunks


def _request_with_retry(fn, *, attempts: int, base_delay_s: float):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            retryable = not isinstance(exc, requests.HTTPError) or any(
                f"code {code}" in str(exc) for code in RETRYABLE_STATUS
            )
            if not retryable or attempt == attempts:
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            log.warning("request failed (%s), retry %d/%d in %.0fs",
                        str(exc).splitlines()[0][:120], attempt, attempts - 1, delay)
            time.sleep(delay)


def fetch_daily(
    site_id: str,
    start: dt.date,
    end: dt.date,
    var: str = "L",
    *,
    pause_s: float = 2.0,
    attempts: int = 5,
    base_delay_s: float = 5.0,
) -> pd.DataFrame:
    """Daily mean readings for one gauge, start..end inclusive.

    Returns columns: site_id, variable, obs_date, value, quality_code, source.
    """
    state = state_for(site_id)
    rows: list[list] = []
    chunks = year_chunks(start, end)
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        got = _request_with_retry(
            lambda: gg.process_gauge_pull(
                [site_id], state, DATASOURCE[state], chunk_start, chunk_end, var, "day", "mean"
            ),
            attempts=attempts,
            base_delay_s=base_delay_s,
        )
        log.info("%s %s %s..%s: %d rows", site_id, var, chunk_start, chunk_end, len(got))
        rows += got
        if i < len(chunks) - 1:
            time.sleep(pause_s)

    raw = pd.DataFrame(rows, columns=["DATASOURCEID", "SITEID", "SUBJECTID", "DATETIME",
                                      "VALUE", "QUALITYCODE"])
    return pd.DataFrame({
        "site_id": raw["SITEID"].astype(str),
        "variable": VARIABLES[var],
        "obs_date": pd.to_datetime(raw["DATETIME"]).dt.date,
        "value": pd.to_numeric(raw["VALUE"], errors="coerce"),
        "quality_code": pd.to_numeric(raw["QUALITYCODE"], errors="coerce").astype("Int32"),
        "source": raw["DATASOURCEID"].astype(str),
    })
