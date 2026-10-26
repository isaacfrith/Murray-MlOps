"""Feature pipeline: raw.gauge_daily → features.barmah_daily (Iceberg) → Valkey (via Feast).

Steps are separate functions so Airflow can run (and retry) them as separate tasks.
"""

import datetime as dt
import logging

from murray.io.catalog import get_catalog
from murray.io.feature_store import get_store
from murray.io.feature_tables import barmah_daily_table, write_barmah_daily
from murray.io.raw_tables import gauge_daily_table
from murray.transforms.features import barmah_features, clean_levels, daily_wide
from murray.transforms.raw import as_of

log = logging.getLogger(__name__)


def build_features(now: dt.datetime | None = None) -> int:
    """Recompute all feature rows from what raw knew at `now`, overwrite the feature table."""
    now = now or dt.datetime.now(dt.timezone.utc)
    catalog = get_catalog()
    raw = gauge_daily_table(catalog).scan().to_pandas()
    wide = daily_wide(clean_levels(as_of(raw, now)))
    features = barmah_features(wide, computed_at=now)
    written = write_barmah_daily(barmah_daily_table(catalog), features)
    log.info("wrote %d feature rows, %s → %s", written,
             features["obs_date"].min(), features["obs_date"].max())
    return written


def apply_definitions() -> list[str]:
    """`feast apply`: register entities and feature views from features/definitions.py."""
    import importlib.util
    import sys

    store = get_store()
    path = store.repo_path / "definitions.py"
    spec = importlib.util.spec_from_file_location("murray_feature_definitions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    objects = [module.site, module.barmah_daily]
    store.apply(objects)
    return [o.name for o in objects]


def materialize(now: dt.datetime | None = None) -> str:
    """Copy the latest feature row per site into Valkey (incremental since last run)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    get_store().materialize_incremental(end_date=now)
    return f"materialized up to {now.isoformat()}"
