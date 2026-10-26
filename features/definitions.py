"""Feast definitions: what features exist, where they come from, how long they stay valid."""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.data_sources.contrib.iceberg_catalog.iceberg_source import IcebergSource
from feast.types import Float64
from feast.value_type import ValueType

from murray.transforms.features import FEATURE_COLUMNS

site = Entity(
    name="site",
    join_keys=["site_id"],
    value_type=ValueType.STRING,
    description="River gauge the forecast is for (409215 = Murray River at Barmah)",
)

barmah_daily_source = IcebergSource(
    name="barmah_daily_iceberg",
    catalog_type="sql",
    catalog_name="murray",  # connection details come from PYICEBERG_CATALOG__MURRAY__* env vars
    warehouse="s3://lakehouse/warehouse",
    namespace="features",
    table="barmah_daily",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    credential_vending=False,
)

barmah_daily = FeatureView(
    name="barmah_daily",
    entities=[site],
    # Online: a row older than this is treated as missing (stale data → "no forecast").
    # Offline: how far back the point-in-time join may look for a row.
    ttl=timedelta(days=2),
    schema=[Field(name=name, dtype=Float64) for name in FEATURE_COLUMNS],
    source=barmah_daily_source,
    online=True,
    description="Barmah level history + Yarrawonga upstream lags; row D known at D+1 10:00 Melbourne",
    tags={"team": "murray", "step": "7"},
)
