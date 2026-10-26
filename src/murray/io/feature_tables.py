"""Iceberg feature tables (the Feast offline store reads these)."""

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

from murray.transforms.features import FEATURE_COLUMNS

NAMESPACE = "features"
BARMAH_DAILY = f"{NAMESPACE}.barmah_daily"

BARMAH_DAILY_SCHEMA = pa.schema(
    [
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("obs_date", pa.date32(), nullable=False),
        *[pa.field(name, pa.float64()) for name in FEATURE_COLUMNS],
        pa.field("event_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("created_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def barmah_daily_table(catalog: Catalog) -> Table:
    catalog.create_namespace_if_not_exists(NAMESPACE)
    return catalog.create_table_if_not_exists(BARMAH_DAILY, schema=BARMAH_DAILY_SCHEMA)


def write_barmah_daily(table: Table, features: pd.DataFrame) -> int:
    """Replace the table contents with a full recompute (one atomic snapshot).

    Features are derived from raw, so a full rebuild is cheap (~20k rows) and keeps the
    table consistent with the current transform code. Old versions stay reachable through
    Iceberg snapshots.
    """
    arrow = pa.Table.from_pandas(
        features[BARMAH_DAILY_SCHEMA.names], schema=BARMAH_DAILY_SCHEMA, preserve_index=False
    )
    table.overwrite(arrow)
    return arrow.num_rows
