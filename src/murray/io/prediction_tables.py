"""Iceberg table of every forecast issued (append-only), for monitoring forecast vs actual."""

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.table import Table

NAMESPACE = "predictions"
BARMAH_FORECAST = f"{NAMESPACE}.barmah_forecast"

BARMAH_FORECAST_SCHEMA = pa.schema([
    pa.field("issued_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("model_version", pa.string(), nullable=False),
    pa.field("obs_date", pa.date32(), nullable=False),       # last observed day D
    pa.field("target_date", pa.date32(), nullable=False),    # D + horizon
    pa.field("horizon", pa.int32(), nullable=False),
    pa.field("level_m", pa.float64()),
    pa.field("wind_kmh", pa.float64()),
    pa.field("rating", pa.string(), nullable=False),
])


def barmah_forecast_table(catalog: Catalog) -> Table:
    catalog.create_namespace_if_not_exists(NAMESPACE)
    return catalog.create_table_if_not_exists(BARMAH_FORECAST, schema=BARMAH_FORECAST_SCHEMA)


def append_forecast(table: Table, rows: pd.DataFrame) -> int:
    arrow = pa.Table.from_pandas(rows[BARMAH_FORECAST_SCHEMA.names],
                                 schema=BARMAH_FORECAST_SCHEMA, preserve_index=False)
    table.append(arrow)
    return arrow.num_rows
