"""Iceberg tables for raw (as-ingested) data."""

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.types import (
    DateType,
    DoubleType,
    IntegerType,
    NestedField,
    StringType,
    TimestamptzType,
)

NAMESPACE = "raw"
GAUGE_DAILY = f"{NAMESPACE}.gauge_daily"
WEATHER_DAILY = f"{NAMESPACE}.weather_daily"

GAUGE_DAILY_SCHEMA = Schema(
    NestedField(1, "site_id", StringType(), required=True, doc="Gauge number, e.g. 409215"),
    NestedField(2, "variable", StringType(), required=True, doc="level_m or flow_mld"),
    NestedField(3, "obs_date", DateType(), required=True, doc="Agency local calendar day"),
    NestedField(4, "value", DoubleType(), required=True, doc="Daily mean"),
    NestedField(5, "quality_code", IntegerType(), required=False, doc="Agency quality code"),
    NestedField(6, "source", StringType(), required=True, doc="VIC or NSW (state agency)"),
    NestedField(7, "ingested_at", TimestamptzType(), required=True, doc="When we fetched it (UTC)"),
)


def gauge_daily_table(catalog: Catalog) -> Table:
    """Load raw.gauge_daily, creating namespace and table on first use."""
    catalog.create_namespace_if_not_exists(NAMESPACE)
    return catalog.create_table_if_not_exists(GAUGE_DAILY, schema=GAUGE_DAILY_SCHEMA)


def append_gauge_daily(table: Table, rows: pd.DataFrame) -> int:
    """Append rows (append-only: never overwrite). Returns the number of rows written."""
    if rows.empty:
        return 0
    arrow = pa.Table.from_pandas(rows, schema=table.schema().as_arrow(), preserve_index=False)
    table.append(arrow)
    return arrow.num_rows


WEATHER_DAILY_SCHEMA = Schema(
    NestedField(1, "location", StringType(), required=True, doc="barmah, yarrawonga, ..."),
    NestedField(2, "obs_date", DateType(), required=True, doc="Melbourne local calendar day"),
    NestedField(3, "precipitation_sum", DoubleType(), required=False, doc="mm"),
    NestedField(4, "temperature_2m_max", DoubleType(), required=False, doc="°C"),
    NestedField(5, "et0_fao_evapotranspiration", DoubleType(), required=False, doc="mm"),
    NestedField(6, "wind_speed_10m_max", DoubleType(), required=False, doc="km/h"),
    NestedField(7, "ingested_at", TimestamptzType(), required=True, doc="When we fetched it (UTC)"),
)


def weather_daily_table(catalog: Catalog) -> Table:
    catalog.create_namespace_if_not_exists(NAMESPACE)
    return catalog.create_table_if_not_exists(WEATHER_DAILY, schema=WEATHER_DAILY_SCHEMA)


def append_rows(table: Table, rows: pd.DataFrame) -> int:
    """Append-only write for any raw table. Returns the number of rows written."""
    if rows.empty:
        return 0
    arrow = pa.Table.from_pandas(rows[table.schema().column_names],
                                 schema=table.schema().as_arrow(), preserve_index=False)
    table.append(arrow)
    return arrow.num_rows
