"""Iceberg table of every forecast issued (append-only), for monitoring forecast vs actual.

Several models write here (LightGBM champion, Timestral second opinion), told apart by
model_name. That column and the q10/q90 range arrived after go-live, via schema evolution:
new columns are optional, so rows written before them read back as null.
"""

import datetime as dt

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
    # Added by schema evolution (2026-10-26). Null model_name = the LightGBM model.
    pa.field("model_name", pa.string()),
    pa.field("q10_m", pa.float64()),     # 80% range, quantile models only
    pa.field("q90_m", pa.float64()),
])


def barmah_forecast_table(catalog: Catalog, name: str = BARMAH_FORECAST) -> Table:
    catalog.create_namespace_if_not_exists(name.split(".")[0])
    table = catalog.create_table_if_not_exists(name, schema=BARMAH_FORECAST_SCHEMA)
    missing = set(BARMAH_FORECAST_SCHEMA.names) - set(table.schema().column_names)
    if missing:  # evolve in place: a metadata-only commit, no data files rewritten
        with table.update_schema() as update:
            update.union_by_name(BARMAH_FORECAST_SCHEMA)
    return table


def append_forecast(table: Table, rows: pd.DataFrame) -> int:
    rows = rows.reindex(columns=BARMAH_FORECAST_SCHEMA.names)  # optional columns may be absent
    arrow = pa.Table.from_pandas(rows, schema=BARMAH_FORECAST_SCHEMA, preserve_index=False)
    table.append(arrow)
    return arrow.num_rows


def forecast_rows(now: dt.datetime, model_name: str, version: str, obs_date: dt.date,
                  days: list[dict]) -> pd.DataFrame:
    """Table rows from a forecast's dashboard days (one per horizon)."""
    return pd.DataFrame([{"issued_at": pd.Timestamp(now), "model_name": model_name,
                          "model_version": str(version), "obs_date": obs_date,
                          "target_date": dt.date.fromisoformat(d["date"]), "horizon": d["horizon"],
                          "level_m": d["level_m"], "q10_m": d.get("q10_m"), "q90_m": d.get("q90_m"),
                          "wind_kmh": d["wind_kmh"], "rating": d["rating"]} for d in days])
