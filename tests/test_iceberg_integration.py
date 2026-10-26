"""Integration test: needs the Step 1 stack running. Run with: uv run pytest -m integration"""

import uuid

import pyarrow as pa
import pytest
from dotenv import find_dotenv, load_dotenv

pytestmark = pytest.mark.integration

load_dotenv(find_dotenv(usecwd=True))

from murray.io.catalog import get_catalog  # noqa: E402
from murray.io.s3 import list_keys  # noqa: E402

NAMESPACE = "pytest"


@pytest.fixture
def catalog():
    try:
        cat = get_catalog()
        cat.create_namespace_if_not_exists(NAMESPACE)
    except Exception as exc:  # stack not running → skip rather than fail
        pytest.skip(f"Iceberg stack not reachable: {exc}")
    return cat


def test_create_append_scan_purge(catalog):
    name = f"{NAMESPACE}.t_{uuid.uuid4().hex[:8]}"
    table = catalog.create_table(name, schema=pa.schema([("site_id", pa.string()), ("level_m", pa.float64())]))
    try:
        table.append(pa.table({"site_id": ["409215", "409025"], "level_m": [2.71, 1.64]}))
        table.append(pa.table({"site_id": ["409215"], "level_m": [2.75]}))

        assert len(table.snapshots()) == 2
        assert table.scan().to_arrow().num_rows == 3
        first = table.snapshots()[0].snapshot_id
        assert table.scan(snapshot_id=first).to_arrow().num_rows == 2  # time travel
    finally:
        location = table.location()
        catalog.purge_table(name)

    assert not catalog.table_exists(name)
    assert list_keys(location.removeprefix("s3://lakehouse/") + "/") == []  # no leaked files


def test_forecast_table_evolves_old_schema_in_place(catalog):
    """A table created before model_name/q10/q90 existed gains them; old rows read back null."""
    import datetime as dt

    from murray.io.prediction_tables import (
        BARMAH_FORECAST_SCHEMA, append_forecast, barmah_forecast_table, forecast_rows)

    name = f"{NAMESPACE}.f_{uuid.uuid4().hex[:8]}"
    old_schema = pa.schema([f for f in BARMAH_FORECAST_SCHEMA
                            if f.name not in ("model_name", "q10_m", "q90_m")])
    old = catalog.create_table(name, schema=old_schema)
    try:
        old.append(pa.table({"issued_at": pa.array([dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)],
                                                   pa.timestamp("us", tz="UTC")),
                             "model_version": ["1"], "obs_date": [dt.date(2026, 10, 24)],
                             "target_date": [dt.date(2026, 10, 25)], "horizon": pa.array([1], pa.int32()),
                             "level_m": [2.7], "wind_kmh": [10.0], "rating": ["good"]}, schema=old_schema))
        table = barmah_forecast_table(catalog, name)
        assert table.schema().column_names == BARMAH_FORECAST_SCHEMA.names
        day = {"date": "2026-10-26", "horizon": 1, "level_m": 2.73, "q10_m": 2.72, "q90_m": 2.74,
               "wind_kmh": None, "rating": "good"}
        append_forecast(table, forecast_rows(dt.datetime.now(dt.timezone.utc), "barmah-level-timestral",
                                             "1", dt.date(2026, 10, 25), [day]))
        df = table.scan().to_pandas().sort_values("issued_at")
        assert df["model_name"].tolist() == [None, "barmah-level-timestral"]
        assert df["q90_m"].isna().tolist() == [True, False]
    finally:
        catalog.purge_table(name)
