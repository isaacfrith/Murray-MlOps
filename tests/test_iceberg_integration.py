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
