"""Integration test: Feast online + point-in-time lookups against the real stack.
Needs the feature pipeline to have run once (features_barmah DAG). Run: uv run pytest -m integration"""

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv

pytestmark = pytest.mark.integration

load_dotenv(find_dotenv(usecwd=True))

from murray.io.feature_store import get_store  # noqa: E402

FEATURES = ["barmah_daily:barmah_level_m", "barmah_daily:yarra_level_lag4"]


@pytest.fixture(scope="module")
def store():
    try:
        s = get_store()
        s.get_feature_view("barmah_daily")
    except Exception as exc:
        pytest.skip(f"Feast registry not available (run features_barmah first): {exc}")
    return s


def test_online_lookup_returns_current_barmah_level(store):
    out = store.get_online_features(features=FEATURES, entity_rows=[{"site_id": "409215"}]).to_dict()
    assert out["barmah_level_m"][0] is not None
    assert 0.5 < out["barmah_level_m"][0] < 10


def test_point_in_time_join_respects_10am_availability(store):
    """Before 10:00 Melbourne the previous day's row is not yet known."""
    ts = pd.to_datetime(["2021-03-03 09:00", "2021-03-03 11:00"]).tz_localize(
        "Australia/Melbourne").tz_convert("UTC")
    entity_df = pd.DataFrame({"site_id": ["409215", "409215"], "event_timestamp": ts})
    hist = store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
    hist = hist.sort_values("event_timestamp")
    before, after = hist["barmah_level_m"].tolist()
    assert before != after  # 09:00 sees 1 March's row, 11:00 sees 2 March's
