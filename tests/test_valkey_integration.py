"""Integration test for Valkey (Feast online store). Run with: uv run pytest -m integration"""

import os
import uuid

import pytest
import redis
from dotenv import find_dotenv, load_dotenv

pytestmark = pytest.mark.integration

load_dotenv(find_dotenv(usecwd=True))


def connect(password: str | None) -> redis.Redis:
    return redis.Redis(
        host=os.environ.get("VALKEY_HOST", "localhost"),
        port=int(os.environ.get("VALKEY_PORT", "6379")),
        password=password,
        socket_timeout=3,
    )


@pytest.fixture
def valkey():
    client = connect(os.environ["VALKEY_PASSWORD"])
    try:
        client.ping()
    except redis.ConnectionError as exc:
        pytest.skip(f"Valkey not reachable: {exc}")
    return client


def test_requires_password(valkey):
    with pytest.raises(redis.AuthenticationError):
        connect(None).ping()


def test_hash_roundtrip_with_ttl(valkey):
    # Feast stores one hash per entity key; this mimics that shape.
    key = f"pytest:{uuid.uuid4().hex[:8]}:site_id=409215"
    valkey.hset(key, mapping={"level_m": "2.71", "flow_mld": "8208"})
    valkey.expire(key, 60)
    try:
        assert valkey.hgetall(key) == {b"level_m": b"2.71", b"flow_mld": b"8208"}
        assert 0 < valkey.ttl(key) <= 60
    finally:
        valkey.delete(key)


def test_never_evicts_silently(valkey):
    # Missing features must be an error, not a silently evicted key.
    assert valkey.config_get("maxmemory-policy")["maxmemory-policy"] == "noeviction"
