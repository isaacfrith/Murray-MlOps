import json

from murray.io import github_publish


def test_skips_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_PUBLISH_TOKEN", raising=False)
    (tmp_path / "forecast.json").write_text("{}")
    assert github_publish.publish(tmp_path) == "skipped: no token"


def test_one_commit_on_existing_branch(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_PUBLISH_TOKEN", "t")
    for name in github_publish.DATA_FILES:
        (tmp_path / name).write_text(json.dumps({"f": name}))
    calls = []

    def fake(self, method, path, **kw):
        calls.append((method, path))
        return {
            ("POST", "/git/blobs"): {"sha": "blob"},
            ("GET", "/git/ref/heads/site-data"): {"object": {"sha": "parent"}},
            ("GET", "/git/commits/parent"): {"tree": {"sha": "oldtree"}},
            ("POST", "/git/trees"): {"sha": "newtree"},
            ("POST", "/git/commits"): {"sha": "abc1234def"},
            ("PATCH", "/git/refs/heads/site-data"): {},
            ("POST", "/actions/workflows/pages.yml/dispatches"): {},
        }[(method, path)]

    monkeypatch.setattr(github_publish._GitHub, "call", fake)
    assert github_publish.publish(tmp_path) == "published abc1234"
    assert calls.count(("POST", "/git/blobs")) == 3
    assert ("PATCH", "/git/refs/heads/site-data") in calls  # branch moved forward, no force
    assert calls[-1] == ("POST", "/actions/workflows/pages.yml/dispatches")  # deploy from main


def test_unchanged_data_makes_no_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_PUBLISH_TOKEN", "t")
    (tmp_path / "forecast.json").write_text("{}")
    calls = []

    def fake(self, method, path, **kw):
        calls.append((method, path))
        return {
            ("POST", "/git/blobs"): {"sha": "blob"},
            ("GET", "/git/ref/heads/site-data"): {"object": {"sha": "parent"}},
            ("GET", "/git/commits/parent"): {"tree": {"sha": "same"}},
            ("POST", "/git/trees"): {"sha": "same"},
        }[(method, path)]

    monkeypatch.setattr(github_publish._GitHub, "call", fake)
    assert github_publish.publish(tmp_path) == "unchanged"
    assert ("POST", "/git/commits") not in calls
