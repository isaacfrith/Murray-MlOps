"""Publish the dashboard data files to GitHub without git: one commit via the Git Data API.

Data goes to a dedicated branch (default `site-data`), not `main`, so code history stays clean
and local runs never conflict with bot commits. The Pages workflow deploys main's site/ (HTML)
plus this branch's JSON.

Needs GITHUB_PUBLISH_TOKEN: a fine-grained token for this repository only, with
"Contents: Read and write". Without it, publishing is skipped (not an error).
"""

import base64
import logging
import os
from pathlib import Path

import requests

log = logging.getLogger(__name__)
API = "https://api.github.com"
DATA_FILES = ("forecast.json", "models.json", "monitoring.json")


class _GitHub:
    def __init__(self, token: str, repo: str):
        self.base = f"{API}/repos/{repo}"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def call(self, method: str, path: str, **kw):
        r = self.s.request(method, self.base + path, timeout=30, **kw)
        if r.status_code == 404 and method == "GET":
            return None
        if r.status_code >= 400:  # never include the token in errors
            raise RuntimeError(f"GitHub {method} {path}: {r.status_code} {r.text[:200]}")
        return r.json()


def publish(site_dir: str | Path, message: str = "Publish forecast data") -> str:
    token = os.environ.get("GITHUB_PUBLISH_TOKEN", "").strip()
    if not token:
        log.warning("GITHUB_PUBLISH_TOKEN not set: skipping publish")
        return "skipped: no token"
    repo = os.environ.get("GITHUB_REPO", "isaacfrith/Murray-MlOps")
    branch = os.environ.get("GITHUB_DATA_BRANCH", "site-data")
    gh = _GitHub(token, repo)

    files = [Path(site_dir) / f for f in DATA_FILES if (Path(site_dir) / f).exists()]
    if not files:
        return "skipped: no data files"
    tree = [{"path": f.name, "mode": "100644", "type": "blob",
             "sha": gh.call("POST", "/git/blobs", json={
                 "content": base64.b64encode(f.read_bytes()).decode(), "encoding": "base64"})["sha"]}
            for f in files]

    ref = gh.call("GET", f"/git/ref/heads/{branch}")
    parent = ref["object"]["sha"] if ref else None
    base_tree = gh.call("GET", f"/git/commits/{parent}")["tree"]["sha"] if parent else None
    new_tree = gh.call("POST", "/git/trees",
                       json={"tree": tree, **({"base_tree": base_tree} if base_tree else {})})["sha"]
    if new_tree == base_tree:
        return "unchanged"

    commit = gh.call("POST", "/git/commits", json={
        "message": message, "tree": new_tree, "parents": [parent] if parent else []})["sha"]
    if ref:
        gh.call("PATCH", f"/git/refs/heads/{branch}", json={"sha": commit})
    else:
        gh.call("POST", "/git/refs", json={"ref": f"refs/heads/{branch}", "sha": commit})

    # Pages only accepts deployments from main (environment protection), so ask GitHub to run
    # the deploy workflow on main; it pulls this branch's JSON in. Needs "Actions: write".
    gh.call("POST", "/actions/workflows/pages.yml/dispatches", json={"ref": "main"})
    log.info("published %d files to %s@%s (%s), deploy requested", len(files), repo, branch,
             commit[:7])
    return f"published {commit[:7]}"
