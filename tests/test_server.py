"""REST server integration — uses FastAPI TestClient (no live port)."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

import pytest


fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def workdir(tmp_path):
    """Return a fresh data root for the server."""
    wd = tmp_path / "server_data"
    wd.mkdir()
    yield str(wd)
    shutil.rmtree(wd, ignore_errors=True)


@pytest.fixture
def client(workdir):
    from spectraltm_db.server import build_app
    from fastapi.testclient import TestClient
    app = build_app(workdir)
    return TestClient(app)


def _make_workspace(client, workdir, dim=32, K=64):
    """Create an index via the control plane and return its name."""
    name = "convo"
    r = client.post("/indexes", json={
        "name": name, "dimension": dim, "metric": "cosine",
        "spec": {"compression": "spectral_k64"},
    })
    assert r.status_code == 200, r.text
    return name


def _rand_vec(dim, seed):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_and_describe(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    r = client.get("/indexes")
    assert r.status_code == 200
    indexes = r.json()["indexes"]
    assert any(i["name"] == name for i in indexes)

    r = client.get(f"/indexes/{name}")
    assert r.status_code == 200
    assert r.json()["name"] == name


def test_create_duplicate_returns_409(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    r = client.post("/indexes", json={"name": name, "dimension": 32})
    assert r.status_code == 409


def test_upsert_then_query(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    vectors = [
        {"id": f"v{i}",
         "values": _rand_vec(32, seed=i),
         "metadata": {"topic": "t1" if i % 2 == 0 else "t2"}}
        for i in range(10)
    ]
    r = client.post(
        "/vectors/upsert",
        headers={"X-Index-Name": name},
        json={"vectors": vectors, "namespace": "default"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["upserted_count"] == 10

    q = _rand_vec(32, seed=999)
    r = client.post(
        "/query",
        headers={"X-Index-Name": name},
        json={"vector": q, "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["matches"]) == 5


def test_query_with_filter(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    vectors = [
        {"id": f"v{i}",
         "values": _rand_vec(32, seed=i),
         "metadata": {"topic": "t1" if i % 2 == 0 else "t2"}}
        for i in range(10)
    ]
    client.post(
        "/vectors/upsert",
        headers={"X-Index-Name": name},
        json={"vectors": vectors},
    )
    r = client.post(
        "/query",
        headers={"X-Index-Name": name},
        json={
            "vector": _rand_vec(32, seed=999),
            "top_k": 5,
            "filter": {"topic": {"$eq": "t1"}},
        },
    )
    assert r.status_code == 200
    matches = r.json()["matches"]
    for m in matches:
        assert m["metadata"]["topic"] == "t1"


def test_fetch_and_delete(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    client.post(
        "/vectors/upsert",
        headers={"X-Index-Name": name},
        json={"vectors": [
            {"id": "a", "values": _rand_vec(32, seed=1), "metadata": {"x": 1}},
            {"id": "b", "values": _rand_vec(32, seed=2), "metadata": {"x": 2}},
        ]},
    )
    r = client.post(
        "/vectors/fetch",
        headers={"X-Index-Name": name},
        json={"ids": ["a", "b"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "a" in body["vectors"]
    assert "b" in body["vectors"]

    r = client.post(
        "/vectors/delete",
        headers={"X-Index-Name": name},
        json={"ids": ["a"]},
    )
    assert r.status_code == 200
    assert r.json()["deleted_count"] == 1


def test_describe_index_stats(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    client.post(
        "/vectors/upsert",
        headers={"X-Index-Name": name},
        json={"vectors": [
            {"id": f"v{i}", "values": _rand_vec(32, seed=i),
             "metadata": {"topic": f"t{i % 2}"}}
            for i in range(20)
        ]},
    )
    r = client.post(
        "/describe_index_stats",
        headers={"X-Index-Name": name},
        json={},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total_vector_count"] == 20
    assert body["dimension"] == 32


def test_list_vectors(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    client.post(
        "/vectors/upsert",
        headers={"X-Index-Name": name},
        json={"vectors": [
            {"id": "alpha-1", "values": _rand_vec(32, seed=1)},
            {"id": "alpha-2", "values": _rand_vec(32, seed=2)},
            {"id": "beta-1", "values": _rand_vec(32, seed=3)},
        ]},
    )
    r = client.post(
        "/vectors/list",
        headers={"X-Index-Name": name},
        json={"prefix": "alpha", "limit": 10},
    )
    assert r.status_code == 200
    ids = r.json()["ids"]
    assert set(ids) == {"alpha-1", "alpha-2"}


def test_delete_index(client, workdir):
    name = _make_workspace(client, workdir, dim=32)
    r = client.delete(f"/indexes/{name}")
    assert r.status_code == 200

    r = client.get(f"/indexes/{name}")
    assert r.status_code == 404
