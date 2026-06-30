"""CLI smoke tests."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

# We use subprocess so the CLI is exercised like a real user would run it.
# Tests below require the package to be importable via `-m`; we set
# PYTHONPATH explicitly so it works on dev checkouts before pip install.

PKG_ROOT = str(Path(__file__).resolve().parent.parent)


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "spectraltm_db", *args],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": PKG_ROOT},
    )


@pytest.fixture
def env_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_create_and_stats(env_path):
    p = env_path
    dim = 32
    r = _run_cli("create", "convo", "--dimension", str(dim),
                 "--compression", "spectral_k64", "--path", str(p))
    assert r.returncode == 0, r.stderr or r.stdout
    info = json.loads(r.stdout)
    assert info["name"] == "convo"
    assert info["dimension"] == dim

    r = _run_cli("stats", "convo", "--path", str(p))
    assert r.returncode == 0, r.stderr or r.stdout
    s = json.loads(r.stdout)
    assert s["dimension"] == dim
    assert s["total_vector_count"] == 0


def test_upsert_query_delete(env_path):
    p = env_path
    dim = 32
    _run_cli("create", "convo", "--dimension", str(dim),
             "--compression", "spectral_k64", "--path", str(p))

    # Build a small JSONL of vectors.
    rng = random.Random(0)
    jsonl = p / "vecs.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(20):
            v = [rng.gauss(0, 1) for _ in range(dim)]
            f.write(json.dumps({"id": f"v{i}", "values": v,
                                "metadata": {"i": i}}) + "\n")
    r = _run_cli("upsert", "convo", "--jsonl", str(jsonl),
                 "--path", str(p))
    assert r.returncode == 0, r.stderr or r.stdout
    assert json.loads(r.stdout)["upserted_count"] == 20

    # Query.
    qv = [rng.gauss(0, 1) for _ in range(dim)]
    r = _run_cli("query", "convo", "--vector", json.dumps(qv),
                 "--top-k", "5", "--path", str(p))
    assert r.returncode == 0, r.stderr or r.stdout
    body = json.loads(r.stdout)
    assert len(body["matches"]) == 5

    # Delete by id.
    r = _run_cli("delete", "convo", "--ids", "v0", "v1", "--path", str(p))
    assert r.returncode == 0, r.stderr or r.stdout
    assert json.loads(r.stdout)["deleted_count"] == 2


def test_compact(env_path):
    p = env_path
    dim = 32
    _run_cli("create", "convo", "--dimension", str(dim),
             "--compression", "spectral_k64", "--path", str(p))
    rng = random.Random(0)
    jsonl = p / "vecs.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"id": f"v{i}",
                                "values": [rng.gauss(0, 1) for _ in range(dim)]})
                    + "\n")
    _run_cli("upsert", "convo", "--jsonl", str(jsonl), "--path", str(p))
    _run_cli("delete", "convo", "--ids", "v0", "v1", "--path", str(p))
    r = _run_cli("compact", "convo", "--path", str(p))
    assert r.returncode == 0, r.stderr or r.stdout
    cleared = json.loads(r.stdout)["cleared"]
    assert cleared >= 2


def test_list_subcommand(env_path):
    p = env_path
    for n in ("alpha", "beta"):
        _run_cli("create", n, "--dimension", "32",
                 "--compression", "spectral_k64", "--path", str(p))
    r = _run_cli("list", "--path", str(p))
    assert r.returncode == 0, r.stderr or r.stdout
    out = json.loads(r.stdout)
    assert sorted(i["name"] for i in out["indexes"]) == ["alpha", "beta"]
