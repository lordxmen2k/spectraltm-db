"""Index — full end-to-end smoke + persistence."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

import spectraltm_db as stm
from spectraltm_db.errors import DimensionMismatch, InvalidArgument, NotFound

from conftest import random_calibration, random_query


DIM = 32


def _make_index(workdir: str, *, compression: str = "spectral_k64",
                dimension: int = DIM, seed: int = 0) -> stm.Index:
    return stm.Index.create(
        name="idx",
        path=workdir,
        dimension=dimension,
        compression=compression,
        calibration_sample=random_calibration(dimension, n_vecs=200, seed=seed),
    )


def test_create_and_reopen(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.flush()
    del idx
    idx2 = stm.Index.open(wd)
    assert idx2.config.name == "idx"
    assert idx2.config.dimension == DIM


def test_create_rejects_existing(tmp_path):
    wd = str(tmp_path / "idx")
    _make_index(wd)
    with pytest.raises(InvalidArgument):
        _make_index(wd)


def test_upsert_and_query_roundtrip(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    rng = random.Random(1)
    vectors = [
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(DIM)],
         "metadata": {"topic": f"t{i % 3}", "seq": i}}
        for i in range(50)
    ]
    r = idx.upsert(vectors=vectors)
    assert r["upserted_count"] == 50

    q = random_query(DIM, seed=2)
    r = idx.query(vector=q, top_k=5)
    assert len(r["matches"]) == 5
    assert r["namespace"] == "default"
    assert all("score" in m for m in r["matches"])
    assert all("id" in m for m in r["matches"])


def test_filter_eq(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    rng = random.Random(1)
    vectors = [
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(DIM)],
         "metadata": {"topic": f"t{i % 2}"}}
        for i in range(40)
    ]
    idx.upsert(vectors=vectors)

    q = random_query(DIM, seed=2)
    r = idx.query(vector=q, top_k=10, filter={"topic": {"$eq": "t0"}})
    assert all(m["metadata"]["topic"] == "t0" for m in r["matches"])


def test_filter_in_or(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    rng = random.Random(1)
    vectors = [
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(DIM)],
         "metadata": {"topic": f"t{i % 3}"}}
        for i in range(40)
    ]
    idx.upsert(vectors=vectors)

    q = random_query(DIM, seed=2)
    r = idx.query(
        vector=q, top_k=10,
        filter={"$or": [{"topic": {"$eq": "t0"}}, {"topic": {"$eq": "t1"}}]},
    )
    topics = {m["metadata"]["topic"] for m in r["matches"]}
    assert topics <= {"t0", "t1"}


def test_fetch(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[
        {"id": "a", "values": [0.1] * DIM, "metadata": {"x": 1}},
        {"id": "b", "values": [0.2] * DIM, "metadata": {"x": 2}},
    ])
    r = idx.fetch(ids=["a", "b", "missing"])
    assert "a" in r["vectors"]
    assert "b" in r["vectors"]
    assert "missing" not in r["vectors"]


def test_delete_by_id_and_filter(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[
        {"id": "v0", "values": [0.1] * DIM, "metadata": {"topic": "t1"}},
        {"id": "v1", "values": [0.1] * DIM, "metadata": {"topic": "t1"}},
        {"id": "v2", "values": [0.1] * DIM, "metadata": {"topic": "t2"}},
    ])
    r = idx.delete(ids=["v0"])
    assert r["deleted_count"] == 1

    r = idx.delete(filter={"topic": {"$eq": "t1"}})
    assert r["deleted_count"] == 1    # only v1 left in t1 after v0 delete

    s = idx.describe_index_stats()
    assert s["total_vector_count"] == 1


def test_delete_delete_all_namespace(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[
        {"id": "v0", "values": [0.1] * DIM, "metadata": {}},
        {"id": "v1", "values": [0.1] * DIM, "metadata": {}},
    ])
    r = idx.delete(delete_all=True)
    assert r["deleted_count"] == 2
    assert idx.describe_index_stats()["total_vector_count"] == 0


def test_update_metadata_only(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[{"id": "v0", "values": [0.1] * DIM, "metadata": {"x": 1}}])
    r = idx.update(id="v0", set_metadata={"flagged": True, "x": 99})
    assert r["updated"] is True
    fetched = idx.fetch(ids=["v0"])["vectors"]["v0"]["metadata"]
    assert fetched["x"] == 99
    assert fetched["flagged"] is True


def test_update_values_replaces_vector(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[{"id": "v0", "values": [0.1] * DIM}])
    r = idx.update(id="v0", values=[0.9] * DIM)
    assert r["updated"] is True
    s = idx.describe_index_stats()
    assert s["total_vector_count"] == 1


def test_upsert_rejects_wrong_dim(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    with pytest.raises(DimensionMismatch):
        idx.upsert(vectors=[{"id": "x", "values": [0.1] * (DIM + 1)}])


def test_query_rejects_wrong_dim(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    with pytest.raises(DimensionMismatch):
        idx.query(vector=[0.1] * (DIM + 1))


def test_stats(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    s = idx.describe_index_stats()
    assert s["dimension"] == DIM
    assert s["compression"] == "spectral_k64"
    assert s["metric"] == "cosine"
    assert s["total_vector_count"] == 0

    idx.upsert(vectors=[
        {"id": f"v{i}", "values": [0.1] * DIM, "metadata": {}}
        for i in range(10)
    ])
    s = idx.describe_index_stats()
    assert s["total_vector_count"] == 10
    assert s["namespaces"]["default"]["vector_count"] == 10


def test_namespaces_isolation(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[{"id": "a", "values": [0.1] * DIM}])
    idx.upsert(vectors=[{"id": "b", "values": [0.5] * DIM}], namespace="staging")

    s = idx.describe_index_stats()
    assert s["total_vector_count"] == 2
    assert sorted(s["namespaces"].keys()) == ["default", "staging"]
    assert s["namespaces"]["default"]["vector_count"] == 1
    assert s["namespaces"]["staging"]["vector_count"] == 1

    # Querying a namespace returns only its vectors.
    r = idx.query(vector=[0.1] * DIM, top_k=10, namespace="default")
    assert {m["id"] for m in r["matches"]} == {"a"}


def test_delete_namespace(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[{"id": "a", "values": [0.1] * DIM}], namespace="to_remove")
    idx.delete_namespace("to_remove")
    assert idx.list_namespaces() == []


def test_persistence_after_reopen(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    rng = random.Random(1)
    vectors = [
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(DIM)]}
        for i in range(30)
    ]
    idx.upsert(vectors=vectors)
    idx.flush()

    # Reopen and verify state.
    idx2 = stm.Index.open(wd)
    assert idx2.describe_index_stats()["total_vector_count"] == 30
    q = random_query(DIM, seed=2)
    r = idx2.query(vector=q, top_k=3)
    assert len(r["matches"]) == 3


def test_unsupported_filter_operator(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[{"id": "a", "values": [0.1] * DIM}])
    with pytest.raises(Exception):
        idx.query(vector=[0.1] * DIM, top_k=1, filter={"x": {"$between": [0, 5]}})


def test_zero_k_returns_empty(tmp_path):
    wd = str(tmp_path / "idx")
    idx = _make_index(wd)
    idx.upsert(vectors=[{"id": "a", "values": [0.1] * DIM}])
    r = idx.query(vector=[0.1] * DIM, top_k=0)
    assert r["matches"] == []


def test_float32_path(tmp_path):
    """Sanity check the uncompressed path is wired through."""
    wd = str(tmp_path / "idx")
    idx = _make_index(wd, compression="float32")
    idx.upsert(vectors=[
        {"id": f"v{i}", "values": [random.Random(i).gauss(0, 1) for _ in range(DIM)]}
        for i in range(20)
    ])
    r = idx.query(vector=[0.5] * DIM, top_k=5)
    assert len(r["matches"]) == 5
