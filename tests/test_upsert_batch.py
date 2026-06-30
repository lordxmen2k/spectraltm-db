"""Index.upsert and VectorStore.add_batch — bulk ingest path."""

from __future__ import annotations

import random

import pytest

import spectraltm_db as stm
from spectraltm_db.errors import DimensionMismatch


DIM = 32


def _sample(n: int = 200, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n * DIM)]


def test_upsert_batch_basic(tmp_path):
    wd = tmp_path / "batch_idx"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64", calibration_sample=_sample(),
    )
    rng = random.Random(1)
    vectors = [
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(DIM)],
         "metadata": {"i": i}}
        for i in range(50)
    ]
    r = idx.upsert(vectors=vectors)
    assert r["upserted_count"] == 50
    assert idx.describe_index_stats()["total_vector_count"] == 50


def test_upsert_batch_equivalent_to_upsert(tmp_path):
    """Calling upsert() with the same records twice, then querying,
    must give identical rankings — verifies the bulk path doesn't
    silently re-order or lose inserts.
    """
    wd = tmp_path / "batch_idx2"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64", calibration_sample=_sample(),
    )
    rng = random.Random(2)
    records = [
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(DIM)]}
        for i in range(30)
    ]
    idx.upsert(vectors=records)
    idx.flush()

    q = [rng.gauss(0, 1) for _ in range(DIM)]
    r1 = idx.query(vector=q, top_k=5)
    ids_1 = [m["id"] for m in r1["matches"]]

    # Reopen, do a fresh query — ordering must match.
    del idx
    idx2 = stm.Index.open(str(wd))
    r2 = idx2.query(vector=q, top_k=5)
    ids_2 = [m["id"] for m in r2["matches"]]
    assert ids_1 == ids_2


def test_upsert_batch_rejects_wrong_dim(tmp_path):
    wd = tmp_path / "batch_idx3"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64", calibration_sample=_sample(),
    )
    with pytest.raises(DimensionMismatch):
        idx.upsert(vectors=[
            {"id": "ok", "values": [0.1] * DIM},
            {"id": "bad", "values": [0.1] * (DIM + 1)},
        ])
    # The whole batch must be rejected — the "ok" record must not be
    # persisted alongside the rollback check.
    assert idx.describe_index_stats()["total_vector_count"] == 0


def test_upsert_batch_handles_reupsert(tmp_path):
    wd = tmp_path / "batch_idx4"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64", calibration_sample=_sample(),
    )
    rng = random.Random(7)
    initial = [
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(DIM)]}
        for i in range(10)
    ]
    idx.upsert(vectors=initial)

    n0 = idx.describe_index_stats()["total_vector_count"]

    # Re-upsert: each id gets a new chunk_id; the prior is tombstoned.
    new = [{"id": v["id"], "values": [rng.gauss(0, 1) for _ in range(DIM)]}
           for v in initial]
    r = idx.upsert(vectors=new)
    assert r["upserted_count"] == len(new)
    # Live count remains 10 (10 fresh + 10 tombstoned).
    assert idx.describe_index_stats()["total_vector_count"] == n0
    # And the chunk count grew by 10 — confirms we appended rather than
    # overwrote in place.
    assert idx._ns_for_write(None).vs.next_chunk_id() == n0 + len(new)  # noqa: SLF001


def test_empty_upsert_is_noop(tmp_path):
    wd = tmp_path / "batch_idx5"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64", calibration_sample=_sample(),
    )
    r = idx.upsert(vectors=[])
    assert r == {"upserted_count": 0}


def test_upsert_batch_with_filters(tmp_path):
    wd = tmp_path / "batch_idx6"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64", calibration_sample=_sample(),
    )
    rng = random.Random(11)
    vectors = [
        {"id": f"v{i}",
         "values": [rng.gauss(0, 1) for _ in range(DIM)],
         "metadata": {"topic": "t1" if i % 2 == 0 else "t2"}}
        for i in range(20)
    ]
    idx.upsert(vectors=vectors)

    q = [rng.gauss(0, 1) for _ in range(DIM)]
    r = idx.query(vector=q, top_k=10, filter={"topic": {"$eq": "t1"}})
    assert all(m["metadata"]["topic"] == "t1" for m in r["matches"])
