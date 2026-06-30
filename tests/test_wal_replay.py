"""WAL replay must not double-count vectors that were already persisted."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import spectraltm_db as stm


DIM = 32


def _sample(n: int = 200, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n * DIM)]


def test_wal_truncates_after_upsert(tmp_path):
    wd = tmp_path / "idx"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64",
        calibration_sample=_sample(),
    )
    rng = random.Random(1)
    idx.upsert(vectors=[
        {"id": f"v{i}",
         "values": [rng.gauss(0, 1) for _ in range(DIM)]}
        for i in range(20)
    ])
    # WAL must be truncated after flush so reopen doesn't replay.
    wal_path = wd / "wal" / "pending.jsonl"
    assert wal_path.exists()
    assert wal_path.read_text() == ""


def test_reopen_does_not_double_upsert(tmp_path):
    wd = tmp_path / "idx"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64",
        calibration_sample=_sample(),
    )
    rng = random.Random(2)
    vectors = [
        {"id": f"v{i}",
         "values": [rng.gauss(0, 1) for _ in range(DIM)]}
        for i in range(20)
    ]
    idx.upsert(vectors=vectors)
    idx.flush()
    del idx

    # Reopen in a fresh Index instance; counts must match.
    idx2 = stm.Index.open(str(wd))
    s = idx2.describe_index_stats()
    assert s["total_vector_count"] == 20

    # And metadata must still resolve each id.
    out = idx2.fetch(ids=["v0", "v10", "v19"])
    assert set(out["vectors"].keys()) == {"v0", "v10", "v19"}


def test_reopen_then_continue_upsert(tmp_path):
    wd = tmp_path / "idx"
    idx = stm.Index.create(
        name="x", path=str(wd), dimension=DIM,
        compression="spectral_k64",
        calibration_sample=_sample(),
    )
    rng = random.Random(3)
    idx.upsert(vectors=[
        {"id": f"v{i}",
         "values": [rng.gauss(0, 1) for _ in range(DIM)]}
        for i in range(10)
    ])
    idx.flush()
    del idx

    # Reopen and add 10 more.
    idx2 = stm.Index.open(str(wd))
    idx2.upsert(vectors=[
        {"id": f"v{i + 10}",
         "values": [rng.gauss(0, 1) for _ in range(DIM)]}
        for i in range(10)
    ])
    assert idx2.describe_index_stats()["total_vector_count"] == 20
