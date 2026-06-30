"""VectorStore.compact() — re-encode and reclaim physical space."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from spectraltm_db.storage.vector_store import VectorStore
from spectraltm_db.config import IndexConfig


DIM = 32


@pytest.fixture
def cfg():
    return IndexConfig(name="x", path=".", dimension=DIM, compression="spectral_k64")


def _sample(n: int = 200, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n * DIM)]


def test_compact_reclaims_tombstones(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns",
        cfg,
        calibration_sample=_sample(),
    )
    rng = random.Random(1)
    for _ in range(20):
        vs.add([rng.gauss(0, 1) for _ in range(DIM)])
    vs.flush()

    # Delete 5 of them.
    vs.mark_deleted([0, 2, 4, 6, 8])
    vs.flush()
    assert len(vs) == 15

    sltm_size_before = os.path.getsize(vs._sltm_path)  # noqa: SLF001
    raw_size_before = os.path.getsize(vs._raw_path)    # noqa: SLF001

    cleared = vs.compact()
    assert cleared == 5
    vs.flush()

    assert vs.tombstones == set()
    assert len(vs) == 15
    assert vs.next_chunk_id() == 15   # chunk_ids re-sequenced to live ones

    # On-disk size should have shrunk (or at least not grown).
    sltm_size_after = os.path.getsize(vs._sltm_path)  # noqa: SLF001
    raw_size_after = os.path.getsize(vs._raw_path)    # noqa: SLF001
    # Raw array is now smaller (15 rows vs 20).
    assert raw_size_after < raw_size_before


def test_compact_search_after_reclaim(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns",
        cfg,
        calibration_sample=_sample(),
    )
    rng = random.Random(1)
    vectors = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(40)]
    for v in vectors:
        vs.add(v)
    vs.flush()

    # Delete half.
    for cid in range(0, 40, 2):
        vs.mark_deleted([cid])
    vs.compact()
    vs.flush()

    # Search should still work and return only live chunk_ids.
    q = vectors[10]
    hits = vs.search(q, top_k=5)
    assert all(h.chunk_id not in vs.tombstones for h in hits)
    assert all(h.chunk_id < 20 for h in hits)   # re-sequenced


def test_compact_then_persistence(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns",
        cfg,
        calibration_sample=_sample(),
    )
    rng = random.Random(1)
    for _ in range(10):
        vs.add([rng.gauss(0, 1) for _ in range(DIM)])
    vs.mark_deleted([0, 1, 2])
    vs.compact()
    vs.flush()

    # Reopen from disk.
    vs2 = VectorStore(Path(tmp_path) / "ns", cfg)
    vs2.ensure_loaded(None)
    assert vs2.tombstones == set()
    assert vs2.next_chunk_id() == 7
    assert len(vs2) == 7


def test_compact_with_no_tombstones_is_noop(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns", cfg, calibration_sample=_sample(),
    )
    for _ in range(5):
        vs.add([0.1] * DIM)
    vs.flush()
    cleared = vs.compact()
    assert cleared == 0
    assert len(vs) == 5
