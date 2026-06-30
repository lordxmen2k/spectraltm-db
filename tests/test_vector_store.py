"""VectorStore round-trip — compressed + float32 paths."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from spectraltm_db.storage.vector_store import VectorStore
from spectraltm_db.config import IndexConfig, CompressionSpec


DIM = 32


@pytest.fixture
def cfg():
    return IndexConfig(name="x", path=".", dimension=DIM, compression="spectral_k64")


def _sample(dim: int = DIM, n: int = 200, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n * dim)]


def test_compressed_round_trip(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns",
        cfg,
        calibration_sample=_sample(DIM),
    )
    rng = random.Random(1)
    for _ in range(20):
        v = [rng.gauss(0, 1) for _ in range(DIM)]
        vs.add(v)
    vs.flush()

    # Persist + reload — open a brand new VectorStore at the same dir.
    vs2 = VectorStore(Path(tmp_path) / "ns", cfg)
    vs2.ensure_loaded(None)
    assert len(vs2) == 20


def test_compressed_search_returns_k_results(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns", cfg, calibration_sample=_sample(DIM),
    )
    rng = random.Random(1)
    vectors = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(100)]
    for v in vectors:
        vs.add(v)
    vs.flush()

    q = vectors[0]    # identical vec -> score should be highest
    hits = vs.search(q, top_k=5)
    assert len(hits) >= 1
    # Top hit should be the index-0 vector itself.
    assert hits[0].chunk_id == 0


def test_tombstone_filtered_from_search(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns", cfg, calibration_sample=_sample(DIM),
    )
    rng = random.Random(1)
    vectors = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(50)]
    for v in vectors:
        vs.add(v)
    vs.flush()

    # Delete chunk 0 then re-query.
    vs.mark_deleted([0])
    vs.flush()

    q = vectors[0]
    hits = vs.search(q, top_k=5)
    assert all(h.chunk_id != 0 for h in hits)


def test_allowed_chunk_ids_post_filter(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns", cfg, calibration_sample=_sample(DIM),
    )
    rng = random.Random(1)
    vectors = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(50)]
    for v in vectors:
        vs.add(v)
    vs.flush()

    q = vectors[0]
    hits = vs.search(q, top_k=5, allowed_chunk_ids={3, 4, 5})
    assert {h.chunk_id for h in hits} <= {3, 4, 5}


def test_float32_path_search(tmp_path):
    cfg = IndexConfig(name="x", path=".", dimension=DIM, compression="float32")
    vs = VectorStore(Path(tmp_path) / "ns", cfg)
    vs.ensure_loaded(None)
    rng = random.Random(1)
    vectors = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(50)]
    for v in vectors:
        vs.add(v)
    q = vectors[0]
    hits = vs.search(q, top_k=5)
    assert len(hits) >= 1
    assert hits[0].chunk_id == 0


def test_n1_cosine_score_is_real(tmp_path, cfg):
    """When n == 1, score must reflect a real similarity (not a 0.0
    sentinel) and the only chunk must rank first.

    Two paths exist: when the spectraltm Rust binding exposes
    ``SpectralIndex.score_one`` (v0.1.3+, optional) we use it and
    return an SSE-normalized cosine — bounded by ~1 but typically much
    smaller due to quantization. When the binding is absent, we fall
    back to a raw-vector numpy dot product (a true cosine in [-1, 1]).
    Either way the score must be finite and two distinct queries must
    rank differently enough to be distinguishable.
    """
    import math

    def near_zero_zero_score(score: float) -> bool:
        # Soft check: SSE scores at K=16 are typically 0.01-0.30 for
        # random corpora, but the absolute value is not what we care
        # about here — we care that *different* queries give *different*
        # scores, not that either is 1.0.
        return math.isfinite(score) and abs(score) < 2.0

    vs = VectorStore(
        Path(tmp_path) / "ns", cfg, calibration_sample=_sample(),
    )
    vs.add([0.1] * DIM)
    hits = vs.search([0.1] * DIM, top_k=5)
    assert len(hits) == 1
    assert hits[0].chunk_id == 0
    assert near_zero_zero_score(hits[0].score)

    ortho = [1.0, 0.0] + [0.0] * (DIM - 2)
    hits2 = vs.search(ortho, top_k=5)
    assert len(hits2) == 1
    assert hits2[0].chunk_id == 0
    assert near_zero_zero_score(hits2[0].score)
    # Ranking matters: distinct queries should produce different scores.
    assert hits[0].score != hits2[0].score


def test_total_bytes_consistency(tmp_path, cfg):
    vs = VectorStore(
        Path(tmp_path) / "ns", cfg, calibration_sample=_sample(DIM),
    )
    for _ in range(10):
        vs.add([random.random() for _ in range(DIM)])
    n = vs.total_bytes()
    assert n > 0
    assert n == vs.bytes_per_chunk() * 10
