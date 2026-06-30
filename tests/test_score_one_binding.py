"""Optional `spectraltm.SpectralIndex.score_one` binding tests."""

from __future__ import annotations

import random

import pytest


@pytest.fixture
def _has_score_one():
    import spectraltm
    return hasattr(spectraltm.SpectralIndex, "score_one")


def test_score_one_returns_normalized_cosine(_has_score_one):
    if not _has_score_one:
        pytest.skip("spectraltm does not expose SpectralIndex.score_one; "
                    "install a v0.1.3+ build with the binding to enable this test")
    import spectraltm

    rng = random.Random(0)
    dim = 32
    enc = spectraltm.SpectralEncoder(dim, 16, 8, 8, 8)
    enc.calibrate([rng.gauss(0, 1) for _ in range(100 * dim)])
    idx = spectraltm.SpectralIndex(enc)
    db = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(20)]
    idx.add_embeddings(db)

    q = [rng.gauss(0, 1) for _ in range(dim)]
    score = idx.score_one(q, 0)
    assert isinstance(score, float)
    # SSE-normalized cosine — bounded by ~1, may be slightly negative.
    assert -1.5 <= score <= 1.5


def test_score_one_safe_at_n1(_has_score_one):
    """The n=1 case where SpectralIndex.search panics; score_one is safe."""
    if not _has_score_one:
        pytest.skip("spectraltm does not expose SpectralIndex.score_one")
    import spectraltm

    rng = random.Random(0)
    dim = 32
    enc = spectraltm.SpectralEncoder(dim, 16, 8, 8, 8)
    enc.calibrate([rng.gauss(0, 1) for _ in range(100 * dim)])
    idx = spectraltm.SpectralIndex(enc)
    idx.add_embeddings([[rng.gauss(0, 1) for _ in range(dim)]])

    q = [rng.gauss(0, 1) for _ in range(dim)]
    score = idx.score_one(q, 0)
    assert isinstance(score, float)


def test_score_one_out_of_range_returns_zero(_has_score_one):
    if not _has_score_one:
        pytest.skip("spectraltm does not expose SpectralIndex.score_one")
    import spectraltm

    rng = random.Random(0)
    dim = 32
    enc = spectraltm.SpectralEncoder(dim, 16, 8, 8, 8)
    enc.calibrate([rng.gauss(0, 1) for _ in range(100 * dim)])
    idx = spectraltm.SpectralIndex(enc)
    idx.add_embeddings([[rng.gauss(0, 1) for _ in range(dim)]])

    score = idx.score_one([0.1] * dim, 999)
    assert score == 0.0
