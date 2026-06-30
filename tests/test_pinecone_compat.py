"""Pinecone-shape parity helpers — API surface mirrors."""

from __future__ import annotations

import random

import spectraltm_db as stm
from conftest import random_calibration


def test_pinecone_shape_response(tmp_path):
    """The Index.query() response should resemble pinecone's contract."""
    wd = str(tmp_path / "parity")
    idx = stm.Index.create(
        name="parity", path=wd, dimension=32,
        compression="spectral_k64",
        calibration_sample=random_calibration(32, n_vecs=200),
    )
    rng = random.Random(7)
    idx.upsert(vectors=[
        {"id": f"v{i}", "values": [rng.gauss(0, 1) for _ in range(32)],
         "metadata": {"topic": "x"}}
        for i in range(10)
    ])
    r = idx.query(vector=[0.1] * 32, top_k=3)
    assert set(r.keys()) >= {"matches", "namespace", "usage"}
    for m in r["matches"]:
        assert set(m.keys()) >= {"id", "score"}
    fetched = idx.fetch(ids=["v0"])
    assert "vectors" in fetched
    assert "namespace" in fetched
    s = idx.delete(ids=["v0"])
    assert "deleted_count" in s
    u = idx.update(id="v1", set_metadata={"flagged": True})
    assert u["updated"] is True


def test_describe_index_stats_shape(tmp_path):
    wd = str(tmp_path / "parity")
    idx = stm.Index.create(
        name="parity", path=wd, dimension=32,
        compression="spectral_k64",
        calibration_sample=random_calibration(32, n_vecs=200),
    )
    s = idx.describe_index_stats()
    assert "dimension" in s
    assert "total_vector_count" in s
    assert "namespaces" in s
