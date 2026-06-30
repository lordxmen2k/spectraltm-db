"""Sparse-dense hybrid wire surface — wire shape only in v0.1.x."""

from __future__ import annotations

import logging

import pytest

import spectraltm_db as stm
from spectraltm_db.sparse import SparseValues


def _cal():
    import random
    rng = random.Random(0)
    return [rng.gauss(0, 1) for _ in range(200 * 32)]


def test_sparse_values_dataclass():
    sv = SparseValues(indices=[1, 2, 3], values=[0.4, 0.5, 0.6])
    assert sv.indices == [1, 2, 3]
    assert sv.values == [0.4, 0.5, 0.6]
    with pytest.raises(ValueError):
        SparseValues(indices=[1, 2], values=[0.4])


def test_from_dict_and_back():
    d = {"indices": [10, 20], "values": [0.5, 0.7]}
    sv = SparseValues.from_dict(d)
    assert sv is not None
    assert sv.indices == [10, 20]
    assert sv.values == [0.5, 0.7]
    assert sv.to_dict() == d


def test_upsert_with_sparse_does_not_raise(tmp_path):
    wd = tmp_path / "sparse_idx"
    idx = stm.Index.create(
        name="sparse", path=str(wd), dimension=32,
        compression="spectral_k64", calibration_sample=_cal(),
    )
    # Sparse fields are accepted, stored in metadata under __sparse_values__,
    # and a runtime warning fires.
    r = idx.upsert(vectors=[
        {"id": "v1",
         "values": [0.1] * 32,
         "sparse_values": {"indices": [0, 4, 9], "values": [0.5, 0.6, 0.7]},
         "metadata": {"topic": "rag"}},
        {"id": "v2", "values": [0.2] * 32, "metadata": {"topic": "memory"}},
    ])
    assert r["upserted_count"] == 2

    fetched = idx.fetch(ids=["v1"])["vectors"]["v1"]["metadata"]
    assert fetched["__sparse_values__"]["indices"] == [0, 4, 9]


def test_query_with_sparse_vector_does_not_raise(tmp_path, caplog):
    wd = tmp_path / "sparse_idx2"
    idx = stm.Index.create(
        name="sparse", path=str(wd), dimension=32,
        compression="spectral_k64", calibration_sample=_cal(),
    )
    idx.upsert(vectors=[{"id": "v1", "values": [0.1] * 32}])

    caplog.set_level(logging.WARNING, logger="spectraltm_db.sparse")
    r = idx.query(
        vector=[0.1] * 32,
        top_k=3,
        sparse_vector={"indices": [0, 1], "values": [0.5, 0.5]},
    )
    assert len(r["matches"]) == 1
    # Warning fired.
    assert any("sparse-dense hybrid" in m for m in caplog.messages)


def test_no_warning_when_sparse_absent(tmp_path, caplog):
    wd = tmp_path / "sparse_idx3"
    idx = stm.Index.create(
        name="sparse", path=str(wd), dimension=32,
        compression="spectral_k64", calibration_sample=_cal(),
    )
    idx.upsert(vectors=[{"id": "v1", "values": [0.1] * 32}])

    caplog.set_level(logging.WARNING, logger="spectraltm_db.sparse")
    idx.query(vector=[0.1] * 32, top_k=3)
    assert not any("sparse-dense hybrid" in m for m in caplog.messages)
