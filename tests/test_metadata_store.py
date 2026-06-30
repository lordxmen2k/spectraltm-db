"""MetadataStore — CRUD, JSON extraction, soft delete, ensure_filter_index."""

from __future__ import annotations

import pytest

from spectraltm_db.storage.metadata_store import MetadataStore
from spectraltm_db.retrieval.filter_executor import execute_filter
from spectraltm_db.retrieval.filter_compiler import compile_filter


def test_insert_and_get(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("foo", 0, {"x": 1, "y": "hi"})
    m.insert("bar", 1, {"x": 2, "y": "lo"})
    out = m.get("foo")
    assert out["chunk_id"] == 0
    assert out["metadata"] == {"x": 1, "y": "hi"}
    assert m.live_count() == 2


def test_update_metadata_merge(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("foo", 0, {"x": 1, "y": "hi"})
    ok = m.update_metadata("foo", {"y": "lo", "z": 99})
    assert ok
    assert m.get("foo")["metadata"] == {"x": 1, "y": "lo", "z": 99}


def test_mark_deleted(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("foo", 0, {"x": 1})
    cid = m.mark_deleted("foo")
    assert cid == 0
    assert m.get("foo") is None
    assert m.live_count() == 0


def test_mark_deleted_chunk(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("foo", 7, {"x": 1})
    id_str = m.mark_deleted_chunk(7)
    assert id_str == "foo"
    assert m.live_count() == 0


def test_get_many_and_list_ids(tmp_path):
    m = MetadataStore(tmp_path)
    for i in range(5):
        m.insert(f"id{i}", i, {"topic": f"t{i % 2}"})
    out = m.get_many(["id0", "id2", "id4", "missing"])
    assert set(out.keys()) == {"id0", "id2", "id4"}
    assert m.list_ids(prefix="id2", limit=10) == ["id2"]


def test_execute_filter_eq(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("a", 0, {"topic": "t1"})
    m.insert("b", 1, {"topic": "t2"})
    m.insert("c", 2, {"topic": "t1"})
    cf = compile_filter({"topic": {"$eq": "t1"}})
    assert execute_filter(m, cf) == {0, 2}


def test_execute_filter_in(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("a", 0, {"topic": "t1"})
    m.insert("b", 1, {"topic": "t2"})
    m.insert("c", 2, {"topic": "t3"})
    cf = compile_filter({"topic": {"$in": ["t1", "t3"]}})
    assert execute_filter(m, cf) == {0, 2}


def test_execute_filter_full_scan(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("a", 0, {"topic": "t1"})
    m.insert("b", 1, {"topic": "t2"})
    cf = compile_filter(None)
    assert execute_filter(m, cf) == {0, 1}


def test_ensure_filter_index(tmp_path):
    m = MetadataStore(tmp_path)
    m.ensure_filter_index("topic")
    # No assertion other than it didn't error — SQLite CREATE INDEX IF NOT EXISTS
    # is hard to verify externally in a unit test short of running EXPLAIN.
    # We rely on the smoke + integration tests for end-to-end correctness.
    m.insert("a", 0, {"topic": "t1"})
    cf = compile_filter({"topic": {"$eq": "t1"}})
    assert execute_filter(m, cf) == {0}


def test_persistence(tmp_path):
    m = MetadataStore(tmp_path)
    m.insert("foo", 0, {"x": 1})
    m.close()
    m2 = MetadataStore(tmp_path)
    assert m2.get("foo")["metadata"] == {"x": 1}
