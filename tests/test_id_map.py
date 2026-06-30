"""IdMap — string ↔ chunk_id, soft delete, conflict detection."""

from __future__ import annotations

import pytest

from spectraltm_db.storage.id_map import IdMap
from spectraltm_db.errors import InvalidArgument


def test_assign_and_get(tmp_path):
    m = IdMap(tmp_path)
    m.assign("foo", 0)
    m.assign("bar", 1)
    assert m.get_chunk_id("foo") == 0
    assert m.get_chunk_id("bar") == 1
    assert m.get_id(0) == "foo"
    assert m.get_id(1) == "bar"
    assert len(m) == 2


def test_mark_deleted(tmp_path):
    m = IdMap(tmp_path)
    m.assign("foo", 0)
    m.assign("bar", 1)
    cid = m.mark_deleted("foo")
    assert cid == 0
    assert m.get_chunk_id("foo") is None
    assert m.get_id(0) is None
    assert len(m) == 1


def test_mark_deleted_chunk(tmp_path):
    m = IdMap(tmp_path)
    m.assign("foo", 5)
    id_str = m.mark_deleted_chunk(5)
    assert id_str == "foo"
    assert m.get_id(5) is None


def test_reassign_after_delete(tmp_path):
    m = IdMap(tmp_path)
    m.assign("foo", 0)
    m.mark_deleted("foo")
    m.assign("foo", 7)            # allowed: previous chunk_id was tombstoned
    assert m.get_chunk_id("foo") == 7


def test_conflict_on_active_assignment(tmp_path):
    m = IdMap(tmp_path)
    m.assign("foo", 0)
    with pytest.raises(InvalidArgument):
        m.assign("foo", 99)


def test_invalid_id(tmp_path):
    m = IdMap(tmp_path)
    with pytest.raises(InvalidArgument):
        m.assign("", 0)
    with pytest.raises(InvalidArgument):
        m.assign(123, 0)
    with pytest.raises(InvalidArgument):
        m.assign("x" * 300, 0)
    with pytest.raises(InvalidArgument):
        m.assign("a\x00b", 0)


def test_all_ids_and_chunk_ids(tmp_path):
    m = IdMap(tmp_path)
    for i in range(5):
        m.assign(f"id{i}", i)
    assert m.all_ids() == ["id0", "id1", "id2", "id3", "id4"]
    assert m.all_chunk_ids() == [0, 1, 2, 3, 4]
    m.mark_deleted("id2")
    assert m.all_ids() == ["id0", "id1", "id3", "id4"]
    assert m.all_chunk_ids() == [0, 1, 3, 4]


def test_persistence(tmp_path):
    m = IdMap(tmp_path)
    m.assign("foo", 0)
    m.assign("bar", 1)
    m.close()
    m2 = IdMap(tmp_path)
    assert m2.get_chunk_id("foo") == 0
    assert m2.get_chunk_id("bar") == 1
