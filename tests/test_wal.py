"""WAL round-trip — append + replay + truncate."""

from __future__ import annotations

from pathlib import Path

from spectraltm_db.wal import WriteAheadLog


def test_append_and_iter(tmp_path):
    w = WriteAheadLog(tmp_path)
    w.append("upsert", id="a", chunk_id=0, values=[0.1, 0.2], namespace="default")
    w.append("delete", id="b", namespace="default")
    entries = list(w.iter_entries())
    assert len(entries) == 2
    assert entries[0]["op"] == "upsert"
    assert entries[1]["op"] == "delete"


def test_truncate(tmp_path):
    w = WriteAheadLog(tmp_path)
    w.append("upsert", id="a", chunk_id=0, values=[0.1])
    w.truncate()
    assert w.size_bytes() == 0
    assert list(w.iter_entries()) == []


def test_corrupted_line_skipped(tmp_path):
    w = WriteAheadLog(tmp_path)
    w.append("upsert", id="a", chunk_id=0, values=[0.1])
    # Manually append a bad JSON line.
    with w.path.open("a", encoding="utf-8") as f:
        f.write("not-json\n")
    w.append("delete", id="b")
    entries = list(w.iter_entries())
    assert len(entries) == 2    # bad line silently skipped
