"""Minimal JSONL write-ahead log for crash recovery.

Spec goal (§8): record every committed operation durably so an
unclean shutdown can replay without losing data.

v0.1 implementation:

  - One JSONL file at ``<path>/wal/pending.jsonl``.
  - Each entry is one JSON object per line.
  - ``Index.upsert``/``delete``/``update`` append a line BEFORE the
    SQLite/spectraltm write succeeds, then truncate the WAL when
    ``flush_wal()`` is called.
  - On ``Index.open()`` we replay any leftover lines.

This is intentionally minimal:

  - We do not ``fdatasync`` after each write (Python-level fsync is
    expensive). For a v0.1 single-process tool this is acceptable.
  - We do not bound the WAL size or rotate. The user should call
    ``Index.flush()`` periodically to truncate it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable


WAL_DIRNAME = "wal"
WAL_FILENAME = "pending.jsonl"


class WriteAheadLog:
    """Append-only JSONL log of committed operations."""

    def __init__(self, base_path: Path) -> None:
        self._base = Path(base_path)
        self._wal_dir = self._base / WAL_DIRNAME
        self._wal_path = self._wal_dir / WAL_FILENAME
        self._wal_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._wal_path

    def append(self, op: str, **fields) -> None:
        """Append an op entry. ``op`` is the operation kind; remaining
        fields become part of the JSON record."""
        record = {
            "op": op,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **fields,
        }
        with self._wal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def iter_entries(self) -> Iterable[dict]:
        if not self._wal_path.exists():
            return
        for line in self._wal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Corrupted line — skip but don't crash.
                continue

    def truncate(self) -> None:
        """Truncate the WAL. Called after a clean checkpoint."""
        if self._wal_path.exists():
            with self._wal_path.open("w", encoding="utf-8") as f:
                f.write("")

    def size_bytes(self) -> int:
        return self._wal_path.stat().st_size if self._wal_path.exists() else 0
