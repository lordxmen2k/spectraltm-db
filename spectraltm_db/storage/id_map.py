"""Bidirectional ``str <-> u32`` chunk_id mapping per namespace.

Pinecone's contract:

  - Each vector has a string ``id`` provided by the user. IDs are
    case-sensitive, must be non-empty, must be unique within an index
    (we enforce uniqueness within a namespace and reject upserts with
    a duplicate id from a different namespace collision — see
    :class:`MetadataStore` for the cross-namespace collision policy).
  - On re-upsert with the same id, the existing chunk_id is reused
    but the underlying vector is overwritten (we tombstone the old
    chunk_id in :class:`VectorStore` and allocate a fresh tail
    chunk_id, so old ranked positions don't persist).

Implementation: a single SQLite table per namespace:

    CREATE TABLE id_map (
        id        TEXT PRIMARY KEY,
        chunk_id  INTEGER UNIQUE NOT NULL,
        is_deleted INTEGER NOT NULL DEFAULT 0
    );

The chunk_id column is unique so we never accidentally map two IDs to
the same chunk. (We could relax this for the float32 no-op-tombstone
case, but enforcing uniqueness is safer and only costs an extra
sqlite check.)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from ..errors import InvalidArgument


# Pinecone's documented ID limits: 1 to 64 chars, [A-Za-z0-9_\-].
# We're permissive — any non-empty string up to 256 chars — but reject
# path separators and NUL bytes on the way in.
_MAX_ID_LEN = 256


class IdMap:
    """SQLite-backed string-id ↔ chunk_id bijection for one namespace."""

    def __init__(self, ns_dir: Path) -> None:
        self._ns_dir = Path(ns_dir)
        self._ns_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._ns_dir / "id_map.sqlite"
        # check_same_thread=False: FastAPI's TestClient (and any
        # threaded HTTP server) reuses these connection objects from
        # request-handler threads. SQLite's internal lock serializes
        # access, which is acceptable for the per-namespace workload.
        self._conn = sqlite3.connect(
            str(self._db_path), isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS id_map (
                    id         TEXT PRIMARY KEY,
                    chunk_id   INTEGER UNIQUE NOT NULL,
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_id_map_chunk_id ON id_map(chunk_id);
                """
            )

    # ---- Public API --------------------------------------------------------

    def get_chunk_id(self, id_string: str) -> int | None:
        """Return the chunk_id for an id string, or None if absent."""
        row = self._conn.execute(
            "SELECT chunk_id FROM id_map WHERE id = ? AND is_deleted = 0",
            (id_string,),
        ).fetchone()
        return None if row is None else int(row[0])

    def get_id(self, chunk_id: int) -> str | None:
        """Inverse of ``get_chunk_id``. Returns None if absent / deleted."""
        row = self._conn.execute(
            "SELECT id FROM id_map WHERE chunk_id = ? AND is_deleted = 0",
            (int(chunk_id),),
        ).fetchone()
        return None if row is None else str(row[0])

    def assign(self, id_string: str, chunk_id: int) -> None:
        """Bind ``id_string -> chunk_id``. Re-activates a previously deleted
        row if one exists, so re-upsert with the same id works.

        Conflicts:
          - Active row with a *different* chunk_id → :class:`InvalidArgument`.
        """
        self._validate_id(id_string)
        cur = self._conn.execute(
            "SELECT chunk_id FROM id_map WHERE id = ? AND is_deleted = 0",
            (id_string,),
        ).fetchone()
        if cur is not None and int(cur[0]) != int(chunk_id):
            raise InvalidArgument(
                f"id {id_string!r} is already assigned to chunk_id {int(cur[0])}"
            )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO id_map(id, chunk_id, is_deleted, updated_at)
                VALUES (?, ?, 0, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    chunk_id   = excluded.chunk_id,
                    is_deleted = 0,
                    updated_at = datetime('now')
                """,
                (id_string, int(chunk_id)),
            )

    def mark_deleted(self, id_string: str) -> int | None:
        """Tombstone an id. Returns the chunk_id that was bound, or None."""
        cur = self._conn.execute(
            "SELECT chunk_id FROM id_map WHERE id = ? AND is_deleted = 0",
            (id_string,),
        ).fetchone()
        if cur is None:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE id_map SET is_deleted = 1, updated_at = datetime('now') "
                "WHERE id = ?",
                (id_string,),
            )
        return int(cur[0])

    def mark_deleted_chunk(self, chunk_id: int) -> str | None:
        """Tombstone by chunk_id, returning the associated id (or None)."""
        cur = self._conn.execute(
            "SELECT id FROM id_map WHERE chunk_id = ? AND is_deleted = 0",
            (int(chunk_id),),
        ).fetchone()
        if cur is None:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE id_map SET is_deleted = 1, updated_at = datetime('now') "
                "WHERE chunk_id = ?",
                (int(chunk_id),),
            )
        return str(cur[0])

    def all_chunk_ids(self) -> list[int]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM id_map WHERE is_deleted = 0 ORDER BY chunk_id"
        ).fetchall()
        return [int(r[0]) for r in rows]

    def all_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT id FROM id_map WHERE is_deleted = 0 ORDER BY chunk_id"
        ).fetchall()
        return [str(r[0]) for r in rows]

    def __len__(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM id_map WHERE is_deleted = 0"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "IdMap":
        return self

    def __exit__(self, *a) -> None:
        self.close()

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _validate_id(id_string: str) -> None:
        if not isinstance(id_string, str):
            raise InvalidArgument(
                f"vector id must be a string, got {type(id_string).__name__}"
            )
        if not id_string:
            raise InvalidArgument("vector id must be a non-empty string")
        if len(id_string) > _MAX_ID_LEN:
            raise InvalidArgument(
                f"vector id length {len(id_string)} exceeds {_MAX_ID_LEN}"
            )
        if any(ord(c) < 0x20 for c in id_string):
            raise InvalidArgument(
                "vector id must not contain control characters"
            )
