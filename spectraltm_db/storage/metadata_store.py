"""SQLite metadata store — per namespace.

Schema (mirrors ``spectraltm_db_spec.md`` §3.3):

    CREATE TABLE vectors (
        id          TEXT PRIMARY KEY,
        chunk_id    INTEGER UNIQUE NOT NULL,
        offset      INTEGER NOT NULL DEFAULT 0,     -- reserved for §3.2 layouts
        is_deleted  INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL,
        metadata    TEXT NOT NULL                   -- JSON blob
    );

The ``metadata`` column is a JSON blob. ``filter_compiler`` extracts
``json_extract(metadata, '$.<key>')`` to compile ``$eq``, ``$gte``, etc.
operators as SQL — see :mod:`spectraltm_db.retrieval.filter_compiler`.

Dynamic filter indexes
----------------------
For keys that get filtered on frequently, ``create_filter_index(key)``
adds an expression index over the JSON path:

    CREATE INDEX IF NOT EXISTS idx_meta_session
        ON vectors(json_extract(metadata, '$.session_id'));

Index management is exposed via :meth:`ensure_filter_index` so the
caller can opt-in at upsert / config time. v0.1 ships with automatic
discovery on first filter use as an opt-in behavior (see Index
configuration).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ..errors import InvalidArgument, NotFound


class MetadataStore:
    """SQLite-backed metadata for one namespace."""

    def __init__(self, ns_dir: Path) -> None:
        self._ns_dir = Path(ns_dir)
        self._ns_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._ns_dir / "metadata.sqlite"
        # See IdMap.__init__ — same threading rationale.
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
                CREATE TABLE IF NOT EXISTS vectors (
                    id          TEXT PRIMARY KEY,
                    chunk_id    INTEGER UNIQUE NOT NULL,
                    offset      INTEGER NOT NULL DEFAULT 0,
                    is_deleted  INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    metadata    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vectors_chunk_id ON vectors(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_vectors_is_deleted ON vectors(is_deleted);
                """
            )

    # ---- Write API ---------------------------------------------------------

    def insert(
        self,
        id_string: str,
        chunk_id: int,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Insert (or replace) a metadata row.

        Replaces any existing row with the same id — caller is
        responsible for tombstoning old chunk_ids.
        """
        meta_json = json.dumps(metadata or {}, separators=(",", ":"))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO vectors(id, chunk_id, is_deleted, metadata, updated_at)
                VALUES (?, ?, 0, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    chunk_id   = excluded.chunk_id,
                    is_deleted = 0,
                    metadata   = excluded.metadata,
                    updated_at = datetime('now')
                """,
                (id_string, int(chunk_id), meta_json),
            )

    def update_metadata(self, id_string: str, set_metadata: dict[str, Any]) -> bool:
        """Partial-update the JSON metadata blob. Returns True if the
        row exists. Caller's responsibility: well-formed values.
        """
        cur = self._conn.execute(
            "SELECT metadata FROM vectors WHERE id = ? AND is_deleted = 0",
            (id_string,),
        ).fetchone()
        if cur is None:
            return False
        existing = json.loads(cur[0]) if cur[0] else {}
        merged = {**existing, **set_metadata}
        with self._conn:
            self._conn.execute(
                "UPDATE vectors SET metadata = ?, updated_at = datetime('now') "
                "WHERE id = ?",
                (json.dumps(merged, separators=(",", ":")), id_string),
            )
        return True

    def mark_deleted(self, id_string: str) -> int | None:
        """Tombstone by id. Returns the chunk_id that was bound, or None."""
        cur = self._conn.execute(
            "SELECT chunk_id FROM vectors WHERE id = ? AND is_deleted = 0",
            (id_string,),
        ).fetchone()
        if cur is None:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE vectors SET is_deleted = 1, updated_at = datetime('now') "
                "WHERE id = ?",
                (id_string,),
            )
        return int(cur[0])

    def mark_deleted_chunk(self, chunk_id: int) -> str | None:
        cur = self._conn.execute(
            "SELECT id FROM vectors WHERE chunk_id = ? AND is_deleted = 0",
            (int(chunk_id),),
        ).fetchone()
        if cur is None:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE vectors SET is_deleted = 1, updated_at = datetime('now') "
                "WHERE chunk_id = ?",
                (int(chunk_id),),
            )
        return str(cur[0])

    # ---- Read API ----------------------------------------------------------

    def get(self, id_string: str) -> dict | None:
        row = self._conn.execute(
            "SELECT id, chunk_id, is_deleted, metadata FROM vectors WHERE id = ?",
            (id_string,),
        ).fetchone()
        if row is None or int(row[2]) == 1:
            return None
        return {
            "id": str(row[0]),
            "chunk_id": int(row[1]),
            "metadata": json.loads(row[3]) if row[3] else {},
        }

    def get_many(self, id_strings: Iterable[str]) -> dict[str, dict]:
        ids = list(id_strings)
        if not ids:
            return {}
        qmarks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, chunk_id, metadata FROM vectors WHERE is_deleted = 0 "
            f"AND id IN ({qmarks})",
            ids,
        ).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            out[str(r[0])] = {
                "id": str(r[0]),
                "chunk_id": int(r[1]),
                "metadata": json.loads(r[2]) if r[2] else {},
            }
        return out

    def get_chunk_ids(self, id_strings: Iterable[str]) -> dict[str, int]:
        ids = list(id_strings)
        if not ids:
            return {}
        qmarks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, chunk_id FROM vectors WHERE is_deleted = 0 "
            f"AND id IN ({qmarks})",
            ids,
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    # ---- Stats -------------------------------------------------------------

    def live_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE is_deleted = 0"
        ).fetchone()
        return int(row[0])

    def list_ids(self, prefix: str | None = None, limit: int | None = None) -> list[str]:
        if prefix:
            like = prefix + "%"
            sql = "SELECT id FROM vectors WHERE is_deleted = 0 AND id LIKE ? ORDER BY id"
            params: tuple = (like,)
        else:
            sql = "SELECT id FROM vectors WHERE is_deleted = 0 ORDER BY id"
            params = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = params + (int(limit),)
        return [str(r[0]) for r in self._conn.execute(sql, params).fetchall()]

    # ---- Filter integration -------------------------------------------------

    def ensure_filter_index(self, key: str) -> None:
        """Add a JSON-expression index on metadata.<key> if not present."""
        safe_key = key.replace('"', '""')
        with self._conn:
            self._conn.execute(
                f'CREATE INDEX IF NOT EXISTS idx_meta_{safe_key} '
                f"ON vectors(json_extract(metadata, '$.{safe_key}')) "
                f"WHERE is_deleted = 0"
            )

    # ---- Maintenance -------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, *a) -> None:
        self.close()
