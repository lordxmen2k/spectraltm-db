"""Run a :class:`CompiledFilter` against a :class:`MetadataStore` and
return the matching chunk_id set.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from .filter_compiler import CompiledFilter

if TYPE_CHECKING:
    from ..storage.metadata_store import MetadataStore


def execute_filter(meta: "MetadataStore", cf: CompiledFilter) -> set[int]:
    """Return the set of live chunk_ids satisfying ``cf``.

    When ``cf.is_full_table_scan`` is True we still call SQL (so the
    caller doesn't have to special-case the empty-filter path) but
    the SQL has no predicates — it's a one-shot ``SELECT chunk_id FROM
    vectors WHERE is_deleted = 0``.
    """
    conn: sqlite3.Connection = meta._conn  # noqa: SLF001
    if cf.is_full_table_scan:
        cur = conn.execute(
            "SELECT chunk_id FROM vectors WHERE is_deleted = 0"
        )
    else:
        cur = conn.execute(
            f"SELECT chunk_id FROM vectors WHERE is_deleted = 0 AND ({cf.sql})",
            cf.params,
        )
    return {int(row[0]) for row in cur.fetchall()}
