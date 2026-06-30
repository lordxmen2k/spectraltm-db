"""Filter pipeline — dict → SQL → chunk_id whitelist.

Filter dict shape mirrors Pinecone's metadata filter contract::

    {"session_id": {"$eq": "abc123"},
     "role":       {"$ne": "user"},
     "timestamp":  {"$gte": "2026-01-01T00:00:00Z"},
     "turn_index": {"$in": [1, 2, 3]},
     "deleted":    {"$exists": False},
     "$and":       [{"session_id": {"$eq": "abc"}},
                    {"role": {"$eq": "user"}}],
     "$or":        [...]}

Operators supported in v0.1 (see :data:`SUPPORTED_OPERATORS`):
  $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $and, $or.

Each leaf operator compiles to a single ``json_extract(metadata, '$.<key>')``
expression on the metadata JSON column. Unsupported operators raise
:class:`~spectraltm_db.errors.FilterError` at compile time with a clear
message including the operator name.
"""

from __future__ import annotations

import json
from typing import Any

from ..errors import FilterError


SUPPORTED_OPERATORS = frozenset({
    "$eq", "$ne",
    "$gt", "$gte", "$lt", "$lte",
    "$in", "$nin",
    "$exists",
    "$and", "$or",
})


def compile_filter(filter_dict: dict | None) -> "CompiledFilter":
    """Compile a Pinecone-shaped filter dict into a SQL fragment.

    Pass ``None`` or an empty dict to get the full-table-scan sentinel
    ``CompiledFilter(is_full_table_scan=True)``.
    """
    if not filter_dict:
        return CompiledFilter(sql="", params=[], is_full_table_scan=True)
    try:
        clauses, params = _compile_node(filter_dict)
    except FilterError:
        raise
    except Exception as e:
        raise FilterError(
            f"failed to compile filter {filter_dict!r}: {e}"
        ) from e
    return CompiledFilter(sql=clauses, params=params, is_full_table_scan=False)


def _compile_node(node: Any) -> tuple[str, list[Any]]:
    if not isinstance(node, dict):
        raise FilterError(f"unexpected node in filter tree: {node!r}")

    logical = [k for k in node if k in ("$and", "$or")]
    if logical:
        if len(node) != 1:
            raise FilterError(
                f"logical operator at top must be the only key, got {node!r}"
            )
        op = logical[0]
        children = node[op]
        if not isinstance(children, list) or not children:
            raise FilterError(
                f"{op} expects a non-empty list, got {children!r}"
            )
        clause_list = []
        param_list: list[Any] = []
        for child in children:
            c, p = _compile_node(child)
            clause_list.append(c)
            param_list.extend(p)
        joiner = " AND " if op == "$and" else " OR "
        return "(" + joiner.join(clause_list) + ")", param_list

    # Leaf dict: ``{"key": value}`` or ``{"key": {"$eq": value}}``.
    clauses: list[str] = []
    params: list[Any] = []
    for key, sub in node.items():
        if not isinstance(sub, dict):
            sub = {"$eq": sub}
        for op, val in sub.items():
            if op not in SUPPORTED_OPERATORS:
                raise FilterError(
                    f"unsupported operator {op!r} on key {key!r}; "
                    f"supported: {sorted(SUPPORTED_OPERATORS)}"
                )
            clause, param_list = _compile_operator(key, op, val)
            clauses.append(clause)
            if isinstance(param_list, list):
                params.extend(param_list)
            else:
                params.append(param_list)
    if not clauses:
        # Empty leaf dict: treat as "no constraint".
        return "1=1", []
    return " AND ".join(f"({c})" for c in clauses), params


def _compile_operator(key: str, op: str, val: Any) -> tuple[str, Any | list[Any]]:
    json_path = _make_json_path(key)
    extract = f"json_extract(metadata, '{json_path}')"
    if op == "$eq":
        return _eq_clause(extract, val)
    if op == "$ne":
        c, p = _eq_clause(extract, val)
        return f"NOT ({c})", p
    if op == "$gt":
        return f"{extract} > ?", val
    if op == "$gte":
        return f"{extract} >= ?", val
    if op == "$lt":
        return f"{extract} < ?", val
    if op == "$lte":
        return f"{extract} <= ?", val
    if op == "$in":
        if not isinstance(val, list):
            raise FilterError(f"$in expects a list, got {type(val).__name__}")
        if not val:
            return "0=1", []
        placeholders = ",".join("?" * len(val))
        return f"{extract} IN ({placeholders})", list(val)
    if op == "$nin":
        if not isinstance(val, list):
            raise FilterError(f"$nin expects a list, got {type(val).__name__}")
        if not val:
            return "1=1", []
        placeholders = ",".join("?" * len(val))
        return f"{extract} NOT IN ({placeholders})", list(val)
    if op == "$exists":
        if bool(val):
            return f"{extract} IS NOT NULL", None
        return f"{extract} IS NULL", None
    raise FilterError(f"unknown operator {op!r}")  # pragma: no cover


def _eq_clause(extract: str, val: Any) -> tuple[str, Any]:
    if isinstance(val, bool):
        # bool before int — bool is a subclass of int in Python.
        return f"{extract} = ?", json.dumps(val)
    if isinstance(val, (int, float)):
        return f"{extract} = ?", val
    if val is None:
        return f"{extract} IS NULL", None
    return f"{extract} = ?", val


def _make_json_path(key: str) -> str:
    # Accept flat dotted paths: "metadata.session_id" or "session_id".
    parts = key.split(".")
    return "$." + ".".join(p for p in parts)


class CompiledFilter:
    """Result of :func:`compile_filter`."""

    __slots__ = ("sql", "params", "is_full_table_scan")

    def __init__(self, sql: str, params: list[Any], is_full_table_scan: bool) -> None:
        self.sql = sql
        self.params = list(params)
        self.is_full_table_scan = is_full_table_scan

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CompiledFilter(is_full_table_scan={self.is_full_table_scan}, "
            f"sql={self.sql!r}, params={self.params!r})"
        )
