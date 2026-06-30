"""Filter compiler — all 11 operators + AND/OR composition."""

from __future__ import annotations

import json

import pytest

from spectraltm_db.retrieval.filter_compiler import compile_filter, SUPPORTED_OPERATORS
from spectraltm_db.errors import FilterError


def test_no_filter_is_full_scan():
    cf = compile_filter(None)
    assert cf.is_full_table_scan
    cf = compile_filter({})
    assert cf.is_full_table_scan


def test_eq_string():
    cf = compile_filter({"session_id": {"$eq": "abc"}})
    assert not cf.is_full_table_scan
    assert "json_extract(metadata, '$.session_id') = ?" in cf.sql
    assert cf.params == ["abc"]


def test_eq_shortcut_dict():
    """`{"k": "v"}` is equivalent to `{"k": {"$eq": "v"}}`."""
    cf = compile_filter({"k": "v"})
    assert "= ?" in cf.sql
    assert cf.params == ["v"]


def test_eq_int():
    cf = compile_filter({"turn_index": {"$eq": 5}})
    assert cf.params == [5]


def test_eq_bool():
    cf = compile_filter({"flagged": {"$eq": True}})
    assert cf.params == [json.dumps(True)]


def test_eq_none():
    cf = compile_filter({"deleted_at": {"$eq": None}})
    assert "IS NULL" in cf.sql


def test_ne():
    cf = compile_filter({"role": {"$ne": "user"}})
    assert "NOT (" in cf.sql


def test_gt_gte_lt_lte():
    for op, expected in [
        ("$gt",  ">"),
        ("$gte", ">="),
        ("$lt",  "<"),
        ("$lte", "<="),
    ]:
        cf = compile_filter({"timestamp": {op: 5}})
        assert expected in cf.sql
        assert cf.params == [5]


def test_in_nin():
    cf = compile_filter({"x": {"$in": [1, 2, 3]}})
    assert "IN (?,?,?)" in cf.sql
    assert cf.params == [1, 2, 3]

    cf = compile_filter({"x": {"$nin": [1, 2]}})
    assert "NOT IN (?,?)" in cf.sql


def test_in_empty_returns_no_match():
    cf = compile_filter({"x": {"$in": []}})
    # The compiler wraps each leaf clause in parens for AND-safety.
    assert cf.sql == "(0=1)"


def test_nin_empty_returns_all_match():
    cf = compile_filter({"x": {"$nin": []}})
    assert cf.sql == "(1=1)"


def test_exists():
    cf = compile_filter({"deleted_at": {"$exists": True}})
    assert "IS NOT NULL" in cf.sql
    cf = compile_filter({"deleted_at": {"$exists": False}})
    assert "IS NULL" in cf.sql


def test_and_at_top():
    f = {"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}
    cf = compile_filter(f)
    assert " AND " in cf.sql
    assert cf.params == [1, 2]


def test_or_at_top():
    f = {"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}
    cf = compile_filter(f)
    assert " OR " in cf.sql
    assert cf.params == [1, 2]


def test_and_with_logical_plus_field_filter():
    """Pinecone-style: leaf filters at top + logical at top are mutually exclusive."""
    f = {"a": {"$eq": 1}, "$and": [{"b": {"$eq": 2}}]}
    with pytest.raises(FilterError):
        compile_filter(f)


def test_unsupported_operator():
    with pytest.raises(FilterError) as exc:
        compile_filter({"x": {"$between": [1, 5]}})
    assert "$between" in str(exc.value)
    assert "supported" in str(exc.value).lower()


def test_in_requires_list():
    with pytest.raises(FilterError):
        compile_filter({"x": {"$in": "abc"}})


def test_and_empty_list():
    with pytest.raises(FilterError):
        compile_filter({"$and": []})


def test_dotted_path_becomes_json_path():
    cf = compile_filter({"a.b.c": {"$eq": 1}})
    assert "json_extract(metadata, '$.a.b.c')" in cf.sql


def test_supported_operators_constant():
    expected = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin",
                "$exists", "$and", "$or"}
    assert expected == SUPPORTED_OPERATORS
