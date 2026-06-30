"""IndexConfig validation."""

from __future__ import annotations

import pytest

import spectraltm_db as stm
from spectraltm_db.config import IndexConfig, CompressionSpec, MetricSpec
from spectraltm_db.errors import InvalidArgument


def test_basics():
    cfg = IndexConfig(name="hello", path="C:/tmp/foo", dimension=128)
    assert cfg.name == "hello"
    assert cfg.metric == MetricSpec.COSINE
    assert cfg.compression == CompressionSpec.SPECTRAL_K64
    assert cfg.is_compressed is True


def test_string_coercion():
    cfg = IndexConfig(
        name="hello",
        path="C:/tmp/foo",
        dimension=64,
        metric="euclidean",
        compression="spectral_k128",
    )
    assert cfg.metric == MetricSpec.EUCLIDEAN
    assert cfg.compression == CompressionSpec.SPECTRAL_K128


def test_invalid_name():
    with pytest.raises(InvalidArgument):
        IndexConfig(name="bad/name", path="C:/tmp/foo", dimension=64)
    with pytest.raises(InvalidArgument):
        IndexConfig(name="", path="C:/tmp/foo", dimension=64)


def test_invalid_dimension():
    with pytest.raises(InvalidArgument):
        IndexConfig(name="x", path="C:/tmp/foo", dimension=0)
    with pytest.raises(InvalidArgument):
        IndexConfig(name="x", path="C:/tmp/foo", dimension=-5)


def test_json_roundtrip():
    cfg = IndexConfig(
        name="x",
        path="C:/tmp/foo",
        dimension=64,
        compression="spectral_k256",
        encoder="my-encoder",
    )
    d = cfg.to_json_dict()
    cfg2 = IndexConfig.from_json_dict(d)
    assert cfg2.name == cfg.name
    assert cfg2.dimension == cfg.dimension
    assert cfg2.compression == CompressionSpec.SPECTRAL_K256
    assert cfg2.encoder == "my-encoder"


def test_encoder_profile_for_compressed():
    cfg = IndexConfig(name="x", path=".", dimension=384, compression="spectral_k64")
    assert cfg.encoder_profile["top_k"] == 64
    assert cfg.is_compressed


def test_encoder_profile_for_float32():
    cfg = IndexConfig(name="x", path=".", dimension=384, compression="float32")
    assert cfg.encoder_profile["top_k"] == 0
    assert not cfg.is_compressed
