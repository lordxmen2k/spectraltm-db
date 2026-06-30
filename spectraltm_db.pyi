"""Type stubs for spectraltm_db public API.

Hand-written for v0.1.x. Update alongside ``__init__.py`` and ``index.py``.
The goal is a usable minimum for IDE assist / mypy --strict callers,
not a complete reproduction of every overload.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# ---- Errors (re-declared here for IDE help; runtime imports are circular) ----

class SpectraltmDbError(Exception): ...
class InvalidArgument(SpectraltmDbError, ValueError): ...
class NotFound(SpectraltmDbError, LookupError): ...
class DimensionMismatch(SpectraltmDbError, ValueError): ...
class StorageError(SpectraltmDbError): ...
class FilterError(SpectraltmDbError): ...


# ---- Compression & metric ----

class CompressionSpec(str):
    SPECTRAL_K64: "CompressionSpec"
    SPECTRAL_K128: "CompressionSpec"
    SPECTRAL_K256: "CompressionSpec"
    FLOAT32: "CompressionSpec"

class MetricSpec(str):
    COSINE: "MetricSpec"
    EUCLIDEAN: "MetricSpec"
    DOTPRODUCT: "MetricSpec"


# ---- Config ----

class IndexConfig:
    name: str
    path: str
    dimension: int
    metric: MetricSpec
    compression: CompressionSpec
    encoder: str | None

    def __init__(
        self,
        name: str,
        path: str,
        dimension: int,
        metric: str | MetricSpec = ...,
        compression: str | CompressionSpec = ...,
        encoder: str | None = ...,
    ) -> None: ...

    @property
    def is_compressed(self) -> bool: ...

    @property
    def encoder_profile(self) -> dict[str, int]: ...

    def to_json_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "IndexConfig": ...


# ---- Index (Pinecone-shaped) ----

class Index:
    """A single index. Multi-namespace via ``namespace=...``."""

    @classmethod
    def create(
        cls,
        name: str,
        path: str,
        dimension: int,
        *,
        metric: str | MetricSpec = ...,
        compression: str | CompressionSpec = ...,
        encoder: str | None = ...,
        calibration_sample: Sequence[float] | None = ...,
        seed: int = ...,
    ) -> "Index": ...

    @classmethod
    def open(cls, path: str) -> "Index": ...

    def upsert(
        self,
        vectors: Sequence[dict[str, Any]],
        namespace: str | None = ...,
    ) -> dict[str, int]: ...

    def query(
        self,
        vector: Sequence[float],
        top_k: int = ...,
        *,
        filter: dict[str, Any] | None = ...,
        sparse_vector: dict[str, Any] | None = ...,
        namespace: str | None = ...,
        include_metadata: bool = ...,
        include_values: bool = ...,
    ) -> dict[str, Any]: ...

    def fetch(
        self,
        ids: Sequence[str],
        namespace: str | None = ...,
    ) -> dict[str, Any]: ...

    def delete(
        self,
        ids: Sequence[str] | None = ...,
        *,
        filter: dict[str, Any] | None = ...,
        delete_all: bool = ...,
        namespace: str | None = ...,
    ) -> dict[str, int]: ...

    def update(
        self,
        id: str,
        *,
        values: Sequence[float] | None = ...,
        set_metadata: dict[str, Any] | None = ...,
        namespace: str | None = ...,
    ) -> dict[str, bool]: ...

    def describe_index_stats(
        self,
        namespace: str | None = ...,
    ) -> dict[str, Any]: ...

    def flush(self) -> None: ...
    def compact(self, namespace: str | None = ...) -> None: ...
    def list_namespaces(self) -> list[str]: ...
    def delete_namespace(self, namespace: str) -> None: ...

    @property
    def config(self) -> IndexConfig: ...
    @property
    def name(self) -> str: ...
    @property
    def path(self) -> str: ...


__version__: str
__all__: list[str]
