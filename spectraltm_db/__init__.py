"""spectraltm_db — a Pinecone-shaped local vector database.

Built on top of the `spectraltm` Rust crate (Sparse Spectral Encoding AVX2
inner loop). Provides a single-machine, on-disk, single-tenant vector DB
with a Pinecone-compatible Python API.

Quickstart
----------
    >>> import spectraltm_db as stm
    >>> idx = stm.Index.create(
    ...     name="conversations",
    ...     path="~/.spectraltm/conversations",
    ...     dimension=384,
    ...     metric="cosine",
    ...     compression="spectral_k64",
    ... )
    >>> idx.upsert(vectors=[{"id": "t1", "values": [...], "metadata": {...}}])
    >>> idx.query(vector=[...], top_k=5)

See :mod:`spectraltm_db.index` for the full API.
"""

from __future__ import annotations

from .config import IndexConfig, CompressionSpec, MetricSpec
from .errors import SpectraltmDbError, NotFound, InvalidArgument, DimensionMismatch
from .index import Index

# Sub-modules exposed for advanced / framework users.
from . import storage
from . import retrieval

__all__ = [
    "Index",
    "IndexConfig",
    "CompressionSpec",
    "MetricSpec",
    "SpectraltmDbError",
    "NotFound",
    "InvalidArgument",
    "DimensionMismatch",
    "storage",
    "retrieval",
]

# `__version__` is read from the installed package metadata (set in
# `pyproject.toml`); falling back to a literal only if the metadata
# resolver is unavailable (e.g. running from an unpacked source tree
# without `pip install`).
try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("spectraltm-db")
except (PackageNotFoundError, ImportError):                # pragma: no cover
    __version__ = "0.1.1.dev0"
