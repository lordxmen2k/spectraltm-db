"""Exception hierarchy for spectraltm_db.

Mirrors the spirit of Pinecone's `PineconeException` family:
  - ``SpectraltmDbError`` is the catch-all root.
  - ``InvalidArgument`` covers type/shape/value errors at the API surface.
  - ``NotFound`` covers missing indexes, namespaces, or vector IDs.
  - ``DimensionMismatch`` surfaces dim/compression-config mismatches.
"""

from __future__ import annotations


class SpectraltmDbError(Exception):
    """Root of the spectraltm_db exception hierarchy."""


class InvalidArgument(SpectraltmDbError, ValueError):
    """Caller passed an ill-formed argument (wrong shape, bad filter, etc)."""


class NotFound(SpectraltmDbError, LookupError):
    """A requested resource (index, namespace, id) does not exist."""


class DimensionMismatch(SpectraltmDbError, ValueError):
    """Vector dimension does not match the index's configured dimension."""


class StorageError(SpectraltmDbError):
    """Underlying storage layer (spectraltm crate or SQLite) raised an error."""


class FilterError(SpectraltmDbError):
    """A filter dict could not be compiled or executed."""
