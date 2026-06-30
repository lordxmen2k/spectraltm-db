"""Per-index layout helper.

Layout on disk::

    <path>/
        config.json                    # IndexConfig snapshot
        namespaces/
            default/
                index.sltm             # spectraltm.SpectralIndex (compressed ns only)
                vector_store.msgpack   # tombstones + next_chunk_id
                calibration.json       # sample buffer used to reconstruct SpectralEncoder
                id_map.sqlite          # id_map
                id_map.sqlite-wal      # SQLite WAL
                metadata.sqlite        # metadata_store
                metadata.sqlite-wal
            <other_ns>/...             # same layout

A new index is created at ``<path>/namespaces/<ns>`` per ``create(ns=...)``.
Namespaces are isolated: each namespace carries its own ``VectorStore``,
``IdMap``, and ``MetadataStore``. There is no cross-namespace ``id``
uniqueness enforced at the SQL level (Pinecone allows the same id in
multiple namespaces). Delete a namespace by removing its directory.
"""

from __future__ import annotations

from pathlib import Path


class Namespaces:
    """Helper to map ``<base_path> + namespace`` to a directory."""

    DEFAULT = "default"

    def __init__(self, base_path: Path) -> None:
        self._base = Path(base_path)

    @property
    def base(self) -> Path:
        return self._base

    def dir_for(self, namespace: str | None) -> Path:
        ns = self._normalize(namespace)
        return self._base / "namespaces" / ns

    def list(self) -> list[str]:
        d = self._base / "namespaces"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def remove(self, namespace: str | None) -> None:
        import shutil
        ns_dir = self.dir_for(namespace)
        if ns_dir.exists():
            shutil.rmtree(ns_dir)

    @staticmethod
    def _normalize(namespace: str | None) -> str:
        # Pinecone allows empty string as the implicit default; we map to
        # "default" the same way the client SDKs do.
        if namespace is None or namespace == "":
            return Namespaces.DEFAULT
        if namespace == Namespaces.DEFAULT:
            return Namespaces.DEFAULT
        # Reject path-traversal characters at this layer; the Index class
        # also enforces this on namespace creation.
        for ch in namespace:
            if ch in ("/", "\\", "\x00"):
                raise ValueError(f"namespace {namespace!r} contains illegal characters")
        return namespace
