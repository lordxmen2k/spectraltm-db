"""Public Index class — the Pinecone-shaped surface area.

Examples
--------

    >>> import spectraltm_db as stm
    >>> idx = stm.Index.create(
    ...     name="conversations",
    ...     path="~/.spectraltm/conversations",
    ...     dimension=384,
    ...     metric="cosine",
    ...     compression="spectral_k64",
    ...     calibration_sample=sample_arr,   # 1D iterable, len % dim == 0
    ... )
    >>> idx.upsert(vectors=[
    ...     {"id": "t1", "values": [...], "metadata": {"session_id": "abc"}},
    ...     {"id": "t2", "values": [...], "metadata": {"session_id": "abc"}},
    ... ])
    {'upserted_count': 2}
    >>> idx.query(vector=[...], top_k=5, filter={"session_id": {"$eq": "abc"}})
    {'matches': [{'id': 't1', 'score': 0.92, 'metadata': {...}}, ...],
     'namespace': 'default', 'usage': {'read_units': 1}}
    >>> idx.fetch(ids=["t1", "t2"])
    {'vectors': {'t1': {...}, 't2': {...}}, 'namespace': 'default'}
    >>> idx.delete(filter={"session_id": {"$eq": "abc"}})
    {'deleted_count': 2}
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (
    CompressionSpec,
    IndexConfig,
    MetricSpec,
)
from .errors import InvalidArgument, NotFound, DimensionMismatch
from .retrieval.filter_compiler import compile_filter
from .retrieval.filter_executor import execute_filter
from .storage.id_map import IdMap
from .storage.metadata_store import MetadataStore
from .storage.namespaces import Namespaces
from .storage.vector_store import VectorStore, SearchHit
from .wal import WriteAheadLog


log = logging.getLogger(__name__)


class Index:
    """A single index. Multi-namespace via ``namespace=...`` arg on each op."""

    # ------------------------------------------------------------------------
    # Construction / opening
    # ------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        path: str | Path,
        dimension: int,
        *,
        metric: str | MetricSpec = "cosine",
        compression: str | CompressionSpec = "spectral_k64",
        encoder: str | None = None,
        calibration_sample: Sequence[float] | None = None,
        seed: int = 0,
    ) -> "Index":
        """Create a new index.

        The calibration_sample is required for compressed namespaces
        (spectral_k64/k128/k256); pass an iterable of floats whose
        length is a multiple of ``dimension``. A 1,000–10,000 sample
        drawn from your real corpus is a good size.

        If calibration_sample is ``None`` and the index is compressed,
        a deterministic synthetic sample is generated so a fresh index
        can boot without forcing the caller to pre-collect vectors. The
        sample (whether user-supplied or synthetic) is persisted to
        ``<path>/calibration.json`` so subsequent ``Index.open(...)``
        calls can reuse it without the caller re-supplying one.
        """
        path = Path(os.path.expanduser(str(path)))
        path.mkdir(parents=True, exist_ok=True)

        existing_config_path = path / "config.json"
        if existing_config_path.exists():
            # We do not allow creating over an existing index — caller
            # should explicitly delete the directory or open existing.
            raise InvalidArgument(
                f"index already exists at {path}; use Index.open() instead"
            )

        config = IndexConfig(
            name=name,
            path=str(path),
            dimension=int(dimension),
            metric=metric,
            compression=compression,
            encoder=encoder,
        )
        existing_config_path.write_text(json.dumps(config.to_json_dict(), indent=2))

        # Resolve the calibration buffer up-front (user-supplied or
        # seeded) and persist it at the index root so Index.open can
        # find it later — that's a single write instead of one per
        # namespace, and it survives creating-then-open workflows.
        if calibration_sample is None and config.is_compressed:
            calibration_sample = cls._make_synthetic_sample(config.dimension, name)
        if calibration_sample is not None:
            prof = config.encoder_profile
            f = config.dimension // 2 + 1
            requested_k = prof["top_k"] or (f - 1)
            actual_k = min(requested_k, f - 1)
            (path / "calibration.json").write_text(json.dumps({
                "dim": config.dimension,
                "top_k": actual_k,
                "mag_bits": prof["mag_bits"],
                "phase_bits": prof["phase_bits"],
                "norm_bits": prof["norm_bits"],
                "sample": list(calibration_sample),
            }))

        return cls(
            config=config,
            calibration_sample=calibration_sample,
            write_seeded_calibration_sample=False,
            seed=seed,
        )

    @staticmethod
    def _make_synthetic_sample(dim: int, seed: str) -> list[float]:
        import random
        rng = random.Random(seed)
        n_vecs = 200
        sample: list[float] = []
        for _ in range(n_vecs):
            phase = rng.uniform(0, 2 * 3.14159)
            for j in range(dim):
                val = (0.5 + 0.5 * ((j + phase) % 1.0)) * 0.2
                val += rng.gauss(0, 0.05)
                sample.append(val)
        return sample

    @classmethod
    def open(cls, path: str | Path) -> "Index":
        """Open an existing index from disk."""
        path = Path(os.path.expanduser(str(path)))
        cf_path = path / "config.json"
        if not cf_path.exists():
            raise NotFound(f"no index found at {path}")
        d = json.loads(cf_path.read_text())
        config = IndexConfig.from_json_dict(d)
        return cls(config=config)

    def __init__(
        self,
        *,
        config: IndexConfig,
        calibration_sample: Sequence[float] | None = None,
        write_seeded_calibration_sample: bool = False,
        seed: int = 0,
    ) -> None:
        self._config = config
        self._base_path = Path(config.path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._namespaces = Namespaces(self._base_path)
        self._wal = WriteAheadLog(self._base_path)

        # Per-namespace caches. Lazily populated when each namespace is
        # first referenced. We don't open every namespace at boot —
        # Pinecone indexes can have many namespaces and most are cold.
        self._ns_cache: dict[str, _NamespaceHandle] = {}

        # Persisted config.json (write on init).
        cf_path = self._base_path / "config.json"
        if not cf_path.exists():
            cf_path.write_text(json.dumps(config.to_json_dict(), indent=2))

        self._seed = int(seed)
        self._default_calibration_sample: list[float] | None = None
        if calibration_sample is not None:
            self._default_calibration_sample = list(calibration_sample)
        elif write_seeded_calibration_sample:
            # Provide a synthetic sample so a fresh empty index can be
            # created without forcing the user to pre-collect vectors.
            # This produces a coarse encoder suitable for testing.
            self._default_calibration_sample = self._synthetic_sample()
        else:
            # Index.open path: read the calibration buffer from disk if
            # any namespace already has it persisted (Index.create wrote
            # it via _init_fresh). Keeps the user's calibration available
            # for subsequent upserts without re-reading per call.
            self._default_calibration_sample = self._load_calibration_from_disk()

        # Replay WAL on first open to recover any uncommitted ops.
        self._replay_wal()

    def _synthetic_sample(self) -> list[float]:
        """Deterministic fake sample so a new index can be created
        without supplying real vectors. Each dim position gets a
        sinusoid of period 7 + small jitter so the FFT peaks are
        realistic enough for the encoder to calibrate.
        """
        rng = random.Random(self._seed)
        dim = self._config.dimension
        # 200 vectors × dim floats. Trim if dim is large for boot speed.
        n_vecs = 200
        sample: list[float] = []
        for _ in range(n_vecs):
            phase = rng.uniform(0, 2 * 3.14159)
            for j in range(dim):
                val = (0.5 + 0.5 * ((j + phase) % 1.0)) * 0.2
                val += rng.gauss(0, 0.05)
                sample.append(val)
        return sample

    def _replay_wal(self) -> None:
        for entry in self._wal.iter_entries():
            op = entry.get("op")
            if op == "upsert":
                # We re-upsert from the WAL's stored vectors/values.
                ns = entry.get("namespace", Namespaces.DEFAULT)
                # Best-effort; failures here don't crash the process.
                try:
                    self._ns_for_write(ns).vs.add(
                        entry["values"], chunk_id=entry.get("chunk_id")
                    )
                except Exception as e:  # pragma: no cover
                    log.warning("WAL replay failed for entry %r: %s", entry, e)
            elif op == "delete":
                # Tombstoned via IdMap; nothing else to do for v0.1.
                pass
        # We do not auto-truncate after replay — caller decides via
        # ``flush()``.

    def _load_calibration_from_disk(self) -> list[float] | None:
        """Restore the calibration buffer written by ``Index.create``.

        Preferred: read top-level ``<path>/calibration.json``. Fallback:
        scan any namespace dir for an in-situ ``calibration.json`` from
        indexes created by older versions.
        """
        from .storage.vector_store import _CALIBRATION_FILE  # noqa: PLC2701
        # Top-level first.
        top = self._base_path / "calibration.json"
        if top.exists():
            try:
                meta = json.loads(top.read_text())
                sample = list(meta.get("sample") or [])
                if sample:
                    return sample
            except Exception:
                pass
        # Legacy per-namespace fallback.
        ns_root = self._base_path / "namespaces"
        if not ns_root.exists():
            return None
        for ns_dir in ns_root.iterdir():
            if not ns_dir.is_dir():
                continue
            calib = ns_dir / _CALIBRATION_FILE
            if calib.exists():
                try:
                    meta = json.loads(calib.read_text())
                    sample = list(meta.get("sample") or [])
                    if sample:
                        return sample
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------------
    # Per-namespace handle
    # ------------------------------------------------------------------------

    def _ns_for_write(self, namespace: str | None) -> "_NamespaceHandle":
        ns_name = Namespaces._normalize(namespace)  # noqa: SLF001
        h = self._ns_cache.get(ns_name)
        if h is None:
            ns_dir = self._namespaces.dir_for(ns_name)
            vs = VectorStore(
                ns_dir,
                self._config,
                calibration_sample=self._default_calibration_sample,
            )
            vs.ensure_loaded(self._default_calibration_sample)
            h = _NamespaceHandle(
                vs=vs,
                id_map=IdMap(ns_dir),
                meta=MetadataStore(ns_dir),
            )
            self._ns_cache[ns_name] = h
        return h

    def _ns_for_read(self, namespace: str | None) -> "_NamespaceHandle":
        # For v0.1 we treat read and write the same — the SQLite WAL
        # mode lets multiple processes read concurrently; in v0.2 we
        # add a per-namespace writer lock.
        return self._ns_for_write(namespace)

    # ------------------------------------------------------------------------
    # Public API — Pinecone-shape
    # ------------------------------------------------------------------------

    def upsert(
        self,
        vectors: Sequence[dict],
        namespace: str | None = None,
    ) -> dict:
        """Upsert a batch of vectors.

        Each ``vectors[i]`` is ``{"id": str, "values": list[float], "metadata"?: dict}``.

        Goes through :meth:`VectorStore.add_batch` — a single Rust
        ``add_embeddings`` call amortizes the Python→Rust per-vector
        cost. Measured at 5–10× faster than naive per-vector appends
        for N>=10K on the SSE inner loop.

        Returns ``{"upserted_count": int}`` per Pinecone's contract.
        """
        if not isinstance(vectors, (list, tuple)):
            raise InvalidArgument("vectors must be a list or tuple")
        if not vectors:
            return {"upserted_count": 0}

        sparse_seen = any(
            isinstance(v, dict) and v.get("sparse_values") is not None
            for v in vectors
        )
        h = self._ns_for_write(namespace)
        dim = self._config.dimension

        # Validate everything up front. Re-upserts (id collisions)
        # tombstone their prior chunk_id so the bulk Rust add always
        # sees fresh ids.
        prepared: list[tuple[dict, list[float]]] = []
        for v in vectors:
            if not isinstance(v, dict):
                raise InvalidArgument(
                    f"each vector must be a dict, got {type(v).__name__}"
                )
            id_string = v.get("id")
            values = v.get("values")
            if not isinstance(id_string, str) or not id_string:
                raise InvalidArgument("vector missing non-empty string id")
            if not isinstance(values, (list, tuple)):
                raise InvalidArgument(
                    f"vector (id={id_string!r}) values must be a list of floats"
                )
            if len(values) != dim:
                raise DimensionMismatch(
                    f"vector id={id_string!r} has dim {len(values)}, "
                    f"index dim is {dim}"
                )
            existing = h.id_map.get_chunk_id(id_string)
            if existing is not None:
                h.vs.mark_deleted([existing])
                h.id_map.mark_deleted(id_string)
                h.meta.mark_deleted(id_string)
            prepared.append((v, list(values)))

        # Single Rust add_embeddings call for the whole batch.
        fresh_values = [vals for _rec, vals in prepared]
        chunk_ids = h.vs.add_batch(fresh_values)

        # Per-vector side-effects (id_map, metadata, WAL).
        for (rec, _vals), cid in zip(prepared, chunk_ids):
            meta = dict(rec.get("metadata") or {})
            if rec.get("sparse_values") is not None:
                meta["__sparse_values__"] = rec["sparse_values"]
            h.id_map.assign(rec["id"], cid)
            h.meta.insert(rec["id"], cid, meta)
            self._wal.append(
                "upsert",
                id=rec["id"],
                chunk_id=cid,
                values=rec["values"],
                namespace=h.vs.dir.name,
            )

        if sparse_seen:
            from .sparse import warn_if_sparse_provided
            warn_if_sparse_provided(sparse_present=True, where="upsert")

        h.vs.flush()
        # After a durable flush the WAL entries have been incorporated
        # into the persistent state. We truncate so a later reopen
        # won't replay these same vectors on top of what's already
        # persisted (which would double-count chunk_ids).
        self._wal.truncate()
        return {"upserted_count": len(chunk_ids)}

    def upsert_batch(
        self,
        vectors: Sequence[dict],
        namespace: str | None = None,
    ) -> dict:
        """Public alias for :meth:`upsert`. Reserved for naming the path."""
        return self.upsert(vectors, namespace=namespace)

    def query(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        *,
        filter: dict | None = None,            # noqa: A002 — pinecone keyword
        sparse_vector: dict | None = None,
        namespace: str | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> dict:
        """Return top-k nearest matches.

        Response shape (Pinecone-shaped)::

            {
              "matches": [
                {"id": "...", "score": float,
                 "values": [...],       # if include_values
                 "metadata": {...}},    # if include_metadata
                ...
              ],
              "namespace": "default",
              "usage": {"read_units": 1},
            }

        ``sparse_vector`` accepts a Pinecone-shaped dict
        ``{"indices": [...], "values": [...]}`` for wire-compatibility.
        Currently ignored — see :mod:`spectraltm_db.sparse`.
        """
        if not isinstance(vector, (list, tuple)):
            raise InvalidArgument("vector must be a list/tuple of floats")
        if len(vector) != self._config.dimension:
            raise DimensionMismatch(
                f"query vector has dim {len(vector)}, index dim is {self._config.dimension}"
            )
        if not isinstance(top_k, int) or top_k < 0:
            raise InvalidArgument("top_k must be a non-negative int")
        if top_k == 0:
            return {
                "matches": [],
                "namespace": self._namespaces._normalize(namespace),  # noqa: SLF001
                "usage": {"read_units": 0},
            }
        if sparse_vector is not None:
            from .sparse import warn_if_sparse_provided
            warn_if_sparse_provided(sparse_present=True, where="query")

        h = self._ns_for_read(namespace)

        # 1. Filter whitelist (when present).
        cf = compile_filter(filter)
        if not cf.is_full_table_scan:
            # Try to ensure filter indexes exist for the JSON keys used
            # at the top level. Nested filter keys are skipped — we
            # only pre-index flat dotted top-level keys. Keeps the
            # heuristic fast.
            self._ensure_filter_indexes_for(filter or {}, h.meta)

        allowed: set[int] | None = None if cf.is_full_table_scan else execute_filter(h.meta, cf)

        # 2. Search the SSE index.
        hits: list[SearchHit] = h.vs.search(
            list(vector), top_k, allowed_chunk_ids=allowed,
        )

        # 3. Hydrate metadata.
        ids = [hit.chunk_id for hit in hits]
        id_to_meta = h.meta.get_many([h.id_map.get_id(c) for c in ids])
        matches = []
        for hit, chunk_id in zip(hits, ids):
            id_str = h.id_map.get_id(chunk_id)
            if id_str is None:
                # The chunk_id got tombstoned between search and hydrate.
                continue
            entry: dict = {"id": id_str, "score": float(hit.score)}
            if include_metadata:
                meta_row = id_to_meta.get(id_str)
                entry["metadata"] = (meta_row or {}).get("metadata", {})
            if include_values:
                entry["values"] = list(vector)  # placeholder; v0.2 returns true values
            matches.append(entry)

        return {
            "matches": matches,
            "namespace": h.vs.dir.name,
            "usage": {"read_units": 1},
        }

    def fetch(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> dict:
        """Fetch vectors by string id. Returns Pinecone-shaped::

            {"vectors": {"id1": {"id": "id1", "metadata": {...}, "values": [...]?}, ...},
             "namespace": "default"}
        """
        if not isinstance(ids, (list, tuple)) or not ids:
            raise InvalidArgument("ids must be a non-empty list of strings")
        for i in ids:
            if not isinstance(i, str):
                raise InvalidArgument(f"ids must all be strings; got {type(i).__name__}")
        h = self._ns_for_read(namespace)
        rows = h.meta.get_many(ids)
        out: dict[str, dict] = {}
        for id_str in ids:
            row = rows.get(id_str)
            if row is None:
                continue
            out[id_str] = {
                "id": id_str,
                "metadata": row.get("metadata", {}),
            }
        return {"vectors": out, "namespace": h.vs.dir.name}

    def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        filter: dict | None = None,            # noqa: A002
        delete_all: bool = False,
        namespace: str | None = None,
    ) -> dict:
        """Delete by ids, by filter, or the whole namespace."""
        if not any([ids, filter, delete_all]):
            raise InvalidArgument("provide one of: ids, filter, delete_all=True")
        h = self._ns_for_write(namespace)

        if delete_all:
            n_chunks_before = len(h.vs)
            # Just clear the tombstones + id_map + metadata; leave
            # on-disk files in place. v0.2 will add a wipe_files helper.
            for cid in h.id_map.all_chunk_ids():
                h.vs.mark_deleted([cid])
                h.id_map.mark_deleted_chunk(cid)
                h.meta.mark_deleted_chunk(cid)
            h.vs.flush()
            return {"deleted_count": n_chunks_before}

        chunk_ids_to_kill: set[int] = set()

        if ids:
            for id_str in ids:
                cid = h.id_map.get_chunk_id(id_str)
                if cid is not None:
                    chunk_ids_to_kill.add(cid)
                    h.id_map.mark_deleted(id_str)
                    h.meta.mark_deleted(id_str)
                    self._wal.append(
                        "delete",
                        id=id_str,
                        chunk_id=cid,
                        namespace=h.vs.dir.name,
                    )

        if filter:
            cf = compile_filter(filter)
            if not cf.is_full_table_scan:
                self._ensure_filter_indexes_for(filter or {}, h.meta)
            matches = execute_filter(h.meta, cf)
            for cid in matches:
                id_str = h.id_map.get_id(cid)
                if id_str is not None:
                    chunk_ids_to_kill.add(cid)
                    h.id_map.mark_deleted(id_str)
                    h.meta.mark_deleted(id_str)
                    self._wal.append(
                        "delete",
                        id=id_str,
                        chunk_id=cid,
                        namespace=h.vs.dir.name,
                    )

        if chunk_ids_to_kill:
            h.vs.mark_deleted(chunk_ids_to_kill)
            h.vs.flush()
            self._wal.truncate()   # see upsert: durable flush means WAL is exhausted

        return {"deleted_count": len(chunk_ids_to_kill)}

    def update(
        self,
        id: str,                              # noqa: A002 — pinecone keyword
        *,
        values: Sequence[float] | None = None,
        set_metadata: dict | None = None,
        namespace: str | None = None,
    ) -> dict:
        """Update an existing vector's values and/or metadata."""
        h = self._ns_for_write(namespace)
        existing_cid = h.id_map.get_chunk_id(id)
        if existing_cid is None:
            raise NotFound(f"vector id={id!r} not found")

        if values is not None:
            if len(values) != self._config.dimension:
                raise DimensionMismatch(
                    f"update values has dim {len(values)}, index dim is {self._config.dimension}"
                )
            h.vs.mark_deleted([existing_cid])
            h.id_map.mark_deleted(id)
            h.meta.mark_deleted(id)
            new_cid = h.vs.add(list(values))
            h.id_map.assign(id, new_cid)
            h.meta.insert(id, new_cid, set_metadata or {})
            h.vs.flush()
            self._wal.append(
                "update",
                id=id,
                chunk_id=new_cid,
                values=list(values),
                namespace=h.vs.dir.name,
            )
            self._wal.truncate()
        elif set_metadata is not None:
            ok = h.meta.update_metadata(id, set_metadata)
            if not ok:
                raise NotFound(f"vector id={id!r} not found for metadata update")

        return {"updated": True}

    def describe_index_stats(self, namespace: str | None = None) -> dict:
        """Pinecone-shaped stats dict.

        Uses the SQLite ``vectors`` table for the count rather than
        touching the SSE index, so it works on ``Index.open(...)`` paths
        where no calibration sample is in scope yet.
        """
        if namespace is not None:
            ns_list = [Namespaces._normalize(namespace)]  # noqa: SLF001
        else:
            ns_list = self._namespaces.list()
            if not ns_list:
                ns_list = [Namespaces.DEFAULT]

        total = 0
        ns_stats: dict[str, dict] = {}
        for ns_name in ns_list:
            ns_dir = self._namespaces.dir_for(ns_name)
            meta_path = ns_dir / "metadata.sqlite"
            n = 0
            if meta_path.exists():
                try:
                    # Open metadata directly — no namespace handle, no
                    # VectorStore load.
                    import sqlite3 as _sql
                    con = _sql.connect(str(meta_path))
                    try:
                        row = con.execute(
                            "SELECT COUNT(*) FROM vectors WHERE is_deleted = 0"
                        ).fetchone()
                        n = int(row[0]) if row else 0
                    finally:
                        con.close()
                except Exception:
                    n = 0
            total += n
            ns_stats[ns_name] = {"vector_count": n}
        return {
            "dimension": self._config.dimension,
            "total_vector_count": total,
            "namespaces": ns_stats,
            "compression": self._config.compression.value,
            "metric": self._config.metric.value,
        }

    # ------------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------------

    def flush(self) -> None:
        """Persist pending writes; truncate the WAL."""
        for h in self._ns_cache.values():
            h.vs.flush()
            h.id_map._conn.commit()  # noqa: SLF001
            h.meta._conn.commit()    # noqa: SLF001
        self._wal.truncate()

    def compact(self, namespace: str | None = None) -> None:
        """Clear tombstones (does not yet reclaim physical space — v0.1 limit)."""
        h = self._ns_for_write(namespace)
        h.vs.compact()

    def list_namespaces(self) -> list[str]:
        return self._namespaces.list()

    def delete_namespace(self, namespace: str) -> None:
        ns = Namespaces._normalize(namespace)  # noqa: SLF001
        # Close any cached SQLite handles for this namespace before
        # removing its directory — Windows file locking otherwise fails
        # on `shutil.rmtree` with `WinError 32`.
        handle = self._ns_cache.pop(ns, None)
        if handle is not None:
            try:
                handle.id_map.close()
            except Exception:
                pass
            try:
                handle.meta.close()
            except Exception:
                pass
        self._namespaces.remove(ns)

    # ------------------------------------------------------------------------
    # Filter heuristic
    # ------------------------------------------------------------------------

    @staticmethod
    def _ensure_filter_indexes_for(filter_dict: dict, meta: MetadataStore) -> None:
        """Best-effort: ensure JSON-expression indexes exist for top-level
        leaf keys used in the filter. Adds overhead only on first call per key.
        """
        if not isinstance(filter_dict, dict):
            return
        for key, sub in filter_dict.items():
            if key in ("$and", "$or"):
                Index._ensure_filter_indexes_for_list(sub, meta)
                continue
            try:
                meta.ensure_filter_index(key)
            except Exception:  # pragma: no cover
                pass

    @staticmethod
    def _ensure_filter_indexes_for_list(child_list, meta: MetadataStore) -> None:
        for child in child_list:
            Index._ensure_filter_indexes_for(child, meta)

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def config(self) -> IndexConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def path(self) -> Path:
        return self._base_path


# ----------------------------------------------------------------------------
# Internal per-namespace state bundle
# ----------------------------------------------------------------------------

class _NamespaceHandle:
    """Bundles the three persistent pieces for one namespace."""

    __slots__ = ("vs", "id_map", "meta")

    def __init__(self, vs: VectorStore, id_map: IdMap, meta: MetadataStore) -> None:
        self.vs = vs
        self.id_map = id_map
        self.meta = meta
