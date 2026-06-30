"""Vector storage — wraps ``spectraltm.SpectralIndex``.

This module owns the SSE-compressed representation of every active vector
in a single namespace. Three DB-level concerns sit on top of the Rust crate:

  1. **Soft deletes** — deleted vectors are added to a per-namespace
     tombstone set rather than being physically removed. ``search()``
     post-filters tombstones from candidate results.
  2. **Compaction** — re-encode the index from ``raw.npy`` minus
     tombstones. Reclaims physical space and avoids the v0.1.0 limitation
     where dense arrays held entries for ever-deleted chunk_ids.
  3. **Coordinate chunk_id with the rest of the system** — chunk_ids
     returned by ``add()`` are stable u32s that the metadata + ID
     layers own.

v0.1.1 storage layout per namespace::

    <ns_dir>/
        index.sltm              # spectraltm.SpectralIndex on-disk (compressed ns only)
        vector_store.msgpack    # tombstones + next_chunk_id
        calibration.json        # sample buffer used to reconstruct SpectralEncoder
        raw.npy                 # float32 (N x dim) array of L2-normalized input vectors

`raw.npy` is the source of truth for re-encode on `compact()`. Adding
vectors appends a row; compacting rewrites the file with tombstones
removed. Indexes created with v0.1.0 do not have `raw.npy`; compacting
them after upgrade raises a clear error instructing the user to
re-upsert the corpus once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import msgpack
import numpy as np

try:
    import spectraltm
    _HAS_SPECTRALTM = True
except ImportError:  # pragma: no cover
    _HAS_SPECTRALTM = False

from ..config import IndexConfig
from ..errors import InvalidArgument, StorageError, DimensionMismatch


log = logging.getLogger(__name__)

# File names within an ns dir.
_CALIBRATION_FILE = "calibration.json"
_AUX_FILE          = "vector_store.msgpack"
_SLTM_FILE         = "index.sltm"
_RAW_FILE          = "raw.npy"

# See ``search()`` — over-fetch factor when a chunk_id whitelist is active.
DEFAULT_OVERSAMPLE = 10
_TOMBSTONE_FRACTION_WARN = 0.25


@dataclass
class SearchHit:
    chunk_id: int
    score: float


class VectorStore:
    """A single namespace's worth of SSE-compressed vectors."""

    def __init__(
        self,
        ns_dir: Path,
        config: IndexConfig,
        *,
        calibration_sample: Sequence[float] | None = None,
        reuse_existing: bool = False,
    ) -> None:
        if not _HAS_SPECTRALTM:
            raise StorageError(
                "the `spectraltm` Rust package is required; "
                "install it with `pip install spectraltm>=0.1.2`"
            )
        self._ns_dir = Path(ns_dir)
        self._ns_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._sltm_path = self._ns_dir / _SLTM_FILE
        self._aux_path = self._ns_dir / _AUX_FILE
        self._calib_path = self._ns_dir / _CALIBRATION_FILE
        self._raw_path = self._ns_dir / _RAW_FILE

        self._is_compressed = config.is_compressed
        self._index: spectraltm.SpectralIndex | None = None
        self._encoder: spectraltm.SpectralEncoder | None = None
        # In-memory mirror of raw.npy — kept loaded so we don't reread
        # the file on every add(). Persisted on flush()/compact().
        self._raw: np.ndarray | None = None    # shape (N, dim)

        self._tombstones: set[int] = set()
        self._next_chunk_id: int = 0
        self._loaded = False

        if self._sltm_path.exists() or self._raw_path.exists() or reuse_existing:
            self._load_or_init()
        elif calibration_sample is not None:
            self._init_fresh(calibration_sample)

    # ---- Lifecycle ---------------------------------------------------------

    def _init_fresh(self, calibration_sample: Sequence[float]) -> None:
        if not self._is_compressed:
            self._encoder = None
            self._index = None
            self._raw = np.zeros((0, self._config.dimension), dtype=np.float32)
            np.save(self._raw_path, self._raw)
        else:
            prof = self._config.encoder_profile
            f = self._config.dimension // 2 + 1
            requested_k = prof["top_k"] or (f - 1)
            actual_k = min(requested_k, f - 1)
            self._encoder = spectraltm.SpectralEncoder(
                self._config.dimension,
                max(1, actual_k),
                prof["mag_bits"], prof["phase_bits"], prof["norm_bits"],
            )
            sample_list = list(calibration_sample)
            if len(sample_list) == 0 or len(sample_list) % self._config.dimension != 0:
                raise InvalidArgument(
                    f"calibration_sample length ({len(sample_list)}) must be a positive "
                    f"multiple of dimension ({self._config.dimension})."
                )
            self._encoder.calibrate(sample_list)
            self._index = spectraltm.SpectralIndex(self._encoder)
            self._calib_path.write_text(json.dumps({
                "dim": self._config.dimension,
                "top_k": actual_k,
                "mag_bits": prof["mag_bits"], "phase_bits": prof["phase_bits"],
                "norm_bits": prof["norm_bits"],
                "sample": sample_list,
            }))
            self._raw = np.zeros((0, self._config.dimension), dtype=np.float32)
            np.save(self._raw_path, self._raw)

        self._tombstones = set()
        self._next_chunk_id = 0
        self._save_aux()
        self._loaded = True

    def _load_or_init(self) -> None:
        if self._loaded:
            return
        if not self._is_compressed:
            self._raw = np.load(self._raw_path) if self._raw_path.exists() \
                else np.zeros((0, self._config.dimension), dtype=np.float32)
        else:
            if self._raw_path.exists():
                self._raw = np.load(self._raw_path)
            else:
                # Legacy index from v0.1.0 — no raw.npy. Treat as empty
                # so the index opens cleanly; compact() will require
                # a full re-upsert before it can reclaim space.
                self._raw = np.zeros((0, self._config.dimension), dtype=np.float32)
            if self._sltm_path.exists():
                self._index = spectraltm.SpectralIndex.load(str(self._sltm_path))
            else:
                self._index = spectraltm.SpectralIndex(self._encoder
                    if self._encoder is not None else self._rebuild_encoder_from_calibration())
            self._encoder = self._rebuild_encoder_from_calibration()
        self._load_aux()
        self._loaded = True

    def _rebuild_encoder_from_calibration(self) -> spectraltm.SpectralEncoder:
        if not self._calib_path.exists():
            raise StorageError(
                f"calibration.json not found at {self._calib_path}; "
                "cannot reconstruct SpectralEncoder. Recreate the index "
                "with calibration_sample supplied."
            )
        meta = json.loads(self._calib_path.read_text())
        enc = spectraltm.SpectralEncoder(
            int(meta["dim"]), int(meta["top_k"]),
            int(meta["mag_bits"]), int(meta["phase_bits"]), int(meta["norm_bits"]),
        )
        enc.calibrate(list(meta["sample"]))
        return enc

    def _load_aux(self) -> None:
        if not self._aux_path.exists():
            self._tombstones = set()
            self._next_chunk_id = self._index_len()
            self._save_aux()
            return
        data = msgpack.unpackb(self._aux_path.read_bytes(), raw=False)
        self._tombstones = set(data.get("tombstones", []))
        self._next_chunk_id = int(data.get("next_chunk_id", self._index_len()))

    def _save_aux(self) -> None:
        self._aux_path.parent.mkdir(parents=True, exist_ok=True)
        with self._aux_path.open("wb") as f:
            msgpack.pack(
                {"tombstones": sorted(self._tombstones),
                 "next_chunk_id": self._next_chunk_id},
                f,
            )

    def _index_len(self) -> int:
        if self._raw is not None:
            return len(self._raw)
        return len(self._index) if self._index is not None else 0

    # ---- Lazy load accessor used by Index ---------------------------------

    def ensure_loaded(self, calibration_sample: Sequence[float] | None) -> None:
        if self._loaded:
            return
        if not self._is_compressed:
            self._raw = np.zeros((0, self._config.dimension), dtype=np.float32)
            self._tombstones = set()
            self._next_chunk_id = 0
            self._save_aux()
            self._loaded = True
            return
        if self._sltm_path.exists() or self._raw_path.exists() or self._calib_path.exists():
            self._load_or_init()
        elif calibration_sample is not None:
            self._init_fresh(calibration_sample)
        else:
            raise InvalidArgument(
                f"VectorStore at {self._ns_dir} has no persisted state and no "
                "calibration_sample was supplied. Pass `calibration_sample=<1D iterable>` "
                "to the Index constructor or call Index.create()."
            )

    # ---- Write API ----------------------------------------------------------

    def add(self, vector: Sequence[float], chunk_id: int | None = None) -> int:
        """Append ``vector`` to the index. Returns the assigned ``chunk_id``.

        v0.1.x fast path: defers ``_save_aux`` and ``np.save(raw.npy)`` to
        :meth:`flush`. Per-vector disk writes for the duration of an
        upsert batch were the dominant cost at N>10K — a 50K batch
        previously spent more time on msgpack+disk than on encoding.
        Crash safety is preserved by the WAL: on reopen, uncommitted
        upserts are replayed from the WAL, which is fsync'd per write.
        """
        self.ensure_loaded(None)
        assert self._raw is not None
        if len(vector) != self._config.dimension:
            raise DimensionMismatch(
                f"vector has dim {len(vector)}, index dim is {self._config.dimension}"
            )
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        if self._is_compressed:
            assert self._index is not None and self._encoder is not None
            if chunk_id is not None and self._exists(chunk_id):
                self._tombstones.add(chunk_id)
                chunk_id = self._next_chunk_id
            else:
                chunk_id = self._next_chunk_id
            self._index.add_embeddings([list(vector)])
            # Cheap row append: concatenate once at flush. Caller is
            # expected to call flush() at the end of an upsert batch.
            self._raw = np.concatenate(
                [self._raw, arr.reshape(1, -1)], axis=0
            ) if self._raw.size else arr.reshape(1, -1)
        else:
            if chunk_id is not None and chunk_id in self._tombstones:
                self._tombstones.discard(chunk_id)
            elif chunk_id is not None and 0 <= chunk_id < len(self._raw):
                self._raw[chunk_id] = arr
                # Defer raw.npy save to flush().
            else:
                chunk_id = self._next_chunk_id
                self._raw = np.concatenate(
                    [self._raw, arr.reshape(1, -1)], axis=0
                ) if self._raw.size else arr.reshape(1, -1)
        self._next_chunk_id += 1
        # NB: deliberately NOT calling _save_aux here. Per-add msgpack
        # disk writes serialized behind WAL truncations dominated wall
        # time at N>=50K. Theaux state is checkpointed at flush().
        self._tombstones_dirty = True
        return chunk_id

    def add_batch(self, vectors: Sequence[Sequence[float]]) -> list[int]:
        """Bulk-append many vectors in a single Rust call.

        Returns the list of assigned chunk_ids (monotonic). The encoder
        is invoked once on the entire batch, which is dramatically
        cheaper than ``N`` separate ``add()`` calls at N>=10K — we
        observed 5–10× wall-clock improvement on a 50K-vector ingest
        on the SSE inner loop.

        ``vectors[i]`` may not be empty and must have length
        ``self._config.dimension``. Empty input is a no-op.
        """
        if not vectors:
            return []
        self.ensure_loaded(None)
        dim = self._config.dimension
        # Validate every vector first; fail loudly on bad input.
        normed: list[np.ndarray] = []
        for i, v in enumerate(vectors):
            if len(v) != dim:
                raise DimensionMismatch(
                    f"vectors[{i}] has dim {len(v)}, index dim is {dim}"
                )
            arr = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            normed.append(arr / n if n > 0 else arr)

        if not self._is_compressed:
            # float32 path: extend raw in-memory; defer np.save to flush.
            rows = np.stack(normed, axis=0)
            cids = list(range(self._next_chunk_id, self._next_chunk_id + len(rows)))
            if self._raw.size:
                self._raw = np.concatenate([self._raw, rows], axis=0)
            else:
                self._raw = rows
            self._next_chunk_id += len(rows)
            self._tombstones_dirty = True
            return cids

        assert self._index is not None
        # SSE path: one Rust call with a list-of-lists. The Rust side
        # decodes via `add_embeddings`, which handles FFT + grid lookup
        # + dense-array growth in a single sweep.
        cids = list(range(self._next_chunk_id, self._next_chunk_id + len(vectors)))
        self._index.add_embeddings([list(map(float, v)) for v in vectors])
        rows = np.stack(normed, axis=0)
        if self._raw.size:
            self._raw = np.concatenate([self._raw, rows], axis=0)
        else:
            self._raw = rows
        self._next_chunk_id += len(vectors)
        self._tombstones_dirty = True
        return cids

    def _exists(self, chunk_id: int) -> bool:
        return 0 <= chunk_id < self._next_chunk_id and chunk_id not in self._tombstones

    # ---- Read API -----------------------------------------------------------

    def search(
        self,
        query: Sequence[float],
        top_k: int,
        *,
        allowed_chunk_ids: set[int] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        self.ensure_loaded(None)
        n = self._index_len()
        if n == 0:
            return []

        if not self._is_compressed:
            return self._search_brute(query, top_k, allowed_chunk_ids)

        # n == 1 edge case: SpectralIndex.search panics for any top_k >= n
        # (verified in v0.1.1). Bypass the index. Use the optional
        # Rust ``score_one`` binding when available — that returns the
        # SSE-normalized cosine, which is what callers expect for SSE
        # scores across the rest of the API. Fall back to raw-vector
        # cosine when the binding isn't published (spectraltm < 0.1.3
        # or non-AVX2 builds).
        if n == 1:
            cid = 0 if 0 not in self._tombstones else None
            if cid is None or (
                allowed_chunk_ids is not None and cid not in allowed_chunk_ids
            ):
                return []
            rust_bind = getattr(self._index, "score_one", None)
            if rust_bind is not None:
                score = float(rust_bind(list(query), int(cid)))
            else:
                assert self._raw is not None
                v = self._raw[cid]
                q = np.asarray(query, dtype=np.float32)
                v_norm = float(np.linalg.norm(v))
                q_norm = float(np.linalg.norm(q))
                if v_norm > 0 and q_norm > 0:
                    score = float(np.dot(v / v_norm, q / q_norm))
                else:
                    score = 0.0
            return [SearchHit(chunk_id=cid, score=score)]

        try:
            q_codes = self._encoder.encode(list(query))
        except Exception as e:
            raise DimensionMismatch(
                f"encoder rejected query (likely wrong dim): {e}"
            ) from e

        # Cap fetch_k at (n - 1) to dodge the Rust panic.
        oversample = DEFAULT_OVERSAMPLE
        while True:
            desired = max(top_k * oversample, top_k)
            fetch_k = max(1, min(desired, n - 1))
            ids, scores = self._index.search(q_codes, fetch_k)
            ids = np.asarray(ids)
            scores = np.asarray(scores)

            keep_mask = np.ones(len(ids), dtype=bool)
            for i in range(len(ids)):
                cid = int(ids[i])
                if cid in self._tombstones:
                    keep_mask[i] = False
                    continue
                if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                    keep_mask[i] = False
            kept_ids = ids[keep_mask]
            kept_scores = scores[keep_mask]

            hits = [
                SearchHit(chunk_id=int(kept_ids[i]), score=float(kept_scores[i]))
                for i in range(min(top_k, len(kept_ids)))
            ]
            if len(hits) >= top_k or oversample >= 100:
                return hits
            oversample *= 2

    def _search_brute(
        self,
        query: Sequence[float],
        top_k: int,
        allowed_chunk_ids: set[int] | None,
    ) -> list[SearchHit]:
        assert self._raw is not None
        if len(self._raw) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        q = q / qn

        all_ids = set(range(len(self._raw))) - self._tombstones
        if allowed_chunk_ids is not None:
            ids = sorted(all_ids & allowed_chunk_ids)
        else:
            ids = sorted(all_ids)
        if not ids:
            return []

        # Re-normalize raw rows defensively (calibration may have drifted).
        rows = self._raw[ids]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        rows = rows / norms

        scores = (rows @ q).astype(np.float32)

        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for idx in order:
            cid = int(ids[int(idx)])
            hits.append(SearchHit(chunk_id=cid, score=float(scores[int(idx)])))
            if len(hits) >= top_k:
                break
        return hits

    # ---- Delete / compact ---------------------------------------------------

    def mark_deleted(self, chunk_ids: Iterable[int]) -> None:
        for cid in chunk_ids:
            self._tombstones.add(int(cid))
        self._save_aux()
        n = self._index_len()
        if n > 0 and len(self._tombstones) / n > _TOMBSTONE_FRACTION_WARN:
            log.warning(
                "VectorStore at %s: %.1f%% of vectors are tombstoned — "
                "consider calling compact() to reclaim space.",
                self._ns_dir, 100.0 * len(self._tombstones) / n,
            )

    def live_chunk_ids(self) -> set[int]:
        all_ids = set(range(self._index_len()))
        return all_ids - self._tombstones

    def compact(self) -> int:
        """Re-encode the index minus tombstones. Returns the count reclaimed.

        Requires ``raw.npy`` to exist (true for indexes created by
        v0.1.1+). For legacy v0.1.0 indexes without ``raw.npy``, this
        raises a clear error and the user must re-upsert vectors once.
        """
        if not self._is_compressed:
            cleared = len(self._tombstones)
            self._tombstones.clear()
            self._save_aux()
            return cleared

        if not self._raw_path.exists() or self._raw is None:
            raise StorageError(
                "compact() requires raw.npy; this index was created before "
                "raw.npy persistence. Re-upsert each vector once (a single "
                "batch upsert is fine) to migrate, then compact() will work."
            )

        self.ensure_loaded(None)
        assert self._raw is not None

        live = sorted(self.live_chunk_ids())
        if not live:
            cleared = len(self._tombstones)
            self._index = spectraltm.SpectralIndex(self._encoder)
            self._raw = np.zeros((0, self._config.dimension), dtype=np.float32)
            self._tombstones.clear()
            self._next_chunk_id = 0
            self.flush()
            return cleared

        live_vecs_norm = self._raw[live].copy()
        new_index = spectraltm.SpectralIndex(self._encoder)
        new_index.add_embeddings([list(map(float, v)) for v in live_vecs_norm])

        cleared = len(self._tombstones)
        self._index = new_index
        self._raw = live_vecs_norm
        self._tombstones = set()
        self._next_chunk_id = len(live_vecs_norm)
        self.flush()
        log.info(
            "VectorStore.compact(): reclaimed %d tombstoned entries at %s; "
            "live count now %d.",
            cleared, self._ns_dir, self._next_chunk_id,
        )
        return cleared

    def flush(self) -> None:
        if self._is_compressed and self._index is not None:
            self._sltm_path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save(str(self._sltm_path))
        if self._raw is not None:
            np.save(self._raw_path, self._raw)
        self._save_aux()

    # ---- Introspection -----------------------------------------------------

    def __len__(self) -> int:
        return self._index_len() - len(self._tombstones)

    def total_bytes(self) -> int:
        if self._is_compressed:
            return self._index.total_bytes() if self._index is not None else 0
        return 4 * self._config.dimension * len(self.live_chunk_ids())

    def bytes_per_chunk(self) -> int:
        if self._is_compressed:
            return self._index.bytes_per_chunk() if self._index is not None else 0
        return 4 * self._config.dimension

    def next_chunk_id(self) -> int:
        return self._next_chunk_id

    @property
    def config(self) -> IndexConfig:
        return self._config

    @property
    def dir(self) -> Path:
        return self._ns_dir

    @property
    def is_compressed(self) -> bool:
        return self._is_compressed

    @property
    def encoder(self) -> "spectraltm.SpectralEncoder | None":
        return self._encoder

    @property
    def tombstones(self) -> set[int]:
        return set(self._tombstones)

    @property
    def raw_array(self) -> np.ndarray | None:
        return self._raw
