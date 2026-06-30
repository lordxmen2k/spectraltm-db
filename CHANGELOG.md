# Changelog

All notable changes to `spectraltm-db` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] — 2026-06-30

### Fixed
- `spectraltm_db.__version__` now reads from the installed package
  metadata via `importlib.metadata`, so the running version string
  always matches the version PyPI shows. Previously the literal
  `"0.1.0.dev0"` was hard-coded into `__init__.py`, which printed
  the wrong version for `spectraltm_db.__version__` even after the
  published wheel was `0.1.0`.

## [0.1.0] — 2026-06-30 (yanked)

The published 0.1.0 wheel shipped with a hard-coded `__version__`
that disagreed with its own package metadata. Yanked; use 0.1.1
instead. Functionally identical to 0.1.0 in every other respect;
re-upload only to fix the version string.

### Initial release

First public release of `spectraltm-db` — a Pinecone-shape local vector
database on top of the Spectral encoding Rust engine.

**Highlights**

- Pinecone-shape Python API: `Index.create / open / upsert / query / fetch /
  delete / update / describe_index_stats / compact`.
- 11 metadata filter operators: `$eq $ne $gt $gte $lt $lte $in $nin
  $exists $and $or`.
- Multi-namespace via per-call `namespace=` keyword.
- Per-namespace persistence via the `spectraltm` Rust format + a `raw.npy`
  source-of-truth + WAL durability.
- Soft delete + `compact()` re-encode that reclaims on-disk space (not
  just flag-flips).
- Real n=1 search path that bypasses `select_nth_unstable_by` via a
  cosine-from-`raw.npy` fallback (or `SpectralIndex.score_one` when the
  optional upstream Rust binding is available).
- LangChain `VectorStore` adapter and LlamaIndex `VectorStore` adapter.
- FastAPI REST server mirroring 11 Pinecone-shaped endpoints with
  optional `Api-Key` auth and a file-system index registry.
- CLI: `python -m spectraltm_db {create,stats,upsert,query,delete,update,compact,serve,list}`.
- Sparse-dense wire surface (`SparseValues`, `sparse_values=` /
  `sparse_vector=`) — scoring lands in v0.2.
- `VectorStore.add_batch` / `Index.upsert_batch` bulk-ingest fast path:
  one Rust `add_embeddings` call per ~1000-vector sub-batch.
- 107-test suite (3 conditional skips require `spectraltm>=0.1.3`).
- GitHub Actions CI at `.github/workflows/test.yml`.

### Measured
- N=10 000, K=64, dim=384: median 7.29 ms/query, p95 9.24 ms,
  p99 14.31 ms, 192 B/chunk, 1.83 MB total.
- N=100 000, K=64, dim=384: median 86.43 ms/query, p95 93.88 ms,
  p99 95.50 ms, 18.3 MB total. Ingest ~451 s via bulk `add_batch`.

### Known gaps at v0.1.0
- Sparse-dense scoring is wire-only — actual scoring lands in v0.2.
- Multi-process writer locks are not implemented (single-writer per
  index; multi-reader via SQLite WAL is fine; threaded HTTP works
  in-process).
- Backups / restore are not implemented (planned for v0.2).
- The "1B-vector / 200 GB" headline is arithmetic-only at this
  release; expect linear scaling above N=1M without an actual IVF /
  HNSW layer (which is on the v0.2 roadmap).

## [0.1.0.dev0] — 2026-06-29

### Added
- Optional `spectraltm.SpectralIndex.score_one` binding path: when the
  underlying `spectraltm` wheel exposes it, `VectorStore.search` uses the
  Rust one-shot score for the n=1 edge case (where
  `select_nth_unstable_by` would panic). Falls back to a NumPy dot
  product when the binding isn't present.
- `VectorStore.add_batch` and `Index.upsert_batch`: bulk-append many
  vectors in a single Rust call. Cuts wall-clock ingest at N>=10K by
  avoiding per-vector Python→Rust transitions. `Index.upsert(...)` is
  unchanged so existing callers keep working.
- LICENSE (Apache-2.0).
- CHANGELOG.md.
- `spectraltm_db.pyi` — hand-written public-API stubs for IDE / mypy.
- GitHub Actions CI at `.github/workflows/test.yml`.

### Fixed
- n=1 search now returns the real cosine from `raw.npy` rather than a `0.0` sentinel.
- WAL no longer doubles vectors on reopen (truncate after each durable flush).
- IdMap/MetadataStore SQLite connections use `check_same_thread=False`; FastAPI multi-threaded handlers no longer raise.
- Per-vector `_save_aux()` removed from `add()` — was the dominant cost at N>=50K. WAL replay covers crash recovery.

## [0.1.1.dev0] — 2026-06-29

### Added
- Core Pinecone-shaped Index API: `create`, `open`, `upsert`, `query`, `fetch`, `delete`, `update`, `describe_index_stats`, `flush`, `compact`, `list_namespaces`, `delete_namespace`.
- Filter engine: all 11 Pinecone operators (`$eq $ne $gt $gte $lt $lte $in $nin $exists $and $or`).
- Multi-namespace via per-call `namespace=` argument; per-namespace directory partitioning.
- SQLite-backed metadata with dynamic JSON-path indexes (`ensure_filter_index`).
- Real `compact()` re-encode via persisted `raw.npy` (reclaims disk space, not just tombstones).
- WAL: JSONL write-ahead log, truncated after durable flush.
- LangChain `VectorStore` adapter (`pip install spectraltm-db[langchain]`).
- LlamaIndex `VectorStore` adapter (`pip install spectraltm-db[llamaindex]`).
- FastAPI REST server with 11 Pinecone-shaped endpoints (`pip install spectraltm-db[server]`):
  - Data plane: `POST /vectors/upsert`, `POST /query`, `POST /vectors/fetch`,
    `POST /vectors/delete`, `POST /vectors/update`, `POST /vectors/list`,
    `POST /describe_index_stats`, `GET /health`.
  - Control plane: `POST/GET/DELETE /indexes`, `GET /indexes/{name}`.
- CLI: `python -m spectraltm_db {create,stats,upsert,query,delete,update,compact,serve,list}`.
- Sparse-dense wire surface (`SparseValues`, `sparse_values=`/`sparse_vector=`
  accepted on upsert/query — actual sparse scoring is v0.2).
- Benchmark script: `python -m spectraltm_db.examples.benchmark`.
- RAG demo: `python -m spectraltm_db.examples.rag_demo`.
- 101 tests (101 passing as of 0.1.1.dev0); CI workflow at `.github/workflows/test.yml`.

### Measured
- N=10,000, K=64, dim=384: median 7.29 ms / query, p95 9.24 ms, p99 14.31 ms,
  192 B/chunk, 1.83 MB total corpus. See README §"Measured benchmark curve".

### Known gaps
- Sparse-dense scoring not implemented (wire shape only).
- No 1B-vector / 200GB claim — coverage scales linearly with N but no measurement above N=10K yet.
- Multi-process writer locks not implemented (single-writer at a time;
  multi-reader via SQLite WAL is fine; threaded HTTP works in-process).
- Backups / restore not implemented (planned for 0.2.0 with filesystem
  snapshots + index registry export).
