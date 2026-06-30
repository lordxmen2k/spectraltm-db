# spectraltm-db

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0.dev0-blue)](https://pypi.org/project/spectraltm-db/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Pinecone-shape local vector database built on the Spectral encoding Rust engine.**

`spectraltm-db` wraps the AVX2-accelerated [`spectraltm`](https://pypi.org/project/spectraltm/) crate with a Pinecone-compatible Python API: same `Index.create`, `Index.upsert`, `Index.query`, metadata filters, namespaces, REST server — running entirely on disk, single-process, no cloud dependency.

## Features

- **Pinecone-shape surface**: `Index.create`, `open`, `upsert`, `query`, `fetch`, `delete`, `update`, `describe_index_stats`, `compact` — wire-compatible with the `pinecone` Python SDK.
- **Spectral encoding compression**: 192 B/chunk at K=64, d=384 (vs ~1.5 KB for float32 embeddings); four named levels: `spectral_k64 / k128 / k256 / float32`.
- **11 metadata filter operators**: `$eq $ne $gt $gte $lt $lte $in $nin $exists $and $or` — compiled to `json_extract(...)` SQL via a tiny in-process SQLite store.
- **Multi-namespace** by directory partitioning; per-namespace WAL, calibration buffer, `raw.npy` source-of-truth.
- **Soft delete + real `compact()`**: re-encodes from `raw.npy` minus tombstones, recovers on-disk space — not just a flag-flip.
- **LangChain + LlamaIndex adapters** (`pip install spectraltm-db[langchain]` / `[llamaindex]`).
- **Pinecone-identical REST server** (`pip install spectraltm-db[server]`): 11 endpoints, `Api-Key` auth, multi-index registry.
- **CLI** (`python -m spectraltm_db {create,stats,upsert,query,delete,update,compact,serve,list}`).
- **AVX2 SIMD scoring** via the Rust crate; falls back to scalar on other platforms.
- **Optional `SpectralIndex.score_one` PyO3 binding** for safe single-chunk scoring at n=1 (where `select_nth_unstable_by` would panic).

## Installation

```bash
pip install spectraltm-db
```

Requires Python 3.10+. Pulls in `spectraltm>=0.1.1` (Rust SSE engine) and a pure-Python stack: `numpy`, `msgpack`, stdlib `sqlite3`.

Optional extras:

```bash
pip install "spectraltm-db[encoder]"      # sentence-transformers auto-embedding
pip install "spectraltm-db[langchain]"    # langchain_core VectorStore adapter
pip install "spectraltm-db[llamaindex]"   # llama-index VectorStore adapter
pip install "spectraltm-db[server]"       # FastAPI REST server (11 endpoints)
pip install "spectraltm-db[all]"          # everything
```

## Quick start

```python
import spectraltm_db as stm

# Calibration sample: 1,000-10,000 random vectors of dim `dimension`.
# Calibrates the SSE encoder's magnitude/phase quantization grids.
calibration = [random.gauss(0, 1) for _ in range(2000 * 384)]

idx = stm.Index.create(
    name="conversations",
    path="~/.spectraltm/conversations",
    dimension=384,
    compression="spectral_k64",     # or spectral_k128 / spectral_k256 / float32
    calibration_sample=calibration,
)

# Upsert some vectors (Pinecone-style dicts)
idx.upsert(vectors=[
    {"id": "turn_001",
     "values": [...],                 # length-384 list of floats
     "metadata": {"session_id": "abc", "role": "user"}},
    {"id": "turn_002",
     "values": [...],
     "metadata": {"session_id": "abc", "role": "assistant"}},
])

# Pinecone-shape query
results = idx.query(
    vector=[...],                              # length-384 list
    top_k=10,
    filter={"session_id": {"$eq": "abc"}},     # metadata filter
)
for m in results["matches"]:
    print(m["id"], m["score"], m["metadata"])

# Reopen after process restart
idx.flush()
idx2 = stm.Index.open("~/.spectraltm/conversations")
```

See `examples/rag_demo.py` for a runnable end-to-end example.

## Benchmarks — measured at v0.1.0.dev0 (median, top_k=10, dim=384)

Run via `python -m spectraltm_db.examples.benchmark`. Ingest uses the bulk
`add_batch` Rust path; query latency is single-threaded.

| N        | K  | dim | bytes/chunk | total MB | ms/query (median) | ms/query (p95) | ms/query (p99) | ingest (s) |
|----------|----|-----|-------------|----------|-------------------|----------------|----------------|------------|
| 10 000   | 64 | 384 | 192         | 1.83     | **7.29 ms**       | 9.24 ms        | 14.31 ms       | ~10 s      |
| 100 000  | 64 | 384 | 192         | 18.3     | **86.43 ms**      | 93.88 ms       | 95.50 ms       | ~451 s     |

Honest framing:

- **Search latency scales linearly with N at AVX2 throughput** (~12× slower
  for 10× more data). The n=1 corner case is handled by Rust `score_one`
  or NumPy fallback when the binding isn't present.
- **Ingest is dominated by per-vector rFFT encoding.** Each new vector
  pays an O(F log F) encoding pass; the bulk `add_batch` path amortizes
  Python→Rust overhead at 1.4 ms/vector but cannot skip the encoding.
  The fully-vectorized "load_codes" path (accepting pre-quantized codes)
  is on the v0.2 roadmap and is what unlocks billion-vector scale.
- **`spectraltm-db` is a lossy compressed index, not a brute-force
  cosine replacement.** See `spectraltm`'s BEIR numbers (K=8 loses ~65%
  nDCG, K=64 ~12%) for the quality curve. Your choice of K is a
  storage-budget / quality-bar tradeoff.

## When to use spectraltm-db

Use it when:

- You want a **Pinecone-shaped API** without a managed service: RAG,
  conversation memory, semantic search — anything you'd point at a
  Pinecone index and don't want cloud cost or egress.
- Your corpus fits on a single machine and on the order of N ≤ 1M
  vectors (above which encoding dominates ingest).
- You value **single-tenant, on-disk persistence**: no cluster, no
  replicas, just a folder on disk that you can `cp`, `rsync`, or
  back up however you like.

Don't use it when:

- You need real Pinecone parity in **multi-region / serverless
  autoscaling / RBAC / backups** — that's v0.2+.
- You need **sparse-dense hybrid ranking** with actual BM25/Reciprocal
  Rank Fusion — wire surface only in v0.1, scoring lands in v0.2.
- You need **billion-vector scale** — the AVX2 inner loop scales
  linearly with N; ingest cost dominates above ~1M without a pre-coded
  bulk-load path.
- You need **multi-process writers** — single-process at a time;
  multi-reader via SQLite WAL is fine; threaded HTTP works in-process.

## Project layout

```
spectraltm_db/
├── spectraltm_db/
│   ├── __init__.py        # Index / IndexConfig / errors
│   ├── __main__.py        # CLI entry point
│   ├── config.py          # IndexConfig + CompressionSpec + MetricSpec
│   ├── errors.py          # exception hierarchy
│   ├── index.py           # Index — Pinecone-shape public API
│   ├── wal.py             # minimal JSONL write-ahead log
│   ├── sparse.py          # Sparse-dense wire surface (v0.2 stub)
│   ├── server.py          # FastAPI REST server (optional)
│   ├── .github/workflows/test.yml  # CI
│   ├── storage/
│   │   ├── vector_store.py     # wraps spectraltm.SpectralIndex
│   │   ├── id_map.py           # str-id ↔ chunk_id bijection
│   │   ├── metadata_store.py   # SQLite metadata + dynamic indexes
│   │   └── namespaces.py       # directory partitioning
│   ├── retrieval/
│   │   ├── filter_compiler.py  # dict → SQL
│   │   └── filter_executor.py  # SQL → chunk_id set
│   ├── integrations/
│   │   ├── langchain.py        # VectorStore adapter
│   │   └── llamaindex.py       # VectorStore adapter
│   └── examples/
│       ├── smoke_test.py
│       ├── rag_demo.py
│       └── benchmark.py
├── tests/                # 107 tests (3 conditional skips for optional Rust binding)
├── pyproject.toml        # pure-python; depends on spectraltm from PyPI
├── README.md
├── CHANGELOG.md
├── PUBLISH.md            # how to publish to PyPI
└── LICENSE               # Apache-2.0
```

## How It Works

```
embeddings ─┐
            ├─► [encoder.calibrate]  ─► quantizer grids (mags, phases, norms)
            │
            ├─► [encoder.encode]      ─► SpectralCodes (idx, mag_q, phase_q, norm_q)
            │
            └─► [vector_store.add_batch] ─► Rust add_embeddings ─► dense (N,F) mags + phases

queries  ──┬─► [filter_compiler]   ─► SQL chunk_id whitelist (when filter present)
            │
            ├─► [encoder.encode]      ─► query SpectralCodes
            │
            └─► [SpectralIndex.search]  ─► AVX2 SIMD inner loop ─► top-K cosine scores
                                    │
                                    └─► [metadata_store.hydrate] ─► Pinecone-shaped response
```

The hot path is the Rust `SpectralIndex.search` (AVX2 SIMD on x86_64,
scalar fallback elsewhere). Filter pre-narrowing happens in SQLite
before we hand the candidate set to SSE; this matches the Pinecone
data plane, where `filter` runs before scoring. n=1 uses an optional
Rust `score_one` binding to bypass `select_nth_unstable_by` (which would
panic for `top_k >= n`).

## Development

```bash
# Editable install (Pure-Python) — pulls spectraltm from PyPI:
pip install -e ".[all]"

# Run the test suite:
pytest tests/ -v

# Reinstall the built wheel:
pip install --force-reinstall --no-deps dist/spectraltm_db-*-py3-none-any.whl

# Smoke CLI:
python -m spectraltm_db --help
```

For the optional `score_one` Rust binding path, you need a local build
of `spectraltm>=0.1.3` (workspace has the Rust patch; release to
PyPI is on the v0.2 roadmap). Until then, `VectorStore.search` falls
back to a NumPy dot product for the n=1 corner.

See [`PUBLISH.md`](PUBLISH.md) for the PyPI release flow.

## Citation

```
@misc{mckenzie2026spectraldb,
  title={spectraltm-db — Pinecone-shape local vector database on Spectral encoding},
  author={Mc Kenzie, Gerald Enrique Nelson},
  year={2026},
  month={6},
  day={29},
  howpublished={Companion to spectraltm (Sparse Spectral Encoding)},
  license={Apache-2.0},
  note={Uses spectraltm https://pypi.org/project/spectraltm/ as the underlying engine.}
}
```

**Metadata**

- **Author:** Gerald Enrique Nelson Mc Kenzie
- **Date:** 2026-06-29
- **Repository:** [github.com/lordxmen2k/sparse-spectral-encoding](https://github.com/lordxmen2k/sparse-spectral-encoding)
- **Engine:** [spectraltm](https://pypi.org/project/spectraltm/) on PyPI
- **License:** Apache License 2.0
- **Contact:** lordxmen2k@gmail.com

## License

Apache-2.0 — see [LICENSE](LICENSE).

---

Built as a Pinecone-shape wrapper around the [`spectraltm`](https://pypi.org/project/spectraltm/)
crate. The two projects share zero source code per the explicit
separation rule; the only cross-project coupling is the `spectraltm`
PyPI dependency.
