"""Honest benchmarking — measures ms/query at multiple corpus sizes.

Run from the repo root::

    python -m spectraltm_db.examples.benchmark --workdir /tmp/bench

Reports ms/query and total MB for N=10K and 100K at three K values
(64, 128, 256) so we publish a curve rather than a single cherry-picked
number. Results are saved to ``--workdir/results.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _run_one(workdir: Path, N: int, K: int, dim: int, n_queries: int = 100,
             seed: int = 7) -> dict:
    # Lazy imports so this script can be inspected without spectraltm installed.
    import spectraltm
    import spectraltm_db as stm

    name = f"bench_n{N}_k{K}"

    rng = random.Random(seed)
    calibration = [rng.gauss(0, 1) for _ in range(300 * dim)]

    idx = stm.Index.create(
        name=name, path=str(workdir / name),
        dimension=dim, compression=f"spectral_k{K}",
        calibration_sample=calibration,
    )

    rng = random.Random(seed + 1)
    vectors = [{"id": f"v{i}",
                "values": [rng.gauss(0, 1) for _ in range(dim)],
                "metadata": {"i": i}}
               for i in range(N)]
    # Batch the upsert in chunks of 1000 — that's 1 Rust call per 1000
    # vectors instead of 1 per vector. The bulk path is what made
    # N=100K tractable inside the time budget.
    ingest_batch = 1000
    t_ingest = time.time()
    for s in range(0, N, ingest_batch):
        idx.upsert(vectors=vectors[s:s + ingest_batch])
    ingest_seconds = time.time() - t_ingest
    idx.flush()

    queries = []
    rng = random.Random(seed + 2)
    for _ in range(n_queries):
        q = [rng.gauss(0, 1) for _ in range(dim)]
        queries.append(q)

    # Warm up.
    idx.query(vector=queries[0], top_k=10)

    # Latency over many queries.
    latencies_ms = []
    for q in queries:
        t0 = time.perf_counter_ns()
        idx.query(vector=q, top_k=10)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

    stats = idx.describe_index_stats()
    handle = next(iter(idx._ns_cache.values()))  # noqa: SLF001
    bytes_per_chunk = handle.vs.bytes_per_chunk()
    total_mb = handle.vs.total_bytes() / (1024 * 1024)

    return {
        "N": N,
        "K": K,
        "dim": dim,
        "n_queries": n_queries,
        "ms_query_mean": statistics.mean(latencies_ms),
        "ms_query_median": statistics.median(latencies_ms),
        "ms_query_p95": _percentile(latencies_ms, 95),
        "ms_query_p99": _percentile(latencies_ms, 99),
        "bytes_per_chunk": bytes_per_chunk,
        "total_mb": total_mb,
        "vector_count": stats["total_vector_count"],
        "ingest_seconds": round(ingest_seconds, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="honest benchmark for spectraltm_db")
    p.add_argument("--workdir", default="/tmp/stm_bench")
    p.add_argument("--sizes", nargs="+", type=int, default=[10000, 100000])
    p.add_argument("--k-list", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--output", default=None,
                   help="results JSON path; default <workdir>/results.json")
    args = p.parse_args(argv)

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    results = []
    for N in args.sizes:
        for K in args.k_list:
            if K >= args.dim // 2:
                continue   # clamp would kick in anyway
            print(f"\n== N={N:,} K={K} dim={args.dim} ==", flush=True)
            t0 = time.time()
            r = _run_one(workdir, N, K, args.dim, args.n_queries)
            elapsed = time.time() - t0
            r["wallclock_s"] = round(elapsed, 2)
            results.append(r)
            print(json.dumps(r, indent=2))

    out_path = Path(args.output) if args.output else workdir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
