"""Command-line interface — ``python -m spectraltm_db``.

Subcommands:

  create     Create a new index.
  stats      Show describe_index_stats.
  upsert     Insert vectors from JSONL.
  query      Run a single query and print top-k.
  delete     Delete by id, filter, or ``--all``.
  update     Update metadata/values.
  compact    Run VectorStore.compact (re-encode + reclaim).
  serve      Start the REST server (delegates to ``server.serve``).

Examples::

    python -m spectraltm_db create conversations --dimension 384 \
        --compression spectral_k64 --path ~/.spectraltm --calibration random:2000

    python -m spectraltm_db upsert conversations --jsonl turns.jsonl

    python -m spectraltm_db query conversations --vector "[0.1, 0.2, ...]" \
        --top-k 5 --filter '{"role": {"$eq": "user"}}'

    python -m spectraltm_db serve --path ~/.spectraltm --port 8080 --api-key dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m spectraltm_db",
        description="spectraltm-db — Pinecone-shape local vector database",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # common args helper
    def add_common(sp, require_dim=True):
        sp.add_argument("--path", required=True, help="index directory (e.g. ~/.spectraltm/conversations)")
        sp.add_argument("--namespace", default=None, help="namespace (default: 'default')")

    # create ---------------------------------------------------------------
    c = sub.add_parser("create", help="create a new index")
    c.add_argument("name", help="index name (subdir under --path)")
    c.add_argument("--dimension", type=int, required=True)
    c.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "dotproduct"])
    c.add_argument("--compression", default="spectral_k64",
                   choices=["spectral_k64", "spectral_k128", "spectral_k256", "float32"])
    c.add_argument("--calibration", default=None,
                   help="Calibration source. Either path to a flat .npy/.bin file "
                        "or 'random:N' for N synthetic vectors.")
    c.add_argument("--path", required=True)

    # stats ----------------------------------------------------------------
    s = sub.add_parser("stats", help="show index stats")
    s.add_argument("name")
    s.add_argument("--path", required=True)
    s.add_argument("--namespace", default=None)

    # upsert ---------------------------------------------------------------
    u = sub.add_parser("upsert", help="insert vectors from a JSONL file")
    u.add_argument("name")
    u.add_argument("--jsonl", required=True, help="path to JSONL with "
                   '{"id": "...", "values": [...], "metadata": {...}} per line')
    u.add_argument("--namespace", default=None)
    u.add_argument("--path", required=True)

    # query ----------------------------------------------------------------
    q = sub.add_parser("query", help="run a single query and print top-k")
    q.add_argument("name")
    q.add_argument("--vector", required=True, help="JSON list of floats")
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--filter", default=None, help="JSON filter dict")
    q.add_argument("--namespace", default=None)
    q.add_argument("--path", required=True)

    # delete ---------------------------------------------------------------
    d = sub.add_parser("delete", help="delete by id / filter / --all")
    d.add_argument("name")
    d.add_argument("--ids", nargs="+", default=None)
    d.add_argument("--filter", default=None)
    d.add_argument("--all", dest="delete_all", action="store_true")
    d.add_argument("--namespace", default=None)
    d.add_argument("--path", required=True)

    # update ---------------------------------------------------------------
    upd = sub.add_parser("update", help="update metadata and/or values of one vector")
    upd.add_argument("name")
    upd.add_argument("--id", required=True)
    upd.add_argument("--values", default=None)
    upd.add_argument("--set-metadata", default=None, help="JSON dict")
    upd.add_argument("--namespace", default=None)
    upd.add_argument("--path", required=True)

    # compact --------------------------------------------------------------
    co = sub.add_parser("compact", help="re-encode the index minus tombstones")
    co.add_argument("name")
    co.add_argument("--namespace", default=None)
    co.add_argument("--path", required=True)

    # serve ----------------------------------------------------------------
    sv = sub.add_parser("serve", help="run the REST server")
    sv.add_argument("--path", required=True, help="data root for all indexes")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--api-key", default=None)

    # list -----------------------------------------------------------------
    ls = sub.add_parser("list", help="list all indexes under --path")
    ls.add_argument("--path", required=True)

    return p


def _resolve_calibration(spec: str, dimension: int) -> list[float]:
    if spec is None:
        # Default: synthetic.
        import random
        rng = random.Random(0)
        return [rng.gauss(0, 1) for _ in range(200 * dimension)]
    if spec.startswith("random:"):
        n = int(spec.split(":", 1)[1])
        import random
        rng = random.Random(0)
        return [rng.gauss(0, 1) for _ in range(n * dimension)]
    p = Path(spec)
    if p.suffix == ".npy":
        import numpy as np
        arr = np.load(str(p))
        return arr.flatten().tolist()
    return json.loads(p.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    from .index import Index
    from .errors import SpectraltmDbError

    try:
        if args.cmd == "create":
            base = Path(os_path := args.path)
            idx_path = base / args.name
            sample = _resolve_calibration(args.calibration, args.dimension)
            idx = Index.create(
                name=args.name, path=str(idx_path),
                dimension=args.dimension, metric=args.metric,
                compression=args.compression,
                calibration_sample=sample,
            )
            print(json.dumps({
                "name": args.name, "dimension": args.dimension,
                "metric": args.metric, "compression": args.compression,
                "path": str(idx_path),
            }, indent=2))
            idx.flush()
            return 0

        if args.cmd == "stats":
            idx = Index.open(Path(args.path) / args.name)
            print(json.dumps(idx.describe_index_stats(args.namespace), indent=2))
            return 0

        if args.cmd == "upsert":
            idx = Index.open(Path(args.path) / args.name)
            n = 0
            batch = []
            with open(args.jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    batch.append(json.loads(line))
                    if len(batch) >= 100:
                        idx.upsert(vectors=batch, namespace=args.namespace)
                        n += len(batch)
                        batch = []
            if batch:
                idx.upsert(vectors=batch, namespace=args.namespace)
                n += len(batch)
            print(json.dumps({"upserted_count": n}))
            return 0

        if args.cmd == "query":
            idx = Index.open(Path(args.path) / args.name)
            r = idx.query(
                vector=json.loads(args.vector),
                top_k=args.top_k,
                filter=json.loads(args.filter) if args.filter else None,
                namespace=args.namespace,
            )
            print(json.dumps(r, indent=2))
            return 0

        if args.cmd == "delete":
            idx = Index.open(Path(args.path) / args.name)
            r = idx.delete(
                ids=args.ids,
                filter=json.loads(args.filter) if args.filter else None,
                delete_all=args.delete_all,
                namespace=args.namespace,
            )
            print(json.dumps(r))
            return 0

        if args.cmd == "update":
            idx = Index.open(Path(args.path) / args.name)
            r = idx.update(
                id=args.id,
                values=json.loads(args.values) if args.values else None,
                set_metadata=json.loads(args.set_metadata) if args.set_metadata else None,
                namespace=args.namespace,
            )
            print(json.dumps(r))
            return 0

        if args.cmd == "compact":
            idx = Index.open(Path(args.path) / args.name)
            handle = idx._ns_for_write(args.namespace)  # noqa: SLF001
            cleared = handle.vs.compact()
            print(json.dumps({"cleared": cleared}))
            return 0

        if args.cmd == "serve":
            from .server import serve as serve_func
            serve_func(args.path, host=args.host, port=args.port, api_key=args.api_key)
            return 0

        if args.cmd == "list":
            from .server import IndexRegistry
            reg = IndexRegistry(args.path)
            print(json.dumps({"indexes": reg.list_indexes()}, indent=2))
            return 0

    except SpectraltmDbError as e:
        print(json.dumps({"error": {"type": type(e).__name__, "message": str(e)}}))
        return 1

    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
