"""Smoke test — verify the full spectraltm_db API round-trips on a tiny index."""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path

# Make the package importable from the workspace source tree.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spectraltm_db"))
sys.path.insert(0, str(ROOT))

import spectraltm_db as stm  # noqa: E402


def main() -> int:
    random.seed(0)
    dim = 32

    workdir = Path(tempfile.mkdtemp(prefix="stm_smoke_"))
    print(f"workdir: {workdir}")
    try:
        # Calibration sample: 200 random vectors, expressed as a flat list.
        sample = [random.gauss(0, 1) for _ in range(200 * dim)]

        idx = stm.Index.create(
            name="smoke",
            path=str(workdir),
            dimension=dim,
            compression="spectral_k64",
            calibration_sample=sample,
        )
        print(f"Created index at {idx.path}")

        # Upsert 50 vectors.
        vectors = [
            {
                "id": f"t{i}",
                "values": [random.gauss(0, 1) for _ in range(dim)],
                "metadata": {"topic": f"t{i % 3}", "seq": i},
            }
            for i in range(50)
        ]
        r = idx.upsert(vectors=vectors)
        print(f"upsert -> {r}")

        # Plain top-5.
        q = [random.gauss(0, 1) for _ in range(dim)]
        r = idx.query(vector=q, top_k=5)
        print(f"query top-5: {[(m['id'], round(m['score'], 4)) for m in r['matches']]}")

        # With filter.
        r = idx.query(vector=q, top_k=5, filter={"topic": {"$eq": "t1"}})
        print(f"filter topic=t1: {[(m['id'], m['metadata']) for m in r['matches']]}")

        # Re-upsert (id collision).
        new_vec = {"id": "t0", "values": [random.gauss(0, 1) for _ in range(dim)],
                   "metadata": {"topic": "rewritten", "seq": 0}}
        r = idx.upsert(vectors=[new_vec])
        print(f"re-upsert t0 -> {r}")

        r = idx.fetch(ids=["t0", "t1", "t99"])
        print(f"fetch t0/t1/t99 -> {[(k, v.get('metadata')) for k, v in r['vectors'].items()]}")

        r = idx.delete(ids=["t1"])
        print(f"delete t1 -> {r}")

        r = idx.delete(filter={"topic": {"$eq": "rewritten"}})
        print(f"delete filter topic=rewritten -> {r}")

        print(f"stats: {idx.describe_index_stats()}")

        # Persist + reopen.
        idx.flush()
        del idx
        idx2 = stm.Index.open(str(workdir))
        r = idx2.query(vector=q, top_k=3)
        print(f"after reopen, top-3: {[(m['id'], round(m['score'], 4)) for m in r['matches']]}")
        print(f"after reopen, stats: {idx2.describe_index_stats()}")

        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
