"""End-to-end RAG-lite example with spectraltm_db.

Shows the full loop: create an index, upsert a knowledge base with
metadata, query semantically with optional metadata filtering, fetch
back the supporting documents.

We embed documents with a deterministic random projection so the example
runs without external model downloads. For a real RAG system, swap in
any encoder (sentence-transformers, OpenAI client, etc.) and pass its
output to ``idx.upsert(vectors=[{"id": ..., "values": <embedding>}])``.

Run::

    python -m spectraltm_db.examples.rag_demo
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import sys
import tempfile
from typing import List


# ---- Deterministic embedder -----------------------------------------------
#
# In a real RAG system you'd use a sentence-transformer or an API model.
# Here we hash the text into a stable seed and project into a fixed-dim
# Gaussian — good enough to demonstrate ranking & filtering behavior.

def embed(text: str, dim: int = 64, seed: int = 7) -> List[float]:
    """Deterministic pseudo-embedding: hash(text) → seed → random Gaussian."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "big") ^ seed
    rng = random.Random(derived_seed)
    return [rng.gauss(0, 1) for _ in range(dim)]


def l2_normalize(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1e-9
    return [x / norm for x in v]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---- Knowledge base -------------------------------------------------------

KB = [
    # ───── RAG / retrieval ─────
    {"id": "rag-1",
     "text": "Retrieval-augmented generation grounds LLM outputs in external knowledge to reduce hallucination.",
     "topic": "rag", "lang": "en"},
    {"id": "rag-2",
     "text": "Top-k dense retrieval returns the K most semantically similar chunks to a query embedding.",
     "topic": "rag", "lang": "en"},
    {"id": "rag-3",
     "text": "Hybrid sparse-dense search combines BM25-style lexical scores with vector similarity via Reciprocal Rank Fusion.",
     "topic": "rag", "lang": "en"},

    # ───── Memory / embeddings ─────
    {"id": "mem-1",
     "text": "Vector memory systems persist embeddings of conversation turns for long-term recall.",
     "topic": "memory", "lang": "en"},
    {"id": "mem-2",
     "text": "Cold-tier storage sacrifices recall to retain much larger corpora on disk cheaply.",
     "topic": "memory", "lang": "en"},
    {"id": "mem-3",
     "text": "Sparse Spectral Encoding compresses embeddings by quantizing the magnitude and phase of their top-K frequency bins.",
     "topic": "memory", "lang": "en"},

    # ───── Database ops ─────
    {"id": "db-1",
     "text": "Pinecone is a managed vector database with serverless autoscaling and per-vector metadata filtering.",
     "topic": "db", "lang": "en"},
    {"id": "db-2",
     "text": "SQLite WAL mode allows multiple concurrent readers alongside a single writer.",
     "topic": "db", "lang": "en"},
    {"id": "db-3",
     "text": "HNSW indexes support fast approximate nearest neighbor over millions of vectors.",
     "topic": "db", "lang": "en"},
]


# ---- Demo loop ------------------------------------------------------------

def main() -> int:
    import spectraltm_db as stm

    workdir = tempfile.mkdtemp(prefix="stm_rag_demo_")
    print(f"workdir: {workdir}\n")

    try:
        dim = 64
        idx = stm.Index.create(
            name="kb",
            path=os.path.join(workdir, "kb"),
            dimension=dim,
            compression="spectral_k64",
            calibration_sample=[rng.gauss(0, 1) for rng in [
                random.Random(i) for i in range(200)
            ] for _ in range(dim)],
        )

        # ---- Upsert knowledge base ---------------------------------------
        print("== Upserting knowledge base ==")
        vectors = [
            {"id": doc["id"],
             "values": l2_normalize(embed(doc["text"], dim=dim)),
             "metadata": {"text": doc["text"], "topic": doc["topic"],
                          "lang": doc["lang"]}}
            for doc in KB
        ]
        n = idx.upsert(vectors=vectors)["upserted_count"]
        print(f"  inserted {n} documents\n")

        # ---- Query 1: plain semantic search ---------------------------
        queries = [
            ("What is hybrid search?", None),
            ("How does cold storage work?", None),
            ("Tell me about conversation memory", None),
            ("What storage engines support ANN?", None),
        ]
        print("== Semantic search (no filter) ==")
        for q, _ in queries:
            r = idx.query(vector=l2_normalize(embed(q, dim=dim)), top_k=2)
            print(f"\n  Q: {q!r}")
            for hit in r["matches"]:
                print(f"    {hit['score']:+.3f}  {hit['id']:<8}  "
                      f"{hit['metadata']['text'][:80]}")

        # ---- Query 2: filter by topic ----------------------------------
        print("\n\n== Filtered search (topic = rag) ==")
        r = idx.query(
            vector=l2_normalize(embed("vector search retrieval", dim=dim)),
            top_k=4,
            filter={"topic": {"$eq": "rag"}},
        )
        for hit in r["matches"]:
            assert hit["metadata"]["topic"] == "rag"
            print(f"    {hit['score']:+.3f}  {hit['id']:<8}  "
                  f"{hit['metadata']['text'][:80]}")

        # ---- Fetch back the canonical doc ------------------------------
        print("\n\n== Direct fetch ==")
        r = idx.fetch(ids=["rag-3", "mem-3"])
        for id_str, body in r["vectors"].items():
            print(f"  {id_str}: {body['metadata']['text']}")

        # ---- Index health ----------------------------------------------
        print("\n\n== Index stats ==")
        print(json_dumps(idx.describe_index_stats()))

        print("\nIndex will be removed below (cleanup).")

        # Simulate post-restart reopen + query to validate persistence.
        idx.flush()
        del idx
        idx = stm.Index.open(os.path.join(workdir, "kb"))
        r = idx.query(
            vector=l2_normalize(embed("what is hybrid search?", dim=dim)),
            top_k=2,
        )
        print("\n== After reopen ==")
        for hit in r["matches"]:
            print(f"    {hit['score']:+.3f}  {hit['id']}")

        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def json_dumps(d: dict) -> str:
    import json
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    sys.exit(main())
