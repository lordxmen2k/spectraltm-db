"""LlamaIndex VectorStore adapter for spectraltm_db.

Implements the surface a LlamaIndex pipeline needs to use a
``SpectralVectorStore`` as its backing vector store.

Install the optional dependency::

    pip install "spectraltm-db[llamaindex]"

Quickstart::

    >>> from llama_index.core.vector_stores.types import VectorStoreQuery
    >>> from spectraltm_db.integrations.llamaindex import SpectralVectorStore
    >>> vs = SpectralVectorStore(
    ...     index_name="conversations",
    ...     path="~/.spectraltm/conversations",
    ...     dimension=384,
    ...     compression="spectral_k64",
    ... )
    >>> vs.add([NodeWithEmbedding(node=TextNode(text="hello"),
    ...                           embedding=[...] )])
    >>> q = VectorStoreQuery(query_embedding=[...], similarity_top_k=5)
    >>> rsp = vs.query(q)
    >>> for n, sid in zip(rsp.nodes, rsp.ids):
    ...     print(sid, n.text)

Note: this is intentionally a thin layer over :mod:`spectraltm_db.index`.
It exposes ``add``/``delete``/``query`` and a ``from_llama_nodes`` helper —
the methods LlamaIndex's ``VectorStoreIndex`` actually invokes. The full
LlamaIndex VectorStore protocol is large and version-sensitive; we wrap
what's needed for the ``VectorStoreIndex.from_vector_store(...)``
construction path. If you hit a missing method, file an issue with the
specific call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

from ..errors import InvalidArgument
from ..index import Index


@dataclass
class _Node:
    """Minimal node structure so we don't require llama-index at import."""

    id_: Optional[str] = None
    text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class _QueryResponse:
    """Minimal VectorStoreQueryResponse — matches the surface LlamaIndex uses."""

    nodes: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    similarities: list = field(default_factory=list)


class SpectralVectorStore:
    """Adapter bridging spectraltm_db.Index ↔ LlamaIndex's VectorStore surface."""

    def __init__(
        self,
        index_name: str,
        path: str,
        dimension: int,
        *,
        compression: str = "spectral_k64",
        metric: str = "cosine",
        namespace: str = "default",
        calibration_sample: Optional[Sequence[float]] = None,
        open_existing: bool = False,
    ) -> None:
        import os
        full = os.path.expanduser(path)
        if open_existing:
            self._index = Index.open(full)
        else:
            sample = calibration_sample or self._synthetic_sample(dimension, index_name)
            self._index = Index.create(
                name=index_name,
                path=full,
                dimension=dimension,
                metric=metric,
                compression=compression,
                calibration_sample=sample,
            )
        self._dimension = dimension
        self._namespace = namespace

    @staticmethod
    def _synthetic_sample(dim: int, seed: str) -> List[float]:
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

    # ---- The LlamaIndex VectorStore surface -----------------------------

    def add(self, nodes: Iterable[Any], **kwargs: Any) -> List[str]:
        """Add a batch of ``Node`` or ``NodeWithEmbedding`` objects.

        Each node must expose ``node_id`` (or ``id_``) and either
        ``embedding`` directly or be a ``NodeWithEmbedding`` wrapper.
        Optional: ``metadata``, ``text`` (stored under metadata['text']).
        """
        vectors = []
        ids = []
        for n in nodes:
            # Unwrap NodeWithEmbedding if needed.
            inner = getattr(n, "node", n)
            nid = getattr(inner, "node_id", None) or getattr(inner, "id_", None) \
                or getattr(n, "node_id", None) or getattr(n, "id_", None)
            if nid is None:
                raise InvalidArgument(
                    f"node {n!r} has neither node_id nor id_"
                )
            emb = getattr(n, "embedding", None) or getattr(inner, "embedding", None)
            if emb is None:
                raise InvalidArgument(
                    f"node {nid!r} has no embedding — "
                    "wrap it in NodeWithEmbedding first"
                )
            meta = dict(getattr(inner, "metadata", {}) or {})
            text = getattr(inner, "text", None) or getattr(inner, "get_content", lambda: "")()
            if isinstance(text, str) and text:
                meta.setdefault("text", text)
            vectors.append({
                "id": str(nid),
                "values": list(map(float, emb)),
                "metadata": meta,
            })
            ids.append(str(nid))
        self._index.upsert(vectors=vectors, namespace=self._namespace)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self._index.delete(ids=[ref_doc_id], namespace=self._namespace)

    def query(self, query: Any, **kwargs: Any) -> _QueryResponse:
        """Run a VectorStoreQuery.

        Accepts either a LlamaIndex ``VectorStoreQuery`` instance or a
        duck-typed object with ``query_embedding`` and ``similarity_top_k``.
        """
        q_emb = getattr(query, "query_embedding", None)
        if q_emb is None:
            q_emb = getattr(query, "embedding", None)
        if q_emb is None:
            raise InvalidArgument("query must expose query_embedding / embedding")
        k = int(getattr(query, "similarity_top_k", 4) or 4)
        filt = getattr(query, "filters", None)
        # Best-effort filter conversion: llama-index uses ``MetadataFilters``
        # which is a tree of operator objects. We don't translate those —
        # users who want filters should use the lower-level Index API.
        if filt is not None:
            try:
                filt_dict = self._convert_filters(filt)
            except Exception:
                filt_dict = None
        else:
            filt_dict = None

        r = self._index.query(
            vector=list(q_emb),
            top_k=k,
            filter=filt_dict,
            namespace=self._namespace,
            include_metadata=True,
        )
        nodes = []
        ids = []
        sims = []
        # Try to import TextNode lazily. If unavailable we return a plain
        # _Node shim.
        try:
            from llama_index.core.schema import TextNode
            NodeCls = TextNode
        except ImportError:
            NodeCls = _Node

        for m in r["matches"]:
            meta = m.get("metadata") or {}
            nodes.append(NodeCls(id_=m["id"], text=meta.get("text", ""), metadata=meta))
            ids.append(m["id"])
            sims.append(float(m["score"]))
        return _QueryResponse(nodes=nodes, ids=ids, similarities=sims)

    @staticmethod
    def _convert_filters(filt: Any) -> Optional[dict]:
        """Best-effort filter conversion. Returns ``None`` if the structure
        isn't recognized; the caller can retry without a filter.
        """
        try:
            conditions = getattr(filt, "filters", None)
            if not conditions:
                return None
            out: dict = {}
            for c in conditions:
                key = getattr(c, "key", None)
                op = getattr(c, "operator", None)
                val = getattr(c, "value", None)
                if key is None or val is None:
                    return None
                op_str = str(op).lower()
                op_token = {
                    "eq": "$eq", "ne": "$ne",
                    "gt": "$gt", "gte": "$gte",
                    "lt": "$lt", "lte": "$lte",
                    "in": "$in", "nin": "$nin",
                }.get(op_str, None)
                if op_token is None:
                    return None
                out[key] = {op_token: val}
            return out
        except Exception:
            return None

    # ---- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._index.flush()
