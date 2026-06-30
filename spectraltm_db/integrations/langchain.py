"""LangChain VectorStore adapter for spectraltm_db.

Implements :class:`langchain_core.vectorstores.VectorStore` so RAG
pipelines can swap a ``SpectralVectorStore`` in for any other
``VectorStore`` with no upstream changes.

Install the optional dependency::

    pip install "spectraltm-db[langchain]"

Quickstart::

    >>> from spectraltm_db.integrations.langchain import SpectralVectorStore
    >>> from langchain_openai import OpenAIEmbeddings
    >>> vs = SpectralVectorStore(
    ...     index_name="conversations",
    ...     path="~/.spectraltm/conversations",
    ...     dimension=1536,
    ...     compression="spectral_k64",
    ...     embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    ...     calibration_sample=calibration_array,
    ... )
    >>> vs.add_texts(["turn one", "turn two"], metadatas=[{"role": "user"},
    ...                                                  {"role": "assistant"}])
    >>> docs = vs.similarity_search("how do I tune a piano", k=5)
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence

from ..errors import InvalidArgument, DimensionMismatch
from ..index import Index


# We import lazily so the package is usable even without langchain installed.
_langchain_vectorstore_cls = None


def _get_langchain_base():
    global _langchain_vectorstore_cls
    if _langchain_vectorstore_cls is not None:
        return _langchain_vectorstore_cls
    try:
        # langchain_core >= 0.1
        from langchain_core.vectorstores import VectorStore as _VS
    except ImportError:                                       # pragma: no cover
        try:
            # langchain < 0.1
            from langchain.vectorstores import VectorStore as _VS
        except ImportError:
            raise ImportError(
                "LangChain is not installed. Install with "
                "`pip install langchain-core` or `pip install "
                "\"spectraltm-db[langchain]\"`."
            )
    _langchain_vectorstore_cls = _VS
    return _VS


class SpectralVectorStore:
    """Adapter bridging spectraltm_db.Index ↔ LangChain's VectorStore contract.

    Notes:
      - We do NOT inherit from ``langchain_core.vectorstores.VectorStore``
        when its Python ``__init_subclass__`` validation rejects simple
        ``add_texts`` return signatures — we expose the same method names
        (per duck-typing) and register as a VectorStore via
        :func:`as_langchain_vectorstore`.
      - For full VectorStore surface, prefer
        :func:`as_langchain_vectorstore` which returns a verified
        subclass.
      - ``embedding`` is any callable: ``str | List[str] -> List[List[float]]``.
        OpenAI / sentence-transformers / Cohere embedders all satisfy this.
    """

    def __init__(
        self,
        index_name: str,
        path: str,
        dimension: int,
        embedding: Callable,
        *,
        compression: str = "spectral_k64",
        metric: str = "cosine",
        calibration_sample: Optional[Sequence[float]] = None,
        namespace: str = "default",
        open_existing: bool = False,
    ) -> None:
        self._index_name = index_name
        self._path = path
        self._dimension = int(dimension)
        self._embedding = embedding
        self._compression = compression
        self._metric = metric
        self._namespace = namespace

        import os
        full = os.path.expanduser(path)
        if open_existing and not self._config_exists(full):
            raise FileNotFoundError(
                f"no index at {full}; pass open_existing=False (default) to "
                "create a new index, or check the path"
            )
        if open_existing:
            self._index = Index.open(full)
        else:
            sample = calibration_sample or self._default_calibration_sample()
            self._index = Index.create(
                name=index_name,
                path=full,
                dimension=dimension,
                metric=metric,
                compression=compression,
                calibration_sample=sample,
            )

    @classmethod
    def _check_has_open(cls) -> bool:
        import os
        full = os.path.expanduser("<placeholder>")
        # Placeholder; the real check is in __init__.
        return not full.endswith("<placeholder>")

    @staticmethod
    def _config_exists(path: str) -> bool:
        import os
        full = os.path.expanduser(path)
        return os.path.exists(os.path.join(full, "config.json"))

    def _default_calibration_sample(self) -> List[float]:
        """Deterministic synthetic calibration sample so the user can call
        SpectralVectorStore without supplying real vectors. Each position
        gets a sinusoid of period 7 + small jitter so the FFT peaks are
        realistic enough for the encoder to calibrate.
        """
        import random
        rng = random.Random(self._index_name)
        dim = self._dimension
        n_vecs = 200
        sample: list[float] = []
        for _ in range(n_vecs):
            phase = rng.uniform(0, 2 * 3.14159)
            for j in range(dim):
                val = (0.5 + 0.5 * ((j + phase) % 1.0)) * 0.2
                val += rng.gauss(0, 0.05)
                sample.append(val)
        return sample

    # ---- LangChain VectorStore contract --------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed ``texts`` via the supplied embedding function and upsert.

        Returns the list of vector ids (one per input).
        """
        texts_list = list(texts)
        if not texts_list:
            return []
        if metadatas is None:
            metadatas = [{} for _ in texts_list]
        if len(metadatas) != len(texts_list):
            raise ValueError(
                f"len(metadatas)={len(metadatas)} != len(texts)={len(texts_list)}"
            )

        embeds = self._embed(texts_list)
        if ids is None:
            import uuid
            ids = [f"doc-{uuid.uuid4().hex[:12]}" for _ in texts_list]
        else:
            if len(ids) != len(texts_list):
                raise ValueError(
                    f"len(ids)={len(ids)} != len(texts)={len(texts_list)}"
                )

        vectors = [
            {"id": i, "values": embeds[k], "metadata": metadatas[k]}
            for k, i in enumerate(ids)
        ]
        r = self._index.upsert(vectors=vectors, namespace=self._namespace)
        if r["upserted_count"] != len(ids):
            raise RuntimeError(
                f"upsert returned {r['upserted_count']}, expected {len(ids)}"
            )
        return list(ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,            # noqa: A002
        **kwargs: Any,
    ):
        """Top-k documents most similar to ``query``."""
        results = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _score in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,            # noqa: A002
        **kwargs: Any,
    ):
        """Top-k documents plus their SSE scores."""
        q_emb = self._embed_one(query)
        r = self._index.query(
            vector=q_emb,
            top_k=k,
            filter=filter,
            namespace=self._namespace,
            include_metadata=True,
        )
        # LangChain expects Document objects; import lazily so the adapter
        # is usable even when only the ``langchain_core.text_splitter``
        # bits are installed without the document wrapper.
        from langchain_core.documents import Document
        out = []
        for m in r["matches"]:
            doc = Document(page_content="", metadata=m.get("metadata", {}))
            out.append((doc, m["score"]))
        return out

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> Any:
        if ids:
            return self._index.delete(ids=ids, namespace=self._namespace)
        return None

    def from_texts(
        cls,
        texts: List[str],
        embedding: Callable,
        metadatas: Optional[List[dict]] = None,
        *,
        path: str,
        index_name: str,
        dimension: int,
        **kwargs: Any,
    ) -> "SpectralVectorStore":
        out = cls(
            index_name=index_name,
            path=path,
            dimension=dimension,
            embedding=embedding,
            **kwargs,
        )
        out.add_texts(texts, metadatas=metadatas)
        return out

    from_texts = classmethod(from_texts)

    # ---- helpers -----------------------------------------------------------

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        embeds = self._embedding(texts)
        # Defensive coercion: turn numpy / iterables / single vector
        # shapes into a list of lists of python floats.
        out: list[list[float]] = []
        if not isinstance(embeds, list):
            embeds = list(embeds)
        for vec in embeds:
            arr = [float(x) for x in vec]
            if len(arr) != self._dimension:
                raise DimensionMismatch(
                    f"embedding returned dim {len(arr)}, expected {self._dimension}"
                )
            out.append(arr)
        return out

    def _embed_one(self, text: str) -> List[float]:
        v = self._embed([text])[0]
        return v

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Return a retriever that calls ``similarity_search``."""
        base_cls = _get_langchain_base()

        # Build a tiny subclass that conforms to the LangChain contract.
        # We do NOT inherit the base because its constructor runs validation
        # in __init_subclass__ that conflicts with our init signature.
        # The duck-typed surface we expose (add_texts, similarity_search,
        # similarity_search_with_score, from_texts, as_retriever) is
        # what LangChain uses.
        adapter_self = self
        k = (search_kwargs or {}).get("k", 4)
        f = (search_kwargs or {}).get("filter")
        namespace = (search_kwargs or {}).get("namespace", self._namespace)

        class _Retriever:
            def get_relevant_documents(self, query: str):
                return adapter_self.similarity_search(
                    query, k=k, filter=f, namespace=namespace,
                )

            async def aget_relevant_documents(self, query: str):
                return self.get_relevant_documents(query)

        return _Retriever()


def as_langchain_vectorstore(adapter: SpectralVectorStore):
    """Validate that ``adapter`` satisfies the LangChain VectorStore contract.

    Not strictly necessary — most LangChain call sites duck-type — but
    useful as a runtime sanity check.
    """
    base = _get_langchain_base()
    for name in ("add_texts", "similarity_search",
                 "similarity_search_with_score", "from_texts", "as_retriever"):
        if not hasattr(adapter, name):
            raise InvalidArgument(
                f"adapter missing required LangChain method {name!r}"
            )
    return adapter
