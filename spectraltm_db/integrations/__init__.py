"""RAG framework adapters.

These modules are imported lazily so installing spectraltm-db does not
require langchain or llama-index. Install extras::

    pip install "spectraltm-db[langchain]"
    pip install "spectraltm-db[llamaindex]"
"""

from __future__ import annotations

# Lazy imports — these modules raise a clear ImportError if the
# corresponding framework is not installed.

__all__ = ["langchain_adapter", "llamaindex_adapter"]


def __getattr__(name):
    if name == "langchain_adapter":
        from . import langchain
        return langchain
    if name == "llamaindex_adapter":
        from . import llamaindex
        return llamaindex
    raise AttributeError(f"no such attribute {name!r}")
