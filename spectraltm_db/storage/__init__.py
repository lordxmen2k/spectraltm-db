"""Storage layer for spectraltm_db.

Three pieces:

  - ``VectorStore``       — wraps the ``spectraltm.SpectralIndex`` Rust type;
                            owns the on-disk ``<ns>.sltm`` file and the
                            in-memory copy.
  - ``IdMap``             — bidirectional ``str <-> u32`` chunk_id
                            bijection, persisted to SQLite alongside
                            metadata.
  - ``MetadataStore``     — SQLite store: vectors table + dynamic filter
                            indexes + per-namespace tombstone set.
  - ``Namespaces``        — pure-directory partitioning helper.

The data flow on ``upsert``:

      user ----> Index.upsert()
                      |
                      +--> IdMap.assign(id)           # allocate chunk_id
                      +--> VectorStore.add(vector)   # append to SSE index
                      +--> MetadataStore.insert(...) # write row
                      +--> WAL.append(...)           # durable record

On ``query``:

      user ----> Index.query()
                      |
                      +--> [filter present?] MetadataStore.filter_chunk_ids(filter)
                      +--> VectorStore.search(query, top_k * oversample)
                      +--> apply chunk_id whitelist (filter) and tombstones
                      +--> MetadataStore.hydrate(ids) -> matches
                      +--> Pinecone-shaped response dict
"""

from .vector_store import VectorStore
from .id_map import IdMap
from .metadata_store import MetadataStore
from .namespaces import Namespaces

__all__ = ["VectorStore", "IdMap", "MetadataStore", "Namespaces"]
