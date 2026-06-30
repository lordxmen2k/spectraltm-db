"""Sparse-dense hybrid support — architectural stub for v0.2+.

Pinecone's hybrid search combines sparse (BM25/TF-IDF-style) and dense
(vector-embedding-style) scores via Reciprocal Rank Fusion. It's a real
production feature; we expose the **wire surface** in v0.1.1 but only
do dense scoring today. Sparse scores are recorded and forwarded but
not aggregated.

What's wired
------------

  - ``SparseValues`` dataclass
  - ``upsert(...)`` accepts ``sparse_values`` per vector
  - ``query(...)`` accepts ``sparse_vector`` (single, top-k fusion)
  - ``update(...)`` accepts ``sparse_values``
  - The Rust engine has no sparse score; we log a warning when a
    query arrives with ``sparse_vector`` so callers can see fallback
    behavior.

What's *not* wired
------------------

  - True sparse ranking (BM25/TF-IDF/SPLADE-style) requires a Rust
    feature on top of ``spectraltm``. Tracked in v0.2; until then
    sparse inputs are stored but ignored.
  - Reciprocal Rank Fusion against the SSE top-K — also v0.2.

The wire surface stays compatible with Pinecone so when v0.2 lands
the on-disk and HTTP contracts don't shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseValues:
    """Pinecone-shaped sparse vector values.

    Pinecone's contract: ``indices`` is a sorted list of u32 indices
    into the (implicit) tokenizer vocabulary; ``values`` is the same
    length list of non-negative floats.

    For now spectraltm-db accepts and stores these for compatibility
    but does not score against them. Plans for v0.2: add an inverted
    index over a configurable tokenizer and integrate score fusion
    into the SSE search path.
    """

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseValues indices/values length mismatch: "
                f"{len(self.indices)} vs {len(self.values)}"
            )

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["SparseValues"]:
        if not d:
            return None
        if not isinstance(d, dict):
            raise ValueError(
                f"sparse_values must be a dict, got {type(d).__name__}"
            )
        indices = list(d.get("indices") or [])
        values = [float(v) for v in (d.get("values") or [])]
        return cls(indices=indices, values=values)

    def to_dict(self) -> dict:
        return {"indices": list(self.indices), "values": list(self.values)}


def warn_if_sparse_provided(*, sparse_present: bool, where: str) -> None:
    """Surface a clear runtime warning when sparse inputs are used."""
    if sparse_present:
        log.warning(
            "sparse-dense hybrid scoring is not implemented in spectraltm_db "
            "v0.1.x (%s); sparse inputs will be stored but ignored at query "
            "time. Track v0.2 for RR-fusion support.", where,
        )
