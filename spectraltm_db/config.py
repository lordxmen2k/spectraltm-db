"""Configuration dataclasses for SpectralTM DB.

The shape mirrors `pinecone.Pinecone.create_index` and friends closely
enough that wrapping a `spectraltm_db.IndexConfig` into a Pinecone-style
``pc.create_index(name=name, dimension=..., metric="cosine", spec=...)``
call is a near-mechanical translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .errors import InvalidArgument


# ---- Compression & metric enums (Pinecone-shaped string aliases) ----

class CompressionSpec(str, Enum):
    """Aliases shipped in v0.1.

    These map to specific SpectralEncoder configurations. Custom values can be
    passed via a raw :class:`EncoderSpec` instead — see ``EncoderSpec.from_str``.
    """

    SPECTRAL_K64 = "spectral_k64"
    SPECTRAL_K128 = "spectral_k128"
    SPECTRAL_K256 = "spectral_k256"
    # Plain float32 — no Spectral compression. Used as the parity / correctness
    # baseline.
    FLOAT32 = "float32"


class MetricSpec(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOTPRODUCT = "dotproduct"


# Pre-validated compression profiles: (top_k, mag_bits, phase_bits, norm_bits).
# These are the only knobs that affect scoring quality or storage size.
COMPRESSION_PROFILES: dict[CompressionSpec, dict] = {
    CompressionSpec.SPECTRAL_K64:  dict(top_k=64,  mag_bits=8, phase_bits=8, norm_bits=8),
    CompressionSpec.SPECTRAL_K128: dict(top_k=128, mag_bits=8, phase_bits=8, norm_bits=8),
    CompressionSpec.SPECTRAL_K256: dict(top_k=256, mag_bits=8, phase_bits=8, norm_bits=8),
    # float32 has no Spectral encoder — handled specially in vector_store.py.
}


@dataclass
class IndexConfig:
    """Validated, hashable-ish handle to an index's static configuration.

    Write-once fields: any change requires a `reindex`. Stored inside
    ``<path>/config.json`` and re-loaded on each ``Index.open``.
    """

    name: str
    path: str                  # absolute path; "~/" expanded at construction
    dimension: int             # > 0
    metric: MetricSpec = MetricSpec.COSINE
    compression: CompressionSpec = CompressionSpec.SPECTRAL_K64
    # Optional auto-encoder (only used by upsert_text/query_text).
    encoder: str | None = None

    # ---- Validated constructors ----------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidArgument("IndexConfig.name must be a non-empty string")
        # Names mirror Pinecone's: lowercase ASCII + dashes, start with letter.
        # We're permissive but reject path separators and obvious garbage.
        for ch in self.name:
            if not (ch.isalnum() or ch in "-_"):
                raise InvalidArgument(
                    f"IndexConfig.name contains illegal character {ch!r}; "
                    "use lowercase ASCII letters, digits, '-', or '_'."
                )
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise InvalidArgument("IndexConfig.dimension must be a positive int")

        if isinstance(self.metric, str):
            self.metric = MetricSpec(self.metric)
        if isinstance(self.compression, str):
            self.compression = CompressionSpec(self.compression)

        # For float32 we still need a SpectralEncoder for the index — but
        # we'll skip compression (effectively store dense float32 vectors and
        # use brute-force cosine over them). See storage/vector_store.py.

    # ---- Convenience ---------------------------------------------------------

    @property
    def is_compressed(self) -> bool:
        return self.compression != CompressionSpec.FLOAT32

    @property
    def encoder_profile(self) -> dict:
        """Top-K and bit-width knobs for the SpectralEncoder."""
        if not self.is_compressed:
            # For float32 we still need a "flat" SpectralEncoder so the
            # crate's ``add_embeddings`` works uniformly. Profile picks K=F
            # so every frequency bin is kept — equivalent to a no-op Spectral
            # pass. ``f`` (=dim/2+1) is the natural upper bound.
            return dict(top_k=0, mag_bits=8, phase_bits=8, norm_bits=8)  # K=0 sentinel
        return COMPRESSION_PROFILES[self.compression]

    def to_json_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "path": self.path,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "compression": self.compression.value,
            "encoder": self.encoder,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "IndexConfig":
        return cls(
            name=d["name"],
            path=d["path"],
            dimension=d["dimension"],
            metric=d.get("metric", "cosine"),
            compression=d.get("compression", "spectral_k64"),
            encoder=d.get("encoder"),
        )
