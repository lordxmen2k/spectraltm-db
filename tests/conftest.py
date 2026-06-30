"""Shared pytest helpers."""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path


# Make the package importable from the workspace source tree, regardless
# of where pytest is invoked from.
HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT / "spectraltm_db"))
sys.path.insert(0, str(PKG_ROOT))


def workdir() -> str:
    """Return a unique, ephemeral, auto-cleaned workdir."""
    return tempfile.mkdtemp(prefix="stm_pytest_")


def random_calibration(dim: int, n_vecs: int = 200, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n_vecs * dim)]


def random_query(dim: int, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]
