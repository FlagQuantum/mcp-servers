"""Shared test configuration.

Pins BLAS and OpenMP thread counts before torch is imported, matching what the
FlagQuantum repository does in its own root conftest. Without it, a reduction
inside torch can pick a different summation order per run and make a numerical
assertion flap.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json

import pytest

BELL_QIR = [{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}]
GHZ3_QIR = [
    {"name": "h", "index": [0]},
    {"name": "cx", "index": [0, 1]},
    {"name": "cx", "index": [1, 2]},
]


@pytest.fixture
def bell_qir() -> str:
    """A two-qubit Bell circuit as a gate-list JSON string."""
    return json.dumps(BELL_QIR)


@pytest.fixture
def ghz3_qir() -> str:
    """A three-qubit GHZ circuit as a gate-list JSON string."""
    return json.dumps(GHZ3_QIR)
