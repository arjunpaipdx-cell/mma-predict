"""Pick GPU libraries when they're available, fall back to CPU otherwise.

The same code path runs on a Colab T4 (cuDF + CuPy + XGBoost CUDA) and on a
laptop with no NVIDIA GPU (pandas + NumPy + XGBoost CPU).
"""
from __future__ import annotations

import os

FORCE_CPU = os.environ.get("MMAPRED_CPU", "0") == "1"


def _has_cuda() -> bool:
    if FORCE_CPU:
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _has_cuda()

if GPU:
    import cupy as xp  # noqa: F401
else:
    import numpy as xp  # noqa: F401

try:
    if not GPU:
        raise ImportError
    import cudf as xdf  # noqa: F401

    CUDF = True
except Exception:
    import pandas as xdf  # noqa: F401

    CUDF = False


def to_host(a):
    """CuPy/cuDF -> NumPy/pandas (no-op on CPU)."""
    mod = type(a).__module__
    if mod.startswith("cudf"):
        return a.to_pandas()
    if mod.startswith("cupy"):
        return a.get()
    return a


def to_device_df(pdf):
    """pandas -> cuDF when available."""
    return xdf.from_pandas(pdf) if CUDF else pdf


def xgb_device() -> str:
    return "cuda" if GPU else "cpu"


def describe() -> str:
    if GPU:
        import cupy as cp

        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        return f"GPU: {name} | cuDF={'yes' if CUDF else 'no'} | CuPy=yes"
    return "CPU fallback (no CUDA device found) | pandas + NumPy"
