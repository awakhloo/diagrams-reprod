"""Array backend switch: numpy (CPU) or cupy (GPU).

Set the environment variable SCS_BACKEND=cupy to use the GPU. Everything else in
the package imports `xp` from here and writes numpy-style code.
"""
import os
import numpy as np

BACKEND = os.environ.get("SCS_BACKEND", "numpy")

if BACKEND == "cupy":
    import cupy as xp  # noqa: F401
else:
    xp = np


def asnumpy(a):
    """Move an array to host memory as a numpy array."""
    if BACKEND == "cupy":
        return xp.asnumpy(a)
    return np.asarray(a)


def to_device(a):
    """Move a numpy array to the compute device (no-op for numpy)."""
    return xp.asarray(a)
