"""OpenImpala — Python bindings for transport-property computation on 3-D voxel images.

Two layers are provided:

* **Low-level (power-user) API** — ``openimpala.core``
  Direct pybind11 wrappers around C++ classes; interoperates with pyAMReX types.

* **High-level (general) API** — module-level functions
  NumPy-native helpers that set up AMReX objects automatically.

Quick-start::

    import openimpala as oi
    import numpy as np

    data = np.random.choice([0, 1], size=(32, 32, 32)).astype(np.int32)

    with oi.Session():
        vf = oi.volume_fraction(data, phase=0)
        print(f"Volume fraction: {vf.fraction:.4f}")
"""

import os
import sys

__version__ = "0.1.0"

# Session context manager (pure Python — always available)
from .session import Session

# Custom exceptions (pure Python — always available)
from .exceptions import (
    OpenImpalaError,
    ConvergenceError,
    PercolationError,
)


def _load_core():
    """Load the _core C extension with the correct dlopen flags.

    Must be called after amrex.space3d is loaded (e.g. inside a Session).
    Repeated calls are cheap — Python caches the import.
    """
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
    try:
        import amrex.space3d  # noqa: F401 — load pyAMReX globally first
        from . import _core
    finally:
        sys.setdlopenflags(old_flags)
    return _core


def __getattr__(name):
    """Lazy-load C extension symbols on first access.

    This allows ``from openimpala.cli import _build_parser`` and other
    pure-Python imports to succeed without loading the compiled backend.
    """
    # Symbols that live in the _core C extension
    _CORE_ATTRS = {
        "core", "_core", "Direction", "CellType", "RawDataType",
        "SolverType", "EffDiffSolverType", "PhysicsType",
    }
    # Symbols that live in the facade module
    _FACADE_ATTRS = {
        "volume_fraction", "percolation_check", "tortuosity", "read_image",
    }

    if name in _CORE_ATTRS:
        _core = _load_core()
        if name in ("core", "_core"):
            return _core
        return getattr(_core, name)

    if name in _FACADE_ATTRS:
        from . import facade
        return getattr(facade, name)

    raise AttributeError(f"module 'openimpala' has no attribute {name!r}")


# Explicitly define the public API for IDEs and static analysis
__all__ = [
    "__version__",
    "core",
    "Direction",
    "CellType",
    "RawDataType",
    "SolverType",
    "EffDiffSolverType",
    "PhysicsType",
    "Session",
    "OpenImpalaError",
    "ConvergenceError",
    "PercolationError",
    "volume_fraction",
    "percolation_check",
    "tortuosity",
    "read_image",
]
