"""OpenImpala — Python bindings for transport-property computation on 3-D voxel images.

Two layers are provided:

* **Low-level (power-user) API** — ``openimpala.core``
  Direct pybind11 wrappers around C++ classes; operates via ``VoxelImage`` handles.

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

import importlib

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openimpala")
except PackageNotFoundError:
    try:
        __version__ = version("openimpala-cuda")
    except PackageNotFoundError:
        __version__ = "unknown"

# Session context manager (pure Python — always available)
from .session import Session

# Custom exceptions (pure Python — always available)
from .exceptions import (
    OpenImpalaError,
    ConvergenceError,
    PercolationError,
)


def _load_core():
    """Load the _core C extension.

    Repeated calls are cheap — Python caches the import.
    """
    return importlib.import_module("openimpala._core")


def __getattr__(name):
    """Lazy-load C extension symbols on first access.

    This allows ``from openimpala.cli import _build_parser`` and other
    pure-Python imports to succeed without loading the compiled backend.
    """
    # Symbols that live in the _core C extension
    _CORE_ATTRS = {
        "core", "_core", "VoxelImage", "Direction", "CellType", "RawDataType",
        "SolverType", "EffDiffSolverType", "PhysicsType",
    }
    # Symbols that live in the facade module
    _FACADE_ATTRS = {
        "volume_fraction", "percolation_check", "tortuosity", "estimate_memory",
        "read_image",
    }

    if name in _CORE_ATTRS:
        try:
            _core = _load_core()
        except ImportError:
            raise AttributeError(
                f"'openimpala.{name}' requires the compiled C++ backend (_core). "
                f"In pure-Python mode, use the high-level API: "
                f"openimpala.volume_fraction(), tortuosity(), etc."
            )
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
    "VoxelImage",
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
    "estimate_memory",
    "read_image",
]
