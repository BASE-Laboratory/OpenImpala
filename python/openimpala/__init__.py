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

# --- CRITICAL PYBIND11 INTEROP FIX ---
# We must force Linux to use RTLD_GLOBAL so that openimpala._core.so can 
# physically see the C++ type registry inside pyAMReX's extension. 
_old_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

try:
    import amrex.space3d as amrex  # 1. Load pyAMReX globally
    from . import _core as core    # 2. Load OpenImpala so it links to pyAMReX
finally:
    sys.setdlopenflags(_old_flags) # 3. Restore safe defaults

# Enums — available at top level for convenience
from ._core import (
    Direction,
    CellType,
    RawDataType,
    SolverType,
    EffDiffSolverType,
    PhysicsType,
)

# Session context manager
from .session import Session

# Custom exceptions
from .exceptions import (
    OpenImpalaError,
    ConvergenceError,
    PercolationError,
)

# High-level facade functions
from .facade import (
    volume_fraction,
    percolation_check,
    tortuosity,
    read_image,
)

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
