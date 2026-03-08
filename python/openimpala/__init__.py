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

__version__ = "0.1.0"

# Re-export the C++ backend under ``openimpala.core``
from . import _core as core  # noqa: F401

# Enums — available at top level for convenience
from ._core import (  # noqa: F401
    Direction,
    CellType,
    RawDataType,
    SolverType,
    EffDiffSolverType,
    PhysicsType,
)

# Session context manager
from .session import Session  # noqa: F401

# Custom exceptions
from .exceptions import (  # noqa: F401
    OpenImpalaError,
    ConvergenceError,
    PercolationError,
)

# High-level facade functions
from .facade import (  # noqa: F401
    volume_fraction,
    percolation_check,
    tortuosity,
    read_image,
)
