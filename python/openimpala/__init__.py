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


def _preload_cuda_libs():
    """Pre-load CUDA shared libs shipped by PyPI nvidia-*-cu12 packages.

    The openimpala-cuda wheel uses ``auditwheel --exclude libcublas.so.12``
    (etc.) to keep the wheel under PyPI's 320 MiB cap, then declares
    ``nvidia-cublas-cu12`` and friends as runtime deps. Those wheels install
    their .so's under ``site-packages/nvidia/<component>/lib/``, which the
    dynamic linker does NOT search by default — so loading ``_core.so``
    fails with ``undefined symbol: cublasSetStream_v2`` (and similar) until
    the libs are dlopened with RTLD_GLOBAL.

    Same trick as PyTorch / CuPy / JAX. For each lib we try the bare soname
    first (so HPC nodes with system CUDA on LD_LIBRARY_PATH win the race),
    then fall back to the PyPI bundled path. No-op for pure-Python wheels
    where the nvidia/ dir doesn't exist.

    Load order matters: cudart first, cublasLt before cublas (cublas links
    against cublasLt at the symbol level).
    """
    import os
    import sys
    import ctypes

    if sys.platform != "linux":
        return

    pkg_root = os.path.dirname(os.path.abspath(__file__))
    site_pkgs = os.path.dirname(pkg_root)
    nvidia_root = os.path.join(site_pkgs, "nvidia")

    libs = (
        ("libcudart.so.12",     "cuda_runtime/lib/libcudart.so.12"),
        ("libnvJitLink.so.12",  "nvjitlink/lib/libnvJitLink.so.12"),
        ("libcublasLt.so.12",   "cublas/lib/libcublasLt.so.12"),
        ("libcublas.so.12",     "cublas/lib/libcublas.so.12"),
        ("libcusparse.so.12",   "cusparse/lib/libcusparse.so.12"),
        ("libcurand.so.10",     "curand/lib/libcurand.so.10"),
    )

    for soname, fallback in libs:
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
            continue
        except OSError:
            pass
        path = os.path.join(nvidia_root, fallback)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cuda_libs()

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
        "read_image", "build_info",
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
    "build_info",
]
