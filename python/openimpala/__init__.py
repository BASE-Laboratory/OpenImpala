"""OpenImpala — Python bindings for transport-property computation on 3-D voxel images.

Two layers are provided:

* **Low-level (power-user) API** — ``openimpala.core``
  Direct pybind11 wrappers around C++ classes; operates via ``VoxelImage`` handles.

* **High-level (general) API** — module-level functions
  NumPy-native helpers that set up AMReX objects automatically (with automatic
  pure-Python SciPy/CuPy fallback on platforms without compiled extensions).

Quick-start::

    import openimpala as oi
    import numpy as np

    data = np.random.choice([0, 1], size=(32, 32, 32)).astype(np.int32)

    with oi.Session():
        vf = oi.volume_fraction(data, phase=0)
        print(f"Volume fraction: {vf.fraction:.4f}")
"""

import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re


def _detect_version() -> str:
    """Resolve package version, preferring local ``pyproject.toml`` in source checkouts."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
            if re.search(r'(?m)^name\s*=\s*"openimpala(?:-cuda)?"\s*$', text):
                m = re.search(
                    r'(?m)^(?:fallback_)?version\s*=\s*"([^"]+)"',
                    text,
                )
                if m:
                    return m.group(1)
        except OSError:
            pass
    for dist_name in ("openimpala", "openimpala-cuda"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return "unknown"


__version__ = _detect_version()


def _preload_cuda_libs():
    """Pre-load CUDA shared libs shipped by PyPI nvidia-*-cu12 packages."""
    import ctypes
    import os
    import sys

    if sys.platform != "linux":
        return

    pkg_root = os.path.dirname(os.path.abspath(__file__))
    site_pkgs = os.path.dirname(pkg_root)
    nvidia_root = os.path.join(site_pkgs, "nvidia")

    libs = (
        ("libcudart.so.12", "cuda_runtime/lib/libcudart.so.12"),
        ("libnvJitLink.so.12", "nvjitlink/lib/libnvJitLink.so.12"),
        ("libcublasLt.so.12", "cublas/lib/libcublasLt.so.12"),
        ("libcublas.so.12", "cublas/lib/libcublas.so.12"),
        ("libcusparse.so.12", "cusparse/lib/libcusparse.so.12"),
        ("libcurand.so.10", "curand/lib/libcurand.so.10"),
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

# Session context manager and datasets (pure Python — always available)
from . import datasets
from .session import Session

# Custom exceptions (pure Python — always available)
from .exceptions import (
    ConvergenceError,
    OpenImpalaError,
    PercolationError,
)


def _augment_core_voxelimage(_core) -> None:
    """Attach pure-Python Otsu and OME-Zarr convenience methods onto ``_core.VoxelImage``."""
    vcls = getattr(_core, "VoxelImage", None)
    if vcls is None or hasattr(vcls, "from_grayscale_otsu"):
        return

    from ._io import binarize_otsu, downsample_volume, read_zarr_volume, write_zarr_volume

    def _from_grayscale_otsu(
        cls,
        arr,
        *,
        pore_value: int = 0,
        solid_value: int = 1,
        dark_is_pore: bool = True,
        nbins: int | None = None,
        max_grid_size: int = 32,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        periodic=(False, False, False),
    ):
        Session.ensure_active()
        seg, _ = binarize_otsu(
            arr,
            pore_value=pore_value,
            solid_value=solid_value,
            dark_is_pore=dark_is_pore,
            nbins=nbins,
        )
        return cls.from_numpy(
            seg,
            max_grid_size=max_grid_size,
            dx=dx,
            dy=dy,
            dz=dz,
            periodic=tuple(bool(b) for b in periodic),
        )

    def _from_zarr(
        cls,
        path,
        *,
        array_path=None,
        level: int = 0,
        threshold=None,
        pore_value: int = 0,
        solid_value: int = 1,
        dark_is_pore: bool = True,
        max_grid_size: int = 32,
        voxel_size=None,
        periodic=(False, False, False),
    ):
        import numpy as np

        Session.ensure_active()
        arr, meta_vz = read_zarr_volume(
            path,
            array_path=array_path,
            level=level,
            threshold=threshold,
            pore_value=pore_value,
            solid_value=solid_value,
            dark_is_pore=dark_is_pore,
        )
        if voxel_size is not None:
            if isinstance(voxel_size, (int, float, np.floating, np.integer)):
                dz = dy = dx = float(voxel_size)
            else:
                dz, dy, dx = (
                    float(voxel_size[0]),
                    float(voxel_size[1]),
                    float(voxel_size[2]),
                )
        elif meta_vz is not None:
            dz, dy, dx = meta_vz
        else:
            dz = dy = dx = 1.0
        return cls.from_numpy(
            np.ascontiguousarray(arr, dtype=np.int32),
            max_grid_size=max_grid_size,
            dx=dx,
            dy=dy,
            dz=dz,
            periodic=tuple(bool(b) for b in periodic),
        )

    def _to_zarr(
        self,
        path,
        *,
        chunks=(64, 64, 64),
        name: str = "volume",
        unit: str = "micrometer",
        overwrite: bool = True,
    ):
        return write_zarr_volume(
            path,
            self.to_numpy(),
            voxel_size=self.voxel_size,
            chunks=chunks,
            name=name,
            unit=unit,
            overwrite=overwrite,
        )

    def _downsample(
        self,
        factor=2,
        *,
        mode: str = "majority",
        max_grid_size: int = 32,
    ):
        import numpy as np

        arr_ds, (dz_new, dy_new, dx_new) = downsample_volume(
            self.to_numpy(),
            factor=factor,
            mode=mode,
            voxel_size=self.voxel_size,
        )
        return type(self).from_numpy(
            np.ascontiguousarray(arr_ds, dtype=np.int32),
            max_grid_size=max_grid_size,
            dx=dx_new,
            dy=dy_new,
            dz=dz_new,
        )

    setattr(vcls, "from_grayscale_otsu", classmethod(_from_grayscale_otsu))
    setattr(vcls, "from_zarr", classmethod(_from_zarr))
    setattr(vcls, "to_zarr", _to_zarr)
    setattr(vcls, "downsample", _downsample)


def _load_core():
    """Load the _core C extension and attach Python convenience helpers."""
    _core = importlib.import_module("openimpala._core")
    _augment_core_voxelimage(_core)
    return _core


def __getattr__(name):
    """Lazy-load C extension, facade, I/O, and synthetic symbols on first access."""
    _CORE_ATTRS = {
        "core",
        "_core",
        "VoxelImage",
        "Direction",
        "CellType",
        "RawDataType",
        "SolverType",
        "PrecondType",
        "EffDiffSolverType",
        "PhysicsType",
    }
    _FACADE_ATTRS = {
        "volume_fraction",
        "percolation_check",
        "tortuosity",
        "tortuosity_all",
        "effective_diffusivity",
        "specific_surface_area",
        "connected_components",
        "particle_size_distribution",
        "through_thickness_profile",
        "rev_study",
        "estimate_memory",
        "read_image",
        "build_info",
        "VolumeFractionResult",
        "PercolationResult",
        "TortuosityResult",
        "EffectiveDiffusivityResult",
        "SpecificSurfaceAreaResult",
        "ConnectedComponentsResult",
        "ParticleSizeDistributionResult",
        "ThroughThicknessResult",
        "REVStudyResult",
    }
    _IO_ATTRS = {
        "calculate_otsu_threshold",
        "otsu_threshold",
        "binarize_otsu",
        "downsample_volume",
        "read_zarr_volume",
        "write_zarr_volume",
        "write_amrex_plotfile",
    }

    if name == "synthetic":
        mod = importlib.import_module("openimpala.synthetic")
        globals()["synthetic"] = mod
        return mod

    if name in _IO_ATTRS:
        _io = importlib.import_module("openimpala._io")
        return getattr(_io, name)

    if name in _CORE_ATTRS:
        from .session import _requested_backend

        if name == "VoxelImage" and (Session._pure_python or _requested_backend() == "pure"):
            _io = importlib.import_module("openimpala._io")
            return _io.PureVoxelImage
        try:
            _core = _load_core()
        except (ImportError, OSError):
            if name == "VoxelImage":
                _io = importlib.import_module("openimpala._io")
                return _io.PureVoxelImage
            raise AttributeError(
                f"'openimpala.{name}' requires the compiled C++ backend (_core). "
                f"In pure-Python mode, use the high-level API: "
                f"openimpala.volume_fraction(), tortuosity(), effective_diffusivity(), etc."
            )
        if name in ("core", "_core"):
            return _core
        return getattr(_core, name)

    if name in _FACADE_ATTRS:
        facade = importlib.import_module("openimpala.facade")
        return getattr(facade, name)

    raise AttributeError(f"module 'openimpala' has no attribute {name!r}")


__all__ = [
    "__version__",
    "datasets",
    "synthetic",
    "core",
    "VoxelImage",
    "Direction",
    "CellType",
    "RawDataType",
    "SolverType",
    "PrecondType",
    "EffDiffSolverType",
    "PhysicsType",
    "Session",
    "OpenImpalaError",
    "ConvergenceError",
    "PercolationError",
    "volume_fraction",
    "percolation_check",
    "tortuosity",
    "tortuosity_all",
    "effective_diffusivity",
    "specific_surface_area",
    "connected_components",
    "particle_size_distribution",
    "through_thickness_profile",
    "rev_study",
    "estimate_memory",
    "read_image",
    "build_info",
    "calculate_otsu_threshold",
    "otsu_threshold",
    "binarize_otsu",
    "downsample_volume",
    "read_zarr_volume",
    "write_zarr_volume",
    "write_amrex_plotfile",
    "VolumeFractionResult",
    "PercolationResult",
    "TortuosityResult",
    "EffectiveDiffusivityResult",
    "SpecificSurfaceAreaResult",
    "ConnectedComponentsResult",
    "ParticleSizeDistributionResult",
    "ThroughThicknessResult",
    "REVStudyResult",
]
