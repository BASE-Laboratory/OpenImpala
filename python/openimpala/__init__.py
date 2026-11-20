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


def _preload_mpi_libs() -> None:
    """Pre-load OpenMPI shared libraries installed via pip/uv/Conda when not on LD_LIBRARY_PATH.

    Checks system / ``LD_LIBRARY_PATH`` / ``DT_RUNPATH`` resolution first so HPC
    ``module load openmpi`` environments always take precedence.  Falls back to
    ``<sys.prefix>/lib`` and ``importlib.metadata.files('openmpi')`` (which handles
    ``uv`` symlinked environments where wheel ``lib/`` files live in separate cache dirs).
    """
    import ctypes
    from importlib.metadata import files as dist_files
    import os
    import sys

    if sys.platform != "linux":
        return
    if os.environ.get("OPENIMPALA_BACKEND", "auto").strip().lower() in (
        "pure",
        "pure-python",
        "pure_python",
        "scipy",
    ):
        return

    pkg_dir = Path(__file__).resolve().parent
    if not any(pkg_dir.glob("_core*.so")):
        return

    # 1. System / LD_LIBRARY_PATH / ELF DT_RUNPATH already resolves libmpi -> nothing to do.
    for soname in ("libmpi.so.40", "libmpi.so.12"):
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            pass

    # 2. Check <sys.prefix>/lib and PyPI `openmpi` package metadata (`pip install openmpi`).
    candidate_files: list[Path] = []
    prefix_lib = Path(sys.prefix) / "lib"
    if prefix_lib.is_dir():
        for pattern in ("libopen-pal.so.*", "libopen-rte.so.*", "libmpi.so.*"):
            candidate_files.extend(sorted(prefix_lib.glob(pattern)))

    try:
        ompi_entries = dist_files("openmpi") or []
    except PackageNotFoundError:
        ompi_entries = []

    pal_paths: list[Path] = []
    rte_paths: list[Path] = []
    mpi_paths: list[Path] = []
    for entry in ompi_entries:
        fname = entry.name
        if (
            fname.startswith("libopen-pal.so.")
            or fname.startswith("libopen-rte.so.")
            or fname.startswith("libmpi.so.")
        ):
            located = Path(entry.locate())
            if fname.startswith("libopen-pal.so."):
                pal_paths.append(located)
            elif fname.startswith("libopen-rte.so."):
                rte_paths.append(located)
            else:
                mpi_paths.append(located)

    ordered = [*candidate_files, *pal_paths, *rte_paths, *mpi_paths]
    seen: set[str] = set()
    for path in ordered:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            # Skip tiny GNU ld text stubs (e.g. unversioned linker scripts)
            if not path.is_file() or path.stat().st_size < 4096:
                continue
            ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


_preload_cuda_libs()
_preload_mpi_libs()

# Session context manager and datasets (pure Python — always available)
from . import datasets
from .session import Session, _core_binary_exists, _probe_core, _requested_backend

# Custom exceptions (pure Python — always available)
from .exceptions import (
    ConvergenceError,
    OpenImpalaError,
    PercolationError,
)


def _warn_if_compiled_core_broken() -> None:
    """Warn once on import when a compiled ``_core`` binary is installed yet fails to load."""
    import warnings

    try:
        if _requested_backend() == "pure":
            return
    except ValueError:
        return

    if not _core_binary_exists():
        return

    ok, err = _probe_core()
    if not ok:
        warnings.warn(
            f"OpenImpala compiled C++ backend (_core) is installed but failed to load "
            f"({err}); falling back to the slower pure-Python SciPy solver. "
            f"To enable the C++/MPI engine, load or install OpenMPI (`module load openmpi`, "
            f"`apt install libopenmpi-dev`, or `pip install openmpi`), or set "
            f"OPENIMPALA_BACKEND=pure to silence this warning.",
            RuntimeWarning,
            stacklevel=3,
        )


_warn_if_compiled_core_broken()


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
        tie_break: str = "lowest",
        max_grid_size: int = 32,
    ):
        import numpy as np

        arr_ds, (dz_new, dy_new, dx_new) = downsample_volume(
            self.to_numpy(),
            factor=factor,
            mode=mode,
            tie_break=tie_break,
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
        "to_pybamm_parameters",
        "export_bpx_json",
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
    "to_pybamm_parameters",
    "export_bpx_json",
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
