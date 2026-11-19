"""High-level, NumPy-native facade for OpenImpala workflows.

These functions accept plain ``numpy.ndarray`` inputs (or ``VoxelImage`` handles)
and return rich Python dataclasses, hiding all AMReX boilerplate from users.

When the compiled C++ backend (``_core``) is unavailable, all high-level analysis
functions transparently fall back to the pure-Python solver (NumPy / SciPy / CuPy).
"""

from __future__ import annotations

import csv
import dataclasses
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .exceptions import ConvergenceError, PercolationError
from .session import Session, _has_core

VoxelSizeArg = Union[float, int, Sequence[float]]
DataInputArg = Union[np.ndarray, str, os.PathLike, "VoxelImage"]


# ---------------------------------------------------------------------------
# Return dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VolumeFractionResult:
    """Result of a volume-fraction calculation."""

    phase_count: int
    total_count: int
    fraction: float

    def _repr_html_(self) -> str:
        pct = self.fraction * 100.0
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Phase count</td><td>{self.phase_count:,}</td></tr>"
            f"<tr><td>Total count</td><td>{self.total_count:,}</td></tr>"
            f"<tr><td>Volume fraction</td><td>{self.fraction:.6f} ({pct:.2f}%)</td></tr>"
            "</table>"
        )


@dataclasses.dataclass
class PercolationResult:
    """Result of a percolation check."""

    percolates: bool
    active_volume_fraction: float
    direction: str

    def _repr_html_(self) -> str:
        status = "Yes" if self.percolates else "No"
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Direction</td><td>{self.direction}</td></tr>"
            f"<tr><td>Percolates</td><td>{status}</td></tr>"
            f"<tr><td>Active VF</td><td>{self.active_volume_fraction:.6f}</td></tr>"
            "</table>"
        )


@dataclasses.dataclass
class TortuosityResult:
    """Result of a tortuosity / multi-phase steady-state transport calculation."""

    tortuosity: float
    solver_converged: bool
    iterations: int
    residual_norm: float
    flux_in: float
    flux_out: float
    active_volume_fraction: float
    solution_field: "np.ndarray | None" = None
    effective_diffusivity: Optional[float] = None
    plotfile_path: Optional[str] = None

    def _repr_html_(self) -> str:
        status = "converged" if self.solver_converged else "NOT converged"
        deff_row = (
            f"<tr><td>Effective diffusivity</td><td>{self.effective_diffusivity:.6e}</td></tr>"
            if self.effective_diffusivity is not None
            else ""
        )
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Tortuosity</td><td>{self.tortuosity:.6f}</td></tr>"
            f"{deff_row}"
            f"<tr><td>Solver status</td><td>{status}</td></tr>"
            f"<tr><td>Iterations</td><td>{self.iterations}</td></tr>"
            f"<tr><td>Residual norm</td><td>{self.residual_norm:.2e}</td></tr>"
            f"<tr><td>Flux in</td><td>{self.flux_in:.6e}</td></tr>"
            f"<tr><td>Flux out</td><td>{self.flux_out:.6e}</td></tr>"
            f"<tr><td>Active VF</td><td>{self.active_volume_fraction:.6f}</td></tr>"
            "</table>"
        )


@dataclasses.dataclass
class EffectiveDiffusivityResult:
    """Result of a 3-D periodic homogenisation effective-diffusivity tensor calculation.

    ``tensor`` is a ``(3, 3)`` ``float64`` NumPy array ordered ``(x, y, z)``:
    ``tensor[0, 0] = D_xx``, ``tensor[1, 1] = D_yy``, ``tensor[2, 2] = D_zz``,
    ``tensor[0, 1] = D_xy``, etc.
    """

    tensor: np.ndarray
    active_volume_fraction: float
    solver_converged: bool
    iterations: tuple[int, int, int]
    residuals: tuple[float, float, float]
    corrector_fields: Optional[dict[str, np.ndarray]] = None
    plotfile_paths: Optional[dict[str, str]] = None

    @property
    def D_eff(self) -> np.ndarray:
        """Alias for ``tensor`` (the symmetric ``(3, 3)`` effective diffusivity tensor)."""
        return self.tensor

    @property
    def diagonal(self) -> tuple[float, float, float]:
        """Principal diagonal entries ``(D_xx, D_yy, D_zz)``."""
        return (
            float(self.tensor[0, 0]),
            float(self.tensor[1, 1]),
            float(self.tensor[2, 2]),
        )

    def _repr_html_(self) -> str:
        dxx, dyy, dzz = self.diagonal
        return (
            "<table>"
            "<tr><th>Component</th><th>Value</th></tr>"
            f"<tr><td>D_xx</td><td>{dxx:.6e}</td></tr>"
            f"<tr><td>D_yy</td><td>{dyy:.6e}</td></tr>"
            f"<tr><td>D_zz</td><td>{dzz:.6e}</td></tr>"
            f"<tr><td>D_xy</td><td>{float(self.tensor[0, 1]):.6e}</td></tr>"
            f"<tr><td>D_xz</td><td>{float(self.tensor[0, 2]):.6e}</td></tr>"
            f"<tr><td>D_yz</td><td>{float(self.tensor[1, 2]):.6e}</td></tr>"
            f"<tr><td>Active VF</td><td>{self.active_volume_fraction:.6f}</td></tr>"
            "</table>"
        )


@dataclasses.dataclass
class SpecificSurfaceAreaResult:
    """Result of an interfacial specific surface area calculation."""

    ssa: float
    ssa_corrected: float
    ssa_voxel: float
    ssa_voxel_corrected: float
    face_count: int
    total_count: int
    face_counts_xyz: tuple[int, int, int]

    def _repr_html_(self) -> str:
        fx, fy, fz = self.face_counts_xyz
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>SSA (raw)</td><td>{self.ssa:.6e}</td></tr>"
            f"<tr><td>SSA (2/3 Cauchy–Crofton)</td><td>{self.ssa_corrected:.6e}</td></tr>"
            f"<tr><td>SSA (voxel⁻¹)</td><td>{self.ssa_voxel:.6f}</td></tr>"
            f"<tr><td>Interfacial faces (x, y, z)</td><td>{self.face_count:,} ({fx:,}, {fy:,}, {fz:,})</td></tr>"
            f"<tr><td>Evaluated voxels</td><td>{self.total_count:,}</td></tr>"
            "</table>"
        )


@dataclasses.dataclass
class ConnectedComponentsResult:
    """Result of 6-connected component labeling."""

    num_components: int
    volumes: np.ndarray
    labels: Optional[np.ndarray] = None

    def _repr_html_(self) -> str:
        largest = int(np.max(self.volumes)) if self.volumes.size > 0 else 0
        median = float(np.median(self.volumes)) if self.volumes.size > 0 else 0.0
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Connected components</td><td>{self.num_components:,}</td></tr>"
            f"<tr><td>Largest volume (voxels)</td><td>{largest:,}</td></tr>"
            f"<tr><td>Median volume (voxels)</td><td>{median:.1f}</td></tr>"
            "</table>"
        )


@dataclasses.dataclass
class ParticleSizeDistributionResult:
    """Result of equivalent-sphere particle size distribution analysis."""

    num_particles: int
    mean_radius: float
    radii: np.ndarray
    volumes: np.ndarray
    radii_voxels: np.ndarray
    volumes_voxels: np.ndarray

    def _repr_html_(self) -> str:
        if self.radii.size > 0:
            p10, p50, p90 = np.percentile(self.radii, [10, 50, 90])
            pct_rows = (
                f"<tr><td>R10 / R50 / R90</td>"
                f"<td>{p10:.4g} / {p50:.4g} / {p90:.4g}</td></tr>"
            )
        else:
            pct_rows = ""
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Particles</td><td>{self.num_particles:,}</td></tr>"
            f"<tr><td>Mean radius</td><td>{self.mean_radius:.6g}</td></tr>"
            f"{pct_rows}"
            "</table>"
        )


@dataclasses.dataclass
class ThroughThicknessResult:
    """Result of a slice-by-slice through-thickness volume fraction profile."""

    direction: str
    num_slices: int
    profile: np.ndarray
    positions: np.ndarray

    def _repr_html_(self) -> str:
        p_min = float(np.min(self.profile)) if self.profile.size > 0 else 0.0
        p_mean = float(np.mean(self.profile)) if self.profile.size > 0 else 0.0
        p_max = float(np.max(self.profile)) if self.profile.size > 0 else 0.0
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Direction</td><td>{self.direction}</td></tr>"
            f"<tr><td>Slices</td><td>{self.num_slices}</td></tr>"
            f"<tr><td>Profile (min / mean / max)</td><td>{p_min:.4f} / {p_mean:.4f} / {p_max:.4f}</td></tr>"
            "</table>"
        )


@dataclasses.dataclass
class REVStudyResult:
    """Result of a Representative Elementary Volume (REV) convergence study."""

    sizes: list[int]
    samples: list[dict]
    mean_diagonal: dict[int, np.ndarray]
    std_diagonal: dict[int, np.ndarray]
    csv_path: Optional[str] = None

    def _repr_html_(self) -> str:
        rows = [
            "<table>",
            "<tr><th>Window L</th><th>Mean D_xx</th><th>Mean D_yy</th><th>Mean D_zz</th><th>Mean ± SD (trace/3)</th></tr>",
        ]
        for sz in self.sizes:
            m = self.mean_diagonal[sz]
            s = self.std_diagonal[sz]
            m_iso = float(np.mean(m))
            s_iso = float(np.mean(s))
            rows.append(
                f"<tr><td>{sz}</td>"
                f"<td>{m[0]:.5f}</td><td>{m[1]:.5f}</td><td>{m[2]:.5f}</td>"
                f"<td>{m_iso:.5f} ± {s_iso:.5f}</td></tr>"
            )
        rows.append("</table>")
        return "".join(rows)


# ---------------------------------------------------------------------------
# Backend detection & input normalisation helpers
# ---------------------------------------------------------------------------

def _is_pure_python() -> bool:
    """Return True if we should use the pure-Python solver backend."""
    return Session._pure_python


def _get_core():
    """Import and return the _core C extension (lazy, cached by Python)."""
    from . import _load_core

    return _load_core()


_DIR_MAP = {"x": 0, "y": 1, "z": 2}
_DIR_NAMES = {0: "x", 1: "y", 2: "z"}


def _normalize_voxel_size(voxel_size: Optional[VoxelSizeArg]) -> tuple[float, float, float]:
    """Normalise scalar or 3-element ``voxel_size`` into ``(dz, dy, dx)`` positive floats."""
    if voxel_size is None:
        return (1.0, 1.0, 1.0)
    if isinstance(voxel_size, (int, float, np.floating, np.integer)):
        v = float(voxel_size)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"voxel_size must be a finite positive number, got {voxel_size!r}")
        return (v, v, v)
    try:
        seq = tuple(float(x) for x in voxel_size)
    except TypeError as exc:
        raise TypeError(
            "voxel_size must be a positive float or a 3-tuple (dz, dy, dx)"
        ) from exc
    if len(seq) != 3 or any((not np.isfinite(v) or v <= 0.0) for v in seq):
        raise ValueError(
            f"voxel_size must have 3 positive finite entries (dz, dy, dx), got {voxel_size!r}"
        )
    return (seq[0], seq[1], seq[2])


def _normalize_phase_diffusivities(
    phase_diffusivities: Optional[Mapping[int, float]],
) -> dict[int, float]:
    """Validate and normalise ``phase_diffusivities`` into ``dict[int, float]``."""
    if phase_diffusivities is None:
        return {}
    if not isinstance(phase_diffusivities, Mapping) or len(phase_diffusivities) == 0:
        raise ValueError("phase_diffusivities must be a non-empty mapping {phase_id: coeff}")
    out: dict[int, float] = {}
    for k, v in phase_diffusivities.items():
        pid = int(k)
        coeff = float(v)
        if not np.isfinite(coeff) or coeff < 0.0:
            raise ValueError(
                f"Diffusivity for phase {pid} must be finite and >= 0, got {v!r}"
            )
        out[pid] = coeff
    return out


def _parse_direction_int(d) -> int:
    """Parse a direction to an integer (0=x, 1=y, 2=z)."""
    if isinstance(d, int):
        if d in (0, 1, 2):
            return d
        raise ValueError(f"Unknown direction '{d}'. Use 'x', 'y', or 'z'.")
    if isinstance(d, str):
        key = d.strip().lower()
        if key in _DIR_MAP:
            return _DIR_MAP[key]
    try:
        return {"X": 0, "Y": 1, "Z": 2}[str(d).split(".")[-1]]
    except (KeyError, AttributeError):
        pass
    raise ValueError(f"Unknown direction '{d}'. Use 'x', 'y', or 'z'.")


def _parse_direction(d):
    """Parse a direction string ('x', 'y', 'z') or Direction enum value."""
    _core = _get_core()
    if isinstance(d, _core.Direction):
        return d
    direction_map = {"x": _core.Direction.X, "y": _core.Direction.Y, "z": _core.Direction.Z}
    key = d.strip().lower()
    if key not in direction_map:
        raise ValueError(f"Unknown direction '{d}'. Use 'x', 'y', or 'z'.")
    return direction_map[key]


def _ensure_initialized():
    """Ensure a Session is active (auto-starting one for interactive/notebook use)."""
    Session.ensure_active()
    if _is_pure_python():
        return
    _core = _get_core()
    if not _core.amrex_initialized():
        raise RuntimeError(
            "OpenImpala AMReX runtime is not initialized. Wrap MPI scripts in:\n\n"
            "with openimpala.Session():\n"
            "    openimpala.volume_fraction(...)"
        )


def _parse_solver(s):
    """Parse a solver string or SolverType enum value."""
    _core = _get_core()
    if isinstance(s, _core.SolverType):
        return s
    key = s.strip().lower() if isinstance(s, str) else s
    if isinstance(key, str) and key in ("mlmg", "auto"):
        return "mlmg"
    solver_map = {
        "jacobi": _core.SolverType.Jacobi,
        "gmres": _core.SolverType.GMRES,
        "flexgmres": _core.SolverType.FlexGMRES,
        "pcg": _core.SolverType.PCG,
        "bicgstab": _core.SolverType.BiCGSTAB,
        "smg": _core.SolverType.SMG,
        "pfmg": _core.SolverType.PFMG,
        "hypre": _core.SolverType.FlexGMRES,
    }
    if not isinstance(key, str) or key not in solver_map:
        raise ValueError(
            f"Unknown solver '{s}'. Options: {list(solver_map) + ['mlmg', 'auto']}"
        )
    return solver_map[key]


def _parse_hypre_solver(s):
    """Parse a HYPRE SolverType for homogenization / REV cell problems."""
    _core = _get_core()
    if isinstance(s, _core.SolverType):
        return s
    key = s.strip().lower() if isinstance(s, str) else s
    if isinstance(key, str) and key in ("auto", "hypre", "flexgmres"):
        return _core.SolverType.FlexGMRES
    solver_map = {
        "jacobi": _core.SolverType.Jacobi,
        "gmres": _core.SolverType.GMRES,
        "pcg": _core.SolverType.PCG,
        "bicgstab": _core.SolverType.BiCGSTAB,
        "smg": _core.SolverType.SMG,
        "pfmg": _core.SolverType.PFMG,
    }
    if not isinstance(key, str) or key not in solver_map:
        raise ValueError(
            f"Unknown homogenization solver '{s}'. Options: {list(solver_map) + ['flexgmres', 'auto']}"
        )
    return solver_map[key]


_KRYLOV_SOLVERS = {"pcg", "gmres", "flexgmres", "bicgstab"}


def _parse_preconditioner(p):
    """Parse a preconditioner string or PrecondType enum value."""
    _core = _get_core()
    if isinstance(p, _core.PrecondType):
        return p
    precond_map = {
        "smg": _core.PrecondType.SMG,
        "pfmg": _core.PrecondType.PFMG,
    }
    key = p.strip().lower()
    if key not in precond_map:
        raise ValueError(f"Unknown preconditioner '{p}'. Options: {list(precond_map)}")
    return precond_map[key]


def _auto_grid_size(shape: tuple[int, ...]) -> int:
    """Pick a good AMReX max_grid_size based on domain dimensions and MPI ranks."""
    n_max = max(shape)
    n_ranks = 1
    if not _is_pure_python():
        try:
            n_ranks = _get_core().mpi_ranks()
        except Exception:
            n_ranks = 1

    if n_ranks <= 1:
        return max(16, min(256, int(n_max)))

    n_min = min(shape)
    total_voxels = int(np.prod(shape))
    mgs = 16
    for candidate in (128, 64, 32, 16):
        if n_min >= candidate and (total_voxels // (candidate**3)) >= n_ranks:
            mgs = candidate
            break
    return mgs


def _is_zarr_path(p: Path) -> bool:
    """Return True if *p* looks like a Zarr directory store."""
    if p.suffix.lower() == ".zarr":
        return True
    if p.is_dir():
        return any((p / marker).exists() for marker in (".zgroup", ".zarray", "zarr.json"))
    return False


def _coerce_phase_array(arr: np.ndarray, *, source_desc: str = "input") -> np.ndarray:
    """Convert *arr* to C-contiguous ``int32`` phase IDs, rejecting unsegmented floats."""
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)) or not np.array_equal(arr, np.trunc(arr)):
            raise ValueError(
                f"Expected integer phase IDs in {source_desc}, but got floating-point "
                f"grayscale values (dtype={arr.dtype}). Segment the volume first with "
                f"oi.read_image(..., threshold='otsu') or oi.binarize_otsu(...)."
            )
    return np.ascontiguousarray(arr, dtype=np.int32)


def _unwrap_external_volume(
    data: object,
    voxel_size: Optional[VoxelSizeArg],
) -> tuple[Optional[np.ndarray], Optional[VoxelSizeArg], str]:
    """Extract ``(ndarray, inferred_voxel_size, source_desc)`` from upstream ecosystem objects.

    Supports STFC Core Imaging Library (CIL) ``ImageData`` (``.as_array()`` +
    ``.geometry.voxel_size_z/y/x``), ``xarray.DataArray`` (``.values`` +
    uniform ``coords`` spacing), and in-memory 3-D array/dataset handles
    (e.g. ``zarr.Array``, ``h5py.Dataset``, ``dask.array``) without requiring
    those packages at install time.
    """
    from ._io import PureVoxelImage

    if isinstance(data, (np.ndarray, str, os.PathLike, PureVoxelImage)):
        return None, voxel_size, "NumPy array"

    # 1. STFC Core Imaging Library (CIL) ImageData / AcquisitionData
    as_array_fn = getattr(data, "as_array", None)
    if callable(as_array_fn):
        arr = np.asarray(as_array_fn())
        eff_vz = voxel_size
        geom = getattr(data, "geometry", None)
        if eff_vz is None and geom is not None:
            vz_attrs = (
                getattr(geom, "voxel_size_z", None),
                getattr(geom, "voxel_size_y", None),
                getattr(geom, "voxel_size_x", None),
            )
            if all(v is not None for v in vz_attrs):
                try:
                    cand = (float(vz_attrs[0]), float(vz_attrs[1]), float(vz_attrs[2]))
                    if all(np.isfinite(c) and c > 0.0 for c in cand):
                        eff_vz = cand
                except (TypeError, ValueError):
                    pass
        return arr, eff_vz, f"CIL {type(data).__name__}"

    # 2. xarray.DataArray (dims + coords + values)
    if hasattr(data, "dims") and hasattr(data, "coords") and hasattr(data, "values"):
        arr = np.asarray(getattr(data, "values"))
        eff_vz = voxel_size
        dims = getattr(data, "dims")
        coords = getattr(data, "coords")
        if eff_vz is None and isinstance(dims, tuple) and len(dims) == 3:
            spacings: list[float] = []
            for dname in dims:
                if dname in coords:
                    cvals = np.asarray(coords[dname])
                    if cvals.ndim == 1 and cvals.size >= 2 and np.issubdtype(cvals.dtype, np.number):
                        step = abs(float(cvals[1]) - float(cvals[0]))
                        if np.isfinite(step) and step > 0.0:
                            spacings.append(step)
            if len(spacings) == 3:
                eff_vz = (spacings[0], spacings[1], spacings[2])
        return arr, eff_vz, f"{type(data).__name__}"

    # 3. Generic 3-D array-protocol or sliceable dataset (zarr.Array, h5py.Dataset, etc.)
    if hasattr(data, "__array__") or (
        hasattr(data, "shape") and hasattr(data, "dtype") and hasattr(data, "__getitem__")
    ):
        try:
            arr = np.asarray(data[...]) if not hasattr(data, "__array__") else np.asarray(data)
            return arr, voxel_size, f"{type(data).__name__}"
        except Exception:
            pass

    return None, voxel_size, "input"


def _resolve_pure_input(
    data: DataInputArg,
    voxel_size: Optional[VoxelSizeArg] = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resolve *data* into ``(arr_int32, (dz, dy, dx))`` for the pure-Python backend."""
    from ._io import PureVoxelImage, read_zarr_volume

    if isinstance(data, (str, os.PathLike)):
        p = Path(data)
        if p.suffix.lower() == ".npy":
            raw_arr = np.load(p)
            if raw_arr.ndim != 3:
                raise ValueError(f"Expected a 3-D array in '{p}', got shape {raw_arr.shape}")
            return (
                _coerce_phase_array(raw_arr, source_desc=f"'{p}'"),
                _normalize_voxel_size(voxel_size),
            )
        if _is_zarr_path(p):
            zarr_arr, meta_vz = read_zarr_volume(p, threshold=None)
            vz = (
                _normalize_voxel_size(voxel_size)
                if voxel_size is not None
                else (meta_vz if meta_vz is not None else (1.0, 1.0, 1.0))
            )
            return _coerce_phase_array(zarr_arr, source_desc=f"'{p}'"), vz
        _, img = read_image(p, voxel_size=voxel_size)
        return np.asarray(img, dtype=np.int32), img.voxel_size

    if isinstance(data, PureVoxelImage):
        vz = data.voxel_size if voxel_size is None else _normalize_voxel_size(voxel_size)
        return np.asarray(data, dtype=np.int32), vz

    ext_arr, ext_vz, ext_desc = _unwrap_external_volume(data, voxel_size)
    if ext_arr is not None:
        if ext_arr.ndim != 3:
            raise ValueError(f"Expected a 3-D volume from {ext_desc}, got shape {ext_arr.shape}")
        return _coerce_phase_array(ext_arr, source_desc=ext_desc), _normalize_voxel_size(ext_vz)

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a 3-D NumPy array, VoxelImage, CIL/xarray volume, or file path")
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {data.shape}")
    return _coerce_phase_array(data, source_desc="NumPy array"), _normalize_voxel_size(voxel_size)


def _numpy_to_voxelimage(
    data: DataInputArg,
    max_grid_size: Union[int, str] = "auto",
    voxel_size: Optional[VoxelSizeArg] = None,
):
    """Convert a 3-D NumPy array, path, CIL/xarray volume, or VoxelImage into a C++ VoxelImage."""
    from ._io import PureVoxelImage, read_zarr_volume

    _core = _get_core()

    if isinstance(data, _core.VoxelImage):
        if voxel_size is None:
            return data
        dz, dy, dx = _normalize_voxel_size(voxel_size)
        return data.with_voxel_size(dx=dx, dy=dy, dz=dz)

    src_desc = "NumPy array"
    if isinstance(data, (str, os.PathLike)):
        p = Path(data)
        src_desc = f"'{p}'"
        if p.suffix.lower() == ".npy":
            data = np.load(p)
        elif _is_zarr_path(p):
            zarr_arr, meta_vz = read_zarr_volume(p, threshold=None)
            data = zarr_arr
            if voxel_size is None and meta_vz is not None:
                voxel_size = meta_vz
        else:
            _, img = read_image(p, voxel_size=voxel_size, max_grid_size=max_grid_size)
            return img

    if isinstance(data, PureVoxelImage):
        vz = data.voxel_size if voxel_size is None else _normalize_voxel_size(voxel_size)
        data = data.to_numpy()
        voxel_size = vz

    if not isinstance(data, np.ndarray):
        ext_arr, ext_vz, ext_desc = _unwrap_external_volume(data, voxel_size)
        if ext_arr is not None:
            data = ext_arr
            voxel_size = ext_vz
            src_desc = ext_desc

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a 3-D NumPy array, VoxelImage, CIL/xarray volume, or file path")

    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {data.shape}")

    if isinstance(max_grid_size, str) and max_grid_size.lower() == "auto":
        max_grid_size = _auto_grid_size(data.shape)

    dz, dy, dx = _normalize_voxel_size(voxel_size)
    arr = _coerce_phase_array(data, source_desc=src_desc)
    return _core.VoxelImage.from_numpy(arr, max_grid_size, dx=dx, dy=dy, dz=dz)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def volume_fraction(
    data: DataInputArg,
    phase: int,
    *,
    max_grid_size: Union[int, str] = "auto",
) -> VolumeFractionResult:
    """Compute the volume fraction of *phase* in a 3-D NumPy array, VoxelImage, or file."""
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver

        arr, _ = _resolve_pure_input(data)
        pc, tc, frac = _solver.volume_fraction(arr, phase)
        return VolumeFractionResult(phase_count=pc, total_count=tc, fraction=frac)

    _core = _get_core()
    img = _numpy_to_voxelimage(data, max_grid_size)
    vf = _core.VolumeFraction(img, phase, 0)
    pc, tc = vf.value()
    frac = pc / tc if tc > 0 else 0.0
    return VolumeFractionResult(phase_count=pc, total_count=tc, fraction=frac)


def percolation_check(
    data: DataInputArg,
    phase: int,
    direction: Union[str, "Direction"] = "x",
    *,
    max_grid_size: Union[int, str] = "auto",
    verbose: int = 0,
) -> PercolationResult:
    """Check whether *phase* percolates across the domain in *direction*."""
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver

        arr, _ = _resolve_pure_input(data)
        d = _parse_direction_int(direction)
        percolates, active_vf, _ = _solver.percolation_check(arr, phase, 2 - d)
        return PercolationResult(
            percolates=percolates,
            active_volume_fraction=active_vf,
            direction=_DIR_NAMES[d].upper(),
        )

    _core = _get_core()
    d = _parse_direction(direction)
    img = _numpy_to_voxelimage(data, max_grid_size)
    pc = _core.PercolationCheck(img, phase, d, verbose)
    return PercolationResult(
        percolates=pc.percolates,
        active_volume_fraction=pc.active_volume_fraction,
        direction=_core.PercolationCheck.direction_string(d),
    )


def tortuosity(
    data: DataInputArg,
    phase: Optional[int] = None,
    direction: Union[str, "Direction"] = "x",
    solver: Union[str, "SolverType"] = "auto",
    *,
    voxel_size: Optional[VoxelSizeArg] = None,
    phase_diffusivities: Optional[Mapping[int, float]] = None,
    return_field: bool = False,
    write_plotfile: bool = False,
    preconditioner: Union[str, "PrecondType"] = "smg",
    max_grid_size: Union[int, str] = "auto",
    results_path: str = ".",
    verbose: int = 0,
    mlmg_eps: Optional[float] = None,
    mlmg_maxiter: Optional[int] = None,
    mlmg_max_coarsening_level: Optional[int] = None,
) -> TortuosityResult:
    """Compute steady-state tortuosity and effective diffusivity along *direction*.

    Parameters
    ----------
    data : np.ndarray, VoxelImage, or path-like
        3-D integer array of phase IDs ``(Nz, Ny, Nx)``, ``VoxelImage`` handle,
        or path to a supported 3-D volume file (TIFF, HDF5, RAW, DAT, NPY, Zarr).
    phase : int
        Conducting phase ID (required when *phase_diffusivities* is ``None``).
    direction : str or Direction
        Flow direction (``'x'``, ``'y'``, or ``'z'``).
    solver : str or SolverType
        ``'auto'`` / ``'mlmg'`` (default matrix-free AMReX geometric multigrid),
        or HYPRE solvers ``'pcg'``, ``'flexgmres'``, ``'gmres'``, ``'bicgstab'``,
        ``'smg'``, ``'pfmg'``, ``'jacobi'``.
    voxel_size : float or tuple of 3 floats, optional
        Physical voxel spacing: either a scalar ``h`` (cubic voxels) or
        ``(dz, dy, dx)`` matching NumPy ``(Nz, Ny, Nx)`` axis order.
    phase_diffusivities : Mapping[int, float], optional
        Multi-phase transport map ``{phase_id: D_phase}``.  Phases with ``D > 0``
        participate in 6-connected transport with harmonic-mean face conductances;
        unlisted phases or phases with ``D == 0`` are insulating.
    return_field : bool
        If ``True``, populate ``result.solution_field`` with the solved 3-D
        potential field ``(Nz, Ny, Nx)`` (``float64``).  Defaults to ``False``
        to avoid host memory copies on large grids.
    write_plotfile : bool
        If ``True``, write an AMReX ``HyperCLaw-V1.1`` plotfile directory under
        *results_path* (readable out-of-core via ``yt.load()`` and ParaView)
        and populate ``result.plotfile_path``.
    """
    phase_map = _normalize_phase_diffusivities(phase_diffusivities)
    if phase is None and not phase_map:
        raise TypeError(
            "tortuosity() missing required argument: 'phase' "
            "(pass e.g. phase=0 or phase=1, or specify phase_diffusivities={...})"
        )
    eff_phase = int(phase) if phase is not None else int(next(iter(phase_map)))
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver
        from ._io import write_amrex_plotfile

        arr, vz = _resolve_pure_input(data, voxel_size=voxel_size)
        d = _parse_direction_int(direction)
        need_field = bool(return_field or write_plotfile)
        result = _solver.solve_tortuosity(
            arr,
            phase=eff_phase,
            direction=2 - d,
            voxel_size=vz,
            phase_diffusivities=phase_map if phase_map else None,
            return_field=need_field,
        )
        if not result["solver_converged"]:
            raise ConvergenceError(
                f"Solver did not converge after {result['iterations']} "
                f"iterations (residual={result['residual_norm']:.2e})"
            )
        plt_path: Optional[str] = None
        if write_plotfile:
            _, _, active_mask = _solver.percolation_check(
                arr, eff_phase, 2 - d, phase_diffusivities=phase_map if phase_map else None
            )
            is_mlmg = isinstance(solver, str) and solver.strip().lower() in ("auto", "mlmg")
            subdir = f"tortuosity_mlmg_{d}" if is_mlmg else f"tortuosity_solution_{d}"
            out_dir = write_amrex_plotfile(
                Path(results_path) / subdir,
                {
                    "solution_potential": result["solution_field"],
                    "phase_id": arr.astype(np.float64),
                    "active_mask": active_mask.astype(np.float64),
                },
                voxel_size=vz,
            )
            plt_path = str(out_dir)

        return TortuosityResult(
            tortuosity=result["tortuosity"],
            solver_converged=result["solver_converged"],
            iterations=result["iterations"],
            residual_norm=result["residual_norm"],
            flux_in=result["flux_in"],
            flux_out=result["flux_out"],
            active_volume_fraction=result["active_volume_fraction"],
            solution_field=result["solution_field"] if return_field else None,
            effective_diffusivity=result["effective_diffusivity"],
            plotfile_path=plt_path,
        )

    # --- C++ backend path ---
    _core = _get_core()
    d = _parse_direction(direction)
    d_int = _parse_direction_int(direction)
    st = _parse_solver(solver)
    img = _numpy_to_voxelimage(data, max_grid_size, voxel_size=voxel_size)

    if phase_map:
        vf_val = sum(
            _core.VolumeFraction(img, pid, 0).value_vf()
            for pid, coeff in phase_map.items()
            if coeff > 0.0
        )
    else:
        vf_val = _core.VolumeFraction(img, eff_phase, 0).value_vf()

    if vf_val <= 0.0:
        dir_name = _core.PercolationCheck.direction_string(d)
        raise PercolationError(
            f"Phase {eff_phase} does not percolate in the {dir_name} direction. "
            f"The Krylov solver cannot converge when the conducting phase is "
            f"disconnected between the inlet and outlet faces."
        )

    if write_plotfile:
        if _core.is_io_rank():
            Path(results_path).mkdir(parents=True, exist_ok=True)
        if _core.mpi_ranks() > 1:
            _core.mpi_bcast_double(0.0)

    if st == "mlmg":
        mlmg_kwargs = {}
        if mlmg_eps is not None:
            mlmg_kwargs["eps"] = float(mlmg_eps)
        if mlmg_maxiter is not None:
            mlmg_kwargs["maxiter"] = int(mlmg_maxiter)
        if mlmg_max_coarsening_level is not None:
            mlmg_kwargs["max_coarsening_level"] = int(mlmg_max_coarsening_level)
        if phase_map:
            mlmg_kwargs["phase_coeffs"] = phase_map
        solver_obj = _core.TortuosityMLMG(
            img,
            vf_val,
            eff_phase,
            d,
            results_path,
            0.0,
            1.0,
            verbose,
            bool(write_plotfile),
            **mlmg_kwargs,
        )
        plt_subdir = f"tortuosity_mlmg_{d_int}"
    else:
        pc = _parse_preconditioner(preconditioner)
        solver_obj = _core.TortuosityHypre(
            img,
            vf_val,
            eff_phase,
            d,
            st,
            results_path,
            0.0,
            1.0,
            verbose,
            bool(write_plotfile),
            pc,
            phase_map,
        )
        plt_subdir = f"tortuosity_solution_{d_int}"

    if solver_obj.active_volume_fraction <= 0.0:
        dir_name = _core.PercolationCheck.direction_string(d)
        raise PercolationError(
            f"Phase {eff_phase} does not percolate in the {dir_name} direction. "
            f"The Krylov solver cannot converge when the conducting phase is "
            f"disconnected between the inlet and outlet faces."
        )

    try:
        tau = solver_obj.value()
    except RuntimeError as exc:
        raise ConvergenceError(str(exc)) from exc

    if not solver_obj.solver_converged:
        solver_name = "MLMG" if st == "mlmg" else "HYPRE"
        raise ConvergenceError(
            f"{solver_name} solver did not converge after {solver_obj.iterations} "
            f"iterations (residual={solver_obj.residual_norm:.2e})"
        )

    field = solver_obj.solution_field() if return_field else None
    plt_path_cpp = str(Path(results_path) / plt_subdir) if write_plotfile else None

    return TortuosityResult(
        tortuosity=tau,
        solver_converged=solver_obj.solver_converged,
        iterations=solver_obj.iterations,
        residual_norm=solver_obj.residual_norm,
        flux_in=solver_obj.flux_in,
        flux_out=solver_obj.flux_out,
        active_volume_fraction=solver_obj.active_volume_fraction,
        solution_field=field,
        effective_diffusivity=float(solver_obj.effective_diffusivity),
        plotfile_path=plt_path_cpp,
    )


def tortuosity_all(
    data: DataInputArg,
    phase: Optional[int] = None,
    solver: Union[str, "SolverType"] = "auto",
    *,
    directions: Sequence[Union[str, "Direction"]] = ("x", "y", "z"),
    strict: bool = False,
    voxel_size: Optional[VoxelSizeArg] = None,
    phase_diffusivities: Optional[Mapping[int, float]] = None,
    return_field: bool = False,
    write_plotfile: bool = False,
    preconditioner: Union[str, "PrecondType"] = "smg",
    max_grid_size: Union[int, str] = "auto",
    results_path: str = ".",
    verbose: int = 0,
    mlmg_eps: Optional[float] = None,
    mlmg_maxiter: Optional[int] = None,
    mlmg_max_coarsening_level: Optional[int] = None,
) -> dict[str, Optional[TortuosityResult]]:
    """Compute steady-state tortuosity across cardinal directions ``('x', 'y', 'z')``.

    Loads or stages *data* once, then solves each direction in *directions*.
    When ``strict=False`` (default), directions that do not percolate map to
    ``None`` instead of raising :class:`~openimpala.PercolationError`.
    """
    phase_map = _normalize_phase_diffusivities(phase_diffusivities)
    if phase is None and not phase_map:
        raise TypeError(
            "tortuosity_all() missing required argument: 'phase' "
            "(pass e.g. phase=0 or phase=1, or specify phase_diffusivities={...})"
        )
    _ensure_initialized()

    if _is_pure_python():
        arr, vz = _resolve_pure_input(data, voxel_size=voxel_size)
        staged: DataInputArg = arr
        effective_vz: Optional[VoxelSizeArg] = vz
    else:
        staged = _numpy_to_voxelimage(data, max_grid_size, voxel_size=voxel_size)
        effective_vz = None

    results: dict[str, Optional[TortuosityResult]] = {}
    for d in directions:
        d_int = _parse_direction_int(d)
        key = _DIR_NAMES[d_int]
        try:
            results[key] = tortuosity(
                staged,
                phase=phase,
                direction=key,
                solver=solver,
                voxel_size=effective_vz,
                phase_diffusivities=phase_diffusivities,
                return_field=return_field,
                write_plotfile=write_plotfile,
                preconditioner=preconditioner,
                max_grid_size=max_grid_size,
                results_path=results_path,
                verbose=verbose,
                mlmg_eps=mlmg_eps,
                mlmg_maxiter=mlmg_maxiter,
                mlmg_max_coarsening_level=mlmg_max_coarsening_level,
            )
        except PercolationError:
            if strict:
                raise
            results[key] = None
    return results


def effective_diffusivity(
    data: DataInputArg,
    phase: Optional[int] = None,
    *,
    voxel_size: Optional[VoxelSizeArg] = None,
    phase_diffusivities: Optional[Mapping[int, float]] = None,
    solver: Union[str, "SolverType"] = "flexgmres",
    return_fields: bool = False,
    write_plotfile: bool = False,
    max_grid_size: Union[int, str] = "auto",
    results_path: str = ".",
    verbose: int = 0,
) -> EffectiveDiffusivityResult:
    """Compute the full 3x3 homogenised effective-diffusivity tensor ``D_eff``.

    Solves the three periodic homogenisation cell problems in ``x``, ``y``, ``z``
    and returns ``EffectiveDiffusivityResult`` with ``tensor`` ordered ``(x, y, z)``.
    """
    phase_map = _normalize_phase_diffusivities(phase_diffusivities)
    if phase is None and not phase_map:
        raise TypeError(
            "effective_diffusivity() missing required argument: 'phase' "
            "(pass e.g. phase=0 or phase=1, or specify phase_diffusivities={...})"
        )
    eff_phase = int(phase) if phase is not None else int(next(iter(phase_map)))
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver
        from ._io import write_amrex_plotfile

        arr, vz = _resolve_pure_input(data, voxel_size=voxel_size)
        need_chi = bool(return_fields or write_plotfile)
        raw = _solver.solve_effective_diffusivity(
            arr,
            phase=eff_phase,
            voxel_size=vz,
            phase_diffusivities=phase_map if phase_map else None,
            return_chi=need_chi,
        )
        if not raw["converged"]:
            raise ConvergenceError(
                f"Effective diffusivity cell-problem solver failed to converge "
                f"(residuals={raw['residuals']})"
            )
        plt_paths: Optional[dict[str, str]] = None
        if write_plotfile:
            if phase_map:
                active_mask = np.isin(
                    arr, [pid for pid, coeff in phase_map.items() if coeff > 0.0]
                ).astype(np.float64)
            else:
                active_mask = (arr == eff_phase).astype(np.float64)
            chi_map = raw["chi"]
            plt_paths = {}
            for d_idx, axis_name in enumerate(("x", "y", "z")):
                out_dir = write_amrex_plotfile(
                    Path(results_path) / f"effdiff_chi_dir{d_idx}",
                    {
                        "chi_k": chi_map[axis_name],
                        "active_mask_from_solver": active_mask,
                    },
                    voxel_size=vz,
                )
                plt_paths[axis_name] = str(out_dir)

        return EffectiveDiffusivityResult(
            tensor=np.asarray(raw["tensor"], dtype=np.float64),
            active_volume_fraction=float(raw["active_volume_fraction"]),
            solver_converged=bool(raw["converged"]),
            iterations=tuple(int(i) for i in raw["iterations"]),
            residuals=tuple(float(r) for r in raw["residuals"]),
            corrector_fields=raw.get("chi") if return_fields else None,
            plotfile_paths=plt_paths,
        )

    _core = _get_core()
    st = _parse_hypre_solver(solver)
    img = _numpy_to_voxelimage(data, max_grid_size, voxel_size=voxel_size)
    if write_plotfile:
        if _core.is_io_rank():
            Path(results_path).mkdir(parents=True, exist_ok=True)
        if _core.mpi_ranks() > 1:
            _core.mpi_bcast_double(0.0)
    raw = _core.compute_effective_diffusivity_tensor(
        img,
        phase_id=eff_phase,
        solver_type=st,
        results_path=results_path,
        verbose=verbose,
        write_plotfile=bool(write_plotfile),
        phase_coeffs=phase_map,
        return_chi=return_fields,
    )
    if not raw["converged"]:
        raise ConvergenceError(
            f"Effective diffusivity HYPRE cell-problem solver failed to converge "
            f"(iterations={raw['iterations']}, residuals={raw['residuals']})"
        )
    plt_paths_cpp: Optional[dict[str, str]] = None
    if write_plotfile:
        plt_paths_cpp = {
            axis_name: str(Path(results_path) / f"effdiff_chi_dir{d_idx}")
            for d_idx, axis_name in enumerate(("x", "y", "z"))
        }

    return EffectiveDiffusivityResult(
        tensor=np.asarray(raw["tensor"], dtype=np.float64),
        active_volume_fraction=float(raw["active_volume_fraction"]),
        solver_converged=bool(raw["converged"]),
        iterations=tuple(int(i) for i in raw["iterations"]),
        residuals=tuple(float(r) for r in raw["residuals"]),
        corrector_fields=dict(raw["chi"]) if return_fields else None,
        plotfile_paths=plt_paths_cpp,
    )


def specific_surface_area(
    data: DataInputArg,
    phase_a: int = 0,
    phase_b: int = 1,
    *,
    voxel_size: Optional[VoxelSizeArg] = None,
    boundary_padding: int = 0,
    max_grid_size: Union[int, str] = "auto",
) -> SpecificSurfaceAreaResult:
    """Compute interfacial specific surface area between *phase_a* and *phase_b*."""
    _ensure_initialized()
    if boundary_padding < 0:
        raise ValueError(f"boundary_padding must be >= 0, got {boundary_padding}")

    if _is_pure_python():
        from . import _solver

        arr, vz = _resolve_pure_input(data, voxel_size=voxel_size)
        raw = _solver.specific_surface_area(
            arr, phase_a=phase_a, phase_b=phase_b, voxel_size=vz, boundary_padding=boundary_padding
        )
        return SpecificSurfaceAreaResult(
            ssa=raw["ssa"],
            ssa_corrected=raw["ssa_corrected"],
            ssa_voxel=raw["ssa_voxel"],
            ssa_voxel_corrected=raw["ssa_voxel_corrected"],
            face_count=raw["face_count"],
            total_count=raw["total_count"],
            face_counts_xyz=raw["face_counts_xyz"],
        )

    _core = _get_core()
    img = _numpy_to_voxelimage(data, max_grid_size, voxel_size=voxel_size)
    dom_shape = img.shape  # (nz, ny, nx)
    if boundary_padding > 0 and any(s <= 2 * boundary_padding for s in dom_shape):
        raise ValueError(
            f"boundary_padding={boundary_padding} is too large for domain of shape {dom_shape}"
        )

    calc = _core.SpecificSurfaceArea(
        img, phase_a=phase_a, phase_b=phase_b, comp=0, boundary_padding=boundary_padding
    )
    (fx, fy, fz), tc = calc.direction_face_counts()
    fc = int(fx + fy + fz)
    ssa_vox = float(fc) / float(tc) if tc > 0 else 0.0
    dx, dy, dz = img.cell_size_xyz
    ssa_phys = (fx / dx + fy / dy + fz / dz) / float(tc) if tc > 0 else 0.0

    return SpecificSurfaceAreaResult(
        ssa=float(ssa_phys),
        ssa_corrected=(2.0 / 3.0) * float(ssa_phys),
        ssa_voxel=ssa_vox,
        ssa_voxel_corrected=(2.0 / 3.0) * ssa_vox,
        face_count=fc,
        total_count=int(tc),
        face_counts_xyz=(int(fx), int(fy), int(fz)),
    )


def connected_components(
    data: DataInputArg,
    phase: int,
    *,
    return_labels: bool = False,
    max_grid_size: Union[int, str] = "auto",
    verbose: int = 0,
) -> ConnectedComponentsResult:
    """Label 6-connected components of *phase*."""
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver

        arr, _ = _resolve_pure_input(data)
        n_comp, vols, labels = _solver.connected_components(
            arr, phase=phase, return_labels=return_labels
        )
        return ConnectedComponentsResult(
            num_components=n_comp,
            volumes=vols,
            labels=labels,
        )

    _core = _get_core()
    img = _numpy_to_voxelimage(data, max_grid_size)
    ccl = _core.ConnectedComponents(img, phase_id=phase, verbose=verbose)
    vols = np.asarray(ccl.component_volumes, dtype=np.int64)
    labels = ccl.label_field() if return_labels else None
    return ConnectedComponentsResult(
        num_components=int(ccl.num_components),
        volumes=vols,
        labels=labels,
    )


def particle_size_distribution(
    data: DataInputArg,
    phase: int,
    *,
    voxel_size: Optional[VoxelSizeArg] = None,
    max_grid_size: Union[int, str] = "auto",
    verbose: int = 0,
) -> ParticleSizeDistributionResult:
    """Compute equivalent-sphere radii ``R = (3V / (4*pi))^(1/3)`` of 6-connected components."""
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver

        arr, vz = _resolve_pure_input(data, voxel_size=voxel_size)
        raw = _solver.particle_size_distribution(arr, phase=phase, voxel_size=vz)
        return ParticleSizeDistributionResult(
            num_particles=raw["num_particles"],
            mean_radius=raw["mean_radius"],
            radii=raw["radii"],
            volumes=raw["volumes"],
            radii_voxels=raw["radii_voxels"],
            volumes_voxels=raw["volumes_voxels"],
        )

    _core = _get_core()
    img = _numpy_to_voxelimage(data, max_grid_size, voxel_size=voxel_size)
    psd = _core.ParticleSizeDistribution(img, phase_id=phase, verbose=verbose)
    dx, dy, dz = img.cell_size_xyz
    cell_vol = float(dx * dy * dz)
    length_scale = cell_vol ** (1.0 / 3.0)

    radii_vox = np.asarray(psd.radii, dtype=np.float64)
    vols_vox = np.asarray(psd.volumes, dtype=np.int64)
    radii_phys = radii_vox * length_scale
    vols_phys = vols_vox.astype(np.float64) * cell_vol
    mean_r_phys = float(psd.mean_radius) * length_scale

    return ParticleSizeDistributionResult(
        num_particles=int(psd.num_particles),
        mean_radius=mean_r_phys,
        radii=radii_phys,
        volumes=vols_phys,
        radii_voxels=radii_vox,
        volumes_voxels=vols_vox,
    )


def through_thickness_profile(
    data: DataInputArg,
    phase: int,
    direction: Union[str, "Direction"] = "x",
    *,
    voxel_size: Optional[VoxelSizeArg] = None,
    max_grid_size: Union[int, str] = "auto",
) -> ThroughThicknessResult:
    """Compute slice-by-slice volume fraction profile of *phase* along *direction*."""
    _ensure_initialized()
    d_int = _parse_direction_int(direction)
    dir_label = _DIR_NAMES[d_int]

    if _is_pure_python():
        from . import _solver

        arr, vz = _resolve_pure_input(data, voxel_size=voxel_size)
        prof, pos = _solver.through_thickness_profile(
            arr, phase=phase, direction_xyz=d_int, voxel_size=vz
        )
        return ThroughThicknessResult(
            direction=dir_label,
            num_slices=int(len(prof)),
            profile=prof,
            positions=pos,
        )

    _core = _get_core()
    d_enum = _parse_direction(dir_label)
    img = _numpy_to_voxelimage(data, max_grid_size, voxel_size=voxel_size)
    ttp = _core.ThroughThicknessProfile(img, phase_id=phase, dir=d_enum, comp=0)
    prof = np.asarray(ttp.volume_fraction_profile, dtype=np.float64)
    cs_xyz = img.cell_size_xyz
    h = float(cs_xyz[d_int])
    pos = (np.arange(len(prof), dtype=np.float64) + 0.5) * h
    return ThroughThicknessResult(
        direction=dir_label,
        num_slices=int(ttp.num_slices),
        profile=prof,
        positions=pos,
    )


def _summarize_rev_records(
    records: list[dict], csv_path: Optional[str]
) -> REVStudyResult:
    seen_sizes: list[int] = []
    grouped: dict[int, list[np.ndarray]] = {}
    for rec in records:
        sz = int(rec["target_size"])
        if sz not in grouped:
            seen_sizes.append(sz)
            grouped[sz] = []
        grouped[sz].append(
            np.array([rec["D_xx"], rec["D_yy"], rec["D_zz"]], dtype=np.float64)
        )

    mean_diag: dict[int, np.ndarray] = {}
    std_diag: dict[int, np.ndarray] = {}
    for sz in seen_sizes:
        arr = np.vstack(grouped[sz])
        mean_diag[sz] = np.mean(arr, axis=0)
        std_diag[sz] = np.std(arr, axis=0)

    return REVStudyResult(
        sizes=seen_sizes,
        samples=records,
        mean_diagonal=mean_diag,
        std_diagonal=std_diag,
        csv_path=csv_path,
    )


def rev_study(
    data: DataInputArg,
    sizes: Sequence[int],
    phase: Optional[int] = None,
    *,
    num_samples: int = 3,
    voxel_size: Optional[VoxelSizeArg] = None,
    solver: Union[str, "SolverType"] = "flexgmres",
    csv_path: Optional[str] = None,
    max_grid_size: Union[int, str] = "auto",
    verbose: int = 0,
) -> REVStudyResult:
    """Run a Representative Elementary Volume (REV) convergence study."""
    if phase is None:
        raise TypeError(
            "rev_study() missing required argument: 'phase' (pass e.g. phase=0 or phase=1)"
        )
    eff_phase = int(phase)
    _ensure_initialized()
    size_list = [int(s) for s in sizes if int(s) > 0]
    if not size_list:
        return REVStudyResult(
            sizes=[], samples=[], mean_diagonal={}, std_diagonal={}, csv_path=csv_path
        )

    if _is_pure_python():
        from . import _solver

        arr, vz = _resolve_pure_input(data, voxel_size=voxel_size)
        records = _solver.rev_study(
            arr,
            sizes=size_list,
            num_samples=num_samples,
            phase=eff_phase,
            voxel_size=vz,
            csv_path=csv_path,
        )
        return _summarize_rev_records(records, csv_path)

    _core = _get_core()
    st = _parse_hypre_solver(solver)
    img = _numpy_to_voxelimage(data, max_grid_size, voxel_size=voxel_size)
    box_sz = 32 if isinstance(max_grid_size, str) else int(max_grid_size)

    def _parse_csv_text(csv_text: str) -> list[dict]:
        rows: list[dict] = []
        if not csv_text.strip():
            return rows
        reader = csv.DictReader(csv_text.splitlines())
        for r in reader:
            dxx = float(r["D_xx"])
            dyy = float(r["D_yy"])
            dzz = float(r["D_zz"])
            dxy = float(r["D_xy"])
            dxz = float(r["D_xz"])
            dyz = float(r["D_yz"])
            tensor = np.array(
                [[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]],
                dtype=np.float64,
            )
            rows.append(
                {
                    "sample": int(r["SampleNo"]),
                    "seed_xyz": (int(r["SeedX"]), int(r["SeedY"]), int(r["SeedZ"])),
                    "target_size": int(r["REV_Size_Target"]),
                    "actual_size_xyz": (
                        int(r["ActualSizeX"]),
                        int(r["ActualSizeY"]),
                        int(r["ActualSizeZ"]),
                    ),
                    "D_xx": dxx,
                    "D_yy": dyy,
                    "D_zz": dzz,
                    "D_xy": dxy,
                    "D_xz": dxz,
                    "D_yz": dyz,
                    "tensor": tensor,
                }
            )
        return rows

    if csv_path is not None:
        target_p = Path(csv_path)
        target_p.parent.mkdir(parents=True, exist_ok=True)
        _written, csv_content = _core.run_rev_study(
            img,
            sizes=size_list,
            num_samples=num_samples,
            phase_id=eff_phase,
            box_size=box_sz,
            solver_type=st,
            results_path=str(target_p.parent),
            csv_filename=target_p.name,
            verbose=verbose,
            write_plotfiles=False,
        )
        records = _parse_csv_text(csv_content)
        return _summarize_rev_records(records, str(target_p))

    with tempfile.TemporaryDirectory() as tmpdir:
        _written, csv_content = _core.run_rev_study(
            img,
            sizes=size_list,
            num_samples=num_samples,
            phase_id=eff_phase,
            box_size=box_sz,
            solver_type=st,
            results_path=tmpdir,
            csv_filename="rev_study_Deff.csv",
            verbose=verbose,
            write_plotfiles=False,
        )
        records = _parse_csv_text(csv_content)
    return _summarize_rev_records(records, None)


def estimate_memory(
    shape: tuple[int, ...],
    num_ranks: int = 1,
    *,
    solver: str = "hypre",
    preconditioner: str = "pfmg",
    mpi_ranks: Optional[int] = None,
) -> dict:
    """Estimate peak per-rank memory usage (in bytes/MB/GB) for a 3-D transport solve.

    Uses empirically calibrated bytes-per-voxel footprints measured on the C++
    AMReX/HYPRE multigrid hierarchies across :math:`64^3`–:math:`256^3` grids:
    ``~310 B/voxel`` for matrix-free ``mlmg``, ``~365 B/voxel`` for ``pcg+pfmg``,
    ``~1030 B/voxel`` for ``flexgmres+pfmg``, and ``~1400 B/voxel`` with ``smg``.
    """
    ranks = max(int(mpi_ranks if mpi_ranks is not None else num_ranks), 1)
    total = int(np.prod(shape))
    per_rank = total / ranks
    s_norm = solver.strip().lower()
    p_norm = preconditioner.strip().lower()
    if s_norm in ("mlmg", "auto"):
        bytes_per_voxel = 310
    elif s_norm == "pure":
        bytes_per_voxel = 340
    elif s_norm == "smg" or p_norm == "smg":
        bytes_per_voxel = 1400 if s_norm in ("flexgmres", "gmres", "hypre") else 780
    elif s_norm in ("pcg", "pfmg", "bicgstab", "jacobi"):
        bytes_per_voxel = 365
    else:
        bytes_per_voxel = 1030
    bytes_per_rank = per_rank * bytes_per_voxel
    gb_val = round(bytes_per_rank / 1e9, 2)
    return {
        "total_voxels": total,
        "voxels_per_rank": int(per_rank),
        "bytes_per_voxel": bytes_per_voxel,
        "bytes_per_rank": int(bytes_per_rank),
        "mb_per_rank": round(bytes_per_rank / 1e6, 1),
        "gb_per_rank": gb_val,
        "per_rank_gb": gb_val,
        "num_ranks": ranks,
        "mpi_ranks": ranks,
        "solver": s_norm,
    }


def build_info() -> dict:
    """Return compile-time + runtime feature flags for the installed wheel."""
    from . import __version__
    from .session import _probe_core, _requested_backend

    req = _requested_backend()
    if req == "pure":
        core_ok = False
        core_err = "Disabled via OPENIMPALA_BACKEND=pure"
    elif _is_pure_python():
        core_ok = False
        core_err = Session._core_load_error
        if core_err is None:
            _, core_err = _probe_core()
    elif not _has_core():
        core_ok = False
        core_err = Session._core_load_error
    else:
        core_ok = True
        core_err = None

    if _is_pure_python() or not core_ok:
        try:
            import cupy  # noqa: F401

            has_cupy = True
        except ImportError:
            has_cupy = False
        return {
            "backend": "pure-python",
            "core_available": False,
            "core_load_error": core_err,
            "cuda_enabled": has_cupy,
            "hip_enabled": False,
            "gpu_enabled": has_cupy,
            "openmp_enabled": False,
            "openmp_max_threads": 1,
            "mpi_enabled": False,
            "mpi_ranks": 1,
            "mpi_rank": 0,
            "tiny_profile": False,
            "hypre_cuda": False,
            "hypre_hip": False,
            "gpu_device_count": -1,
            "version": __version__,
        }

    _core = _get_core()
    info = dict(_core.build_info())
    if info.get("cuda_enabled"):
        info["backend"] = "cpp-cuda"
    elif info.get("hip_enabled"):
        info["backend"] = "cpp-hip"
    else:
        info["backend"] = "cpp-cpu"
    info["core_available"] = True
    info["core_load_error"] = None
    info["version"] = __version__
    return info


def _detect_file_format(path_str: str, file_format: Optional[str]) -> str:
    if file_format is not None:
        return file_format.strip().lower()
    clean = path_str.rstrip("/\\")
    lower = clean.lower()
    if lower.endswith((".tif", ".tiff")):
        return "tiff"
    if lower.endswith((".h5", ".hdf5", ".hdf")):
        return "hdf5"
    if lower.endswith(".raw"):
        return "raw"
    if lower.endswith(".dat"):
        return "dat"
    if lower.endswith(".npy"):
        return "npy"
    if lower.endswith(".zarr") or _is_zarr_path(Path(clean)):
        return "zarr"
    raise ValueError(f"Cannot auto-detect format for '{path_str}'. Set file_format explicitly.")


_NUMPY_STR_TO_RAW_ENUM: dict[str, str] = {
    "|u1": "UINT8",
    "<u1": "UINT8",
    ">u1": "UINT8",
    "|i1": "INT8",
    "<i1": "INT8",
    ">i1": "INT8",
    "<i2": "INT16_LE",
    ">i2": "INT16_BE",
    "<u2": "UINT16_LE",
    ">u2": "UINT16_BE",
    "<i4": "INT32_LE",
    ">i4": "INT32_BE",
    "<u4": "UINT32_LE",
    ">u4": "UINT32_BE",
    "<f4": "FLOAT32_LE",
    ">f4": "FLOAT32_BE",
    "<f8": "FLOAT64_LE",
    ">f8": "FLOAT64_BE",
}


def _coerce_core_raw_data_type(_core, raw_data_type):
    """Map None, string names, or NumPy dtypes to ``_core.RawDataType``."""
    if raw_data_type is None:
        return _core.RawDataType.UINT8
    if isinstance(raw_data_type, _core.RawDataType):
        return raw_data_type
    if isinstance(raw_data_type, str):
        name = raw_data_type.strip().upper()
        if name != "UNKNOWN" and hasattr(_core.RawDataType, name):
            return getattr(_core.RawDataType, name)
        try:
            dt = np.dtype(raw_data_type)
        except TypeError as exc:
            raise ValueError(f"Unsupported raw_data_type: {raw_data_type!r}") from exc
        enum_name = _NUMPY_STR_TO_RAW_ENUM.get(dt.str)
        if enum_name is not None:
            return getattr(_core.RawDataType, enum_name)
    if isinstance(raw_data_type, np.dtype):
        enum_name = _NUMPY_STR_TO_RAW_ENUM.get(raw_data_type.str)
        if enum_name is not None:
            return getattr(_core.RawDataType, enum_name)
    raise ValueError(f"Unsupported raw_data_type: {raw_data_type!r}")


def read_image(
    path: Union[str, os.PathLike],
    threshold: Union[float, str] = 0.5,
    *,
    dark_is_pore: bool = True,
    pore_value: int = 0,
    solid_value: int = 1,
    voxel_size: Optional[VoxelSizeArg] = None,
    file_format: Optional[str] = None,
    hdf5_dataset: str = "/data",
    zarr_array_path: Optional[str] = None,
    zarr_level: int = 0,
    raw_width: int = 0,
    raw_height: int = 0,
    raw_depth: int = 0,
    raw_data_type=None,
    max_grid_size: Union[int, str] = "auto",
) -> tuple:
    """Read a 3-D image file and threshold it into pore and solid phase IDs.

    When *dark_is_pore* is ``True`` (default X-ray CT contrast), voxels with
    ``pixel > threshold`` map to *solid_value* (default ``1``) and ``<= threshold``
    map to *pore_value* (default ``0``).  Set ``dark_is_pore=False`` for inverted
    (bright-pore) images so ``pixel > threshold`` maps to *pore_value* (``0``).

    *threshold* may be a numeric cutoff or ``'otsu'`` to automatically determine
    the optimal global threshold via Otsu's between-class variance method.
    Supports TIFF, HDF5, OME-Zarr (``.zarr``), RAW, DAT, and NPY volumes across
    both compiled C++ and pure-Python backends.
    """
    _ensure_initialized()
    path_str = os.fspath(path)
    fmt = _detect_file_format(path_str, file_format)

    if isinstance(threshold, str) and threshold.strip().lower() != "otsu":
        raise ValueError(
            f"Unsupported string threshold {threshold!r}; expected a number or 'otsu'."
        )

    v_true = int(solid_value) if dark_is_pore else int(pore_value)
    v_false = int(pore_value) if dark_is_pore else int(solid_value)

    if _is_pure_python() or fmt in ("npy", "zarr"):
        from ._io import (
            PureReader,
            read_dat_volume,
            read_hdf5_volume,
            read_raw_volume,
            read_tiff_volume,
            read_zarr_volume,
        )

        meta_vz: Optional[tuple[float, float, float]] = None
        if fmt == "tiff":
            raw_vol = read_tiff_volume(path_str)
            reader = PureReader(raw_vol, kind="TiffReader")
        elif fmt == "hdf5":
            raw_vol = read_hdf5_volume(path_str, dataset=hdf5_dataset)
            reader = PureReader(raw_vol, kind="HDF5Reader")
        elif fmt == "raw":
            raw_vol = read_raw_volume(
                path_str, raw_width, raw_height, raw_depth, raw_data_type=raw_data_type
            )
            reader = PureReader(raw_vol, kind="RawReader")
        elif fmt == "dat":
            raw_vol = read_dat_volume(path_str)
            reader = PureReader(raw_vol, kind="DatReader")
        elif fmt == "npy":
            raw_vol = np.load(path_str)
            reader = PureReader(raw_vol, kind="NpyReader")
        elif fmt == "zarr":
            raw_vol, meta_vz = read_zarr_volume(
                path_str, array_path=zarr_array_path, level=zarr_level, threshold=None
            )
            reader = PureReader(raw_vol, kind="ZarrReader")
        else:
            raise ValueError(f"Unknown file_format '{fmt}'")

        pure_img = reader.threshold(threshold, v_true, v_false)
        eff_vz = voxel_size if voxel_size is not None else meta_vz
        if eff_vz is not None:
            dz, dy, dx = _normalize_voxel_size(eff_vz)
            pure_img = pure_img.with_voxel_size(dx=dx, dy=dy, dz=dz)

        if not _is_pure_python():
            cpp_img = _numpy_to_voxelimage(pure_img, max_grid_size=max_grid_size)
            return reader, cpp_img
        return reader, pure_img

    _core = _get_core()
    core_raw_dt = _coerce_core_raw_data_type(_core, raw_data_type)

    if isinstance(threshold, str):
        from ._io import (
            calculate_otsu_threshold,
            read_dat_volume,
            read_hdf5_volume,
            read_raw_volume,
            read_tiff_volume,
        )

        # Compute Otsu threshold on the I/O rank only and broadcast to avoid
        # replicating full-volume host reads across N MPI ranks per node.
        if _core.is_io_rank():
            if fmt == "tiff":
                otsu_src = read_tiff_volume(path_str)
            elif fmt == "hdf5":
                otsu_src = read_hdf5_volume(path_str, dataset=hdf5_dataset)
            elif fmt == "raw":
                otsu_src = read_raw_volume(
                    path_str, raw_width, raw_height, raw_depth, raw_data_type=raw_data_type
                )
            elif fmt == "dat":
                otsu_src = read_dat_volume(path_str)
            else:
                raise ValueError(f"Unknown file_format '{fmt}'")
            numeric_threshold = float(calculate_otsu_threshold(otsu_src))
            del otsu_src
        else:
            numeric_threshold = 0.0
        if _core.mpi_ranks() > 1:
            numeric_threshold = float(_core.mpi_bcast_double(numeric_threshold))
    else:
        numeric_threshold = float(threshold)

    if fmt == "tiff":
        reader = _core.TiffReader(path_str)
    elif fmt == "hdf5":
        reader = _core.HDF5Reader(path_str, hdf5_dataset)
    elif fmt == "raw":
        reader = _core.RawReader(path_str, raw_width, raw_height, raw_depth, core_raw_dt)
    elif fmt == "dat":
        reader = _core.DatReader(path_str)
    else:
        raise ValueError(f"Unknown file_format '{fmt}'")

    if isinstance(max_grid_size, str) and max_grid_size.lower() == "auto":
        max_grid_size = 256 if _core.mpi_ranks() <= 1 else 64

    if fmt == "dat":
        import math

        eff_threshold: float | int = max(0, min(255, int(math.floor(numeric_threshold))))
    else:
        eff_threshold = numeric_threshold

    if (v_true, v_false) == (1, 0):
        img = reader.threshold(eff_threshold, max_grid_size)
    else:
        img = reader.threshold(eff_threshold, v_true, v_false, max_grid_size)
    if voxel_size is not None:
        dz, dy, dx = _normalize_voxel_size(voxel_size)
        img = img.with_voxel_size(dx=dx, dy=dy, dz=dz)

    return reader, img
