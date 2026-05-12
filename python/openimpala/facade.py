"""High-level, NumPy-native facade for common OpenImpala workflows.

These functions accept plain ``numpy.ndarray`` inputs and return rich
Python dataclasses, hiding all AMReX boilerplate from general users.

When the compiled C++ backend (_core) is unavailable, functions
transparently fall back to the pure-Python solver (SciPy/CuPy).
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Union

import numpy as np

from .exceptions import ConvergenceError, PercolationError
from .session import Session


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


@dataclasses.dataclass
class TortuosityResult:
    """Result of a tortuosity calculation."""

    tortuosity: float
    solver_converged: bool
    iterations: int
    residual_norm: float
    flux_in: float
    flux_out: float
    active_volume_fraction: float
    solution_field: "np.ndarray | None" = None

    def _repr_html_(self) -> str:
        status = "converged" if self.solver_converged else "NOT converged"
        return (
            "<table>"
            "<tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Tortuosity</td><td>{self.tortuosity:.6f}</td></tr>"
            f"<tr><td>Solver status</td><td>{status}</td></tr>"
            f"<tr><td>Iterations</td><td>{self.iterations}</td></tr>"
            f"<tr><td>Residual norm</td><td>{self.residual_norm:.2e}</td></tr>"
            f"<tr><td>Flux in</td><td>{self.flux_in:.6e}</td></tr>"
            f"<tr><td>Flux out</td><td>{self.flux_out:.6e}</td></tr>"
            f"<tr><td>Active VF</td><td>{self.active_volume_fraction:.6f}</td></tr>"
            "</table>"
        )


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _is_pure_python() -> bool:
    """Return True if we should use the pure-Python solver backend."""
    return Session._pure_python


def _get_core():
    """Import and return the _core C extension (lazy, cached by Python)."""
    import importlib
    return importlib.import_module("openimpala._core")


# ---------------------------------------------------------------------------
# Direction / solver helpers
# ---------------------------------------------------------------------------

_DIR_MAP = {"x": 0, "y": 1, "z": 2}
_DIR_NAMES = {0: "x", 1: "y", 2: "z"}


def _parse_direction_int(d) -> int:
    """Parse a direction to an integer (0=x, 1=y, 2=z)."""
    if isinstance(d, int):
        return d
    if isinstance(d, str):
        key = d.strip().lower()
        if key in _DIR_MAP:
            return _DIR_MAP[key]
    # Might be a _core.Direction enum
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
    """Check that a Session is active."""
    if _is_pure_python():
        if Session._depth == 0:
            raise RuntimeError(
                "OpenImpala is not initialized! Please wrap your code in a session block:\n\n"
                "with openimpala.Session():\n"
                "    openimpala.volume_fraction(...)"
            )
        return
    _core = _get_core()
    if not _core.amrex_initialized():
        raise RuntimeError(
            "OpenImpala is not initialized! Please wrap your code in a session block:\n\n"
            "with openimpala.Session():\n"
            "    openimpala.volume_fraction(...)"
        )


def _parse_solver(s):
    """Parse a solver string or SolverType enum value.

    The special value ``"auto"`` selects PCG — the optimal solver for
    single-phase steady-state diffusion (the Laplacian with harmonic-mean
    face coefficients is symmetric positive-definite). Works on both CPU
    and GPU wheels.

    The value ``"mlmg"`` selects AMReX's matrix-free geometric multigrid
    solver. Bypasses HYPRE entirely; the most performant choice on GPU
    hardware for structured-grid problems and fully matrix-free (no
    assembly cost).
    """
    _core = _get_core()
    if isinstance(s, _core.SolverType):
        return s
    # "mlmg" is handled as a special string, not a SolverType enum
    if isinstance(s, str) and s.strip().lower() == "mlmg":
        return "mlmg"
    solver_map = {
        "jacobi": _core.SolverType.Jacobi,
        "gmres": _core.SolverType.GMRES,
        "flexgmres": _core.SolverType.FlexGMRES,
        "pcg": _core.SolverType.PCG,
        "bicgstab": _core.SolverType.BiCGSTAB,
        "smg": _core.SolverType.SMG,
        "pfmg": _core.SolverType.PFMG,
        "hypre": _core.SolverType.FlexGMRES,  # convenience alias
        "auto": _core.SolverType.PCG,  # SPD-optimal for single-phase diffusion
    }
    key = s.strip().lower()
    if key not in solver_map:
        raise ValueError(f"Unknown solver '{s}'. Options: {list(solver_map) + ['mlmg']}")
    return solver_map[key]


# Krylov solvers that accept a multigrid preconditioner
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
    """Pick a good AMReX max_grid_size based on domain dimensions.

    Heuristic: use the largest power-of-two that evenly divides the
    smallest dimension, clamped to [16, 128].  For small domains this
    avoids unnecessary box splitting; for large domains it keeps MPI
    load-balanced.
    """
    n_min = min(shape)
    mgs = 16
    for candidate in (128, 64, 32, 16):
        if n_min >= candidate and n_min % candidate == 0:
            mgs = candidate
            break
    return mgs


def _numpy_to_voxelimage(
    data: np.ndarray,
    max_grid_size: Union[int, str] = 32,
):
    """Convert a 3-D int32 NumPy array to a VoxelImage (native C++ ingestion).

    Returns a VoxelImage handle that encapsulates all AMReX objects.
    """
    _core = _get_core()

    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {data.shape}")

    if isinstance(max_grid_size, str) and max_grid_size.lower() == "auto":
        max_grid_size = _auto_grid_size(data.shape)

    data = np.ascontiguousarray(data, dtype=np.int32)
    return _core.VoxelImage.from_numpy(data, max_grid_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def volume_fraction(
    data: np.ndarray,
    phase: int = 0,
    *,
    max_grid_size: Union[int, str] = 32,
) -> VolumeFractionResult:
    """Compute the volume fraction of *phase* in a 3-D NumPy array.

    Parameters
    ----------
    data : np.ndarray
        3-D integer array of phase IDs (shape: z, y, x).
    phase : int
        Phase ID to count.
    max_grid_size : int or str
        AMReX box decomposition size.  ``'auto'`` picks based on domain size.

    Returns
    -------
    VolumeFractionResult
    """
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver
        data = np.ascontiguousarray(data, dtype=np.int32)
        pc, tc, frac = _solver.volume_fraction(data, phase)
        return VolumeFractionResult(phase_count=pc, total_count=tc, fraction=frac)

    _core = _get_core()

    if isinstance(data, np.ndarray):
        img = _numpy_to_voxelimage(data, max_grid_size)
    elif isinstance(data, _core.VoxelImage):
        img = data
    else:
        raise TypeError("data must be a NumPy array or a VoxelImage")

    vf = _core.VolumeFraction(img, phase, 0)
    pc, tc = vf.value()
    frac = pc / tc if tc > 0 else 0.0
    return VolumeFractionResult(phase_count=pc, total_count=tc, fraction=frac)


def percolation_check(
    data: np.ndarray,
    phase: int = 0,
    direction: Union[str, "Direction"] = "x",
    *,
    max_grid_size: Union[int, str] = 32,
    verbose: int = 0,
) -> PercolationResult:
    """Check whether *phase* percolates across the domain in *direction*.

    Parameters
    ----------
    data : np.ndarray
        3-D integer array of phase IDs.
    phase : int
        Phase ID to check.
    direction : str or Direction
        'x', 'y', or 'z'.
    max_grid_size : int or str
        AMReX box decomposition size.  ``'auto'`` picks based on domain size.

    Returns
    -------
    PercolationResult
    """
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver
        data = np.ascontiguousarray(data, dtype=np.int32)
        d = _parse_direction_int(direction)
        percolates, active_vf, _ = _solver.percolation_check(data, phase, d)
        return PercolationResult(
            percolates=percolates,
            active_volume_fraction=active_vf,
            direction=_DIR_NAMES[d],
        )

    _core = _get_core()
    d = _parse_direction(direction)

    if isinstance(data, np.ndarray):
        img = _numpy_to_voxelimage(data, max_grid_size)
    elif isinstance(data, _core.VoxelImage):
        img = data
    else:
        raise TypeError("data must be a NumPy array or a VoxelImage")

    pc = _core.PercolationCheck(img, phase, d, verbose)
    return PercolationResult(
        percolates=pc.percolates,
        active_volume_fraction=pc.active_volume_fraction,
        direction=_core.PercolationCheck.direction_string(d),
    )


def tortuosity(
    data: np.ndarray,
    phase: int = 0,
    direction: Union[str, "Direction"] = "x",
    solver: Union[str, "SolverType"] = "auto",
    *,
    preconditioner: Union[str, "PrecondType"] = "smg",
    max_grid_size: Union[int, str] = 32,
    results_path: str = ".",
    verbose: int = 0,
    mlmg_eps: Optional[float] = None,
    mlmg_maxiter: Optional[int] = None,
    mlmg_max_coarsening_level: Optional[int] = None,
) -> TortuosityResult:
    """Compute the tortuosity of *phase* in *direction*.

    Parameters
    ----------
    data : np.ndarray
        3-D integer array of phase IDs.
    phase : int
        Phase ID for the conducting phase.
    direction : str or Direction
        Flow direction ('x', 'y', 'z').
    solver : str or SolverType
        Solver algorithm.  ``'auto'`` (default) selects HYPRE PCG, the
        optimal choice for the symmetric Poisson-like operator with
        harmonic-mean face coefficients on both CPU and GPU.  ``'mlmg'``
        uses AMReX's matrix-free geometric multigrid (often the fastest
        option on GPU hardware).  Other HYPRE options: ``'flexgmres'``,
        ``'gmres'``, ``'bicgstab'``, ``'pcg'``, ``'smg'``, ``'pfmg'``,
        ``'jacobi'``.
    preconditioner : str or PrecondType, keyword-only
        Multigrid preconditioner for Krylov solvers (PCG/GMRES/FlexGMRES/BiCGSTAB):
        ``'smg'`` (default) or ``'pfmg'``.  Ignored for standalone SMG/PFMG/Jacobi
        and for MLMG.  SMG is the default because its point-based smoothing
        handles the decoupled inactive rows that arise from masked porous-media
        inputs; PFMG's semicoarsening tends to stall on those rows and may not
        converge.  Use PFMG for fully-active grids (no masked cells) — it scales
        better.  Both work on CPU and GPU.
    max_grid_size : int or str
        AMReX box decomposition size.  ``'auto'`` picks a value based on the
        domain dimensions.
    mlmg_eps, mlmg_maxiter, mlmg_max_coarsening_level
        Tuning knobs for ``solver='mlmg'``.  ``None`` leaves the C++ default
        (``mlmg_eps=1e-11``, two orders tighter than HYPRE's default because
        MLMG's relative residual norm is referenced to a different baseline —
        see TortuosityMLMG.H for the derivation).  Loosen ``mlmg_eps`` only
        for trivial geometries; tightening it further is rarely needed.

    Returns
    -------
    TortuosityResult

    Raises
    ------
    ConvergenceError
        If the solver does not converge.
    PercolationError
        If the phase does not percolate in the given direction.
    """
    _ensure_initialized()

    if _is_pure_python():
        from . import _solver
        data = np.ascontiguousarray(data, dtype=np.int32)
        d = _parse_direction_int(direction)
        result = _solver.solve_tortuosity(data, phase, d)
        if not result["solver_converged"]:
            raise ConvergenceError(
                f"Solver did not converge after {result['iterations']} "
                f"iterations (residual={result['residual_norm']:.2e})"
            )
        return TortuosityResult(
            tortuosity=result["tortuosity"],
            solver_converged=result["solver_converged"],
            iterations=result["iterations"],
            residual_norm=result["residual_norm"],
            flux_in=result["flux_in"],
            flux_out=result["flux_out"],
            active_volume_fraction=result["active_volume_fraction"],
            solution_field=result["solution_field"],
        )

    # --- C++ backend path (unchanged) ---
    _core = _get_core()
    d = _parse_direction(direction)
    st = _parse_solver(solver)

    if isinstance(data, np.ndarray):
        img = _numpy_to_voxelimage(data, max_grid_size)
    elif isinstance(data, _core.VoxelImage):
        img = data
    else:
        raise TypeError("data must be a NumPy array or a VoxelImage")

    # Pre-flight percolation check — fail fast before expensive solver
    # construction and HYPRE matrix assembly.
    pc = _core.PercolationCheck(img, phase, d, verbose)
    if not pc.percolates:
        dir_name = _core.PercolationCheck.direction_string(d)
        raise PercolationError(
            f"Phase {phase} does not percolate in the {dir_name} direction. "
            f"The Krylov solver cannot converge when the conducting phase is "
            f"disconnected between the inlet and outlet faces."
        )

    # Volume fraction (cheap — pure counting, no flood fill)
    vf_calc = _core.VolumeFraction(img, phase, 0)
    vf_val = vf_calc.value_vf()

    # Route to the appropriate solver backend
    if st == "mlmg":
        # Matrix-free AMReX MLMG solver — no HYPRE, lower memory
        mlmg_kwargs = {}
        if mlmg_eps is not None:
            mlmg_kwargs["eps"] = float(mlmg_eps)
        if mlmg_maxiter is not None:
            mlmg_kwargs["maxiter"] = int(mlmg_maxiter)
        if mlmg_max_coarsening_level is not None:
            mlmg_kwargs["max_coarsening_level"] = int(mlmg_max_coarsening_level)
        solver_obj = _core.TortuosityMLMG(
            img, vf_val, phase, d, results_path,
            0.0, 1.0, verbose, False,
            **mlmg_kwargs,
        )
    else:
        # HYPRE-based solver (PCG, FlexGMRES, etc.). The preconditioner is applied
        # to Krylov solvers; HYPRE ignores it for standalone SMG/PFMG/Jacobi.
        pc = _parse_preconditioner(preconditioner)
        solver_obj = _core.TortuosityHypre(
            img, vf_val, phase, d, st, results_path,
            0.0, 1.0, verbose, False,
            pc,
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

    return TortuosityResult(
        tortuosity=tau,
        solver_converged=solver_obj.solver_converged,
        iterations=solver_obj.iterations,
        residual_norm=solver_obj.residual_norm,
        flux_in=solver_obj.flux_in,
        flux_out=solver_obj.flux_out,
        active_volume_fraction=solver_obj.active_volume_fraction,
    )


def estimate_memory(
    shape: tuple[int, ...],
    num_ranks: int = 1,
) -> dict:
    """Estimate per-rank memory usage for a tortuosity solve.

    Uses the rule of thumb: ~80 bytes per active voxel (4 bytes phase data,
    56 bytes HYPRE matrix for 7-point stencil, 8 bytes solution field,
    ~12 bytes work arrays and ghost cells).

    Parameters
    ----------
    shape : tuple of int
        Domain dimensions (Nz, Ny, Nx).
    num_ranks : int
        Number of MPI ranks.

    Returns
    -------
    dict
        Keys: ``total_voxels``, ``voxels_per_rank``, ``bytes_per_rank``,
        ``mb_per_rank``, ``gb_per_rank``, ``num_ranks``.
    """
    total = int(np.prod(shape))
    per_rank = total / max(num_ranks, 1)
    bytes_per_rank = per_rank * 80
    return {
        "total_voxels": total,
        "voxels_per_rank": int(per_rank),
        "bytes_per_rank": int(bytes_per_rank),
        "mb_per_rank": round(bytes_per_rank / 1e6, 1),
        "gb_per_rank": round(bytes_per_rank / 1e9, 2),
        "num_ranks": num_ranks,
    }


def build_info() -> dict:
    """Return compile-time + runtime feature flags for the installed wheel.

    Critical for Colab users who need to verify they got the GPU wheel
    (``pip install openimpala-cuda``) and not the default CPU wheel.

    Returns
    -------
    dict
        Keys:

        * ``backend``: ``"cpp-cuda"``, ``"cpp-hip"``, ``"cpp-cpu"``, or ``"pure-python"``
        * ``cuda_enabled`` / ``hip_enabled`` / ``gpu_enabled``: bool
        * ``openmp_enabled``: bool; ``openmp_max_threads``: int
        * ``mpi_enabled``: bool
        * ``tiny_profile``: bool (BL_PROFILE regions emit a table at shutdown)
        * ``hypre_cuda`` / ``hypre_hip``: bool (HYPRE solver device support)
        * ``gpu_device_count``: int (runtime — ``-1`` if AMReX not yet initialised)
        * ``version``: str (package version)
    """
    from . import __version__

    if _is_pure_python():
        # Pure-Python fallback (SciPy/CuPy) — no compiled backend at all.
        # CuPy CG is GPU-accelerated but has no OpenMP / HYPRE / TinyProfile.
        try:
            import cupy  # noqa: F401
            has_cupy = True
        except ImportError:
            has_cupy = False
        return {
            "backend": "pure-python",
            "cuda_enabled": has_cupy,
            "hip_enabled": False,
            "gpu_enabled": has_cupy,
            "openmp_enabled": False,
            "openmp_max_threads": 1,
            "mpi_enabled": False,
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
    info["version"] = __version__
    return info


def read_image(
    path: str,
    threshold: float = 0.5,
    *,
    file_format: Optional[str] = None,
    hdf5_dataset: str = "/data",
    raw_width: int = 0,
    raw_height: int = 0,
    raw_depth: int = 0,
    raw_data_type=None,
    max_grid_size: Union[int, str] = 32,
) -> tuple:
    """Read a 3-D image file and threshold it into a VoxelImage.

    Automatically detects format from the file extension unless *file_format*
    is explicitly set.

    Parameters
    ----------
    path : str
        Path to the image file.
    threshold : float
        Threshold value for binarisation.
    file_format : str, optional
        Force format: 'tiff', 'hdf5', 'raw', or 'dat'.
    max_grid_size : int or str
        AMReX box decomposition size.  ``'auto'`` picks based on domain size.

    Returns
    -------
    (reader, VoxelImage)
        The reader object and the VoxelImage handle.
    """
    _ensure_initialized()

    if _is_pure_python():
        raise NotImplementedError(
            "read_image() requires the compiled C++ backend (_core). "
            "In pure-Python mode, load your image with tifffile/h5py/numpy "
            "and pass the array directly to volume_fraction() or tortuosity()."
        )

    _core = _get_core()

    if raw_data_type is None:
        raw_data_type = _core.RawDataType.UINT8

    # Auto-detect format
    if file_format is None:
        lower = path.lower()
        if lower.endswith((".tif", ".tiff")):
            file_format = "tiff"
        elif lower.endswith((".h5", ".hdf5", ".hdf")):
            file_format = "hdf5"
        elif lower.endswith(".raw"):
            file_format = "raw"
        elif lower.endswith(".dat"):
            file_format = "dat"
        else:
            raise ValueError(f"Cannot auto-detect format for '{path}'. Set file_format explicitly.")

    # Create reader and read metadata
    if file_format == "tiff":
        reader = _core.TiffReader(path)
    elif file_format == "hdf5":
        reader = _core.HDF5Reader(path, hdf5_dataset)
    elif file_format == "raw":
        reader = _core.RawReader(path, raw_width, raw_height, raw_depth, raw_data_type)
    elif file_format == "dat":
        reader = _core.DatReader(path)
    else:
        raise ValueError(f"Unknown file_format '{file_format}'")

    # Resolve "auto" — for file-based reads we don't know dimensions until
    # after the reader is constructed, so default to 32 (good general choice).
    if isinstance(max_grid_size, str) and max_grid_size.lower() == "auto":
        max_grid_size = 32

    # Threshold directly into a VoxelImage (all AMReX setup happens in C++)
    img = reader.threshold(threshold, max_grid_size)

    return reader, img
