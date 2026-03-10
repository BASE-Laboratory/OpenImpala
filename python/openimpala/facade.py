"""High-level, NumPy-native facade for common OpenImpala workflows.

These functions accept plain ``numpy.ndarray`` inputs and return rich
Python dataclasses, hiding all AMReX boilerplate from general users.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Union

import numpy as np

from .exceptions import ConvergenceError, PercolationError


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
# Helpers
# ---------------------------------------------------------------------------

def _get_core():
    """Import and return the _core C extension (lazy, cached by Python)."""
    import importlib
    return importlib.import_module("openimpala._core")


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
    _core = _get_core()
    if not _core.amrex_initialized():
        raise RuntimeError(
            "OpenImpala is not initialized! Please wrap your code in a session block:\n\n"
            "with openimpala.Session():\n"
            "    openimpala.volume_fraction(...)"
        )


def _parse_solver(s):
    """Parse a solver string or SolverType enum value."""
    _core = _get_core()
    if isinstance(s, _core.SolverType):
        return s
    solver_map = {
        "jacobi": _core.SolverType.Jacobi,
        "gmres": _core.SolverType.GMRES,
        "flexgmres": _core.SolverType.FlexGMRES,
        "pcg": _core.SolverType.PCG,
        "bicgstab": _core.SolverType.BiCGSTAB,
        "smg": _core.SolverType.SMG,
        "pfmg": _core.SolverType.PFMG,
        "hypre": _core.SolverType.FlexGMRES,  # convenience alias
    }
    key = s.strip().lower()
    if key not in solver_map:
        raise ValueError(f"Unknown solver '{s}'. Options: {list(solver_map)}")
    return solver_map[key]


def _numpy_to_voxelimage(
    data: np.ndarray,
    max_grid_size: int = 32,
):
    """Convert a 3-D int32 NumPy array to a VoxelImage (native C++ ingestion).

    Returns a VoxelImage handle that encapsulates all AMReX objects.
    """
    _core = _get_core()

    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {data.shape}")

    data = np.ascontiguousarray(data, dtype=np.int32)
    return _core.VoxelImage.from_numpy(data, max_grid_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def volume_fraction(
    data: np.ndarray,
    phase: int = 0,
    *,
    max_grid_size: int = 32,
) -> VolumeFractionResult:
    """Compute the volume fraction of *phase* in a 3-D NumPy array.

    Parameters
    ----------
    data : np.ndarray
        3-D integer array of phase IDs (shape: z, y, x).
    phase : int
        Phase ID to count.
    max_grid_size : int
        AMReX box decomposition size.

    Returns
    -------
    VolumeFractionResult
    """
    _ensure_initialized()
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
    max_grid_size: int = 32,
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

    Returns
    -------
    PercolationResult
    """
    _ensure_initialized()
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
    solver: Union[str, "SolverType"] = "flexgmres",
    *,
    max_grid_size: int = 32,
    results_path: str = ".",
    verbose: int = 0,
) -> TortuosityResult:
    """Compute the tortuosity of *phase* in *direction* using the HYPRE solver.

    Parameters
    ----------
    data : np.ndarray
        3-D integer array of phase IDs.
    phase : int
        Phase ID for the conducting phase.
    direction : str or Direction
        Flow direction ('x', 'y', 'z').
    solver : str or SolverType
        HYPRE solver algorithm.  Use 'hypre' or 'flexgmres' for a good default.

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
    _core = _get_core()
    d = _parse_direction(direction)
    st = _parse_solver(solver)

    if isinstance(data, np.ndarray):
        img = _numpy_to_voxelimage(data, max_grid_size)
    elif isinstance(data, _core.VoxelImage):
        img = data
    else:
        raise TypeError("data must be a NumPy array or a VoxelImage")

    # Percolation check first
    pc = _core.PercolationCheck(img, phase, d, verbose)
    if not pc.percolates:
        raise PercolationError(
            f"Phase {phase} does not percolate in direction "
            f"{_core.PercolationCheck.direction_string(d)}"
        )

    # Volume fraction
    vf_calc = _core.VolumeFraction(img, phase, 0)
    vf_val = vf_calc.value_vf()

    # Solve
    solver_obj = _core.TortuosityHypre(
        img, vf_val, phase, d, st, results_path,
        0.0, 1.0, verbose, False,
    )

    try:
        tau = solver_obj.value()
    except RuntimeError as exc:
        raise ConvergenceError(str(exc)) from exc

    return TortuosityResult(
        tortuosity=tau,
        solver_converged=solver_obj.solver_converged,
        iterations=solver_obj.iterations,
        residual_norm=solver_obj.residual_norm,
        flux_in=solver_obj.flux_in,
        flux_out=solver_obj.flux_out,
        active_volume_fraction=solver_obj.active_volume_fraction,
    )


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
    max_grid_size: int = 32,
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
    max_grid_size : int
        AMReX box decomposition size.

    Returns
    -------
    (reader, VoxelImage)
        The reader object and the VoxelImage handle.
    """
    _ensure_initialized()
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

    # Threshold directly into a VoxelImage (all AMReX setup happens in C++)
    img = reader.threshold(threshold, max_grid_size)

    return reader, img
