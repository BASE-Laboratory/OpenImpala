"""Pure-Python transport and microstructure solvers using NumPy / SciPy (or CuPy).

This module provides a drop-in pure-Python backend for OpenImpala's high-level API,
enabling ``pip install openimpala`` to work out-of-the-box on macOS, Windows,
Linux ARM64, and new Python versions without requiring AMReX, HYPRE, or MPI.
GPU acceleration is used automatically when CuPy is installed.

Supported features (matching the C++ backend):
  * Volume fraction and 6-connected percolation checks (single- and multi-phase)
  * Steady-state tortuosity and effective diffusivity (anisotropic voxels,
    multi-phase harmonic-mean conductances, 3-D potential field output)
  * Homogenised 3x3 effective diffusivity tensor D_eff (periodic cell problems)
  * Specific surface area (voxel and physical units, Cauchy-Crofton 2/3 correction)
  * Connected-component labeling and equivalent-sphere particle size distribution
  * Through-thickness volume-fraction profiles
  * Representative Elementary Volume (REV) convergence studies
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Backend detection: prefer CuPy (GPU), fall back to SciPy (CPU)
# ---------------------------------------------------------------------------

_USE_CUPY = False
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    import cupyx.scipy.sparse.linalg as cusp_linalg

    # Verify a GPU is actually available
    cp.cuda.runtime.getDeviceCount()
    _USE_CUPY = True
except Exception:
    cp = None

import scipy.ndimage as ndimage
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg


def _xp():
    """Return the active array module (cupy or numpy)."""
    return cp if _USE_CUPY else np


def backend_name() -> str:
    """Return a human-readable name for the active compute backend."""
    if _USE_CUPY:
        dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"]
        return f"CuPy (GPU: {name})"
    return "SciPy (CPU)"


def _build_phase_diff_field(
    data: np.ndarray,
    phase_diffusivities: Mapping[int, float],
) -> np.ndarray:
    """Map integer phase IDs in *data* to non-negative float64 diffusivities."""
    diff = np.zeros(data.shape, dtype=np.float64)
    for pid, val in phase_diffusivities.items():
        d_val = float(val)
        if d_val < 0.0:
            raise ValueError(f"Diffusivity for phase {pid} must be >= 0, got {d_val}")
        if d_val > 0.0:
            diff[data == int(pid)] = d_val
    return diff


# ---------------------------------------------------------------------------
# Volume fraction
# ---------------------------------------------------------------------------

def volume_fraction(data: np.ndarray, phase: int = 0) -> tuple[int, int, float]:
    """Return (phase_count, total_count, fraction) for *phase*."""
    mask = data == phase
    pc = int(np.count_nonzero(mask))
    tc = int(data.size)
    return pc, tc, pc / tc if tc > 0 else 0.0


# ---------------------------------------------------------------------------
# Percolation check (6-connected flood fill)
# ---------------------------------------------------------------------------

def percolation_check(
    data: np.ndarray,
    phase: int = 0,
    direction: int = 0,
    phase_diffusivities: Optional[Mapping[int, float]] = None,
) -> tuple[bool, float, np.ndarray]:
    """Check if conducting cells percolate along NumPy axis *direction* (0=z, 1=y, 2=x).

    Returns (percolates, active_vf, active_mask).
    The active_mask marks cells reachable from BOTH inlet and outlet.
    """
    if phase_diffusivities:
        conducting_mask = np.zeros(data.shape, dtype=bool)
        for pid, val in phase_diffusivities.items():
            if float(val) > 0.0:
                conducting_mask |= data == int(pid)
    else:
        conducting_mask = data == phase

    # Label connected components using 6-connectivity (faces only)
    struct = ndimage.generate_binary_structure(3, 1)
    labels, num_features = ndimage.label(conducting_mask, structure=struct)

    if num_features == 0:
        return False, 0.0, np.zeros_like(data, dtype=np.int32)

    inlet_slice = [slice(None)] * 3
    inlet_slice[direction] = 0
    outlet_slice = [slice(None)] * 3
    outlet_slice[direction] = data.shape[direction] - 1

    inlet_labels = set(labels[tuple(inlet_slice)].ravel()) - {0}
    outlet_labels = set(labels[tuple(outlet_slice)].ravel()) - {0}
    active_labels = inlet_labels & outlet_labels

    if not active_labels:
        return False, 0.0, np.zeros_like(data, dtype=np.int32)

    active_mask = np.isin(labels, list(active_labels)).astype(np.int32)
    active_vf = float(np.count_nonzero(active_mask)) / float(data.size)
    return True, active_vf, active_mask


# ---------------------------------------------------------------------------
# Sparse matrix assembly (7-point stencil, harmonic-mean face coefficients)
# ---------------------------------------------------------------------------

def _build_laplacian(
    active_mask: np.ndarray,
    diff_coeff: np.ndarray,
    direction: int,
    dx: tuple[float, float, float],
    vlo: float = 0.0,
    vhi: float = 1.0,
):
    """Assemble the sparse linear system Ax = b for steady-state flow-through diffusion.

    Parameters
    ----------
    active_mask : (Nz, Ny, Nx) int32 — 1 for active cells, 0 for inactive
    diff_coeff  : (Nz, Ny, Nx) float  — diffusion coefficient per cell
    direction   : NumPy axis index (0=z, 1=y, 2=x)
    dx          : grid spacing (dz, dy, dx) matching array axis order
    vlo, vhi    : Dirichlet BC values at inlet/outlet
    """
    shape = active_mask.shape  # (Nz, Ny, Nx)
    N = int(np.prod(shape))

    mask_flat = active_mask.ravel()
    diff_flat = diff_coeff.ravel()

    strides = [shape[1] * shape[2], shape[2], 1]
    dxinv2 = [1.0 / (dx[d] * dx[d]) for d in range(3)]

    d_max = float(np.max(diff_flat)) if N > 0 else 1.0
    d_scale = d_max if d_max > 0.0 else 1.0
    pin_diag = d_scale * max(dxinv2)

    neighbors = []
    for axis in range(3):
        neighbors.append((axis, -1, -strides[axis]))
        neighbors.append((axis, +1, +strides[axis]))

    rows = []
    cols = []
    vals = []
    rhs = np.zeros(N, dtype=np.float64)

    indices = np.arange(N)
    i3d = np.unravel_index(indices, shape)
    is_active = mask_flat == 1

    is_inlet = np.zeros(N, dtype=bool)
    is_outlet = np.zeros(N, dtype=bool)
    is_inlet[is_active] = i3d[direction][is_active] == 0
    is_outlet[is_active] = i3d[direction][is_active] == shape[direction] - 1

    # Inactive cells: scale-matched diagonal pin, rhs = 0
    inactive_idx = indices[~is_active]
    rows.append(inactive_idx)
    cols.append(inactive_idx)
    vals.append(np.full(len(inactive_idx), pin_diag, dtype=np.float64))

    active_idx = indices[is_active]
    if len(active_idx) > 0:
        D_center = diff_flat[active_idx]
        diag = np.zeros_like(D_center, dtype=np.float64)
        for boundary, value in ((is_inlet, vlo), (is_outlet, vhi)):
            weight = 2.0 * diff_flat[boundary] * dxinv2[direction]
            diag[boundary[active_idx]] += weight
            rhs[boundary] += weight * value

        for axis, delta, stride in neighbors:
            nbr_idx = active_idx + stride
            coord = i3d[axis][active_idx] + delta

            in_bounds = (coord >= 0) & (coord < shape[axis])
            valid_nbr = nbr_idx[in_bounds]

            nbr_active = np.zeros(len(active_idx), dtype=bool)
            nbr_active[in_bounds] = mask_flat[valid_nbr] == 1

            coupled = nbr_active
            coupled_idx = active_idx[coupled]
            coupled_nbr = active_idx[coupled] + stride
            D_c = D_center[coupled]
            D_n = diff_flat[coupled_nbr]

            denom = D_c + D_n
            D_face = np.where(denom > 0.0, 2.0 * D_c * D_n / denom, 0.0)
            coeff = -D_face * dxinv2[axis]

            rows.append(coupled_idx)
            cols.append(coupled_nbr)
            vals.append(coeff)
            diag[coupled] -= coeff

        rows.append(active_idx)
        cols.append(active_idx)
        vals.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    A = sp.coo_matrix((all_vals, (all_rows, all_cols)), shape=(N, N)).tocsr()
    return A, rhs


# ---------------------------------------------------------------------------
# Flux computation
# ---------------------------------------------------------------------------

def _compute_flux(
    solution: np.ndarray,
    active_mask: np.ndarray,
    diff_coeff: np.ndarray,
    direction: int,
    dx: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute inlet flux, outlet flux, and D_eff from the solved potential."""
    shape = active_mask.shape  # (Nz, Ny, Nx)
    sol = solution.reshape(shape)

    perp_axes = [a for a in range(3) if a != direction]
    cross_area = dx[perp_axes[0]] * dx[perp_axes[1]]
    L = shape[direction] * dx[direction]

    n_planes = shape[direction] - 1
    plane_fluxes = np.zeros(n_planes, dtype=np.float64)

    for p in range(n_planes):
        left_sl = [slice(None)] * 3
        left_sl[direction] = p
        right_sl = [slice(None)] * 3
        right_sl[direction] = p + 1

        left_sl_t = tuple(left_sl)
        right_sl_t = tuple(right_sl)

        mask_l = active_mask[left_sl_t]
        mask_r = active_mask[right_sl_t]
        both_active = (mask_l == 1) & (mask_r == 1)
        if not np.any(both_active):
            continue

        D_l = diff_coeff[left_sl_t][both_active]
        D_r = diff_coeff[right_sl_t][both_active]
        denom = D_l + D_r
        D_face = np.where(denom > 0.0, 2.0 * D_l * D_r / denom, 0.0)

        grad = (sol[right_sl_t][both_active] - sol[left_sl_t][both_active]) / dx[direction]
        plane_fluxes[p] = np.sum(-D_face * grad * cross_area)

    conductance = 2.0 * active_mask * diff_coeff * cross_area / dx[direction]
    flux_in = -float(np.sum(np.take(conductance * sol, 0, axis=direction)))
    flux_out = -float(np.sum(np.take(conductance * (1.0 - sol), -1, axis=direction)))

    if n_planes > 0:
        avg_flux = float(np.mean(np.abs(plane_fluxes)))
    else:
        avg_flux = 0.5 * (abs(flux_in) + abs(flux_out))

    total_cross_area = 1.0
    for a in perp_axes:
        total_cross_area *= shape[a] * dx[a]

    vlo, vhi = 0.0, 1.0
    grad_imposed = abs(vhi - vlo) / L

    if total_cross_area * grad_imposed > 0.0:
        D_eff = avg_flux / (total_cross_area * grad_imposed)
    else:
        D_eff = 0.0

    return float(flux_in), float(flux_out), float(D_eff)


def _solve_spd_cg(
    A_sp: sp.csr_matrix,
    rhs: np.ndarray,
    x0: np.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, bool, int, float]:
    """Solve SPD system A_sp x = rhs via Jacobi-preconditioned CG."""
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros_like(rhs), True, 0, 0.0

    if _USE_CUPY:
        A_gpu = cusp.csr_matrix(A_sp)
        rhs_gpu = cp.asarray(rhs)
        x0_gpu = cp.asarray(x0)
        diag_gpu = A_gpu.diagonal()
        inv_diag_gpu = cp.where(cp.abs(diag_gpu) > 0.0, 1.0 / diag_gpu, 1.0)
        M_gpu = cusp.diags(inv_diag_gpu)
        try:
            solution_gpu, info = cusp_linalg.cg(
                A_gpu, rhs_gpu, x0=x0_gpu, M=M_gpu, rtol=tol, maxiter=maxiter
            )
        except TypeError:
            solution_gpu, info = cusp_linalg.cg(
                A_gpu, rhs_gpu, x0=x0_gpu, M=M_gpu, tol=tol, maxiter=maxiter
            )
        solution = cp.asnumpy(solution_gpu)
        converged = info == 0
        iterations = maxiter if not converged else -1
        res = cp.asnumpy(rhs_gpu - A_gpu @ solution_gpu)
        residual_norm = float(np.linalg.norm(res) / max(rhs_norm, 1e-300))
        return solution, converged, iterations, residual_norm

    diag = A_sp.diagonal()
    inv_diag = np.where(np.abs(diag) > 0.0, 1.0 / diag, 1.0)
    M = sp.diags(inv_diag)

    iter_count = [0]

    def _callback(_xk):
        iter_count[0] += 1

    try:
        solution, info = sp_linalg.cg(
            A_sp, rhs, x0=x0, M=M, rtol=tol, atol=0.0, maxiter=maxiter, callback=_callback
        )
    except TypeError:
        solution, info = sp_linalg.cg(
            A_sp, rhs, x0=x0, M=M, tol=tol, maxiter=maxiter, callback=_callback
        )
    converged = info == 0
    iterations = iter_count[0]
    res = rhs - A_sp @ solution
    residual_norm = float(np.linalg.norm(res) / max(rhs_norm, 1e-300))
    return solution, converged, iterations, residual_norm


# ---------------------------------------------------------------------------
# Main tortuosity solver entry point
# ---------------------------------------------------------------------------

def solve_tortuosity(
    data: np.ndarray,
    phase: int = 0,
    direction: int = 0,
    tol: float = 1e-9,
    maxiter: int = 2000,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    phase_diffusivities: Optional[Mapping[int, float]] = None,
    return_field: bool = True,
) -> dict:
    """Compute tortuosity and effective diffusivity along NumPy axis *direction* (0=z, 1=y, 2=x)."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {data.shape}")

    dx = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
    if any(h <= 0.0 for h in dx):
        raise ValueError(f"voxel_size components must all be > 0, got {dx}")

    shape = data.shape  # (Nz, Ny, Nx)

    percolates, active_vf, active_mask = percolation_check(
        data, phase, direction, phase_diffusivities=phase_diffusivities
    )
    if not percolates:
        from .exceptions import PercolationError

        dir_names = {0: "z", 1: "y", 2: "x"}
        if phase_diffusivities:
            raise PercolationError(
                f"Conducting phases do not percolate in the {dir_names[direction]} direction. "
                f"The solver cannot converge when conducting phases are "
                f"disconnected between the inlet and outlet faces."
            )
        raise PercolationError(
            f"Phase {phase} does not percolate in the {dir_names[direction]} direction. "
            f"The solver cannot converge when the conducting phase is "
            f"disconnected between the inlet and outlet faces."
        )

    if phase_diffusivities:
        raw_diff = _build_phase_diff_field(data, phase_diffusivities)
        diff_coeff = np.where(active_mask == 1, raw_diff, 0.0)
    else:
        diff_coeff = np.where(active_mask == 1, 1.0, 0.0)

    A_sp, rhs = _build_laplacian(active_mask, diff_coeff, direction, dx)

    coords = (np.arange(shape[direction], dtype=np.float64) + 0.5) / max(shape[direction], 1)
    bcast_shape = [1, 1, 1]
    bcast_shape[direction] = shape[direction]
    tile_shape = list(shape)
    tile_shape[direction] = 1
    x0 = np.tile(coords.reshape(bcast_shape), tile_shape).ravel()
    x0[active_mask.ravel() == 0] = 0.0

    solution, converged, iterations, residual_norm = _solve_spd_cg(
        A_sp, rhs, x0, tol=tol, maxiter=maxiter
    )

    flux_in, flux_out, D_eff = _compute_flux(solution, active_mask, diff_coeff, direction, dx)

    d_scale = float(np.max(diff_coeff))
    if D_eff > 1e-15 * max(d_scale, 1e-300):
        tau = active_vf / D_eff
    else:
        tau = float("inf")

    solution_field = solution.reshape(shape).copy() if return_field else None

    return {
        "tortuosity": tau,
        "effective_diffusivity": D_eff,
        "solver_converged": converged,
        "iterations": iterations,
        "residual_norm": residual_norm,
        "flux_in": flux_in,
        "flux_out": flux_out,
        "active_volume_fraction": active_vf,
        "solution_field": solution_field,
        "backend": backend_name(),
    }


# ---------------------------------------------------------------------------
# Periodic homogenization 3x3 D_eff tensor solver
# ---------------------------------------------------------------------------

def _build_cell_problem_operator(
    active_mask: np.ndarray,
    diff_coeff: np.ndarray,
    voxel_size: tuple[float, float, float],
) -> tuple[sp.csr_matrix, list[np.ndarray], Optional[int]]:
    """Assemble the SPD conservative finite-volume periodic cell-problem matrix.

    Solves ``-div(D (e_k + grad chi_k)) = 0`` on the periodic 3-torus with
    harmonic-mean face conductances ``D_{i+1/2}`` between active neighbours and
    zero-flux insulating faces wherever either side has ``D == 0``.
    """
    shape = active_mask.shape  # (Nz, Ny, Nx)
    N = int(np.prod(shape))
    idx_grid = np.arange(N, dtype=np.int64).reshape(shape)

    is_active_3d = active_mask == 1
    active_idx = idx_grid[is_active_3d]
    inactive_idx = idx_grid[~is_active_3d]

    d_max = float(np.max(diff_coeff)) if N > 0 else 1.0
    d_scale = d_max if d_max > 0.0 else 1.0
    inv_h2_sum = sum(1.0 / (h * h) for h in voxel_size)
    pin_scale = d_scale * max(1.0 / (h * h) for h in voxel_size)

    # Precompute +d face harmonic diffusivities D_{i+1/2}^{(d)} for d_xyz = 0(X), 1(Y), 2(Z)
    d_face_plus: list[np.ndarray] = []
    for d_xyz in range(3):
        ax = 2 - d_xyz
        d_nbr = np.roll(diff_coeff, -1, axis=ax)
        both = is_active_3d & np.roll(is_active_3d, -1, axis=ax)
        denom = np.where(both, diff_coeff + d_nbr, 1.0)
        d_face_plus.append(np.where(both, 2.0 * diff_coeff * d_nbr / denom, 0.0))

    rows = [inactive_idx]
    cols = [inactive_idx]
    vals = [np.full(len(inactive_idx), pin_scale, dtype=np.float64)]
    pin_idx: Optional[int] = None

    if len(active_idx) > 0:
        diag = np.zeros(N, dtype=np.float64)
        for d_xyz in range(3):
            ax = 2 - d_xyz
            inv_h2 = 1.0 / (voxel_size[ax] * voxel_size[ax])
            dp = d_face_plus[d_xyz]
            dm = np.roll(dp, 1, axis=ax)
            p_idx = np.roll(idx_grid, -1, axis=ax)
            m_idx = np.roll(idx_grid, 1, axis=ax)

            has_p = dp > 0.0
            if np.any(has_p):
                rows.append(idx_grid[has_p])
                cols.append(p_idx[has_p])
                vals.append(-dp[has_p] * inv_h2)

            has_m = dm > 0.0
            if np.any(has_m):
                rows.append(idx_grid[has_m])
                cols.append(m_idx[has_m])
                vals.append(-dm[has_m] * inv_h2)

            diag += ((dp + dm) * inv_h2).ravel()

        # Tiny regularisation ensures disconnected active pockets are nonsingular
        diag[active_idx] += 1e-12 * diff_coeff[is_active_3d] * inv_h2_sum
        rows.append(active_idx)
        cols.append(active_idx)
        vals.append(diag[active_idx])

        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_vals = np.concatenate(vals)

        # Symmetrically pin the first active cell to chi = 0 to remove the constant nullspace
        pin_idx = int(active_idx[0])
        keep = (all_rows != pin_idx) & (all_cols != pin_idx)
        all_rows = np.append(all_rows[keep], pin_idx)
        all_cols = np.append(all_cols[keep], pin_idx)
        all_vals = np.append(all_vals[keep], pin_scale)
    else:
        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_vals = np.concatenate(vals)

    A = sp.coo_matrix((all_vals, (all_rows, all_cols)), shape=(N, N)).tocsr()
    return A, d_face_plus, pin_idx


def solve_effective_diffusivity(
    data: np.ndarray,
    phase: int = 1,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    phase_diffusivities: Optional[Mapping[int, float]] = None,
    tol: float = 1e-9,
    maxiter: int = 2000,
    return_chi: bool = False,
) -> dict:
    """Solve the 3 periodic cell problems and return the 3x3 D_eff tensor in (X, Y, Z) order."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {data.shape}")

    vz = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
    if any(h <= 0.0 for h in vz):
        raise ValueError(f"voxel_size components must all be > 0, got {vz}")

    shape = data.shape
    N = int(data.size)

    if phase_diffusivities:
        diff_coeff = _build_phase_diff_field(data, phase_diffusivities)
        active_mask = (diff_coeff > 0.0).astype(np.int32)
    else:
        active_mask = (data == phase).astype(np.int32)
        diff_coeff = active_mask.astype(np.float64)

    active_count = int(np.count_nonzero(active_mask))
    active_vf = float(active_count) / float(N) if N > 0 else 0.0

    if active_count == 0:
        out = {
            "tensor": np.zeros((3, 3), dtype=np.float64),
            "active_volume_fraction": 0.0,
            "converged": True,
            "iterations": (0, 0, 0),
            "residuals": (0.0, 0.0, 0.0),
        }
        if return_chi:
            zeros = np.zeros(shape, dtype=np.float64)
            out["chi"] = {"x": zeros.copy(), "y": zeros.copy(), "z": zeros.copy()}
        return out

    A_sp, d_face_plus, pin_idx = _build_cell_problem_operator(active_mask, diff_coeff, vz)

    chi_fields: list[np.ndarray] = []
    dir_iters: list[int] = []
    dir_res: list[float] = []
    all_converged = True
    x0_zero = np.zeros(N, dtype=np.float64)

    for dir_k_xyz in range(3):
        ax_k = 2 - dir_k_xyz
        dp_k = d_face_plus[dir_k_xyz]
        dm_k = np.roll(dp_k, 1, axis=ax_k)
        rhs = ((dp_k - dm_k) / vz[ax_k]).ravel()
        if pin_idx is not None:
            rhs[pin_idx] = 0.0

        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0.0:
            chi_fields.append(np.zeros(shape, dtype=np.float64))
            dir_iters.append(0)
            dir_res.append(0.0)
            continue

        sol, conv, iters, res_norm = _solve_spd_cg(
            A_sp, rhs, x0_zero, tol=tol, maxiter=maxiter
        )
        conv = conv or res_norm <= tol * 10.0
        all_converged = all_converged and conv
        chi_fields.append(sol.reshape(shape))
        dir_iters.append(iters)
        dir_res.append(res_norm)

    tensor = np.full((3, 3), np.nan, dtype=np.float64)
    if all_converged and N > 0:
        for r in range(3):
            ax_r = 2 - r
            h_r = vz[ax_r]
            dp_r = d_face_plus[r]
            for c in range(3):
                chi_c = chi_fields[c]
                dchi_face = (np.roll(chi_c, -1, axis=ax_r) - chi_c) / h_r
                delta_rc = 1.0 if r == c else 0.0
                tensor[r, c] = float(np.sum(dp_r * (delta_rc + dchi_face))) / float(N)

    out = {
        "tensor": tensor,
        "active_volume_fraction": active_vf,
        "converged": all_converged,
        "iterations": (dir_iters[0], dir_iters[1], dir_iters[2]),
        "residuals": (dir_res[0], dir_res[1], dir_res[2]),
    }
    if return_chi:
        out["chi"] = {
            "x": chi_fields[0].copy(),
            "y": chi_fields[1].copy(),
            "z": chi_fields[2].copy(),
        }
    return out


# ---------------------------------------------------------------------------
# Microstructural metrics (pure-Python parity with C++ SpecificSurfaceArea,
# ConnectedComponents, ParticleSizeDistribution, ThroughThicknessProfile, REV)
# ---------------------------------------------------------------------------

def specific_surface_area(
    data: np.ndarray,
    phase_a: int = 0,
    phase_b: int = 1,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    boundary_padding: int = 0,
) -> dict:
    """Compute interface face counts and specific surface area between *phase_a* and *phase_b*."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {data.shape}")
    if boundary_padding < 0:
        raise ValueError(f"boundary_padding must be >= 0, got {boundary_padding}")

    dz, dy, dx = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    if dz <= 0.0 or dy <= 0.0 or dx <= 0.0:
        raise ValueError(f"voxel_size components must all be > 0, got {voxel_size}")

    if boundary_padding > 0:
        p = int(boundary_padding)
        if any(s <= 2 * p for s in data.shape):
            raise ValueError(
                f"boundary_padding={p} is too large for array of shape {data.shape}"
            )
        sub = data[p:-p, p:-p, p:-p]
    else:
        sub = data

    total_count = int(sub.size)

    def _count_axis(a: np.ndarray, b: np.ndarray) -> int:
        return int(
            np.count_nonzero(((a == phase_a) & (b == phase_b)) | ((a == phase_b) & (b == phase_a)))
        )

    fx = _count_axis(sub[:, :, :-1], sub[:, :, 1:]) if sub.shape[2] > 1 else 0
    fy = _count_axis(sub[:, :-1, :], sub[:, 1:, :]) if sub.shape[1] > 1 else 0
    fz = _count_axis(sub[:-1, :, :], sub[1:, :, :]) if sub.shape[0] > 1 else 0
    face_count = fx + fy + fz

    if total_count > 0:
        ssa_voxel = float(face_count) / float(total_count)
        ssa_physical = (fx / dx + fy / dy + fz / dz) / float(total_count)
    else:
        ssa_voxel = 0.0
        ssa_physical = 0.0

    return {
        "face_count": face_count,
        "face_counts_xyz": (fx, fy, fz),
        "total_count": total_count,
        "ssa_voxel": ssa_voxel,
        "ssa_voxel_corrected": (2.0 / 3.0) * ssa_voxel,
        "ssa": ssa_physical,
        "ssa_corrected": (2.0 / 3.0) * ssa_physical,
    }


def connected_components(
    data: np.ndarray,
    phase: int = 1,
    return_labels: bool = False,
) -> tuple[int, np.ndarray, Optional[np.ndarray]]:
    """Label 6-connected components of *phase* and return (num_components, volumes, labels)."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {data.shape}")

    struct = ndimage.generate_binary_structure(3, 1)
    labels, num_comp = ndimage.label(data == phase, structure=struct)
    if num_comp > 0:
        volumes = np.bincount(labels.ravel())[1 : num_comp + 1].astype(np.int64)
    else:
        volumes = np.zeros(0, dtype=np.int64)
    label_arr = labels.astype(np.int32) if return_labels else None
    return int(num_comp), volumes, label_arr


def particle_size_distribution(
    data: np.ndarray,
    phase: int = 1,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Compute equivalent-sphere radii R = (3V / (4*pi))^(1/3) for 6-connected components."""
    dz, dy, dx = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    if dz <= 0.0 or dy <= 0.0 or dx <= 0.0:
        raise ValueError(f"voxel_size components must all be > 0, got {voxel_size}")

    num_particles, volumes_vox, _ = connected_components(data, phase=phase, return_labels=False)
    cell_vol = dz * dy * dx
    length_scale = cell_vol ** (1.0 / 3.0)

    if num_particles > 0:
        radii_vox = ((3.0 * volumes_vox.astype(np.float64)) / (4.0 * np.pi)) ** (1.0 / 3.0)
        radii_phys = radii_vox * length_scale
        mean_radius_vox = float(np.mean(radii_vox))
        mean_radius_phys = float(np.mean(radii_phys))
    else:
        radii_vox = np.zeros(0, dtype=np.float64)
        radii_phys = np.zeros(0, dtype=np.float64)
        mean_radius_vox = 0.0
        mean_radius_phys = 0.0

    return {
        "num_particles": num_particles,
        "volumes_voxels": volumes_vox,
        "volumes": volumes_vox.astype(np.float64) * cell_vol,
        "radii_voxels": radii_vox,
        "radii": radii_phys,
        "mean_radius_voxels": mean_radius_vox,
        "mean_radius": mean_radius_phys,
    }


def through_thickness_profile(
    data: np.ndarray,
    phase: int = 1,
    direction_xyz: int = 0,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-slice volume fraction profile along *direction_xyz* (0=X, 1=Y, 2=Z)."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {data.shape}")

    ax = 2 - direction_xyz
    perp_axes = tuple(a for a in range(3) if a != ax)
    profile = np.mean(data == phase, axis=perp_axes, dtype=np.float64)
    h = float(voxel_size[ax])
    positions = (np.arange(data.shape[ax], dtype=np.float64) + 0.5) * h
    return profile, positions


def rev_study(
    data: np.ndarray,
    sizes: Sequence[int],
    num_samples: int = 3,
    phase: int = 1,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    csv_path: Optional[str] = None,
    seed: int = 42,
) -> list[dict]:
    """Run an REV convergence study in pure Python and optionally write CSV matching REVStudy.cpp."""
    data = np.ascontiguousarray(data, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {data.shape}")

    nz, ny, nx = data.shape
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    if csv_path is not None:
        out_path = Path(csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    for target in sizes:
        t = int(target)
        if t <= 0:
            continue
        act_x = min(t, nx)
        act_y = min(t, ny)
        act_z = min(t, nz)
        for s_idx in range(1, int(num_samples) + 1):
            sx = int(rng.integers(0, nx - act_x + 1)) if nx > act_x else 0
            sy = int(rng.integers(0, ny - act_y + 1)) if ny > act_y else 0
            sz = int(rng.integers(0, nz - act_z + 1)) if nz > act_z else 0
            sub = data[sz : sz + act_z, sy : sy + act_y, sx : sx + act_x]
            eff = solve_effective_diffusivity(sub, phase=phase, voxel_size=voxel_size)
            D = eff["tensor"]
            records.append(
                {
                    "sample": s_idx,
                    "seed_xyz": (sx, sy, sz),
                    "target_size": t,
                    "actual_size_xyz": (act_x, act_y, act_z),
                    "D_xx": float(D[0, 0]),
                    "D_yy": float(D[1, 1]),
                    "D_zz": float(D[2, 2]),
                    "D_xy": float(D[0, 1]),
                    "D_xz": float(D[0, 2]),
                    "D_yz": float(D[1, 2]),
                    "tensor": D.copy(),
                }
            )

    if out_path is not None:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "SampleNo",
                    "SeedX",
                    "SeedY",
                    "SeedZ",
                    "REV_Size_Target",
                    "ActualSizeX",
                    "ActualSizeY",
                    "ActualSizeZ",
                    "D_xx",
                    "D_yy",
                    "D_zz",
                    "D_xy",
                    "D_xz",
                    "D_yz",
                ]
            )
            for r in records:
                sx, sy, sz = r["seed_xyz"]
                ax_x, ax_y, ax_z = r["actual_size_xyz"]
                writer.writerow(
                    [
                        r["sample"],
                        sx,
                        sy,
                        sz,
                        r["target_size"],
                        ax_x,
                        ax_y,
                        ax_z,
                        f"{r['D_xx']:.8e}",
                        f"{r['D_yy']:.8e}",
                        f"{r['D_zz']:.8e}",
                        f"{r['D_xy']:.8e}",
                        f"{r['D_xz']:.8e}",
                        f"{r['D_yz']:.8e}",
                    ]
                )

    return records
