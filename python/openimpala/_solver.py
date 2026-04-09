"""Pure-Python tortuosity solver using sparse matrices (SciPy / CuPy).

This module provides a drop-in replacement for the C++ HYPRE-based solver,
enabling ``pip install openimpala`` to work without any compiled extensions.
GPU acceleration is available automatically when CuPy is installed.

The algorithm matches the C++ implementation exactly:
  1. Flood-fill percolation check (6-connected BFS)
  2. 7-point stencil Laplacian with harmonic-mean face coefficients
  3. Dirichlet BCs at inlet/outlet, zero-flux Neumann on sides
  4. Sparse CG solve (CuPy on GPU, SciPy on CPU)
  5. Flux integration for D_eff, then tau = active_vf / D_eff
"""

from __future__ import annotations

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

import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
import scipy.ndimage as ndimage


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
) -> tuple[bool, float, np.ndarray]:
    """Check if *phase* percolates in *direction* (0=x, 1=y, 2=z).

    Returns (percolates, active_vf, active_mask).
    The active_mask marks cells reachable from BOTH inlet and outlet.
    """
    phase_mask = data == phase

    # Label connected components using 6-connectivity (faces only)
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    labels, num_features = ndimage.label(phase_mask, structure=struct)

    if num_features == 0:
        return False, 0.0, np.zeros_like(data, dtype=np.int32)

    # Find labels touching inlet face (first slice along direction)
    # and outlet face (last slice along direction)
    inlet_slice = [slice(None)] * 3
    inlet_slice[direction] = 0
    outlet_slice = [slice(None)] * 3
    outlet_slice[direction] = data.shape[direction] - 1

    inlet_labels = set(labels[tuple(inlet_slice)].ravel()) - {0}
    outlet_labels = set(labels[tuple(outlet_slice)].ravel()) - {0}

    # Active labels are those touching both inlet and outlet
    active_labels = inlet_labels & outlet_labels

    if not active_labels:
        return False, 0.0, np.zeros_like(data, dtype=np.int32)

    # Build active mask
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
    """Assemble the sparse linear system Ax = b.

    Parameters
    ----------
    active_mask : (Nz, Ny, Nx) int32 — 1 for active cells, 0 for inactive
    diff_coeff  : (Nz, Ny, Nx) float  — diffusion coefficient per cell
    direction   : 0=x, 1=y, 2=z flow direction
    dx          : grid spacing (dz, dy, dx) matching array axis order
    vlo, vhi    : Dirichlet BC values at inlet/outlet

    Returns
    -------
    A : sparse CSR matrix (N x N)
    b : RHS vector (N,)
    """
    shape = active_mask.shape  # (Nz, Ny, Nx)
    N = int(np.prod(shape))

    # Flat index helper
    def flat(i, j, k):
        return i * shape[1] * shape[2] + j * shape[2] + k

    # Pre-flatten arrays for fast indexing
    mask_flat = active_mask.ravel()
    diff_flat = diff_coeff.ravel()

    # Strides for each axis in the flattened array
    strides = [shape[1] * shape[2], shape[2], 1]

    # Inverse squared grid spacing for each axis
    dxinv2 = [1.0 / (dx[d] * dx[d]) for d in range(3)]

    # Neighbor offsets: (axis, delta, stride)
    neighbors = []
    for axis in range(3):
        neighbors.append((axis, -1, -strides[axis]))
        neighbors.append((axis, +1, +strides[axis]))

    # Build COO data
    rows = []
    cols = []
    vals = []
    rhs = np.zeros(N, dtype=np.float64)

    # Vectorized assembly using NumPy
    indices = np.arange(N)
    i3d = np.unravel_index(indices, shape)  # (iz, iy, ix) arrays

    # For each cell: if inactive, diagonal=1, rhs=0
    # If active but on Dirichlet boundary: diagonal=1, rhs=bc_value
    # If active interior: build 7-point stencil

    is_active = mask_flat == 1

    # Identify Dirichlet boundary cells (inlet/outlet faces)
    is_inlet = np.zeros(N, dtype=bool)
    is_outlet = np.zeros(N, dtype=bool)
    is_inlet[is_active] = i3d[direction][is_active] == 0
    is_outlet[is_active] = i3d[direction][is_active] == shape[direction] - 1

    is_dirichlet = is_inlet | is_outlet
    is_interior = is_active & ~is_dirichlet

    # --- Inactive cells: A_ii = 1, rhs = 0 ---
    inactive_idx = indices[~is_active]
    rows.append(inactive_idx)
    cols.append(inactive_idx)
    vals.append(np.ones(len(inactive_idx)))

    # --- Dirichlet cells: A_ii = 1, rhs = bc_value ---
    dirichlet_idx = indices[is_dirichlet]
    rows.append(dirichlet_idx)
    cols.append(dirichlet_idx)
    vals.append(np.ones(len(dirichlet_idx)))
    rhs[is_inlet] = vlo
    rhs[is_outlet] = vhi

    # --- Interior active cells: 7-point stencil ---
    interior_idx = indices[is_interior]
    if len(interior_idx) > 0:
        diag = np.zeros(len(interior_idx), dtype=np.float64)
        D_center = diff_flat[interior_idx]

        for axis, delta, stride in neighbors:
            nbr_idx = interior_idx + stride
            coord = i3d[axis][interior_idx] + delta

            # Check bounds
            in_bounds = (coord >= 0) & (coord < shape[axis])

            # For out-of-bounds neighbors (domain boundary), zero-flux Neumann
            # means we simply don't add a coupling term.
            valid = in_bounds.copy()
            valid_nbr = nbr_idx[valid]

            # Check neighbor is active
            nbr_active = np.zeros(len(interior_idx), dtype=bool)
            nbr_active[valid] = mask_flat[valid_nbr] == 1

            # Compute harmonic mean face coefficient for valid+active neighbors
            coupled = nbr_active
            coupled_idx = interior_idx[coupled]
            coupled_nbr = interior_idx[coupled] + stride
            D_c = D_center[coupled]
            D_n = diff_flat[coupled_nbr]

            denom = D_c + D_n
            # Harmonic mean: 2*D_c*D_n / (D_c + D_n), zero if denom==0
            D_face = np.where(denom > 0, 2.0 * D_c * D_n / denom, 0.0)

            coeff = -D_face * dxinv2[axis]

            rows.append(coupled_idx)
            cols.append(coupled_nbr)
            vals.append(coeff)

            # Accumulate diagonal (negative of off-diagonal sum)
            diag[coupled] -= coeff

        rows.append(interior_idx)
        cols.append(interior_idx)
        vals.append(diag)

    # Assemble sparse matrix
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
    """Compute inlet flux, outlet flux, and D_eff from the solved potential.

    Returns (flux_in, flux_out, D_eff).
    """
    shape = active_mask.shape  # (Nz, Ny, Nx)
    sol = solution.reshape(shape)

    # Cross-sectional area perpendicular to flow direction
    perp_axes = [a for a in range(3) if a != direction]
    cross_area = dx[perp_axes[0]] * dx[perp_axes[1]]

    # Domain length in flow direction
    L = shape[direction] * dx[direction]

    # --- Compute flux through each interior plane perpendicular to flow ---
    # For plane between cell i and i+1:
    #   flux = sum over plane of: -D_face * (phi[i+1] - phi[i]) / dx * cell_area
    n_planes = shape[direction] - 1
    plane_fluxes = np.zeros(n_planes)

    for p in range(n_planes):
        # Build slices for left and right cells
        left_sl = [slice(None)] * 3
        left_sl[direction] = p
        right_sl = [slice(None)] * 3
        right_sl[direction] = p + 1

        left_sl = tuple(left_sl)
        right_sl = tuple(right_sl)

        mask_l = active_mask[left_sl]
        mask_r = active_mask[right_sl]
        both_active = (mask_l == 1) & (mask_r == 1)

        if not np.any(both_active):
            continue

        D_l = diff_coeff[left_sl][both_active]
        D_r = diff_coeff[right_sl][both_active]
        denom = D_l + D_r
        D_face = np.where(denom > 0, 2.0 * D_l * D_r / denom, 0.0)

        grad = (sol[right_sl][both_active] - sol[left_sl][both_active]) / dx[direction]
        plane_fluxes[p] = np.sum(-D_face * grad * cross_area)

    # Flux in = first plane, flux out = last plane
    flux_in = plane_fluxes[0]
    flux_out = plane_fluxes[-1]

    # Average flux magnitude from all interior planes
    nonzero_planes = plane_fluxes[plane_fluxes != 0.0]
    if len(nonzero_planes) > 0:
        avg_flux = float(np.mean(np.abs(nonzero_planes)))
    else:
        avg_flux = 0.5 * (abs(flux_in) + abs(flux_out))

    # Total cross-section area of the domain
    total_cross_area = 1.0
    for a in perp_axes:
        total_cross_area *= shape[a] * dx[a]

    # Imposed gradient
    vlo, vhi = 0.0, 1.0
    grad_imposed = abs(vhi - vlo) / L

    # D_eff = avg_flux / (A_total * grad_imposed)
    if total_cross_area * grad_imposed > 0:
        D_eff = avg_flux / (total_cross_area * grad_imposed)
    else:
        D_eff = 0.0

    return float(flux_in), float(flux_out), float(D_eff)


# ---------------------------------------------------------------------------
# Main solver entry point
# ---------------------------------------------------------------------------

def solve_tortuosity(
    data: np.ndarray,
    phase: int = 0,
    direction: int = 0,
    tol: float = 1e-9,
    maxiter: int = 2000,
) -> dict:
    """Compute tortuosity of *phase* in *direction*.

    Parameters
    ----------
    data : (Nz, Ny, Nx) int32 array of phase IDs
    phase : target phase ID
    direction : 0=x, 1=y, 2=z  (axis index in the NumPy array)
    tol : solver tolerance
    maxiter : max solver iterations

    Returns
    -------
    dict with keys: tortuosity, solver_converged, iterations, residual_norm,
                    flux_in, flux_out, active_volume_fraction, backend
    """
    data = np.ascontiguousarray(data, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {data.shape}")

    shape = data.shape  # (Nz, Ny, Nx)

    # --- Percolation check ---
    percolates, active_vf, active_mask = percolation_check(data, phase, direction)
    if not percolates:
        from .exceptions import PercolationError
        dir_names = {0: "x", 1: "y", 2: "z"}
        raise PercolationError(
            f"Phase {phase} does not percolate in the {dir_names[direction]} direction. "
            f"The solver cannot converge when the conducting phase is "
            f"disconnected between the inlet and outlet faces."
        )

    # --- Build diffusion coefficient field ---
    # All active cells get D=1.0, inactive get D=0.0
    diff_coeff = np.where(active_mask == 1, 1.0, 0.0)

    # --- Grid spacing (uniform unit grid) ---
    dx = (1.0, 1.0, 1.0)

    # --- Assemble sparse system ---
    A_sp, rhs = _build_laplacian(active_mask, diff_coeff, direction, dx)

    # --- Initial guess: linear gradient in flow direction ---
    x0 = np.zeros(A_sp.shape[0], dtype=np.float64)
    coords = np.arange(shape[direction], dtype=np.float64)
    grad_init = coords / max(shape[direction] - 1, 1)

    # Broadcast linear gradient to 3D then flatten
    bcast_shape = [1, 1, 1]
    bcast_shape[direction] = shape[direction]
    tile_shape = list(shape)
    tile_shape[direction] = 1
    x0_3d = np.tile(grad_init.reshape(bcast_shape), tile_shape)
    x0 = x0_3d.ravel()
    # Zero out inactive cells
    x0[active_mask.ravel() == 0] = 0.0

    # --- Solve ---
    if _USE_CUPY:
        A_gpu = cusp.csr_matrix(A_sp)
        rhs_gpu = cp.asarray(rhs)
        x0_gpu = cp.asarray(x0)

        # CuPy CG solve — CuPy >= 13 renamed tol → rtol (mirroring scipy).
        try:
            solution_gpu, info = cusp_linalg.cg(A_gpu, rhs_gpu, x0=x0_gpu,
                                                 rtol=tol, maxiter=maxiter)
        except TypeError:
            solution_gpu, info = cusp_linalg.cg(A_gpu, rhs_gpu, x0=x0_gpu,
                                                 tol=tol, maxiter=maxiter)
        solution = cp.asnumpy(solution_gpu)
        converged = info == 0
        # CuPy doesn't return iteration count directly; estimate from info
        iterations = maxiter if not converged else -1  # unknown exact count
        # Compute residual
        res = cp.asnumpy(rhs_gpu - A_gpu @ solution_gpu)
        residual_norm = float(np.linalg.norm(res) / max(np.linalg.norm(rhs), 1e-30))
    else:
        # SciPy CG solve with callback to count iterations
        iter_count = [0]

        def _callback(xk):
            iter_count[0] += 1

        # scipy >= 1.12 renamed tol → rtol; older versions use tol.
        try:
            solution, info = sp_linalg.cg(A_sp, rhs, x0=x0, rtol=tol,
                                           maxiter=maxiter, callback=_callback)
        except TypeError:
            solution, info = sp_linalg.cg(A_sp, rhs, x0=x0, tol=tol,
                                           maxiter=maxiter, callback=_callback)
        converged = info == 0
        iterations = iter_count[0]
        res = rhs - A_sp @ solution
        residual_norm = float(np.linalg.norm(res) / max(np.linalg.norm(rhs), 1e-30))

    # --- Compute flux and D_eff ---
    flux_in, flux_out, D_eff = _compute_flux(
        solution, active_mask, diff_coeff, direction, dx,
    )

    # --- Tortuosity: tau = active_vf / D_eff ---
    if D_eff > 0:
        tau = active_vf / D_eff
    else:
        tau = float("inf")

    return {
        "tortuosity": tau,
        "solver_converged": converged,
        "iterations": iterations,
        "residual_norm": residual_norm,
        "flux_in": flux_in,
        "flux_out": flux_out,
        "active_volume_fraction": active_vf,
        "backend": backend_name(),
    }
