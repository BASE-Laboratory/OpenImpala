/** @file TortuosityHypreFill.H
 *  @brief C++ kernel for filling HYPRE structured-grid matrix coefficients
 *         for the tortuosity (flow-through) solve: div(D grad(phi)) = 0.
 *
 *  Replaces the legacy Fortran kernel TortuosityHypreFill.F90.
 *  Uses a 7-point stencil with harmonic-mean face coefficients and
 *  supports Dirichlet BCs perpendicular to the flow direction.
 */
#ifndef OPENIMPALA_TORTUOSITY_HYPRE_FILL_H_
#define OPENIMPALA_TORTUOSITY_HYPRE_FILL_H_

#include <AMReX_REAL.H>
#include <AMReX_Print.H>

#include <cmath>
#include <cstdio>

namespace OpenImpala {

/// Stencil indices (0-based, matching HYPRE_StructStencilSetElement order)
enum StencilIdx : int {
    STN_C = 0,  ///< Center
    STN_MX = 1, ///< -X (West)
    STN_PX = 2, ///< +X (East)
    STN_MY = 3, ///< -Y (South)
    STN_PY = 4, ///< +Y (North)
    STN_MZ = 5, ///< -Z (Bottom)
    STN_PZ = 6, ///< +Z (Top)
    NSTENCIL = 7
};

/// Mask values
constexpr int CELL_INACTIVE = 0;
constexpr int CELL_ACTIVE = 1;

/// Small tolerance for floating-point comparisons
constexpr amrex::Real SMALL_REAL = 1.0e-15;

/**
 * @brief Fill HYPRE matrix coefficients for the tortuosity solve.
 *
 * Discretises div(D grad(phi)) = 0 on a 7-point stencil with harmonic-mean
 * face coefficients.  Inactive cells are decoupled (A_ii=1, A_ij=0, b=0).
 * Dirichlet BCs are applied on domain faces perpendicular to the flow direction.
 *
 * @param[out] a          Flattened stencil array, size nval*7 (C,W,E,S,N,B,T).
 * @param[out] rhs        Right-hand side, size nval.
 * @param[out] xinit      Initial guess, size nval.
 * @param[in]  nval       Number of cells in this box.
 * @param[in]  active_mask  3-D activity mask (1=active, 0=inactive), with ghosts.
 * @param[in]  mask_lo    Lower bounds of active_mask (3 ints).
 * @param[in]  mask_hi    Upper bounds of active_mask (3 ints).
 * @param[in]  diff_coeff 3-D cell-centred diffusion coefficients, with ghosts.
 * @param[in]  dc_lo      Lower bounds of diff_coeff (3 ints).
 * @param[in]  dc_hi      Upper bounds of diff_coeff (3 ints).
 * @param[in]  bxlo       Lower corner of valid box (3 ints).
 * @param[in]  bxhi       Upper corner of valid box (3 ints).
 * @param[in]  domlo      Domain lower corner (3 ints).
 * @param[in]  domhi      Domain upper corner (3 ints).
 * @param[in]  dxinv      Inverse grid spacing squared: [1/dx^2, 1/dy^2, 1/dz^2].
 * @param[in]  vlo        Dirichlet BC value on the low face.
 * @param[in]  vhi        Dirichlet BC value on the high face.
 * @param[in]  dir        Flow direction (0=X, 1=Y, 2=Z).
 * @param[in]  verbose    Debug print level.
 */
inline void tortuosityFillMatrix(amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit, int nval,
                                 const int* active_mask, const int* mask_lo, const int* mask_hi,
                                 const amrex::Real* diff_coeff, const int* dc_lo, const int* dc_hi,
                                 const int* bxlo, const int* bxhi, const int* domlo,
                                 const int* domhi, const amrex::Real* dxinv, amrex::Real vlo,
                                 amrex::Real vhi, int dir, int verbose) {

    const int len_x = bxhi[0] - bxlo[0] + 1;
    const int len_y = bxhi[1] - bxlo[1] + 1;

    if (nval <= 0) return;

    const amrex::Real coeff_x = dxinv[0];
    const amrex::Real coeff_y = dxinv[1];
    const amrex::Real coeff_z = dxinv[2];

    // Helper lambdas to index into 3-D arrays stored in column-major (Fortran) order
    // active_mask dimensions
    const int mask_nx = mask_hi[0] - mask_lo[0] + 1;
    const int mask_ny = mask_hi[1] - mask_lo[1] + 1;
    auto mask = [&](int i, int j, int k) -> int {
        return active_mask[(i - mask_lo[0]) + (j - mask_lo[1]) * mask_nx +
                           (k - mask_lo[2]) * mask_nx * mask_ny];
    };

    // diff_coeff dimensions
    const int dc_nx = dc_hi[0] - dc_lo[0] + 1;
    const int dc_ny = dc_hi[1] - dc_lo[1] + 1;
    auto dc = [&](int i, int j, int k) -> amrex::Real {
        return diff_coeff[(i - dc_lo[0]) + (j - dc_lo[1]) * dc_nx +
                          (k - dc_lo[2]) * dc_nx * dc_ny];
    };

    int m_idx = 0;

    for (int k = bxlo[2]; k <= bxhi[2]; ++k) {
        for (int j = bxlo[1]; j <= bxhi[1]; ++j) {
            for (int i = bxlo[0]; i <= bxhi[0]; ++i) {
                const int stn_start = NSTENCIL * m_idx;

                // --- Inactive cell: decouple ---
                if (mask(i, j, k) == CELL_INACTIVE) {
                    for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                    a[stn_start + STN_C] = 1.0;
                    rhs[m_idx] = 0.0;
                    xinit[m_idx] = 0.0;
                    ++m_idx;
                    continue;
                }

                // --- Active cell ---
                const amrex::Real D_c = dc(i, j, k);
                amrex::Real diag_val = 0.0;
                for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                bool has_inactive_neighbor = false;

                // -X face
                if (mask(i - 1, j, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i - 1, j, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a[stn_start + STN_MX] = -D_face * coeff_x;
                    diag_val += D_face * coeff_x;
                } else {
                    has_inactive_neighbor = true;
                }

                // +X face
                if (mask(i + 1, j, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i + 1, j, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a[stn_start + STN_PX] = -D_face * coeff_x;
                    diag_val += D_face * coeff_x;
                } else {
                    has_inactive_neighbor = true;
                }

                // -Y face
                if (mask(i, j - 1, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j - 1, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a[stn_start + STN_MY] = -D_face * coeff_y;
                    diag_val += D_face * coeff_y;
                } else {
                    has_inactive_neighbor = true;
                }

                // +Y face
                if (mask(i, j + 1, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j + 1, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a[stn_start + STN_PY] = -D_face * coeff_y;
                    diag_val += D_face * coeff_y;
                } else {
                    has_inactive_neighbor = true;
                }

                // -Z face
                if (mask(i, j, k - 1) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j, k - 1);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a[stn_start + STN_MZ] = -D_face * coeff_z;
                    diag_val += D_face * coeff_z;
                } else {
                    has_inactive_neighbor = true;
                }

                // +Z face
                if (mask(i, j, k + 1) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j, k + 1);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a[stn_start + STN_PZ] = -D_face * coeff_z;
                    diag_val += D_face * coeff_z;
                } else {
                    has_inactive_neighbor = true;
                }

                a[stn_start + STN_C] = diag_val;

                // Check for zero diagonal in an active cell
                if (std::abs(diag_val) < SMALL_REAL) {
                    std::printf("WARNING: Zero diagonal in ACTIVE cell at (i,j,k)=%d %d %d\n",
                                i, j, k);
                    for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                    a[stn_start + STN_C] = 1.0;
                    rhs[m_idx] = 0.0;
                    xinit[m_idx] = 0.0;
                    ++m_idx;
                    continue;
                }

                rhs[m_idx] = 0.0;

                // --- Overwrite for Dirichlet BCs perpendicular to flow ---
                bool on_dirichlet_boundary = false;
                if (dir == 0) {
                    if (i == domlo[0]) {
                        for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = vlo;
                        on_dirichlet_boundary = true;
                    } else if (i == domhi[0]) {
                        for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = vhi;
                        on_dirichlet_boundary = true;
                    }
                } else if (dir == 1) {
                    if (j == domlo[1]) {
                        for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = vlo;
                        on_dirichlet_boundary = true;
                    } else if (j == domhi[1]) {
                        for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = vhi;
                        on_dirichlet_boundary = true;
                    }
                } else if (dir == 2) {
                    if (k == domlo[2]) {
                        for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = vlo;
                        on_dirichlet_boundary = true;
                    } else if (k == domhi[2]) {
                        for (int s = 0; s < NSTENCIL; ++s) a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = vhi;
                        on_dirichlet_boundary = true;
                    }
                }

                // --- Initial guess: linear interpolation along flow direction ---
                if (std::abs(a[stn_start + STN_C] - 1.0) > SMALL_REAL || on_dirichlet_boundary) {
                    amrex::Real domain_extent = 0.0;
                    amrex::Real coord = 0.0;
                    if (dir == 0) {
                        domain_extent = static_cast<amrex::Real>(domhi[0] - domlo[0]);
                        coord = static_cast<amrex::Real>(i - domlo[0]);
                    } else if (dir == 1) {
                        domain_extent = static_cast<amrex::Real>(domhi[1] - domlo[1]);
                        coord = static_cast<amrex::Real>(j - domlo[1]);
                    } else {
                        domain_extent = static_cast<amrex::Real>(domhi[2] - domlo[2]);
                        coord = static_cast<amrex::Real>(k - domlo[2]);
                    }
                    amrex::Real factor =
                        (std::abs(domain_extent) < SMALL_REAL) ? 0.0 : 1.0 / domain_extent;
                    xinit[m_idx] = vlo + (vhi - vlo) * coord * factor;
                }

                // --- Debug printing ---
                if (verbose >= 3) {
                    bool near_boundary =
                        (i <= domlo[0] + 1 || i >= domhi[0] - 1 || j <= domlo[1] + 1 ||
                         j >= domhi[1] - 1 || k <= domlo[2] + 1 || k >= domhi[2] - 1);

                    if (near_boundary || has_inactive_neighbor) {
                        amrex::Real off_diag_sum = 0.0;
                        for (int s = 1; s < NSTENCIL; ++s)
                            off_diag_sum += std::abs(a[stn_start + s]);
                        amrex::Real dv = a[stn_start + STN_C];
                        amrex::Real diag_ratio;
                        if (std::abs(off_diag_sum) < SMALL_REAL)
                            diag_ratio = (std::abs(dv) < SMALL_REAL) ? 1.0 : 1.0e+30;
                        else
                            diag_ratio = std::abs(dv) / off_diag_sum;

                        std::printf("DEBUG Stencil at (%d,%d,%d) Active=1 Dirichlet=%d "
                                    "Interface=%d D_c=%.4e\n",
                                    i, j, k, static_cast<int>(on_dirichlet_boundary),
                                    static_cast<int>(has_inactive_neighbor), D_c);
                        std::printf("  RHS =%.4e\n", rhs[m_idx]);
                        std::printf("  Stencil (C, -X,+X, -Y,+Y, -Z,+Z) = %.4e %.4e %.4e %.4e "
                                    "%.4e %.4e %.4e\n",
                                    a[stn_start + STN_C], a[stn_start + STN_MX],
                                    a[stn_start + STN_PX], a[stn_start + STN_MY],
                                    a[stn_start + STN_PY], a[stn_start + STN_MZ],
                                    a[stn_start + STN_PZ]);
                        std::printf("  Diag Dominance Ratio = %.4e (OffDiagSum = %.4e)\n",
                                    diag_ratio, off_diag_sum);
                    }
                }

                ++m_idx;
            }
        }
    }
}

} // namespace OpenImpala

#endif // OPENIMPALA_TORTUOSITY_HYPRE_FILL_H_
