/** @file EffDiffFillMtx.H
 *  @brief C++ kernel for filling HYPRE structured-grid matrix coefficients
 *         for the effective-diffusivity cell problem:
 *           div(D grad(chi_k)) = -div(D e_k)
 *
 *  Replaces the legacy Fortran kernel EffDiffFillMtx.F90.
 *  Uses a 7-point stencil with harmonic-mean face coefficients,
 *  internal Neumann BCs at pore-solid interfaces, and a central-
 *  difference source term for the RHS.
 */
#ifndef OPENIMPALA_EFFDIFF_FILL_MTX_H_
#define OPENIMPALA_EFFDIFF_FILL_MTX_H_

#include "TortuosityHypreFill.H" // Shared constants: StencilIdx, CELL_ACTIVE/INACTIVE, SMALL_REAL

#include <AMReX_BLassert.H>
#include <AMReX_REAL.H>
#include <AMReX_Print.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>
#include <cstdio>

namespace OpenImpala {

/**
 * @brief Fill HYPRE matrix for the effective-diffusivity cell problem.
 *
 * @param[out] a_out       Flattened stencil array, size npts*7.
 * @param[out] rhs_out     RHS vector, size npts.
 * @param[out] xinit_out   Initial guess, size npts.
 * @param[in]  npts        Number of cells in this box.
 * @param[in]  active_mask 3-D activity mask (1=active, 0=inactive).
 * @param[in]  mask_lo     Lower bounds of mask (3 ints).
 * @param[in]  mask_hi     Upper bounds of mask (3 ints).
 * @param[in]  diff_coeff  3-D cell-centred diffusion coefficients.
 * @param[in]  dc_lo       Lower bounds of diff_coeff (3 ints).
 * @param[in]  dc_hi       Upper bounds of diff_coeff (3 ints).
 * @param[in]  bxlo        Lower corner of valid box (3 ints).
 * @param[in]  bxhi        Upper corner of valid box (3 ints).
 * @param[in]  domlo       Domain lower corner (3 ints).
 * @param[in]  domhi       Domain upper corner (3 ints).
 * @param[in]  cell_sizes  Grid spacing [dx, dy, dz].
 * @param[in]  dir_k       Direction of unit vector e_k (0=X, 1=Y, 2=Z).
 * @param[in]  verbose     Debug print level.
 */
inline void effDiffFillMatrix(amrex::Real* a_out, amrex::Real* rhs_out, amrex::Real* xinit_out,
                              int npts, const int* active_mask, const int* mask_lo,
                              const int* mask_hi, const amrex::Real* diff_coeff, const int* dc_lo,
                              const int* dc_hi, const int* bxlo, const int* bxhi, const int* domlo,
                              const int* domhi, const amrex::Real* cell_sizes, int dir_k,
                              int verbose) {

    if (npts <= 0)
        return;

    const amrex::Real dx = cell_sizes[0];
    const amrex::Real dy = cell_sizes[1];
    const amrex::Real dz = cell_sizes[2];

    const amrex::Real inv_dx2 = 1.0 / (dx * dx);
    const amrex::Real inv_dy2 = 1.0 / (dy * dy);
    const amrex::Real inv_dz2 = 1.0 / (dz * dz);
    const amrex::Real inv_2dx = 1.0 / (2.0 * dx);
    const amrex::Real inv_2dy = 1.0 / (2.0 * dy);
    const amrex::Real inv_2dz = 1.0 / (2.0 * dz);

    // Array indexing helpers (column-major / Fortran order)
    const int mask_nx = mask_hi[0] - mask_lo[0] + 1;
    const int mask_ny = mask_hi[1] - mask_lo[1] + 1;
    auto mask = [&](int i, int j, int k) -> int {
        return active_mask[(i - mask_lo[0]) + (j - mask_lo[1]) * mask_nx +
                           (k - mask_lo[2]) * mask_nx * mask_ny];
    };

    const int dc_nx = dc_hi[0] - dc_lo[0] + 1;
    const int dc_ny = dc_hi[1] - dc_lo[1] + 1;
    auto dc = [&](int i, int j, int k) -> amrex::Real {
        return diff_coeff[(i - dc_lo[0]) + (j - dc_lo[1]) * dc_nx + (k - dc_lo[2]) * dc_nx * dc_ny];
    };

    constexpr int NSTEN = 7;
    int m_idx = 0;

    for (int k = bxlo[2]; k <= bxhi[2]; ++k) {
        for (int j = bxlo[1]; j <= bxhi[1]; ++j) {
            for (int i = bxlo[0]; i <= bxhi[0]; ++i) {
                const int stn = NSTEN * m_idx;

                // Initialize
                for (int s = 0; s < NSTEN; ++s)
                    a_out[stn + s] = 0.0;
                rhs_out[m_idx] = 0.0;
                xinit_out[m_idx] = 0.0;

                // Inactive cell: decouple
                if (mask(i, j, k) == CELL_INACTIVE) {
                    a_out[stn + STN_C] = 1.0;
                    ++m_idx;
                    continue;
                }

                // --- Active cell ---
                const amrex::Real D_c = dc(i, j, k);
                amrex::Real diag_val = 0.0;
                amrex::Real flux_bc_rhs = 0.0;

                // === LHS: div(D grad(chi_k)) with harmonic mean at faces ===

                // -X face
                if (mask(i - 1, j, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i - 1, j, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a_out[stn + STN_MX] = -D_face * inv_dx2;
                    diag_val += D_face * inv_dx2;
                } else {
                    // Internal Neumann: n_hat = (-1,0,0)
                    diag_val += D_c * inv_dx2;
                    if (dir_k == 0)
                        flux_bc_rhs += D_c / dx;
                }

                // +X face
                if (mask(i + 1, j, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i + 1, j, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a_out[stn + STN_PX] = -D_face * inv_dx2;
                    diag_val += D_face * inv_dx2;
                } else {
                    diag_val += D_c * inv_dx2;
                    if (dir_k == 0)
                        flux_bc_rhs -= D_c / dx;
                }

                // -Y face
                if (mask(i, j - 1, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j - 1, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a_out[stn + STN_MY] = -D_face * inv_dy2;
                    diag_val += D_face * inv_dy2;
                } else {
                    diag_val += D_c * inv_dy2;
                    if (dir_k == 1)
                        flux_bc_rhs += D_c / dy;
                }

                // +Y face
                if (mask(i, j + 1, k) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j + 1, k);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a_out[stn + STN_PY] = -D_face * inv_dy2;
                    diag_val += D_face * inv_dy2;
                } else {
                    diag_val += D_c * inv_dy2;
                    if (dir_k == 1)
                        flux_bc_rhs -= D_c / dy;
                }

#if (AMREX_SPACEDIM == 3)
                // -Z face
                if (mask(i, j, k - 1) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j, k - 1);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a_out[stn + STN_MZ] = -D_face * inv_dz2;
                    diag_val += D_face * inv_dz2;
                } else {
                    diag_val += D_c * inv_dz2;
                    if (dir_k == 2)
                        flux_bc_rhs += D_c / dz;
                }

                // +Z face
                if (mask(i, j, k + 1) == CELL_ACTIVE) {
                    amrex::Real D_nbr = dc(i, j, k + 1);
                    amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    a_out[stn + STN_PZ] = -D_face * inv_dz2;
                    diag_val += D_face * inv_dz2;
                } else {
                    diag_val += D_c * inv_dz2;
                    if (dir_k == 2)
                        flux_bc_rhs -= D_c / dz;
                }
#endif

                a_out[stn + STN_C] = diag_val;

                // === RHS: -div(D e_k) via central differences on D ===
                amrex::Real rhs_div_De = 0.0;
                if (dir_k == 0) {
                    rhs_div_De = -(dc(i + 1, j, k) - dc(i - 1, j, k)) * inv_2dx;
                } else if (dir_k == 1) {
                    rhs_div_De = -(dc(i, j + 1, k) - dc(i, j - 1, k)) * inv_2dy;
                }
#if (AMREX_SPACEDIM == 3)
                else if (dir_k == 2) {
                    rhs_div_De = -(dc(i, j, k + 1) - dc(i, j, k - 1)) * inv_2dz;
                }
#endif

                rhs_out[m_idx] = rhs_div_De + flux_bc_rhs;

                // Safety: decouple cells with near-zero diagonal
                if (std::abs(diag_val) < SMALL_REAL) {
                    if (verbose > 0) {
                        std::printf("effdiff_fillmtx WARNING: Near-zero diagonal in ACTIVE "
                                    "cell (%d,%d,%d), diag_val=%.4e. Decoupling.\n",
                                    i, j, k, diag_val);
                    }
                    for (int s = 0; s < NSTEN; ++s)
                        a_out[stn + s] = 0.0;
                    a_out[stn + STN_C] = 1.0;
                    rhs_out[m_idx] = 0.0;
                    xinit_out[m_idx] = 0.0;
                }

                if (verbose >= 3) {
                    std::printf("DEBUG effdiff_fillmtx: Cell (%d,%d,%d) dir_k=%d D_c=%.4e\n", i, j,
                                k, dir_k, D_c);
                    std::printf("  Stencil A: %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n", a_out[stn],
                                a_out[stn + 1], a_out[stn + 2], a_out[stn + 3], a_out[stn + 4],
                                a_out[stn + 5], a_out[stn + 6]);
                    std::printf("  RHS terms: div_De=%.4e flux_bc=%.4e Total_RHS=%.4e\n",
                                rhs_div_De, flux_bc_rhs, rhs_out[m_idx]);
                }

                ++m_idx;
            }
        }
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_idx == npts, "effDiffFillMatrix: m_idx != npts.");
}

// =============================================================================
// GPU-accelerated effective diffusivity matrix fill kernel
// =============================================================================
#ifdef OPENIMPALA_USE_GPU

/**
 * @brief GPU-compatible effective diffusivity matrix fill using AMReX ParallelFor.
 *
 * Fills HYPRE stencil coefficients, RHS, and initial guess for the cell problem
 * div(D grad(chi_k)) = -div(D e_k) on GPU. Uses Array4<> accessors.
 *
 * @param bx         Valid box defining the region.
 * @param a_out      Device pointer: flattened stencil, size bx.numPts()*7.
 * @param rhs_out    Device pointer: RHS, size bx.numPts().
 * @param xinit_out  Device pointer: initial guess, size bx.numPts().
 * @param mask_arr   Array4<const int> for activity mask (with ghosts).
 * @param dc_arr     Array4<const Real> for diffusion coefficients (with ghosts).
 * @param cell_sizes Grid spacing [dx, dy, dz].
 * @param dir_k      Direction of unit vector e_k (0=X, 1=Y, 2=Z).
 */
inline void effDiffFillMatrixGpu(const amrex::Box& bx, amrex::Real* a_out, amrex::Real* rhs_out,
                                 amrex::Real* xinit_out, const amrex::Array4<const int>& mask_arr,
                                 const amrex::Array4<const amrex::Real>& dc_arr,
                                 const amrex::Real* cell_sizes, int dir_k) {
    const int bx_lo0 = bx.smallEnd(0);
    const int bx_lo1 = bx.smallEnd(1);
    const int bx_lo2 = bx.smallEnd(2);
    const int bx_nx = bx.length(0);
    const int bx_ny = bx.length(1);

    const amrex::Real dx = cell_sizes[0];
    const amrex::Real dy = cell_sizes[1];
    const amrex::Real dz = cell_sizes[2];

    const amrex::Real inv_dx2 = 1.0 / (dx * dx);
    const amrex::Real inv_dy2 = 1.0 / (dy * dy);
    const amrex::Real inv_dz2 = 1.0 / (dz * dz);
    const amrex::Real inv_2dx = 1.0 / (2.0 * dx);
    const amrex::Real inv_2dy = 1.0 / (2.0 * dy);
    const amrex::Real inv_2dz = 1.0 / (2.0 * dz);

    constexpr int NSTEN = 7;

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const int m_idx = (i - bx_lo0) + (j - bx_lo1) * bx_nx + (k - bx_lo2) * bx_nx * bx_ny;
        const int stn = NSTEN * m_idx;

        // Initialize
        for (int s = 0; s < NSTEN; ++s)
            a_out[stn + s] = 0.0;
        rhs_out[m_idx] = 0.0;
        xinit_out[m_idx] = 0.0;

        // Inactive cell: decouple
        if (mask_arr(i, j, k) == CELL_INACTIVE) {
            a_out[stn + STN_C] = 1.0;
            return;
        }

        // Active cell
        const amrex::Real D_c = dc_arr(i, j, k);
        amrex::Real diag_val = 0.0;
        amrex::Real flux_bc_rhs = 0.0;

        // -X face
        if (mask_arr(i - 1, j, k) == CELL_ACTIVE) {
            amrex::Real D_nbr = dc_arr(i - 1, j, k);
            amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            a_out[stn + STN_MX] = -D_face * inv_dx2;
            diag_val += D_face * inv_dx2;
        } else {
            diag_val += D_c * inv_dx2;
            if (dir_k == 0)
                flux_bc_rhs += D_c / dx;
        }

        // +X face
        if (mask_arr(i + 1, j, k) == CELL_ACTIVE) {
            amrex::Real D_nbr = dc_arr(i + 1, j, k);
            amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            a_out[stn + STN_PX] = -D_face * inv_dx2;
            diag_val += D_face * inv_dx2;
        } else {
            diag_val += D_c * inv_dx2;
            if (dir_k == 0)
                flux_bc_rhs -= D_c / dx;
        }

        // -Y face
        if (mask_arr(i, j - 1, k) == CELL_ACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j - 1, k);
            amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            a_out[stn + STN_MY] = -D_face * inv_dy2;
            diag_val += D_face * inv_dy2;
        } else {
            diag_val += D_c * inv_dy2;
            if (dir_k == 1)
                flux_bc_rhs += D_c / dy;
        }

        // +Y face
        if (mask_arr(i, j + 1, k) == CELL_ACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j + 1, k);
            amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            a_out[stn + STN_PY] = -D_face * inv_dy2;
            diag_val += D_face * inv_dy2;
        } else {
            diag_val += D_c * inv_dy2;
            if (dir_k == 1)
                flux_bc_rhs -= D_c / dy;
        }

#if (AMREX_SPACEDIM == 3)
        // -Z face
        if (mask_arr(i, j, k - 1) == CELL_ACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j, k - 1);
            amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            a_out[stn + STN_MZ] = -D_face * inv_dz2;
            diag_val += D_face * inv_dz2;
        } else {
            diag_val += D_c * inv_dz2;
            if (dir_k == 2)
                flux_bc_rhs += D_c / dz;
        }

        // +Z face
        if (mask_arr(i, j, k + 1) == CELL_ACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j, k + 1);
            amrex::Real D_face = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            a_out[stn + STN_PZ] = -D_face * inv_dz2;
            diag_val += D_face * inv_dz2;
        } else {
            diag_val += D_c * inv_dz2;
            if (dir_k == 2)
                flux_bc_rhs -= D_c / dz;
        }
#endif

        a_out[stn + STN_C] = diag_val;

        // RHS: -div(D e_k) via central differences
        amrex::Real rhs_div_De = 0.0;
        if (dir_k == 0) {
            rhs_div_De = -(dc_arr(i + 1, j, k) - dc_arr(i - 1, j, k)) * inv_2dx;
        } else if (dir_k == 1) {
            rhs_div_De = -(dc_arr(i, j + 1, k) - dc_arr(i, j - 1, k)) * inv_2dy;
        }
#if (AMREX_SPACEDIM == 3)
        else if (dir_k == 2) {
            rhs_div_De = -(dc_arr(i, j, k + 1) - dc_arr(i, j, k - 1)) * inv_2dz;
        }
#endif

        rhs_out[m_idx] = rhs_div_De + flux_bc_rhs;

        // Safety: decouple near-zero diagonal
        if (amrex::Math::abs(diag_val) < SMALL_REAL) {
            for (int s = 0; s < NSTEN; ++s)
                a_out[stn + s] = 0.0;
            a_out[stn + STN_C] = 1.0;
            rhs_out[m_idx] = 0.0;
            xinit_out[m_idx] = 0.0;
        }
    });
}

#endif // OPENIMPALA_USE_GPU

} // namespace OpenImpala

#endif // OPENIMPALA_EFFDIFF_FILL_MTX_H_
