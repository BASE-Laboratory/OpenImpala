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

/// Pinned active cell marker (chi = 0 Dirichlet pin for per-component nullspace removal)
constexpr int CELL_PINNED = 2;

/**
 * @brief Fill HYPRE matrix for the effective-diffusivity cell problem.
 *
 * @param[out] a_out       Flattened stencil array, size npts*7.
 * @param[out] rhs_out     RHS vector, size npts.
 * @param[out] xinit_out   Initial guess, size npts.
 * @param[in]  npts        Number of cells in this box.
 * @param[in]  active_mask 3-D cell-state mask (0=inactive, 1=active, 2=pinned active).
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
 * @param[in]  pin_scale   Diagonal scale for decoupled inactive and pinned cells.
 */
inline void effDiffFillMatrix(amrex::Real* a_out, amrex::Real* rhs_out, amrex::Real* xinit_out,
                              int npts, const int* active_mask, const int* mask_lo,
                              const int* mask_hi, const amrex::Real* diff_coeff, const int* dc_lo,
                              const int* dc_hi, const int* bxlo, const int* bxhi, const int* domlo,
                              const int* domhi, const amrex::Real* cell_sizes, int dir_k,
                              int verbose, amrex::Real pin_scale = 1.0) {
    (void)domlo;
    (void)domhi;
    (void)verbose;

    if (npts <= 0)
        return;

    const amrex::Real dx = cell_sizes[0];
    const amrex::Real dy = cell_sizes[1];
    const amrex::Real dz = cell_sizes[2];

    const amrex::Real inv_dx = 1.0 / dx;
    const amrex::Real inv_dy = 1.0 / dy;
    const amrex::Real inv_dz = 1.0 / dz;
    const amrex::Real inv_dx2 = inv_dx * inv_dx;
    const amrex::Real inv_dy2 = inv_dy * inv_dy;
    const amrex::Real inv_dz2 = inv_dz * inv_dz;

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

                const int cell_state = mask(i, j, k);
                // Inactive cell or pinned active cell (chi = 0): decouple symmetrically
                if (cell_state == CELL_INACTIVE || cell_state == CELL_PINNED) {
                    a_out[stn + STN_C] = pin_scale;
                    ++m_idx;
                    continue;
                }

                // --- Active cell: conservative finite-volume -div(D (e_k + grad chi_k)) = 0 ---
                const amrex::Real D_c = dc(i, j, k);
                amrex::Real diag_val = 0.0;

                const int mx_state = mask(i - 1, j, k);
                const int px_state = mask(i + 1, j, k);
                const int my_state = mask(i, j - 1, k);
                const int py_state = mask(i, j + 1, k);

                amrex::Real D_mx = 0.0, D_px = 0.0;
                amrex::Real D_my = 0.0, D_py = 0.0;
                amrex::Real D_mz = 0.0, D_pz = 0.0;

                // -X / +X faces (insulating faces contribute zero conductance;
                // faces into CELL_PINNED contribute to diagonal and RHS with chi_pinned = 0)
                if (mx_state != CELL_INACTIVE) {
                    amrex::Real D_nbr = dc(i - 1, j, k);
                    if (D_c + D_nbr > 0.0)
                        D_mx = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    if (mx_state == CELL_ACTIVE)
                        a_out[stn + STN_MX] = -D_mx * inv_dx2;
                    diag_val += D_mx * inv_dx2;
                }
                if (px_state != CELL_INACTIVE) {
                    amrex::Real D_nbr = dc(i + 1, j, k);
                    if (D_c + D_nbr > 0.0)
                        D_px = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    if (px_state == CELL_ACTIVE)
                        a_out[stn + STN_PX] = -D_px * inv_dx2;
                    diag_val += D_px * inv_dx2;
                }

                // -Y / +Y faces
                if (my_state != CELL_INACTIVE) {
                    amrex::Real D_nbr = dc(i, j - 1, k);
                    if (D_c + D_nbr > 0.0)
                        D_my = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    if (my_state == CELL_ACTIVE)
                        a_out[stn + STN_MY] = -D_my * inv_dy2;
                    diag_val += D_my * inv_dy2;
                }
                if (py_state != CELL_INACTIVE) {
                    amrex::Real D_nbr = dc(i, j + 1, k);
                    if (D_c + D_nbr > 0.0)
                        D_py = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    if (py_state == CELL_ACTIVE)
                        a_out[stn + STN_PY] = -D_py * inv_dy2;
                    diag_val += D_py * inv_dy2;
                }

#if (AMREX_SPACEDIM == 3)
                const int mz_state = mask(i, j, k - 1);
                const int pz_state = mask(i, j, k + 1);

                // -Z / +Z faces
                if (mz_state != CELL_INACTIVE) {
                    amrex::Real D_nbr = dc(i, j, k - 1);
                    if (D_c + D_nbr > 0.0)
                        D_mz = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    if (mz_state == CELL_ACTIVE)
                        a_out[stn + STN_MZ] = -D_mz * inv_dz2;
                    diag_val += D_mz * inv_dz2;
                }
                if (pz_state != CELL_INACTIVE) {
                    amrex::Real D_nbr = dc(i, j, k + 1);
                    if (D_c + D_nbr > 0.0)
                        D_pz = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    if (pz_state == CELL_ACTIVE)
                        a_out[stn + STN_PZ] = -D_pz * inv_dz2;
                    diag_val += D_pz * inv_dz2;
                }
#endif

                a_out[stn + STN_C] = diag_val;

                // Conservative face-flux RHS: (D_{i+1/2}^{(k)} - D_{i-1/2}^{(k)}) / h_k
                amrex::Real rhs_val = 0.0;
                if (dir_k == 0) {
                    rhs_val = (D_px - D_mx) * inv_dx;
                } else if (dir_k == 1) {
                    rhs_val = (D_py - D_my) * inv_dy;
                }
#if (AMREX_SPACEDIM == 3)
                else if (dir_k == 2) {
                    rhs_val = (D_pz - D_mz) * inv_dz;
                }
#endif
                rhs_out[m_idx] = rhs_val;

                // Safety: decouple cells with near-zero diagonal
                if (std::abs(diag_val) < SMALL_REAL) {
                    for (int s = 0; s < NSTEN; ++s)
                        a_out[stn + s] = 0.0;
                    a_out[stn + STN_C] = pin_scale;
                    rhs_out[m_idx] = 0.0;
                    xinit_out[m_idx] = 0.0;
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
 * Fills HYPRE stencil coefficients, RHS, and initial guess for the conservative
 * finite-volume cell problem -div(D (e_k + grad chi_k)) = 0 on GPU.
 *
 * @param bx         Valid box defining the region.
 * @param a_out      Device pointer: flattened stencil, size bx.numPts()*7.
 * @param rhs_out    Device pointer: RHS, size bx.numPts().
 * @param xinit_out  Device pointer: initial guess, size bx.numPts().
 * @param mask_arr   Array4<const int> for cell-state mask (with ghosts).
 * @param dc_arr     Array4<const Real> for diffusion coefficients (with ghosts).
 * @param cell_sizes Grid spacing [dx, dy, dz].
 * @param dir_k      Direction of unit vector e_k (0=X, 1=Y, 2=Z).
 * @param pin_scale  Diagonal scale for decoupled inactive and pinned cells.
 */
inline void effDiffFillMatrixGpu(const amrex::Box& bx, amrex::Real* a_out, amrex::Real* rhs_out,
                                 amrex::Real* xinit_out, const amrex::Array4<const int>& mask_arr,
                                 const amrex::Array4<const amrex::Real>& dc_arr,
                                 const amrex::Real* cell_sizes, int dir_k,
                                 amrex::Real pin_scale = 1.0) {
    const int bx_lo0 = bx.smallEnd(0);
    const int bx_lo1 = bx.smallEnd(1);
    const int bx_lo2 = bx.smallEnd(2);
    const int bx_nx = bx.length(0);
    const int bx_ny = bx.length(1);

    const amrex::Real dx = cell_sizes[0];
    const amrex::Real dy = cell_sizes[1];
    const amrex::Real dz = cell_sizes[2];

    const amrex::Real inv_dx = 1.0 / dx;
    const amrex::Real inv_dy = 1.0 / dy;
    const amrex::Real inv_dz = 1.0 / dz;
    const amrex::Real inv_dx2 = inv_dx * inv_dx;
    const amrex::Real inv_dy2 = inv_dy * inv_dy;
    const amrex::Real inv_dz2 = inv_dz * inv_dz;

    constexpr int NSTEN = 7;

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const int m_idx = (i - bx_lo0) + (j - bx_lo1) * bx_nx + (k - bx_lo2) * bx_nx * bx_ny;
        const int stn = NSTEN * m_idx;

        // Initialize
        for (int s = 0; s < NSTEN; ++s)
            a_out[stn + s] = 0.0;
        rhs_out[m_idx] = 0.0;
        xinit_out[m_idx] = 0.0;

        const int cell_state = mask_arr(i, j, k);
        // Inactive cell or pinned active cell (chi = 0): decouple symmetrically
        if (cell_state == CELL_INACTIVE || cell_state == CELL_PINNED) {
            a_out[stn + STN_C] = pin_scale;
            return;
        }

        const amrex::Real D_c = dc_arr(i, j, k);
        amrex::Real diag_val = 0.0;

        const int mx_state = mask_arr(i - 1, j, k);
        const int px_state = mask_arr(i + 1, j, k);
        const int my_state = mask_arr(i, j - 1, k);
        const int py_state = mask_arr(i, j + 1, k);

        amrex::Real D_mx = 0.0, D_px = 0.0;
        amrex::Real D_my = 0.0, D_py = 0.0;
        amrex::Real D_mz = 0.0, D_pz = 0.0;

        if (mx_state != CELL_INACTIVE) {
            amrex::Real D_nbr = dc_arr(i - 1, j, k);
            if (D_c + D_nbr > 0.0)
                D_mx = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            if (mx_state == CELL_ACTIVE)
                a_out[stn + STN_MX] = -D_mx * inv_dx2;
            diag_val += D_mx * inv_dx2;
        }
        if (px_state != CELL_INACTIVE) {
            amrex::Real D_nbr = dc_arr(i + 1, j, k);
            if (D_c + D_nbr > 0.0)
                D_px = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            if (px_state == CELL_ACTIVE)
                a_out[stn + STN_PX] = -D_px * inv_dx2;
            diag_val += D_px * inv_dx2;
        }

        if (my_state != CELL_INACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j - 1, k);
            if (D_c + D_nbr > 0.0)
                D_my = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            if (my_state == CELL_ACTIVE)
                a_out[stn + STN_MY] = -D_my * inv_dy2;
            diag_val += D_my * inv_dy2;
        }
        if (py_state != CELL_INACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j + 1, k);
            if (D_c + D_nbr > 0.0)
                D_py = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            if (py_state == CELL_ACTIVE)
                a_out[stn + STN_PY] = -D_py * inv_dy2;
            diag_val += D_py * inv_dy2;
        }

#if (AMREX_SPACEDIM == 3)
        const int mz_state = mask_arr(i, j, k - 1);
        const int pz_state = mask_arr(i, j, k + 1);

        if (mz_state != CELL_INACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j, k - 1);
            if (D_c + D_nbr > 0.0)
                D_mz = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            if (mz_state == CELL_ACTIVE)
                a_out[stn + STN_MZ] = -D_mz * inv_dz2;
            diag_val += D_mz * inv_dz2;
        }
        if (pz_state != CELL_INACTIVE) {
            amrex::Real D_nbr = dc_arr(i, j, k + 1);
            if (D_c + D_nbr > 0.0)
                D_pz = 2.0 * D_c * D_nbr / (D_c + D_nbr);
            if (pz_state == CELL_ACTIVE)
                a_out[stn + STN_PZ] = -D_pz * inv_dz2;
            diag_val += D_pz * inv_dz2;
        }
#endif

        a_out[stn + STN_C] = diag_val;

        amrex::Real rhs_val = 0.0;
        if (dir_k == 0) {
            rhs_val = (D_px - D_mx) * inv_dx;
        } else if (dir_k == 1) {
            rhs_val = (D_py - D_my) * inv_dy;
        }
#if (AMREX_SPACEDIM == 3)
        else if (dir_k == 2) {
            rhs_val = (D_pz - D_mz) * inv_dz;
        }
#endif
        rhs_out[m_idx] = rhs_val;

        if (amrex::Math::abs(diag_val) < SMALL_REAL) {
            for (int s = 0; s < NSTEN; ++s)
                a_out[stn + s] = 0.0;
            a_out[stn + STN_C] = pin_scale;
            rhs_out[m_idx] = 0.0;
            xinit_out[m_idx] = 0.0;
        }
    });
}

#endif // OPENIMPALA_USE_GPU

} // namespace OpenImpala

#endif // OPENIMPALA_EFFDIFF_FILL_MTX_H_
