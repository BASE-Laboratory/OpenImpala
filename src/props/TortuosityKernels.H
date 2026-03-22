/** @file TortuosityKernels.H
 *  @brief C++ kernels replacing the legacy Fortran routines in
 *         Tortuosity_filcc.F90 and Tortuosity_poisson_3d.F90.
 *
 *  Only the routines that are actually called from C++ are included:
 *    - removeIsolatedCells()    replaces tortuosity_remspot
 *    - computeFaceFluxes()      replaces tortuosity_poisson_flux
 *    - forwardEulerUpdate()     replaces tortuosity_poisson_update
 *    - integrateBoundaryFluxes() replaces tortuosity_poisson_fio
 *
 *  Legacy routines tortuosity_filct, tortuosity_filbc, and tortuosity_filic
 *  are no longer called and have been removed.
 */
#ifndef OPENIMPALA_TORTUOSITY_KERNELS_H_
#define OPENIMPALA_TORTUOSITY_KERNELS_H_

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace OpenImpala {

/// Cell-type constants (matching Tortuosity.H conventions)
constexpr int CELL_TYPE_BLOCKED = 0;
constexpr int CELL_TYPE_FREE = 1;

/// Sentinel value for neighbors outside the physical domain
constexpr int NEIGHBOR_OUTSIDE = -1;

// =========================================================================
// removeIsolatedCells  —  replaces tortuosity_remspot
// =========================================================================
/**
 * @brief Remove isolated single cells by checking 6-connectivity.
 *
 * For each cell in bxlo:bxhi, if none of the 6 face-connected neighbours
 * share the same phase, the cell's phase is flipped (0<->1).
 *
 * @param[in,out] q       Pointer to integer phase data (first component).
 * @param[in]     q_lo    Lower bounds of q array (3 ints, includes ghosts).
 * @param[in]     q_hi    Upper bounds of q array (3 ints, includes ghosts).
 * @param[in]     ncomp   Number of components (only component 0 is used).
 * @param[in]     bxlo    Lower corner of valid box to iterate (3 ints).
 * @param[in]     bxhi    Upper corner of valid box to iterate (3 ints).
 * @param[in]     domlo   Physical domain lower corner (3 ints).
 * @param[in]     domhi   Physical domain upper corner (3 ints).
 */
inline void removeIsolatedCells(int* q, const int* q_lo, const int* q_hi, int /*ncomp*/,
                                const int* bxlo, const int* bxhi, const int* domlo,
                                const int* domhi) {
    // 3-D column-major indexing for component 0
    const int nx = q_hi[0] - q_lo[0] + 1;
    const int ny = q_hi[1] - q_lo[1] + 1;
    auto idx = [&](int i, int j, int k) -> int& {
        return q[(i - q_lo[0]) + (j - q_lo[1]) * nx + (k - q_lo[2]) * nx * ny];
    };

    for (int k = bxlo[2]; k <= bxhi[2]; ++k) {
        for (int j = bxlo[1]; j <= bxhi[1]; ++j) {
            for (int i = bxlo[0]; i <= bxhi[0]; ++i) {
                const int center = idx(i, j, k);

                // Gather 6 neighbour phases; use NEIGHBOR_OUTSIDE at domain edges
                int nbr[6];
                nbr[0] = (i == domlo[0]) ? NEIGHBOR_OUTSIDE : idx(i - 1, j, k);
                nbr[1] = (i == domhi[0]) ? NEIGHBOR_OUTSIDE : idx(i + 1, j, k);
                nbr[2] = (j == domlo[1]) ? NEIGHBOR_OUTSIDE : idx(i, j - 1, k);
                nbr[3] = (j == domhi[1]) ? NEIGHBOR_OUTSIDE : idx(i, j + 1, k);
                nbr[4] = (k == domlo[2]) ? NEIGHBOR_OUTSIDE : idx(i, j, k - 1);
                nbr[5] = (k == domhi[2]) ? NEIGHBOR_OUTSIDE : idx(i, j, k + 1);

                bool connected = false;
                for (int n = 0; n < 6; ++n) {
                    if (center == nbr[n]) {
                        connected = true;
                        break;
                    }
                }

                if (!connected) {
                    idx(i, j, k) = (center == 0) ? 1 : 0;
                }
            }
        }
    }
}

// =========================================================================
// computeFaceFluxes  —  replaces tortuosity_poisson_flux
// =========================================================================
/**
 * @brief Compute face fluxes using finite differences on a cell-centred solution.
 *
 * Sets flux to zero if either adjacent cell is blocked.
 * Uses AMReX Array4 for clean multi-dimensional indexing.
 *
 * @param bx       Valid cell-centred box defining the region.
 * @param fx       X-face flux (output), Array4<Real>.
 * @param fy       Y-face flux (output), Array4<Real>.
 * @param fz       Z-face flux (output), Array4<Real>.
 * @param sol      Solution field with component 0 = phi, component 1 = cell type.
 * @param dxinv    Inverse cell sizes [1/dx, 1/dy, 1/dz].
 */
inline void computeFaceFluxes(const amrex::Box& bx, const amrex::Array4<amrex::Real>& fx,
                              const amrex::Array4<amrex::Real>& fy,
                              const amrex::Array4<amrex::Real>& fz,
                              const amrex::Array4<const amrex::Real>& sol,
                              const amrex::Real* dxinv) {
    const auto lo = bx.loVect3d();
    const auto hi = bx.hiVect3d();
    const amrex::Real dhx = dxinv[0];
    const amrex::Real dhy = dxinv[1];
    const amrex::Real dhz = dxinv[2];

    constexpr int comp_phi = 0;
    constexpr int comp_ct = 1;

    // X-fluxes on faces i = lo.x .. hi.x+1
    {
        amrex::Box xbx(bx);
        xbx.surroundingNodes(0); // face-centred in X
        amrex::ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (static_cast<int>(std::round(sol(i, j, k, comp_ct))) == CELL_TYPE_BLOCKED ||
                static_cast<int>(std::round(sol(i - 1, j, k, comp_ct))) == CELL_TYPE_BLOCKED) {
                fx(i, j, k) = 0.0;
            } else {
                fx(i, j, k) = dhx * (sol(i, j, k, comp_phi) - sol(i - 1, j, k, comp_phi));
            }
        });
    }

    // Y-fluxes on faces j = lo.y .. hi.y+1
    {
        amrex::Box ybx(bx);
        ybx.surroundingNodes(1);
        amrex::ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (static_cast<int>(std::round(sol(i, j, k, comp_ct))) == CELL_TYPE_BLOCKED ||
                static_cast<int>(std::round(sol(i, j - 1, k, comp_ct))) == CELL_TYPE_BLOCKED) {
                fy(i, j, k) = 0.0;
            } else {
                fy(i, j, k) = dhy * (sol(i, j, k, comp_phi) - sol(i, j - 1, k, comp_phi));
            }
        });
    }

    // Z-fluxes on faces k = lo.z .. hi.z+1
    {
        amrex::Box zbx(bx);
        zbx.surroundingNodes(2);
        amrex::ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (static_cast<int>(std::round(sol(i, j, k, comp_ct))) == CELL_TYPE_BLOCKED ||
                static_cast<int>(std::round(sol(i, j, k - 1, comp_ct))) == CELL_TYPE_BLOCKED) {
                fz(i, j, k) = 0.0;
            } else {
                fz(i, j, k) = dhz * (sol(i, j, k, comp_phi) - sol(i, j, k - 1, comp_phi));
            }
        });
    }
}

// =========================================================================
// forwardEulerUpdate  —  replaces tortuosity_poisson_update
// =========================================================================
/**
 * @brief Forward Euler update: phi_new = phi_old + dt * div(flux).
 *
 * Only updates component 0 (phi). Cell type (component 1) is untouched.
 *
 * @param bx       Valid cell-centred box.
 * @param p_arr    Old solution (input), Array4<const Real>.
 * @param n_arr    New solution (output), Array4<Real>.
 * @param fx       X-face flux (input).
 * @param fy       Y-face flux (input).
 * @param fz       Z-face flux (input).
 * @param dxinv    Inverse cell sizes [1/dx, 1/dy, 1/dz].
 * @param dt       Time step / relaxation parameter.
 */
inline void forwardEulerUpdate(const amrex::Box& bx, const amrex::Array4<const amrex::Real>& p_arr,
                               const amrex::Array4<amrex::Real>& n_arr,
                               const amrex::Array4<const amrex::Real>& fx,
                               const amrex::Array4<const amrex::Real>& fy,
                               const amrex::Array4<const amrex::Real>& fz, const amrex::Real* dxinv,
                               amrex::Real dt) {
    const amrex::Real dtdx = dt * dxinv[0];
    const amrex::Real dtdy = dt * dxinv[1];
    const amrex::Real dtdz = dt * dxinv[2];

    constexpr int comp_phi = 0;

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        n_arr(i, j, k, comp_phi) =
            p_arr(i, j, k, comp_phi) + dtdx * (fx(i + 1, j, k) - fx(i, j, k)) +
            dtdy * (fy(i, j + 1, k) - fy(i, j, k)) + dtdz * (fz(i, j, k + 1) - fz(i, j, k));
    });
}

// =========================================================================
// integrateBoundaryFluxes  —  replaces tortuosity_poisson_fio
// =========================================================================
/**
 * @brief Sum fluxes on the inlet (low) and outlet (high) domain faces.
 *
 * @param cell_bx   Cell-centred valid box for this FAB.
 * @param fx        X-face flux Array4 (const).
 * @param fy        Y-face flux Array4 (const).
 * @param fz        Z-face flux Array4 (const).
 * @param fxbox     Allocated memory box for fx (to check face ownership).
 * @param fybox     Allocated memory box for fy.
 * @param fzbox     Allocated memory box for fz.
 * @param dir       Flow direction (0=X, 1=Y, 2=Z).
 * @param flux_in   (output) accumulated flux entering through the low face.
 * @param flux_out  (output) accumulated flux exiting through the high face.
 */
inline void integrateBoundaryFluxes(const amrex::Box& cell_bx,
                                    const amrex::Array4<const amrex::Real>& fx,
                                    const amrex::Array4<const amrex::Real>& fy,
                                    const amrex::Array4<const amrex::Real>& fz,
                                    const amrex::Box& fxbox, const amrex::Box& fybox,
                                    const amrex::Box& fzbox, int dir, amrex::Real& flux_in,
                                    amrex::Real& flux_out) {
    const auto lo = cell_bx.loVect3d();
    const auto hi = cell_bx.hiVect3d();

    flux_in = 0.0;
    flux_out = 0.0;

    if (dir == 0) {
        // X-direction: sum fx on low-X and high-X faces
        if (fxbox.smallEnd(0) <= lo[0]) {
            for (int k = lo[2]; k <= hi[2]; ++k)
                for (int j = lo[1]; j <= hi[1]; ++j)
                    flux_in += fx(lo[0], j, k);
        }
        if (fxbox.bigEnd(0) >= hi[0] + 1) {
            for (int k = lo[2]; k <= hi[2]; ++k)
                for (int j = lo[1]; j <= hi[1]; ++j)
                    flux_out += fx(hi[0] + 1, j, k);
        }
    } else if (dir == 1) {
        // Y-direction
        if (fybox.smallEnd(1) <= lo[1]) {
            for (int k = lo[2]; k <= hi[2]; ++k)
                for (int i = lo[0]; i <= hi[0]; ++i)
                    flux_in += fy(i, lo[1], k);
        }
        if (fybox.bigEnd(1) >= hi[1] + 1) {
            for (int k = lo[2]; k <= hi[2]; ++k)
                for (int i = lo[0]; i <= hi[0]; ++i)
                    flux_out += fy(i, hi[1] + 1, k);
        }
    } else {
        // Z-direction
        if (fzbox.smallEnd(2) <= lo[2]) {
            for (int j = lo[1]; j <= hi[1]; ++j)
                for (int i = lo[0]; i <= hi[0]; ++i)
                    flux_in += fz(i, j, lo[2]);
        }
        if (fzbox.bigEnd(2) >= hi[2] + 1) {
            for (int j = lo[1]; j <= hi[1]; ++j)
                for (int i = lo[0]; i <= hi[0]; ++i)
                    flux_out += fz(i, j, hi[2] + 1);
        }
    }
}

} // namespace OpenImpala

#endif // OPENIMPALA_TORTUOSITY_KERNELS_H_
