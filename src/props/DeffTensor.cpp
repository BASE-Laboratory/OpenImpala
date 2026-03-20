/** @file DeffTensor.cpp
 *  @brief Implementation of the effective diffusivity tensor assembly.
 */

#include "DeffTensor.H"

#include <AMReX_Loop.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

namespace OpenImpala {

void calculateDeffTensor(amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
                         const amrex::MultiFab& mf_chi_x, const amrex::MultiFab& mf_chi_y,
                         const amrex::MultiFab& mf_chi_z, const amrex::iMultiFab& active_mask,
                         const amrex::Geometry& geom, int verbose) {
    BL_PROFILE("OpenImpala::calculateDeffTensor");

    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            Deff_tensor[i][j] = 0.0;
        }
    }

    AMREX_ASSERT(mf_chi_x.nGrow() >= 1);
    AMREX_ASSERT(mf_chi_y.nGrow() >= 1);
    if (AMREX_SPACEDIM == 3) {
        AMREX_ASSERT(mf_chi_z.isDefined() && mf_chi_z.nGrow() >= 1);
    }
    AMREX_ASSERT(active_mask.nGrow() == 0);

    const amrex::Real* dx_arr = geom.CellSize();
    amrex::Real inv_2dx[AMREX_SPACEDIM];
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        inv_2dx[i] = 1.0 / (2.0 * dx_arr[i]);
    }

    amrex::Real sum_local[AMREX_SPACEDIM][AMREX_SPACEDIM];
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            sum_local[i][j] = 0.0;
        }
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        amrex::Real sum_thread[AMREX_SPACEDIM][AMREX_SPACEDIM];
        for (int r = 0; r < AMREX_SPACEDIM; ++r) {
            for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                sum_thread[r][c] = 0.0;
            }
        }

        for (amrex::MFIter mfi(active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<const int> const mask_arr = active_mask.const_array(mfi);
            amrex::Array4<const amrex::Real> const chi_x_arr = mf_chi_x.const_array(mfi);
            amrex::Array4<const amrex::Real> const chi_y_arr = mf_chi_y.const_array(mfi);
            amrex::Array4<const amrex::Real> const chi_z_arr =
                (AMREX_SPACEDIM == 3 && mf_chi_z.isDefined()) ? mf_chi_z.const_array(mfi)
                                                               : mf_chi_x.const_array(mfi);

            amrex::LoopOnCpu(bx, [=, &sum_thread](int i, int j, int k) noexcept {
                if (mask_arr(i, j, k, 0) == 1) {
                    amrex::Real grad_chi_x[AMREX_SPACEDIM] = {0.0};
                    amrex::Real grad_chi_y[AMREX_SPACEDIM] = {0.0};
                    amrex::Real grad_chi_z[AMREX_SPACEDIM] = {0.0};

                    grad_chi_x[0] =
                        (chi_x_arr(i + 1, j, k, 0) - chi_x_arr(i - 1, j, k, 0)) * inv_2dx[0];
                    grad_chi_x[1] =
                        (chi_x_arr(i, j + 1, k, 0) - chi_x_arr(i, j - 1, k, 0)) * inv_2dx[1];
                    if (AMREX_SPACEDIM == 3)
                        grad_chi_x[2] =
                            (chi_x_arr(i, j, k + 1, 0) - chi_x_arr(i, j, k - 1, 0)) * inv_2dx[2];

                    grad_chi_y[0] =
                        (chi_y_arr(i + 1, j, k, 0) - chi_y_arr(i - 1, j, k, 0)) * inv_2dx[0];
                    grad_chi_y[1] =
                        (chi_y_arr(i, j + 1, k, 0) - chi_y_arr(i, j - 1, k, 0)) * inv_2dx[1];
                    if (AMREX_SPACEDIM == 3)
                        grad_chi_y[2] =
                            (chi_y_arr(i, j, k + 1, 0) - chi_y_arr(i, j, k - 1, 0)) * inv_2dx[2];

                    if (AMREX_SPACEDIM == 3) {
                        grad_chi_z[0] =
                            (chi_z_arr(i + 1, j, k, 0) - chi_z_arr(i - 1, j, k, 0)) * inv_2dx[0];
                        grad_chi_z[1] =
                            (chi_z_arr(i, j + 1, k, 0) - chi_z_arr(i, j - 1, k, 0)) * inv_2dx[1];
                        grad_chi_z[2] =
                            (chi_z_arr(i, j, k + 1, 0) - chi_z_arr(i, j, k - 1, 0)) * inv_2dx[2];
                    }

                    sum_thread[0][0] += (1.0 - grad_chi_x[0]);
                    sum_thread[0][1] += (-grad_chi_y[0]);
                    sum_thread[1][0] += (-grad_chi_x[1]);
                    sum_thread[1][1] += (1.0 - grad_chi_y[1]);

                    if (AMREX_SPACEDIM == 3) {
                        sum_thread[0][2] += (-grad_chi_z[0]);
                        sum_thread[2][0] += (-grad_chi_x[2]);
                        sum_thread[1][2] += (-grad_chi_z[1]);
                        sum_thread[2][1] += (-grad_chi_y[2]);
                        sum_thread[2][2] += (1.0 - grad_chi_z[2]);
                    }
                }
            });
        }

#ifdef AMREX_USE_OMP
#pragma omp critical
#endif
        {
            for (int r = 0; r < AMREX_SPACEDIM; ++r) {
                for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                    sum_local[r][c] += sum_thread[r][c];
                }
            }
        }
    }

    // MPI reduction
    for (int r = 0; r < AMREX_SPACEDIM; ++r) {
        for (int c = 0; c < AMREX_SPACEDIM; ++c) {
            amrex::ParallelDescriptor::ReduceRealSum(sum_local[r][c]);
        }
    }

    // Normalize by total domain cell count
    amrex::Long N_total = geom.Domain().numPts();
    if (N_total > 0) {
        for (int r = 0; r < AMREX_SPACEDIM; ++r) {
            for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                Deff_tensor[r][c] = sum_local[r][c] / static_cast<amrex::Real>(N_total);
            }
        }
    } else {
        if (amrex::ParallelDescriptor::IOProcessor() && verbose > 0) {
            amrex::Warning("Total cells in domain is zero, D_eff cannot be calculated.");
        }
    }

    if (verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  [calculateDeffTensor] Raw summed (1-dchi_x_dx): " << sum_local[0][0]
                       << ", N_total: " << N_total << std::endl;
    }
}

} // namespace OpenImpala
