/** @file DeffTensor.cpp
 *  @brief Implementation of the effective diffusivity tensor assembly.
 */

#include "DeffTensor.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

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

    AMREX_ALWAYS_ASSERT(mf_chi_x.nGrow() >= 1);
    AMREX_ALWAYS_ASSERT(mf_chi_y.nGrow() >= 1);
    if (AMREX_SPACEDIM == 3) {
        AMREX_ALWAYS_ASSERT(mf_chi_z.isDefined() && mf_chi_z.nGrow() >= 1);
    }
    AMREX_ALWAYS_ASSERT(active_mask.nGrow() == 0);

    const amrex::Real* dx_arr = geom.CellSize();
    amrex::Real inv_2dx_0 = 1.0 / (2.0 * dx_arr[0]);
    amrex::Real inv_2dx_1 = 1.0 / (2.0 * dx_arr[1]);
    amrex::Real inv_2dx_2 = (AMREX_SPACEDIM == 3) ? 1.0 / (2.0 * dx_arr[2]) : 0.0;

    // GPU-compatible 9-component reduction for the full D_eff tensor
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum>
        reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real, amrex::Real>
        reduce_data(reduce_op);

    for (amrex::MFIter mfi(active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<const int> const mask_arr = active_mask.const_array(mfi);
        amrex::Array4<const amrex::Real> const chi_x_arr = mf_chi_x.const_array(mfi);
        amrex::Array4<const amrex::Real> const chi_y_arr = mf_chi_y.const_array(mfi);
        amrex::Array4<const amrex::Real> const chi_z_arr =
            (AMREX_SPACEDIM == 3 && mf_chi_z.isDefined()) ? mf_chi_z.const_array(mfi)
                                                          : mf_chi_x.const_array(mfi);

        reduce_op.eval(
            bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                               amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
                if (mask_arr(i, j, k, 0) != 1) {
                    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                }

                amrex::Real dchix_dx =
                    (chi_x_arr(i + 1, j, k, 0) - chi_x_arr(i - 1, j, k, 0)) * inv_2dx_0;
                amrex::Real dchix_dy =
                    (chi_x_arr(i, j + 1, k, 0) - chi_x_arr(i, j - 1, k, 0)) * inv_2dx_1;
                amrex::Real dchiy_dx =
                    (chi_y_arr(i + 1, j, k, 0) - chi_y_arr(i - 1, j, k, 0)) * inv_2dx_0;
                amrex::Real dchiy_dy =
                    (chi_y_arr(i, j + 1, k, 0) - chi_y_arr(i, j - 1, k, 0)) * inv_2dx_1;

                amrex::Real dchix_dz = 0.0, dchiy_dz = 0.0;
                amrex::Real dchiz_dx = 0.0, dchiz_dy = 0.0, dchiz_dz = 0.0;
                if (AMREX_SPACEDIM == 3) {
                    dchix_dz = (chi_x_arr(i, j, k + 1, 0) - chi_x_arr(i, j, k - 1, 0)) * inv_2dx_2;
                    dchiy_dz = (chi_y_arr(i, j, k + 1, 0) - chi_y_arr(i, j, k - 1, 0)) * inv_2dx_2;
                    dchiz_dx = (chi_z_arr(i + 1, j, k, 0) - chi_z_arr(i - 1, j, k, 0)) * inv_2dx_0;
                    dchiz_dy = (chi_z_arr(i, j + 1, k, 0) - chi_z_arr(i, j - 1, k, 0)) * inv_2dx_1;
                    dchiz_dz = (chi_z_arr(i, j, k + 1, 0) - chi_z_arr(i, j, k - 1, 0)) * inv_2dx_2;
                }

                // D_eff[r][c] = delta_rc + d(chi_c)/d(x_r), summed over active cells
                // Cell problem: div(D grad chi_k) = -div(D e_k) => plus sign
                return {1.0 + dchix_dx, dchiy_dx, dchiz_dx, dchix_dy,      1.0 + dchiy_dy,
                        dchiz_dy,       dchix_dz, dchiy_dz, 1.0 + dchiz_dz};
            });
    }

    auto hv = reduce_data.value();
    amrex::Real sum_local[AMREX_SPACEDIM][AMREX_SPACEDIM];
    sum_local[0][0] = amrex::get<0>(hv);
    sum_local[0][1] = amrex::get<1>(hv);
    sum_local[1][0] = amrex::get<3>(hv);
    sum_local[1][1] = amrex::get<4>(hv);
    if (AMREX_SPACEDIM == 3) {
        sum_local[0][2] = amrex::get<2>(hv);
        sum_local[2][0] = amrex::get<6>(hv);
        sum_local[1][2] = amrex::get<5>(hv);
        sum_local[2][1] = amrex::get<7>(hv);
        sum_local[2][2] = amrex::get<8>(hv);
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
        amrex::Print() << "  [calculateDeffTensor] Raw summed (1+dchi_x_dx): " << sum_local[0][0]
                       << ", N_total: " << N_total << std::endl;
    }
}

} // namespace OpenImpala
