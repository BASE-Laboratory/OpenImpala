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
                         const amrex::Geometry& geom, int verbose,
                         const amrex::MultiFab* mf_diff_coeff) {
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

    const amrex::BoxArray& ba = active_mask.boxArray();
    const amrex::DistributionMapping& dm = active_mask.DistributionMap();

    // Populate 1-ghost periodic copies of active_mask and diff_coeff so +x/+y/+z
    // harmonic face conductances are exact across tile and periodic boundaries.
    amrex::iMultiFab mask_1g(ba, dm, 1, 1);
    mask_1g.setVal(0);
    amrex::Copy(mask_1g, active_mask, 0, 0, 1, 0);
    mask_1g.FillBoundary(geom.periodicity());

    const bool has_dc = (mf_diff_coeff != nullptr && mf_diff_coeff->isDefined());
    amrex::MultiFab dc_1g(ba, dm, 1, 1);
    dc_1g.setVal(0.0);
    if (has_dc) {
        amrex::Copy(dc_1g, *mf_diff_coeff, 0, 0, 1, 0);
    } else {
        for (amrex::MFIter mfi(dc_1g, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tbx = mfi.tilebox();
            amrex::Array4<amrex::Real> const dc_arr = dc_1g.array(mfi);
            amrex::Array4<const int> const m_arr = mask_1g.const_array(mfi);
            amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                dc_arr(i, j, k, 0) = (m_arr(i, j, k, 0) == 1) ? 1.0 : 0.0;
            });
        }
    }
    dc_1g.FillBoundary(geom.periodicity());

    const amrex::Real* dx_arr = geom.CellSize();
    const amrex::Real inv_dx_0 = 1.0 / dx_arr[0];
    const amrex::Real inv_dx_1 = 1.0 / dx_arr[1];
    const amrex::Real inv_dx_2 = (AMREX_SPACEDIM == 3) ? 1.0 / dx_arr[2] : 0.0;

    // GPU-compatible 9-component reduction for the full D_eff tensor
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum>
        reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                      amrex::Real, amrex::Real, amrex::Real>
        reduce_data(reduce_op);

    for (amrex::MFIter mfi(mask_1g); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<const int> const mask_arr = mask_1g.const_array(mfi);
        amrex::Array4<const amrex::Real> const dc_arr = dc_1g.const_array(mfi);
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

                const amrex::Real D_c = dc_arr(i, j, k, 0);
                if (D_c <= 0.0) {
                    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                }

                amrex::Real D_px = 0.0;
                if (mask_arr(i + 1, j, k, 0) == 1) {
                    const amrex::Real D_nbr = dc_arr(i + 1, j, k, 0);
                    if (D_c + D_nbr > 0.0) {
                        D_px = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    }
                }

                amrex::Real D_py = 0.0;
                if (mask_arr(i, j + 1, k, 0) == 1) {
                    const amrex::Real D_nbr = dc_arr(i, j + 1, k, 0);
                    if (D_c + D_nbr > 0.0) {
                        D_py = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                    }
                }

                const amrex::Real dchix_dx =
                    (chi_x_arr(i + 1, j, k, 0) - chi_x_arr(i, j, k, 0)) * inv_dx_0;
                const amrex::Real dchiy_dx =
                    (chi_y_arr(i + 1, j, k, 0) - chi_y_arr(i, j, k, 0)) * inv_dx_0;
                const amrex::Real dchix_dy =
                    (chi_x_arr(i, j + 1, k, 0) - chi_x_arr(i, j, k, 0)) * inv_dx_1;
                const amrex::Real dchiy_dy =
                    (chi_y_arr(i, j + 1, k, 0) - chi_y_arr(i, j, k, 0)) * inv_dx_1;

                amrex::Real D_pz = 0.0;
                amrex::Real dchiz_dx = 0.0, dchiz_dy = 0.0;
                amrex::Real dchix_dz = 0.0, dchiy_dz = 0.0, dchiz_dz = 0.0;
                if (AMREX_SPACEDIM == 3) {
                    if (mask_arr(i, j, k + 1, 0) == 1) {
                        const amrex::Real D_nbr = dc_arr(i, j, k + 1, 0);
                        if (D_c + D_nbr > 0.0) {
                            D_pz = 2.0 * D_c * D_nbr / (D_c + D_nbr);
                        }
                    }
                    dchiz_dx = (chi_z_arr(i + 1, j, k, 0) - chi_z_arr(i, j, k, 0)) * inv_dx_0;
                    dchiz_dy = (chi_z_arr(i, j + 1, k, 0) - chi_z_arr(i, j, k, 0)) * inv_dx_1;
                    dchix_dz = (chi_x_arr(i, j, k + 1, 0) - chi_x_arr(i, j, k, 0)) * inv_dx_2;
                    dchiy_dz = (chi_y_arr(i, j, k + 1, 0) - chi_y_arr(i, j, k, 0)) * inv_dx_2;
                    dchiz_dz = (chi_z_arr(i, j, k + 1, 0) - chi_z_arr(i, j, k, 0)) * inv_dx_2;
                }

                // Conservative face flux: D_{i+1/2}^{(r)} * (delta_rc + (chi_c(i+e_r) - chi_c(i)) /
                // h_r)
                return {D_px * (1.0 + dchix_dx), D_px * dchiy_dx,         D_px * dchiz_dx,
                        D_py * dchix_dy,         D_py * (1.0 + dchiy_dy), D_py * dchiz_dy,
                        D_pz * dchix_dz,         D_pz * dchiy_dz,         D_pz * (1.0 + dchiz_dz)};
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
        amrex::Print() << "  [calculateDeffTensor] Raw summed flux_xx: " << sum_local[0][0]
                       << ", N_total: " << N_total << std::endl;
    }
}

} // namespace OpenImpala
