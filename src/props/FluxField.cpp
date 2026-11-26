/** @file FluxField.cpp
 *  @brief Implementation of cell-centred 3-D diffusive flux field and statistics.
 */

#include "FluxField.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace OpenImpala {

void computeTortuosityCellFluxField(amrex::MultiFab& mf_flux, const amrex::Geometry& geom,
                                    const amrex::MultiFab& mf_soln,
                                    const amrex::iMultiFab& mf_active_mask,
                                    const amrex::MultiFab& mf_diff_coeff,
                                    OpenImpala::Direction flow_dir, amrex::Real vlo,
                                    amrex::Real vhi, int soln_comp) {
    BL_PROFILE("OpenImpala::computeTortuosityCellFluxField");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf_soln.nGrow() >= 1,
                                     "computeTortuosityCellFluxField: mf_soln requires >=1 ghost");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mf_active_mask.nGrow() >= 1,
        "computeTortuosityCellFluxField: mf_active_mask requires >=1 ghost");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mf_diff_coeff.nGrow() >= 1,
        "computeTortuosityCellFluxField: mf_diff_coeff requires >=1 ghost");

    if (!mf_flux.ok() || mf_flux.nComp() < 4 || mf_flux.boxArray() != mf_soln.boxArray() ||
        mf_flux.DistributionMap() != mf_soln.DistributionMap()) {
        mf_flux.define(mf_soln.boxArray(), mf_soln.DistributionMap(), 4, 0);
    }
    mf_flux.setVal(0.0);

    const amrex::Box& domain = geom.Domain();
    const amrex::GpuArray<int, AMREX_SPACEDIM> dom_lo{
        AMREX_D_DECL(domain.smallEnd(0), domain.smallEnd(1), domain.smallEnd(2))};
    const amrex::GpuArray<int, AMREX_SPACEDIM> dom_hi{
        AMREX_D_DECL(domain.bigEnd(0), domain.bigEnd(1), domain.bigEnd(2))};
    const amrex::Real* dx_raw = geom.CellSize();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{
        AMREX_D_DECL(dx_raw[0], dx_raw[1], dx_raw[2])};
    const int idir = static_cast<int>(flow_dir);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf_flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto flux_arr = mf_flux.array(mfi);
        const auto soln_arr = mf_soln.const_array(mfi, soln_comp);
        const auto mask_arr = mf_active_mask.const_array(mfi, 0);
        const auto dc_arr = mf_diff_coeff.const_array(mfi, 0);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (mask_arr(i, j, k) != 1) {
                flux_arr(i, j, k, 0) = 0.0;
                flux_arr(i, j, k, 1) = 0.0;
                flux_arr(i, j, k, 2) = 0.0;
                flux_arr(i, j, k, 3) = 0.0;
                return;
            }

            const amrex::IntVect iv(i, j, k);
            const amrex::Real d_c = dc_arr(iv);
            const amrex::Real phi_c = soln_arr(iv);
            amrex::Real j_vec[3] = {0.0, 0.0, 0.0};

            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                const amrex::IntVect shift = amrex::IntVect::TheDimensionVector(d);
                const amrex::Real h_d = dx[d];

                amrex::Real j_minus = 0.0;
                if (iv[d] == dom_lo[d]) {
                    if (d == idir) {
                        j_minus = -2.0 * d_c * (phi_c - vlo) / h_d;
                    }
                } else {
                    const amrex::IntVect lo_iv = iv - shift;
                    if (mask_arr(lo_iv) == 1) {
                        const amrex::Real d_lo = dc_arr(lo_iv);
                        const amrex::Real d_face =
                            (d_lo + d_c > 0.0) ? (2.0 * d_lo * d_c / (d_lo + d_c)) : 0.0;
                        j_minus = -d_face * (phi_c - soln_arr(lo_iv)) / h_d;
                    }
                }

                amrex::Real j_plus = 0.0;
                if (iv[d] == dom_hi[d]) {
                    if (d == idir) {
                        j_plus = -2.0 * d_c * (vhi - phi_c) / h_d;
                    }
                } else {
                    const amrex::IntVect hi_iv = iv + shift;
                    if (mask_arr(hi_iv) == 1) {
                        const amrex::Real d_hi = dc_arr(hi_iv);
                        const amrex::Real d_face =
                            (d_c + d_hi > 0.0) ? (2.0 * d_c * d_hi / (d_c + d_hi)) : 0.0;
                        j_plus = -d_face * (soln_arr(hi_iv) - phi_c) / h_d;
                    }
                }

                j_vec[d] = 0.5 * (j_minus + j_plus);
            }

            flux_arr(i, j, k, 0) = j_vec[0];
            flux_arr(i, j, k, 1) = j_vec[1];
            flux_arr(i, j, k, 2) = j_vec[2];
            flux_arr(i, j, k, 3) =
                std::sqrt(j_vec[0] * j_vec[0] + j_vec[1] * j_vec[1] + j_vec[2] * j_vec[2]);
        });
    }
}

FluxStats computeTortuosityFluxStats(const amrex::Geometry& geom, const amrex::MultiFab& mf_soln,
                                     const amrex::iMultiFab& mf_active_mask,
                                     const amrex::MultiFab& mf_diff_coeff,
                                     OpenImpala::Direction flow_dir, amrex::Real vlo,
                                     amrex::Real vhi, int num_bins, int soln_comp) {
    BL_PROFILE("OpenImpala::computeTortuosityFluxStats");

    const int bins = std::max(1, num_bins);
    amrex::MultiFab mf_flux;
    computeTortuosityCellFluxField(mf_flux, geom, mf_soln, mf_active_mask, mf_diff_coeff, flow_dir,
                                   vlo, vhi, soln_comp);

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpMax, amrex::ReduceOpSum>
        reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, long> reduce_data(reduce_op);

    for (amrex::MFIter mfi(mf_flux); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto flux_arr = mf_flux.const_array(mfi, 3);
        const auto mask_arr = mf_active_mask.const_array(mfi, 0);
        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                       -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, long> {
                           if (mask_arr(i, j, k) == 1) {
                               const amrex::Real mag = flux_arr(i, j, k);
                               return {mag, mag * mag, mag, 1L};
                           }
                           return {0.0, 0.0, 0.0, 0L};
                       });
    }

    auto rv = reduce_data.value();
    amrex::Real sum_mag = amrex::get<0>(rv);
    amrex::Real sum_sq = amrex::get<1>(rv);
    amrex::Real max_mag = amrex::get<2>(rv);
    long active_count = amrex::get<3>(rv);

    amrex::ParallelDescriptor::ReduceRealSum(sum_mag);
    amrex::ParallelDescriptor::ReduceRealSum(sum_sq);
    amrex::ParallelDescriptor::ReduceRealMax(max_mag);
    amrex::ParallelDescriptor::ReduceLongSum(active_count);

    FluxStats stats;
    stats.histogram_edges.resize(static_cast<std::size_t>(bins) + 1U, 0.0);
    stats.histogram_counts.assign(static_cast<std::size_t>(bins), 0LL);

    if (active_count <= 0) {
        return stats;
    }

    const amrex::Real inv_n = 1.0 / static_cast<amrex::Real>(active_count);
    stats.mean_magnitude = sum_mag * inv_n;
    stats.max_magnitude = max_mag;
    const amrex::Real var =
        std::max<amrex::Real>(0.0, sum_sq * inv_n - stats.mean_magnitude * stats.mean_magnitude);
    stats.std_magnitude = std::sqrt(var);

    const amrex::Real upper = (max_mag > 0.0) ? max_mag : 1.0;
    for (int b = 0; b <= bins; ++b) {
        stats.histogram_edges[static_cast<std::size_t>(b)] =
            upper * static_cast<amrex::Real>(b) / static_cast<amrex::Real>(bins);
    }

    std::vector<long long> local_counts(static_cast<std::size_t>(bins), 0LL);
    for (amrex::MFIter mfi(mf_flux); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const std::size_t npts = static_cast<std::size_t>(bx.numPts());
        if (npts == 0) {
            continue;
        }
        const auto flux_arr = mf_flux.const_array(mfi, 3);
        const auto mask_arr = mf_active_mask.const_array(mfi, 0);
        const auto lo = amrex::lbound(bx);
        const int bx_nx = bx.length(0);
        const int bx_ny = bx.length(1);

#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<amrex::Real> d_mag(npts, -1.0);
        amrex::Real* mag_ptr = d_mag.data();
#else
        std::vector<amrex::Real> h_mag(npts, -1.0);
        amrex::Real* mag_ptr = h_mag.data();
#endif
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const std::size_t lin =
                (static_cast<std::size_t>(k - lo.z) * static_cast<std::size_t>(bx_ny) +
                 static_cast<std::size_t>(j - lo.y)) *
                    static_cast<std::size_t>(bx_nx) +
                static_cast<std::size_t>(i - lo.x);
            mag_ptr[lin] = (mask_arr(i, j, k) == 1) ? flux_arr(i, j, k) : -1.0;
        });
#ifdef AMREX_USE_GPU
        std::vector<amrex::Real> h_mag(npts);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_mag.begin(), d_mag.end(), h_mag.begin());
#endif
        for (std::size_t idx = 0; idx < npts; ++idx) {
            const amrex::Real mag = h_mag[idx];
            if (mag >= 0.0) {
                int bin = (max_mag > 0.0) ? static_cast<int>(std::floor(
                                                (mag / max_mag) * static_cast<amrex::Real>(bins)))
                                          : 0;
                bin = std::clamp(bin, 0, bins - 1);
                local_counts[static_cast<std::size_t>(bin)] += 1LL;
            }
        }
    }

    for (int b = 0; b < bins; ++b) {
        long c = static_cast<long>(local_counts[static_cast<std::size_t>(b)]);
        amrex::ParallelDescriptor::ReduceLongSum(c);
        stats.histogram_counts[static_cast<std::size_t>(b)] = static_cast<long long>(c);
    }

    return stats;
}

} // namespace OpenImpala
