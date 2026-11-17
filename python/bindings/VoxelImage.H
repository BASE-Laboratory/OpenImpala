/** @file VoxelImage.H
 *  @brief Opaque container for AMReX grid objects and field-export helpers for pybind11.
 *
 *  Encapsulates the Geometry, BoxArray, DistributionMapping, and iMultiFab
 *  so that Python users never need to interact with AMReX types directly.
 *  This eliminates the pyamrex dependency entirely.
 */
#ifndef OPENIMPALA_VOXEL_IMAGE_H
#define OPENIMPALA_VOXEL_IMAGE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_iMultiFab.H>

#ifdef AMREX_USE_GPU
#include <AMReX_Gpu.H>
#endif

namespace OpenImpala {

/** @brief Opaque handle that bundles AMReX grid infrastructure for a 3-D voxel image.
 *
 *  Created from a NumPy array (via pybind11) or from a native C++ reader.
 *  Passed by shared_ptr so pybind11 can safely prevent premature destruction.
 */
struct VoxelImage {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    std::shared_ptr<amrex::iMultiFab> mf;
};

/** @brief Gather a 3-D AMReX MultiFab component into a C-contiguous (nz, ny, nx) float64 NumPy
 * array.
 *
 *  Safe across CPU, OpenMP, CUDA device memory, and multi-rank MPI decompositions.
 */
inline pybind11::array_t<double> multifab_to_numpy(const amrex::MultiFab& mf,
                                                   const amrex::Geometry& geom, int comp = 0) {
    const amrex::Box& domain = geom.Domain();
    const int lo_x = domain.smallEnd(0);
    const int lo_y = domain.smallEnd(1);
    const int lo_z = domain.smallEnd(2);
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    const std::size_t total =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    pybind11::array_t<double> out({static_cast<pybind11::ssize_t>(nz),
                                   static_cast<pybind11::ssize_t>(ny),
                                   static_cast<pybind11::ssize_t>(nx)});
    auto* host_buf = static_cast<double*>(out.request().ptr);
    if (total == 0) {
        return out;
    }
    std::memset(host_buf, 0, total * sizeof(double));

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.validbox() & domain;
        if (!bx.ok()) {
            continue;
        }
        const auto arr = mf.const_array(mfi, comp);
        const auto lo = amrex::lbound(bx);
        const int bx_nx = bx.length(0);
        const int bx_ny = bx.length(1);
        const int bx_nz = bx.length(2);
        const std::size_t bx_pts = static_cast<std::size_t>(bx.numPts());

#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<double> d_tmp(bx_pts);
        double* dst_ptr = d_tmp.data();
#else
        std::vector<double> h_tmp(bx_pts);
        double* dst_ptr = h_tmp.data();
#endif

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const std::size_t lin =
                (static_cast<std::size_t>(k - lo.z) * static_cast<std::size_t>(bx_ny) +
                 static_cast<std::size_t>(j - lo.y)) *
                    static_cast<std::size_t>(bx_nx) +
                static_cast<std::size_t>(i - lo.x);
            dst_ptr[lin] = static_cast<double>(arr(i, j, k));
        });

#ifdef AMREX_USE_GPU
        std::vector<double> h_tmp(bx_pts);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_tmp.begin(), d_tmp.end(), h_tmp.begin());
#endif

        for (int k = 0; k < bx_nz; ++k) {
            for (int j = 0; j < bx_ny; ++j) {
                const std::size_t src_row =
                    (static_cast<std::size_t>(k) * static_cast<std::size_t>(bx_ny) +
                     static_cast<std::size_t>(j)) *
                    static_cast<std::size_t>(bx_nx);
                const std::size_t dst_row =
                    (static_cast<std::size_t>(lo.z + k - lo_z) * static_cast<std::size_t>(ny) +
                     static_cast<std::size_t>(lo.y + j - lo_y)) *
                        static_cast<std::size_t>(nx) +
                    static_cast<std::size_t>(lo.x - lo_x);
                std::memcpy(&host_buf[dst_row], &h_tmp[src_row],
                            static_cast<std::size_t>(bx_nx) * sizeof(double));
            }
        }
    }

    if (amrex::ParallelDescriptor::NProcs() > 1) {
        constexpr std::size_t max_chunk = 1ULL << 29;
        std::size_t offset = 0;
        while (offset < total) {
            const std::size_t chunk = std::min(max_chunk, total - offset);
            amrex::ParallelAllReduce::Sum(host_buf + offset, static_cast<int>(chunk),
                                          amrex::ParallelContext::CommunicatorSub());
            offset += chunk;
        }
    }

    return out;
}

/** @brief Gather a 3-D AMReX iMultiFab component into a C-contiguous (nz, ny, nx) int32 NumPy
 * array.
 *
 *  Safe across CPU, OpenMP, CUDA device memory, and multi-rank MPI decompositions.
 */
inline pybind11::array_t<std::int32_t>
imultifab_to_numpy(const amrex::iMultiFab& mf, const amrex::Geometry& geom, int comp = 0) {
    const amrex::Box& domain = geom.Domain();
    const int lo_x = domain.smallEnd(0);
    const int lo_y = domain.smallEnd(1);
    const int lo_z = domain.smallEnd(2);
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    const std::size_t total =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    pybind11::array_t<std::int32_t> out({static_cast<pybind11::ssize_t>(nz),
                                         static_cast<pybind11::ssize_t>(ny),
                                         static_cast<pybind11::ssize_t>(nx)});
    auto* host_buf = static_cast<std::int32_t*>(out.request().ptr);
    if (total == 0) {
        return out;
    }
    std::memset(host_buf, 0, total * sizeof(std::int32_t));

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.validbox() & domain;
        if (!bx.ok()) {
            continue;
        }
        const auto arr = mf.const_array(mfi, comp);
        const auto lo = amrex::lbound(bx);
        const int bx_nx = bx.length(0);
        const int bx_ny = bx.length(1);
        const int bx_nz = bx.length(2);
        const std::size_t bx_pts = static_cast<std::size_t>(bx.numPts());

#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<std::int32_t> d_tmp(bx_pts);
        std::int32_t* dst_ptr = d_tmp.data();
#else
        std::vector<std::int32_t> h_tmp(bx_pts);
        std::int32_t* dst_ptr = h_tmp.data();
#endif

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const std::size_t lin =
                (static_cast<std::size_t>(k - lo.z) * static_cast<std::size_t>(bx_ny) +
                 static_cast<std::size_t>(j - lo.y)) *
                    static_cast<std::size_t>(bx_nx) +
                static_cast<std::size_t>(i - lo.x);
            dst_ptr[lin] = static_cast<std::int32_t>(arr(i, j, k));
        });

#ifdef AMREX_USE_GPU
        std::vector<std::int32_t> h_tmp(bx_pts);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_tmp.begin(), d_tmp.end(), h_tmp.begin());
#endif

        for (int k = 0; k < bx_nz; ++k) {
            for (int j = 0; j < bx_ny; ++j) {
                const std::size_t src_row =
                    (static_cast<std::size_t>(k) * static_cast<std::size_t>(bx_ny) +
                     static_cast<std::size_t>(j)) *
                    static_cast<std::size_t>(bx_nx);
                const std::size_t dst_row =
                    (static_cast<std::size_t>(lo.z + k - lo_z) * static_cast<std::size_t>(ny) +
                     static_cast<std::size_t>(lo.y + j - lo_y)) *
                        static_cast<std::size_t>(nx) +
                    static_cast<std::size_t>(lo.x - lo_x);
                std::memcpy(&host_buf[dst_row], &h_tmp[src_row],
                            static_cast<std::size_t>(bx_nx) * sizeof(std::int32_t));
            }
        }
    }

    if (amrex::ParallelDescriptor::NProcs() > 1) {
        constexpr std::size_t max_chunk = 1ULL << 29;
        std::size_t offset = 0;
        while (offset < total) {
            const std::size_t chunk = std::min(max_chunk, total - offset);
            amrex::ParallelAllReduce::Sum(host_buf + offset, static_cast<int>(chunk),
                                          amrex::ParallelContext::CommunicatorSub());
            offset += chunk;
        }
    }

    return out;
}

} // namespace OpenImpala

#endif // OPENIMPALA_VOXEL_IMAGE_H
