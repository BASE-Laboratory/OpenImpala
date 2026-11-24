/** @file PeriodicComponentPin.cpp
 *  @brief Implementation of periodic 6-connected component nullspace-pinning helper.
 */

#include "PeriodicComponentPin.H"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

namespace OpenImpala {

int buildPeriodicCellTypeMask(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm,
                              const amrex::iMultiFab& mf_active_mask,
                              amrex::iMultiFab& mf_cell_type, int mask_comp) {
    BL_PROFILE("OpenImpala::buildPeriodicCellTypeMask");

    const amrex::Box& domain = geom.Domain();
    const int dom_lo_x = domain.smallEnd(0);
    const int dom_lo_y = domain.smallEnd(1);
    const int dom_lo_z = domain.smallEnd(2);
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    const long total_pts_long = domain.numPts();

    mf_cell_type.setVal(CELL_INACTIVE);

#ifndef AMREX_USE_GPU
    // Fast single-rank O(N) periodic BFS on CPU: visits every active cell once and
    // marks the lexicographically smallest (k, j, i) cell of each periodic 6-connected
    // component as CELL_PINNED (chi = 0).
    if (amrex::ParallelDescriptor::NProcs() == 1) {
        const auto total_pts = static_cast<std::size_t>(total_pts_long);
        const auto snx = static_cast<std::size_t>(nx);
        const auto sny = static_cast<std::size_t>(ny);
        auto flat_idx = [=](int ix, int iy, int iz) noexcept -> std::size_t {
            return (static_cast<std::size_t>(iz) * sny + static_cast<std::size_t>(iy)) * snx +
                   static_cast<std::size_t>(ix);
        };

        // 0 = inactive, 1 = unvisited active, 2 = visited active, 3 = pinned active
        std::vector<std::uint8_t> state(total_pts, 0u);
        for (amrex::MFIter mfi(mf_active_mask); mfi.isValid(); ++mfi) {
            const amrex::Box& vbox = mfi.validbox();
            const auto mask_arr = mf_active_mask.const_array(mfi);
            amrex::LoopOnCpu(vbox, [&](int i, int j, int k) {
                if (mask_arr(i, j, k, mask_comp) == CELL_ACTIVE) {
                    state[flat_idx(i - dom_lo_x, j - dom_lo_y, k - dom_lo_z)] = 1u;
                }
            });
        }

        const bool per_x = geom.isPeriodic(0);
        const bool per_y = geom.isPeriodic(1);
        const bool per_z = geom.isPeriodic(2);

        struct Coord {
            int x;
            int y;
            int z;
        };
        std::vector<Coord> queue;
        queue.reserve(total_pts / 8 + 16);

        auto try_visit = [&](int nbx, int nby, int nbz) {
            const std::size_t nb_idx = flat_idx(nbx, nby, nbz);
            if (state[nb_idx] == 1u) {
                state[nb_idx] = 2u;
                queue.push_back({nbx, nby, nbz});
            }
        };

        int num_components = 0;
        for (int kz = 0; kz < nz; ++kz) {
            for (int jy = 0; jy < ny; ++jy) {
                for (int ix = 0; ix < nx; ++ix) {
                    const std::size_t seed_idx = flat_idx(ix, jy, kz);
                    if (state[seed_idx] != 1u) {
                        continue;
                    }
                    state[seed_idx] = 3u;
                    ++num_components;
                    queue.clear();
                    queue.push_back({ix, jy, kz});

                    std::size_t head = 0;
                    while (head < queue.size()) {
                        const Coord c = queue[head++];
                        if (c.x > 0) {
                            try_visit(c.x - 1, c.y, c.z);
                        } else if (per_x) {
                            try_visit(nx - 1, c.y, c.z);
                        }
                        if (c.x + 1 < nx) {
                            try_visit(c.x + 1, c.y, c.z);
                        } else if (per_x) {
                            try_visit(0, c.y, c.z);
                        }
                        if (c.y > 0) {
                            try_visit(c.x, c.y - 1, c.z);
                        } else if (per_y) {
                            try_visit(c.x, ny - 1, c.z);
                        }
                        if (c.y + 1 < ny) {
                            try_visit(c.x, c.y + 1, c.z);
                        } else if (per_y) {
                            try_visit(c.x, 0, c.z);
                        }
                        if (c.z > 0) {
                            try_visit(c.x, c.y, c.z - 1);
                        } else if (per_z) {
                            try_visit(c.x, c.y, nz - 1);
                        }
                        if (c.z + 1 < nz) {
                            try_visit(c.x, c.y, c.z + 1);
                        } else if (per_z) {
                            try_visit(c.x, c.y, 0);
                        }
                    }
                }
            }
        }

        for (amrex::MFIter mfi(mf_cell_type); mfi.isValid(); ++mfi) {
            const amrex::Box& vbox = mfi.validbox();
            auto ct_arr = mf_cell_type.array(mfi);
            amrex::LoopOnCpu(vbox, [&](int i, int j, int k) {
                const std::uint8_t s = state[flat_idx(i - dom_lo_x, j - dom_lo_y, k - dom_lo_z)];
                if (s == 3u) {
                    ct_arr(i, j, k, 0) = CELL_PINNED;
                } else if (s == 2u) {
                    ct_arr(i, j, k, 0) = CELL_ACTIVE;
                } else {
                    ct_arr(i, j, k, 0) = CELL_INACTIVE;
                }
            });
        }

        if (mf_cell_type.nGrow() > 0) {
            mf_cell_type.FillBoundary(geom.periodicity());
        }
        return num_components;
    }
#endif

    // General MPI + CUDA GPU path: double-buffered parallel min-label propagation.
    AMREX_ALWAYS_ASSERT(total_pts_long < static_cast<long>(std::numeric_limits<int>::max()));
    const int sentinel = std::numeric_limits<int>::max();

    amrex::iMultiFab mf_curr(ba, dm, 1, 1);
    amrex::iMultiFab mf_next(ba, dm, 1, 1);
    mf_curr.setVal(sentinel);
    mf_next.setVal(sentinel);

    for (amrex::MFIter mfi(mf_curr, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto curr_arr = mf_curr.array(mfi);
        const auto mask_arr = mf_active_mask.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (mask_arr(i, j, k, mask_comp) == CELL_ACTIVE) {
                curr_arr(i, j, k, 0) =
                    (i - dom_lo_x) + nx * ((j - dom_lo_y) + ny * (k - dom_lo_z)) + 1;
            } else {
                curr_arr(i, j, k, 0) = sentinel;
            }
        });
    }

    const int max_label_iter = static_cast<int>(std::min<long>(total_pts_long + 2L, 100000L));
    int iter = 0;
    bool changed_globally = true;

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceScalar<int> d_changed(0);
#else
    int h_changed = 0;
#endif

    while (changed_globally && iter < max_label_iter) {
        ++iter;
        mf_curr.FillBoundary(geom.periodicity());

        int* d_flag_ptr = nullptr;
#ifdef AMREX_USE_GPU
        int zero = 0;
        amrex::Gpu::htod_memcpy(d_changed.dataPtr(), &zero, sizeof(int));
        d_flag_ptr = d_changed.dataPtr();
#else
        h_changed = 0;
        d_flag_ptr = &h_changed;
#endif

        for (amrex::MFIter mfi(mf_curr, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto curr_arr = mf_curr.const_array(mfi);
            auto next_arr = mf_next.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int cur = curr_arr(i, j, k, 0);
                if (cur == sentinel) {
                    next_arr(i, j, k, 0) = sentinel;
                    return;
                }
                int best = cur;
                best = amrex::min(best, curr_arr(i - 1, j, k, 0));
                best = amrex::min(best, curr_arr(i + 1, j, k, 0));
                best = amrex::min(best, curr_arr(i, j - 1, k, 0));
                best = amrex::min(best, curr_arr(i, j + 1, k, 0));
                best = amrex::min(best, curr_arr(i, j, k - 1, 0));
                best = amrex::min(best, curr_arr(i, j, k + 1, 0));
                next_arr(i, j, k, 0) = best;
                if (best < cur) {
                    amrex::Gpu::Atomic::Max(d_flag_ptr, 1);
                }
            });
        }

#ifdef AMREX_USE_GPU
        changed_globally = (d_changed.dataValue() != 0);
#else
        changed_globally = (h_changed != 0);
#endif
        amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);
        std::swap(mf_curr, mf_next);
    }

    amrex::ReduceOps<amrex::ReduceOpSum> pin_count_op;
    amrex::ReduceData<long> pin_count_data(pin_count_op);
    for (amrex::MFIter mfi(mf_cell_type); mfi.isValid(); ++mfi) {
        const amrex::Box& vbox = mfi.validbox();
        auto ct_arr = mf_cell_type.array(mfi);
        const auto curr_arr = mf_curr.const_array(mfi);
        const auto mask_arr = mf_active_mask.const_array(mfi);
        pin_count_op.eval(
            vbox, pin_count_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<long> {
                if (mask_arr(i, j, k, mask_comp) != CELL_ACTIVE) {
                    ct_arr(i, j, k, 0) = CELL_INACTIVE;
                    return {0L};
                }
                const int self_lbl =
                    (i - dom_lo_x) + nx * ((j - dom_lo_y) + ny * (k - dom_lo_z)) + 1;
                if (curr_arr(i, j, k, 0) == self_lbl) {
                    ct_arr(i, j, k, 0) = CELL_PINNED;
                    return {1L};
                }
                ct_arr(i, j, k, 0) = CELL_ACTIVE;
                return {0L};
            });
    }

    long total_components = amrex::get<0>(pin_count_data.value());
    amrex::ParallelDescriptor::ReduceLongSum(total_components);

    if (mf_cell_type.nGrow() > 0) {
        mf_cell_type.FillBoundary(geom.periodicity());
    }
    return static_cast<int>(total_components);
}

} // namespace OpenImpala
