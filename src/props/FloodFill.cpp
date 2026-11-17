/** @file FloodFill.cpp
 *  @brief GPU-compatible parallel flood fill implementation.
 */
#include "FloodFill.H"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <mpi.h>

namespace OpenImpala {

// =========================================================================
// collectBoundarySeeds
// =========================================================================
void collectBoundarySeeds(const amrex::iMultiFab& phaseFab, int phaseID, int dir,
                          const amrex::Geometry& geom, amrex::Vector<amrex::IntVect>& inletSeeds,
                          amrex::Vector<amrex::IntVect>& outletSeeds) {
    const amrex::Box& domain = geom.Domain();

    amrex::Box domain_lo_face = domain;
    domain_lo_face.setBig(dir, domain.smallEnd(dir));
    amrex::Box domain_hi_face = domain;
    domain_hi_face.setSmall(dir, domain.bigEnd(dir));

    amrex::Vector<amrex::IntVect> local_inlet;
    amrex::Vector<amrex::IntVect> local_outlet;

    // Seed collection runs on CPU (it builds host-side IntVect vectors).
    // On a CUDA build, phaseFab.const_array(mfi) is a view into device-resident
    // memory — reading it from a host LoopOnCpu segfaults on T4 / A100. We
    // therefore snapshot phaseFab into a pinned-host iMultiFab once, and
    // iterate that copy on the CPU. On CPU builds the snapshot is a no-op
    // assignment (host data already accessible) so we just alias the input.
    amrex::Gpu::streamSynchronize();

#ifdef AMREX_USE_GPU
    amrex::iMultiFab phaseFab_host_storage(phaseFab.boxArray(), phaseFab.DistributionMap(),
                                           phaseFab.nComp(), phaseFab.nGrow(),
                                           amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::Copy(phaseFab_host_storage, phaseFab, 0, 0, phaseFab.nComp(), phaseFab.nGrow());
    amrex::Gpu::streamSynchronize();
    const amrex::iMultiFab& phaseFab_host = phaseFab_host_storage;
#else
    const amrex::iMultiFab& phaseFab_host = phaseFab;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phaseFab_host); mfi.isValid(); ++mfi) {
        const amrex::Box& validBox = mfi.validbox();
        const auto phase_arr = phaseFab_host.const_array(mfi);

        amrex::Box inlet_intersect = validBox & domain_lo_face;
        if (!inlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(inlet_intersect, [&](int i, int j, int k) {
                if (phase_arr(i, j, k, 0) == phaseID) {
#ifdef AMREX_USE_OMP
#pragma omp critical(flood_inlet_seed)
#endif
                    local_inlet.push_back(amrex::IntVect(i, j, k));
                }
            });
        }

        amrex::Box outlet_intersect = validBox & domain_hi_face;
        if (!outlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(outlet_intersect, [&](int i, int j, int k) {
                if (phase_arr(i, j, k, 0) == phaseID) {
#ifdef AMREX_USE_OMP
#pragma omp critical(flood_outlet_seed)
#endif
                    local_outlet.push_back(amrex::IntVect(i, j, k));
                }
            });
        }
    }

    // Gather seeds across MPI ranks
    MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
    int mpi_size = amrex::ParallelDescriptor::NProcs();

    auto gatherAndDeduplicate =
        [&](const amrex::Vector<amrex::IntVect>& local_seeds) -> amrex::Vector<amrex::IntVect> {
        const size_t n_local = local_seeds.size();
        std::vector<int> flat_local(n_local * AMREX_SPACEDIM);
        for (size_t i = 0; i < n_local; ++i) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                flat_local[i * AMREX_SPACEDIM + d] = local_seeds[i][d];
        }

        int local_count = static_cast<int>(flat_local.size());
        std::vector<int> recv_counts(mpi_size);
        MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

        std::vector<int> displacements(mpi_size, 0);
        int total = recv_counts[0];
        for (int i = 1; i < mpi_size; ++i) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
            total += recv_counts[i];
        }

        std::vector<int> flat_gathered(total);
        MPI_Allgatherv(flat_local.data(), local_count, MPI_INT, flat_gathered.data(),
                       recv_counts.data(), displacements.data(), MPI_INT, comm);

        amrex::Vector<amrex::IntVect> seeds;
        seeds.reserve(total / AMREX_SPACEDIM);
        for (size_t i = 0; i < flat_gathered.size(); i += AMREX_SPACEDIM) {
            seeds.emplace_back(flat_gathered[i], flat_gathered[i + 1], flat_gathered[i + 2]);
        }
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
        return seeds;
    };

    inletSeeds = gatherAndDeduplicate(local_inlet);
    outletSeeds = gatherAndDeduplicate(local_outlet);
}

// =========================================================================
// parallelFloodFill
// =========================================================================
void parallelFloodFill(amrex::iMultiFab& reachabilityMask, const amrex::iMultiFab& phaseFab,
                       int phaseID, const amrex::Vector<amrex::IntVect>& seedPoints,
                       const amrex::Geometry& geom, int verbose, int label) {
    BL_PROFILE("OpenImpala::parallelFloodFill");
    AMREX_ASSERT(reachabilityMask.nGrow() >= 1);
    AMREX_ASSERT(phaseFab.nGrow() >= 1);
    AMREX_ASSERT(label != FLOOD_INACTIVE); // label must differ from the "empty" marker

#ifndef AMREX_USE_GPU
    // Fast single-rank O(N) host BFS path for non-periodic CPU runs.
    // Replaces O(N * path_length) Jacobi wavefront sweeps + FillBoundary calls
    // with a single linear-time queue traversal while preserving multi-label
    // semantics used by ConnectedComponents.
    if (!geom.isAnyPeriodic() && amrex::ParallelDescriptor::NProcs() == 1) {
        const amrex::Box& domain = geom.Domain();
        if (seedPoints.empty()) {
            reachabilityMask.FillBoundary(geom.periodicity());
            return;
        }

        const amrex::IntVect offsets[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                           {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
        const std::size_t total_pts = static_cast<std::size_t>(domain.numPts());
        std::vector<amrex::IntVect> queue;
        queue.reserve(total_pts / 4 + 16);

        if (reachabilityMask.size() == 1 && reachabilityMask.box(0) == domain &&
            phaseFab.size() == 1 && phaseFab.box(0) == domain) {
            auto mask_arr = reachabilityMask.array(0);
            const auto phase_arr = phaseFab.const_array(0);

            for (const auto& seed : seedPoints) {
                if (domain.contains(seed) && phase_arr(seed, 0) == phaseID &&
                    mask_arr(seed, 0) != label) {
                    mask_arr(seed, 0) = label;
                    queue.push_back(seed);
                }
            }

            std::size_t head = 0;
            while (head < queue.size()) {
                const amrex::IntVect cur = queue[head++];
                for (int n = 0; n < 6; ++n) {
                    const amrex::IntVect nb = cur + offsets[n];
                    if (domain.contains(nb) && mask_arr(nb, 0) == FLOOD_INACTIVE &&
                        phase_arr(nb, 0) == phaseID) {
                        mask_arr(nb, 0) = label;
                        queue.push_back(nb);
                    }
                }
            }
        } else {
            const amrex::IntVect dom_lo = domain.smallEnd();
            const std::size_t nx = static_cast<std::size_t>(domain.length(0));
            const std::size_t ny = static_cast<std::size_t>(domain.length(1));
            auto flat_index = [&](int i, int j, int k) noexcept -> std::size_t {
                return (static_cast<std::size_t>(k - dom_lo[2]) * ny +
                        static_cast<std::size_t>(j - dom_lo[1])) *
                           nx +
                       static_cast<std::size_t>(i - dom_lo[0]);
            };

            constexpr std::uint8_t kTargetPhase = 1u;
            constexpr std::uint8_t kInactive = 2u;
            constexpr std::uint8_t kLabeled = 4u;
            constexpr std::uint8_t kEligible = kTargetPhase | kInactive;

            std::vector<std::uint8_t> state(total_pts, 0u);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(reachabilityMask, amrex::TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                const auto mask_arr = reachabilityMask.const_array(mfi);
                const auto phase_arr = phaseFab.const_array(mfi, 0);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    std::uint8_t flags = 0u;
                    if (phase_arr(i, j, k) == phaseID) {
                        flags |= kTargetPhase;
                    }
                    const int cur_lbl = mask_arr(i, j, k, 0);
                    if (cur_lbl == FLOOD_INACTIVE) {
                        flags |= kInactive;
                    } else if (cur_lbl == label) {
                        flags |= kLabeled;
                    }
                    state[flat_index(i, j, k)] = flags;
                });
            }

            for (const auto& seed : seedPoints) {
                if (!domain.contains(seed)) {
                    continue;
                }
                const std::size_t idx = flat_index(seed[0], seed[1], seed[2]);
                if ((state[idx] & kTargetPhase) != 0u && (state[idx] & kLabeled) == 0u) {
                    state[idx] = static_cast<std::uint8_t>((state[idx] & ~kInactive) | kLabeled);
                    queue.push_back(seed);
                }
            }

            std::size_t head = 0;
            while (head < queue.size()) {
                const amrex::IntVect cur = queue[head++];
                for (int n = 0; n < 6; ++n) {
                    const amrex::IntVect nb = cur + offsets[n];
                    if (!domain.contains(nb)) {
                        continue;
                    }
                    const std::size_t nb_idx = flat_index(nb[0], nb[1], nb[2]);
                    if ((state[nb_idx] & kEligible) == kEligible) {
                        state[nb_idx] = kTargetPhase | kLabeled;
                        queue.push_back(nb);
                    }
                }
            }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(reachabilityMask, amrex::TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto mask_arr = reachabilityMask.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    if ((state[flat_index(i, j, k)] & kLabeled) != 0u) {
                        mask_arr(i, j, k, 0) = label;
                    }
                });
            }
        }

        if (verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "    Flood fill (CPU BFS) labeled " << queue.size() << " cells.\n";
        }
        reachabilityMask.FillBoundary(geom.periodicity());
        return;
    }
#endif

    // --- Phase 1: Plant seeds ---
    // Seeds arrive as a small host-side list. The mask_arr writes below need
    // to run on the device that owns the iMultiFab data: on a CUDA build that
    // is the GPU, and writing to mask_arr(iv, 0) from a host loop would
    // segfault (T4 / A100 etc.). Filter seeds per tile on the host, stage
    // them in a device buffer, then plant them via ParallelFor.
    //
    // Note: we do NOT clear the mask here — callers that use multiple labels
    // (ConnectedComponents) may call this repeatedly on the same mask.
    // Callers doing a fresh fill should setVal(FLOOD_INACTIVE) beforehand.
    amrex::Gpu::streamSynchronize();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = reachabilityMask.array(mfi);
        const auto phase_arr = phaseFab.const_array(mfi, 0);

        amrex::Vector<amrex::IntVect> tile_seeds;
        for (const auto& seed : seedPoints) {
            if (tileBox.contains(seed)) {
                tile_seeds.push_back(seed);
            }
        }
        if (tile_seeds.empty()) {
            continue;
        }

        const int n_seeds = static_cast<int>(tile_seeds.size());
        const int pid = phaseID;
        const int lbl = label;

#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<amrex::IntVect> d_seeds(n_seeds);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, tile_seeds.data(),
                              tile_seeds.data() + n_seeds, d_seeds.data());
        amrex::Gpu::streamSynchronize();
        const amrex::IntVect* seed_ptr = d_seeds.data();
#else
        const amrex::IntVect* seed_ptr = tile_seeds.data();
#endif

        amrex::ParallelFor(n_seeds, [=] AMREX_GPU_DEVICE(int s) noexcept {
            const amrex::IntVect iv = seed_ptr[s];
            if (phase_arr(iv) == pid) {
                mask_arr(iv, 0) = lbl;
            }
        });
    }
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif

    // --- Phase 2: Iterative wavefront expansion ---
    // Each iteration expands the reachable set by 1 cell in each direction.
    // The inner loop is GPU-parallel via ParallelFor; inter-patch communication
    // uses FillBoundary (MPI ghost exchange).
    //
    // Early termination: a device-compatible atomic flag tracks whether any
    // cell changed.  On GPU this avoids reading back the full mask each
    // iteration; on CPU it reduces to a simple int with atomic store.

    amrex::IntVect domain_size = geom.Domain().size();
    const int max_flood_iter = domain_size[0] + domain_size[1] + domain_size[2] + 2;
    int iter = 0;
    bool changed_globally = true;

    // Device-accessible flag: 0 = no change, 1 = something changed
    int h_changed = 0;
    int* d_flag_ptr = nullptr;
#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceScalar<int> d_changed(0);
    d_flag_ptr = d_changed.dataPtr();
#else
    d_flag_ptr = &h_changed;
#endif

    while (changed_globally && iter < max_flood_iter) {
        ++iter;
        reachabilityMask.FillBoundary(geom.periodicity());

        // Reset device flag
#ifdef AMREX_USE_GPU
        {
            int zero = 0;
            amrex::Gpu::htod_memcpy(d_changed.dataPtr(), &zero, sizeof(int));
        }
        d_flag_ptr = d_changed.dataPtr();
#else
        h_changed = 0;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(reachabilityMask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tileBox = mfi.tilebox();
            auto mask_arr = reachabilityMask.array(mfi);
            const auto phase_arr = phaseFab.const_array(mfi, 0);
            // The grown box defines valid memory for neighbor access
            const amrex::Box grownBox = amrex::grow(mfi.validbox(), reachabilityMask.nGrow());
            const int pid = phaseID;
            const int lbl = label;

            amrex::ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Skip already-labeled or wrong-phase cells
                if (mask_arr(i, j, k, 0) != FLOOD_INACTIVE || phase_arr(i, j, k) != pid) {
                    return;
                }
                // Check 6-connected neighbors for matching label
                const amrex::IntVect offsets[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                                   {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
                amrex::IntVect iv(i, j, k);
                for (int n = 0; n < 6; ++n) {
                    amrex::IntVect nb = iv + offsets[n];
                    if (grownBox.contains(nb)) {
                        if (mask_arr(nb, 0) == lbl) {
                            mask_arr(i, j, k, 0) = lbl;
                            amrex::Gpu::Atomic::Max(d_flag_ptr, 1);
                            return;
                        }
                    }
                }
            });
        }

        // Read back the device flag
#ifdef AMREX_USE_GPU
        changed_globally = (d_changed.dataValue() != 0);
#else
        changed_globally = (h_changed != 0);
#endif
        amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);
    }

    if (iter >= max_flood_iter && changed_globally) {
        amrex::Warning("OpenImpala::parallelFloodFill reached max iterations.");
    }
    if (verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "    Flood fill completed in " << iter << " iterations.\n";
    }
    reachabilityMask.FillBoundary(geom.periodicity());
}

} // namespace OpenImpala
