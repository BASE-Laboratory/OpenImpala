/** @file FloodFill.cpp
 *  @brief GPU-compatible parallel flood fill implementation.
 */
#include "FloodFill.H"

#include <algorithm>
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
                           const amrex::Geometry& geom,
                           amrex::Vector<amrex::IntVect>& inletSeeds,
                           amrex::Vector<amrex::IntVect>& outletSeeds) {
    const amrex::Box& domain = geom.Domain();

    amrex::Box domain_lo_face = domain;
    domain_lo_face.setBig(dir, domain.smallEnd(dir));
    amrex::Box domain_hi_face = domain;
    domain_hi_face.setSmall(dir, domain.bigEnd(dir));

    amrex::Vector<amrex::IntVect> local_inlet;
    amrex::Vector<amrex::IntVect> local_outlet;

    // Seed collection must run on CPU (building host-side IntVect vectors).
    // Synchronize first to ensure phase data is available on host.
    amrex::Gpu::streamSynchronize();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phaseFab); mfi.isValid(); ++mfi) {
        const amrex::Box& validBox = mfi.validbox();
        const auto phase_arr = phaseFab.const_array(mfi);

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

    // --- Phase 1: Plant seeds ---
    // Seeds are a small host-side list; planting is cheap either way.
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
        for (const auto& seed : seedPoints) {
            if (tileBox.contains(seed)) {
                if (phase_arr(seed) == phaseID) {
                    mask_arr(seed, 0) = label;
                }
            }
        }
    }

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
    amrex::Gpu::DeviceScalar<int> d_changed(0);

    while (changed_globally && iter < max_flood_iter) {
        ++iter;
        reachabilityMask.FillBoundary(geom.periodicity());

        // Reset device flag
        d_changed.setVal(0);
        int* d_flag_ptr = d_changed.dataPtr();

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
                const amrex::IntVect offsets[6] = {
                    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
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
        changed_globally = (d_changed.dataValue() != 0);
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
