#include "PercolationCheck.H"

#include <algorithm>
#include <vector>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_Box.H>

#include <mpi.h>

namespace {
constexpr int cell_inactive = 0;
constexpr int cell_active = 1;
} // namespace

namespace OpenImpala {

PercolationCheck::PercolationCheck(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                   const amrex::DistributionMapping& dm,
                                   const amrex::iMultiFab& mf_phase, int phase_id,
                                   OpenImpala::Direction dir, int verbose)
    : m_geom(geom), m_ba(ba), m_dm(dm), m_verbose(verbose) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf_phase.nGrow() >= 1,
                                     "PercolationCheck: input iMultiFab must have >= 1 ghost cell.");
    run(mf_phase, phase_id, dir);
}

std::string PercolationCheck::directionString(OpenImpala::Direction dir) {
    switch (dir) {
    case OpenImpala::Direction::X: return "X";
    case OpenImpala::Direction::Y: return "Y";
    case OpenImpala::Direction::Z: return "Z";
    default: return "?";
    }
}

void PercolationCheck::run(const amrex::iMultiFab& mf_phase, int phase_id,
                           OpenImpala::Direction dir) {
    BL_PROFILE("PercolationCheck::run");

    const amrex::Box& domain = m_geom.Domain();
    const int idir = static_cast<int>(dir);

    // Find seed points on inlet (low) and outlet (high) faces
    amrex::Vector<amrex::IntVect> local_inlet_seeds;
    amrex::Vector<amrex::IntVect> local_outlet_seeds;

    amrex::Box domain_lo_face = domain;
    domain_lo_face.setBig(idir, domain.smallEnd(idir));
    amrex::Box domain_hi_face = domain;
    domain_hi_face.setSmall(idir, domain.bigEnd(idir));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf_phase); mfi.isValid(); ++mfi) {
        const amrex::Box& validBox = mfi.validbox();
        const auto phase_arr = mf_phase.const_array(mfi);

        amrex::Box inlet_intersect = validBox & domain_lo_face;
        if (!inlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(inlet_intersect, [&](int i, int j, int k) {
                if (phase_arr(i, j, k, 0) == phase_id) {
#ifdef AMREX_USE_OMP
#pragma omp critical(perc_inlet_seed)
#endif
                    local_inlet_seeds.push_back(amrex::IntVect(i, j, k));
                }
            });
        }

        amrex::Box outlet_intersect = validBox & domain_hi_face;
        if (!outlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(outlet_intersect, [&](int i, int j, int k) {
                if (phase_arr(i, j, k, 0) == phase_id) {
#ifdef AMREX_USE_OMP
#pragma omp critical(perc_outlet_seed)
#endif
                    local_outlet_seeds.push_back(amrex::IntVect(i, j, k));
                }
            });
        }
    }

    // Gather seeds across MPI ranks
    MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
    int mpi_size = amrex::ParallelDescriptor::NProcs();

    auto gatherSeeds = [&](const amrex::Vector<amrex::IntVect>& local_seeds)
        -> amrex::Vector<amrex::IntVect> {
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

    auto inlet_seeds = gatherSeeds(local_inlet_seeds);
    auto outlet_seeds = gatherSeeds(local_outlet_seeds);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "    PercolationCheck: " << inlet_seeds.size() << " inlet seeds, "
                       << outlet_seeds.size() << " outlet seeds in " << directionString(dir)
                       << " direction.\n";
    }

    // Early exit if either face has no phase cells
    if (inlet_seeds.empty() || outlet_seeds.empty()) {
        m_percolates = false;
        m_active_vf = 0.0;
        return;
    }

    // Flood fill from inlet and outlet
    amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
    amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "    PercolationCheck: Flood fill from inlet...\n";
    parallelFloodFill(mf_reached_inlet, mf_phase, phase_id, inlet_seeds);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "    PercolationCheck: Flood fill from outlet...\n";
    parallelFloodFill(mf_reached_outlet, mf_phase, phase_id, outlet_seeds);

    // Build active mask: cells reachable from BOTH inlet and outlet
    amrex::iMultiFab mf_active_mask(m_ba, m_dm, 1, 0);
    mf_active_mask.setVal(cell_inactive);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf_active_mask, true); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = mf_active_mask.array(mfi);
        const auto inlet_arr = mf_reached_inlet.const_array(mfi);
        const auto outlet_arr = mf_reached_outlet.const_array(mfi);
        amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) {
            if (inlet_arr(i, j, k, 0) == cell_active && outlet_arr(i, j, k, 0) == cell_active) {
                mask_arr(i, j, k, 0) = cell_active;
            }
        });
    }

    long num_active = mf_active_mask.sum(0);
    long total_cells = m_geom.Domain().numPts();
    m_active_vf =
        (total_cells > 0) ? static_cast<amrex::Real>(num_active) / static_cast<amrex::Real>(total_cells)
                          : 0.0;
    m_percolates = (num_active > 0);
}

void PercolationCheck::parallelFloodFill(amrex::iMultiFab& reachabilityMask,
                                         const amrex::iMultiFab& phaseFab, int phaseID,
                                         const amrex::Vector<amrex::IntVect>& seedPoints) {
    BL_PROFILE("PercolationCheck::parallelFloodFill");
    AMREX_ASSERT(reachabilityMask.nGrow() >= 1);
    AMREX_ASSERT(phaseFab.nGrow() >= 1);

    reachabilityMask.setVal(cell_inactive);

    // Plant seeds
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
                    mask_arr(seed, 0) = cell_active;
                }
            }
        }
    }

    // Iterative expansion
    int iter = 0;
    amrex::IntVect domain_size = m_geom.Domain().size();
    const int max_flood_iter = domain_size[0] + domain_size[1] + domain_size[2] + 2;
    bool changed_globally = true;
    const std::vector<amrex::IntVect> offsets = {amrex::IntVect{1, 0, 0}, amrex::IntVect{-1, 0, 0},
                                                 amrex::IntVect{0, 1, 0}, amrex::IntVect{0, -1, 0},
                                                 amrex::IntVect{0, 0, 1}, amrex::IntVect{0, 0, -1}};

    while (changed_globally && iter < max_flood_iter) {
        ++iter;
        changed_globally = false;
        reachabilityMask.FillBoundary(m_geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        {
            bool changed_locally = false;
            for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
                const amrex::Box& tileBox = mfi.tilebox();
                auto mask_arr = reachabilityMask.array(mfi);
                const auto phase_arr = phaseFab.const_array(mfi, 0);
                const amrex::Box& grownTileBox = amrex::grow(tileBox, reachabilityMask.nGrow());
                amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) {
                    amrex::IntVect current_cell(i, j, k);
                    if (mask_arr(current_cell, 0) == cell_active ||
                        phase_arr(current_cell) != phaseID) {
                        return;
                    }
                    for (const auto& offset : offsets) {
                        amrex::IntVect neighbor_cell = current_cell + offset;
                        if (grownTileBox.contains(neighbor_cell)) {
                            if (mask_arr(neighbor_cell, 0) == cell_active) {
                                mask_arr(current_cell, 0) = cell_active;
                                changed_locally = true;
                                break;
                            }
                        }
                    }
                });
            }
#ifdef AMREX_USE_OMP
#pragma omp critical(perc_flood_fill)
#endif
            {
                if (changed_locally) {
                    changed_globally = true;
                }
            }
        }
        amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);
    }

    if (iter >= max_flood_iter && changed_globally) {
        amrex::Warning(
            "PercolationCheck::parallelFloodFill reached max iterations - result may be incomplete.");
    }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "    Flood fill completed in " << iter << " iterations.\n";
    }
    reachabilityMask.FillBoundary(m_geom.periodicity());
}

} // namespace OpenImpala
