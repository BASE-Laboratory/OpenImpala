#include "PercolationCheck.H"
#include "FloodFill.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_MFIter.H>
#include <AMReX_Box.H>

namespace OpenImpala {

PercolationCheck::PercolationCheck(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                   const amrex::DistributionMapping& dm,
                                   const amrex::iMultiFab& mf_phase, int phase_id,
                                   OpenImpala::Direction dir, int verbose)
    : m_geom(geom), m_ba(ba), m_dm(dm), m_verbose(verbose) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mf_phase.nGrow() >= 1, "PercolationCheck: input iMultiFab must have >= 1 ghost cell.");
    run(mf_phase, phase_id, dir);
}

std::string PercolationCheck::directionString(OpenImpala::Direction dir) {
    switch (dir) {
    case OpenImpala::Direction::X:
        return "X";
    case OpenImpala::Direction::Y:
        return "Y";
    case OpenImpala::Direction::Z:
        return "Z";
    default:
        return "?";
    }
}

void PercolationCheck::run(const amrex::iMultiFab& mf_phase, int phase_id,
                           OpenImpala::Direction dir) {
    BL_PROFILE("PercolationCheck::run");

    const int idir = static_cast<int>(dir);

    // Collect boundary seeds (shared utility handles MPI gather + dedup)
    amrex::Vector<amrex::IntVect> inlet_seeds;
    amrex::Vector<amrex::IntVect> outlet_seeds;
    OpenImpala::collectBoundarySeeds(mf_phase, phase_id, idir, m_geom, inlet_seeds, outlet_seeds);

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

    // GPU-compatible flood fill from inlet and outlet
    amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
    amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);
    mf_reached_inlet.setVal(FLOOD_INACTIVE);
    mf_reached_outlet.setVal(FLOOD_INACTIVE);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "    PercolationCheck: Flood fill from inlet...\n";
    OpenImpala::parallelFloodFill(mf_reached_inlet, mf_phase, phase_id, inlet_seeds, m_geom,
                                  m_verbose);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "    PercolationCheck: Flood fill from outlet...\n";
    OpenImpala::parallelFloodFill(mf_reached_outlet, mf_phase, phase_id, outlet_seeds, m_geom,
                                  m_verbose);

    // Build active mask: cells reachable from BOTH inlet and outlet (GPU-compatible)
    amrex::iMultiFab mf_active_mask(m_ba, m_dm, 1, 0);
    mf_active_mask.setVal(FLOOD_INACTIVE);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = mf_active_mask.array(mfi);
        const auto inlet_arr = mf_reached_inlet.const_array(mfi);
        const auto outlet_arr = mf_reached_outlet.const_array(mfi);
        amrex::ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            mask_arr(i, j, k, 0) =
                (inlet_arr(i, j, k, 0) == FLOOD_ACTIVE && outlet_arr(i, j, k, 0) == FLOOD_ACTIVE)
                    ? FLOOD_ACTIVE
                    : FLOOD_INACTIVE;
        });
    }

    long num_active = mf_active_mask.sum(0);
    long total_cells = m_geom.Domain().numPts();
    m_active_vf = (total_cells > 0)
                      ? static_cast<amrex::Real>(num_active) / static_cast<amrex::Real>(total_cells)
                      : 0.0;
    m_percolates = (num_active > 0);
}

} // namespace OpenImpala
