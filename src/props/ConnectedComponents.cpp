#include "ConnectedComponents.H"
#include "FloodFill.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_Box.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Gpu.H>

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace OpenImpala {

ConnectedComponents::ConnectedComponents(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm,
                                         const amrex::iMultiFab& mf_phase, int phase_id,
                                         int verbose)
    : m_geom(geom), m_ba(ba), m_dm(dm), m_verbose(verbose), m_labels(ba, dm, 1, 1) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mf_phase.nGrow() >= 1, "ConnectedComponents: input iMultiFab must have >= 1 ghost cell.");
    run(mf_phase, phase_id);
}

amrex::IntVect ConnectedComponents::findNextUnlabeled(const amrex::iMultiFab& labelMF,
                                                      const amrex::iMultiFab& phaseFab,
                                                      int phaseID) const {
    // Find the first unlabeled cell of the target phase on this rank
    amrex::IntVect local_seed(-1, -1, -1);
    bool found = false;

    for (amrex::MFIter mfi(labelMF); mfi.isValid() && !found; ++mfi) {
        const amrex::Box& vbox = mfi.validbox();
        const auto label_arr = labelMF.const_array(mfi);
        const auto phase_arr = phaseFab.const_array(mfi, 0);

        amrex::LoopOnCpu(vbox, [&](int i, int j, int k) {
            if (!found && phase_arr(i, j, k) == phaseID && label_arr(i, j, k, 0) == 0) {
                local_seed = amrex::IntVect(i, j, k);
                found = true;
            }
        });
    }

    // Gather all candidate seeds and pick the lexicographically smallest
    // to ensure deterministic labeling across MPI ranks
    MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
    int flat[3] = {local_seed[0], local_seed[1], local_seed[2]};

    int mpi_size = amrex::ParallelDescriptor::NProcs();
    std::vector<int> all_seeds(mpi_size * 3);
    MPI_Allgather(flat, 3, MPI_INT, all_seeds.data(), 3, MPI_INT, comm);

    // Find the smallest valid seed across all ranks
    amrex::IntVect best(-1, -1, -1);
    for (int r = 0; r < mpi_size; ++r) {
        int si = all_seeds[r * 3 + 0];
        if (si < 0) {
            continue;
        }
        amrex::IntVect candidate(all_seeds[r * 3 + 0], all_seeds[r * 3 + 1], all_seeds[r * 3 + 2]);
        if (best[0] < 0 || candidate < best) {
            best = candidate;
        }
    }
    return best;
}

void ConnectedComponents::run(const amrex::iMultiFab& mf_phase, int phase_id) {
    BL_PROFILE("ConnectedComponents::run");

    m_labels.setVal(0);
    m_num_components = 0;

    // Iteratively find seeds and flood-fill until all cells are labeled.
    // Each component gets a unique label (1, 2, 3, ...) via the shared
    // GPU-compatible flood fill utility.
    while (true) {
        amrex::IntVect seed = findNextUnlabeled(m_labels, mf_phase, phase_id);
        if (seed[0] < 0) {
            break; // No more unlabeled cells
        }

        m_num_components++;
        int label = m_num_components;

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "    CCL: labeling component " << label << " from seed (" << seed[0]
                           << ", " << seed[1] << ", " << seed[2] << ")\n";
        }

        // Use shared flood fill with a single-seed vector and custom label
        amrex::Vector<amrex::IntVect> seedVec = {seed};
        OpenImpala::parallelFloodFill(m_labels, mf_phase, phase_id, seedVec, m_geom, m_verbose,
                                      label);
    }

    if (m_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  ConnectedComponents: found " << m_num_components << " components\n";
    }

    // Compute volume of each component using GPU-compatible atomic scatter-add
    m_volumes.resize(m_num_components, 0);

    amrex::Gpu::DeviceVector<long long> d_volumes(m_num_components, 0);
    long long* d_vol_ptr = d_volumes.data();
    const int num_comp = m_num_components;

    for (amrex::MFIter mfi(m_labels); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto label_arr = m_labels.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int lbl = label_arr(i, j, k, 0);
            if (lbl > 0 && lbl <= num_comp) {
                amrex::Gpu::Atomic::Add(&d_vol_ptr[lbl - 1], 1LL);
            }
        });
    }

    std::vector<long long> local_volumes(m_num_components);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_volumes.begin(), d_volumes.end(),
                     local_volumes.begin());

    if (m_num_components > 0) {
        amrex::ParallelAllReduce::Sum(local_volumes.data(), m_num_components,
                                      amrex::ParallelContext::CommunicatorSub());
    }
    m_volumes = local_volumes;
}

} // namespace OpenImpala
