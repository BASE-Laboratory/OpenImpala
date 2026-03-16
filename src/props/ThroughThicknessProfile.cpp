#include "ThroughThicknessProfile.H"

#include <AMReX_ParallelReduce.H>
#include <AMReX_MFIter.H>
#include <AMReX_Loop.H>
#include <AMReX.H>

namespace OpenImpala {

ThroughThicknessProfile::ThroughThicknessProfile(const amrex::Geometry& geom,
                                                 const amrex::iMultiFab& mf_phase, int phase_id,
                                                 OpenImpala::Direction dir, int comp) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(comp >= 0 && comp < mf_phase.nComp(),
                                     "ThroughThicknessProfile: Component index out of bounds.");

    const amrex::Box& domain = geom.Domain();
    const int idir = static_cast<int>(dir);
    const int n_slices = domain.length(idir);

    // Per-slice counts: phase_count and total_count
    std::vector<long long> slice_phase(n_slices, 0);
    std::vector<long long> slice_total(n_slices, 0);

    const int target_phase = phase_id;
    const int phase_comp = comp;
    const int slice_lo = domain.smallEnd(idir);

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        // Thread-local accumulators to avoid false sharing
        std::vector<long long> thr_phase(n_slices, 0);
        std::vector<long long> thr_total(n_slices, 0);

        for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto& fab = mf_phase.const_array(mfi, phase_comp);

            amrex::Loop(bx, [&](int i, int j, int k) {
                int coord = (idir == 0) ? i : (idir == 1) ? j : k;
                int idx = coord - slice_lo;
                thr_total[idx] += 1;
                if (fab(i, j, k) == target_phase) {
                    thr_phase[idx] += 1;
                }
            });
        }

#ifdef AMREX_USE_OMP
#pragma omp critical(ttp_accumulate)
#endif
        {
            for (int s = 0; s < n_slices; ++s) {
                slice_phase[s] += thr_phase[s];
                slice_total[s] += thr_total[s];
            }
        }
    }

    // MPI reduction across all ranks
    amrex::ParallelAllReduce::Sum(slice_phase.data(), n_slices,
                                  amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Sum(slice_total.data(), n_slices,
                                  amrex::ParallelContext::CommunicatorSub());

    // Compute VF per slice
    m_vf_profile.resize(n_slices);
    for (int s = 0; s < n_slices; ++s) {
        m_vf_profile[s] = (slice_total[s] > 0) ? static_cast<amrex::Real>(slice_phase[s]) /
                                                     static_cast<amrex::Real>(slice_total[s])
                                               : 0.0;
    }
}

} // namespace OpenImpala
