#include "ThroughThicknessProfile.H"

#include <AMReX_ParallelReduce.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Gpu.H>
#include <AMReX.H>

namespace OpenImpala {

ThroughThicknessProfile::ThroughThicknessProfile(const amrex::Geometry& geom,
                                                 const amrex::iMultiFab& mf_phase, int phase_id,
                                                 OpenImpala::Direction dir, int comp) {
    compute(geom, mf_phase, phase_id, dir, comp);
}

void ThroughThicknessProfile::compute(const amrex::Geometry& geom, const amrex::iMultiFab& mf_phase,
                                      int phase_id, OpenImpala::Direction dir, int comp) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(comp >= 0 && comp < mf_phase.nComp(),
                                     "ThroughThicknessProfile: Component index out of bounds.");

    const amrex::Box& domain = geom.Domain();
    const int idir = static_cast<int>(dir);
    const int n_slices = domain.length(idir);

    // GPU-compatible per-slice accumulation using device vectors with atomic scatter-add
    amrex::Gpu::DeviceVector<int> d_slice_phase(n_slices, 0);
    amrex::Gpu::DeviceVector<int> d_slice_total(n_slices, 0);
    int* d_phase_ptr = d_slice_phase.data();
    int* d_total_ptr = d_slice_total.data();

    const int target_phase = phase_id;
    const int phase_comp = comp;
    const int slice_lo = domain.smallEnd(idir);

    for (amrex::MFIter mfi(mf_phase); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& fab = mf_phase.const_array(mfi, phase_comp);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int coord = (idir == 0) ? i : (idir == 1) ? j : k;
            int idx = coord - slice_lo;
            amrex::Gpu::Atomic::Add(&d_total_ptr[idx], 1);
            if (fab(i, j, k) == target_phase) {
                amrex::Gpu::Atomic::Add(&d_phase_ptr[idx], 1);
            }
        });
    }

    // Copy device results to host
    std::vector<int> slice_phase(n_slices);
    std::vector<int> slice_total(n_slices);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_slice_phase.begin(), d_slice_phase.end(),
                     slice_phase.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_slice_total.begin(), d_slice_total.end(),
                     slice_total.begin());

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
