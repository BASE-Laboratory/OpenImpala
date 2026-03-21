#include "VolumeFraction.H"
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Reduce.H>
#include <AMReX.H>

namespace OpenImpala {

VolumeFraction::VolumeFraction(const amrex::iMultiFab& fm, const int phase, int comp)
    : m_mf(fm), m_phase(phase), m_comp(comp) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_comp >= 0 && m_comp < m_mf.nComp(),
                                     "VolumeFraction: Component index out of bounds.");
}

void VolumeFraction::value(long long& phase_count, long long& total_count, bool local) const {
    const int target_phase = m_phase;
    const int phase_comp = m_comp;

    // GPU-compatible reduction for phase counting
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<long long> reduce_data(reduce_op);

    long long local_total_count = 0;

    for (amrex::MFIter mfi(m_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& fab = m_mf.const_array(mfi, phase_comp);

        reduce_op.eval(
            bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<long long> {
                return {(fab(i, j, k) == target_phase) ? 1LL : 0LL};
            });

        local_total_count += bx.numPts();
    }

    long long local_phase_count = amrex::get<0>(reduce_data.value());

    if (!local) {
        amrex::ParallelAllReduce::Sum(local_phase_count, amrex::ParallelContext::CommunicatorSub());
        amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());
    }

    phase_count = local_phase_count;
    total_count = local_total_count;
}

} // namespace OpenImpala
