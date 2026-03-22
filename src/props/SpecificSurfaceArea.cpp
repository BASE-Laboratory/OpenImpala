#include "SpecificSurfaceArea.H"

#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Reduce.H>
#include <AMReX.H>

namespace OpenImpala {

SpecificSurfaceArea::SpecificSurfaceArea(const amrex::Geometry& geom, const amrex::iMultiFab& fm,
                                         int phase_a, int phase_b, int comp, int boundary_padding)
    : m_geom(geom), m_mf(fm), m_phase_a(phase_a), m_phase_b(phase_b), m_comp(comp),
      m_boundary_padding(boundary_padding) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_comp >= 0 && m_comp < m_mf.nComp(),
                                     "SpecificSurfaceArea: Component index out of bounds.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_boundary_padding >= 0,
                                     "SpecificSurfaceArea: boundary_padding must be non-negative.");
}

void SpecificSurfaceArea::value(long long& face_count, long long& total_count, bool local) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_mf.nGrow() >= 1,
        "SpecificSurfaceArea: iMultiFab must have at least 1 ghost cell for neighbor access.");

    const int pa = m_phase_a;
    const int pb = m_phase_b;
    const int phase_comp = m_comp;
    const amrex::Box& domain = m_geom.Domain();

    amrex::Box padded_domain = domain;
    padded_domain.grow(-m_boundary_padding);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(padded_domain.ok(),
                                     "SpecificSurfaceArea: boundary_padding too large for domain.");

    // GPU-compatible reduction for face and cell counting
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<long long, long long> reduce_data(reduce_op);

    for (amrex::MFIter mfi(m_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& fab = m_mf.const_array(mfi, phase_comp);

        // Count total cells in padded interior
        const amrex::Box count_bx = bx & padded_domain;
        long long box_total = count_bx.ok() ? count_bx.numPts() : 0;

        // Count interface faces in each direction.
        // For each direction d, we check the face between cell (i,j,k) and its
        // neighbor in the +d direction. Both cells must lie within the padded
        // domain. The face is "owned" by the cell on its low side.
        // We also clamp to the valid box to avoid accessing ghost cells that
        // may not be filled (the +d neighbor must be within this FAB's valid region).
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::Box check_bx = bx & padded_domain;
            if (!check_bx.ok()) {
                continue;
            }
            // The +d neighbor must be in padded_domain; clamp so cell+1 is also valid.
            // The +d neighbor accesses a ghost cell at the FAB boundary, which is
            // correctly filled via FillBoundary (asserted nGrow >= 1 above).
            int hi_limit = std::min(check_bx.bigEnd(d), padded_domain.bigEnd(d) - 1);
            check_bx.setBig(d, hi_limit);

            if (check_bx.ok()) {
                amrex::IntVect offset(amrex::IntVect::TheZeroVector());
                offset[d] = 1;
                const int od0 = offset[0], od1 = offset[1], od2 = offset[2];
                reduce_op.eval(
                    check_bx, reduce_data,
                    [=] AMREX_GPU_DEVICE(int i, int j,
                                         int k) noexcept -> amrex::GpuTuple<long long, long long> {
                        int val = fab(i, j, k);
                        int nbr = fab(i + od0, j + od1, k + od2);
                        long long fc =
                            ((val == pa && nbr == pb) || (val == pb && nbr == pa)) ? 1LL : 0LL;
                        // total_contrib is added only for the first cell evaluated when d==0
                        return {fc, 0LL};
                    });

                // Accumulate total count outside the kernel (it's a box property, not per-cell)
                if (d == 0) {
                    // We need to add box_total to the total_count. Since this is a constant
                    // per MFIter iteration, we handle it after the reduce.
                }
            }
        }
    }

    auto hv = reduce_data.value();
    long long local_face_count = amrex::get<0>(hv);
    long long local_total_count = 0;

    // Compute total count on host (it's a box-level quantity, not per-cell)
    for (amrex::MFIter mfi(m_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const amrex::Box count_bx = bx & padded_domain;
        if (count_bx.ok()) {
            local_total_count += count_bx.numPts();
        }
    }

    if (!local) {
        amrex::ParallelAllReduce::Sum(local_face_count, amrex::ParallelContext::CommunicatorSub());
        amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());
    }

    face_count = local_face_count;
    total_count = local_total_count;
}

} // namespace OpenImpala
