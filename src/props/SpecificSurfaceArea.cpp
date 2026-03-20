#include "SpecificSurfaceArea.H"

#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Loop.H>
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
    long long local_face_count = 0;
    long long local_total_count = 0;

    const int pa = m_phase_a;
    const int pb = m_phase_b;
    const int phase_comp = m_comp;
    const amrex::Box& domain = m_geom.Domain();

    // Shrink domain by boundary_padding on all sides to exclude outermost layers
    const amrex::Box padded_domain = domain.grow(-m_boundary_padding);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(padded_domain.ok(),
                                     "SpecificSurfaceArea: boundary_padding too large for domain.");

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+ : local_face_count, local_total_count)
#endif
    for (amrex::MFIter mfi(m_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto& fab = m_mf.const_array(mfi, phase_comp);

        // Restrict cell counting to the padded interior
        const amrex::Box count_bx = bx & padded_domain;
        if (count_bx.ok()) {
            local_total_count += count_bx.numPts();
        }

        // Count interface faces in each direction.
        // For each direction d, we check the face between cell (i,j,k) and its
        // neighbor in the +d direction. Both cells must lie within the padded
        // domain. The face is "owned" by the cell on its low side.
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::Box check_bx = bx & padded_domain;
            if (!check_bx.ok()) {
                continue;
            }
            // The +d neighbor must also be in padded_domain, so shrink high end by 1
            int hi_limit = std::min(check_bx.bigEnd(d), padded_domain.bigEnd(d) - 1);
            check_bx.setBig(d, hi_limit);

            if (check_bx.ok()) {
                amrex::IntVect offset(amrex::IntVect::TheZeroVector());
                offset[d] = 1;

                amrex::Loop(check_bx, [&](int i, int j, int k) {
                    int val = fab(i, j, k);
                    int nbr = fab(i + offset[0], j + offset[1], k + offset[2]);
                    if ((val == pa && nbr == pb) || (val == pb && nbr == pa)) {
                        local_face_count += 1;
                    }
                });
            }
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
