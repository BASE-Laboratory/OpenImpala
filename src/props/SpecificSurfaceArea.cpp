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
                                         int phase_a, int phase_b, int comp, int boundary_padding,
                                         const amrex::iMultiFab* active_mask, int active_phase)
    : m_geom(geom), m_mf(fm), m_phase_a(phase_a), m_phase_b(phase_b), m_comp(comp),
      m_boundary_padding(boundary_padding), m_active_mask(active_mask),
      m_active_phase((active_phase >= 0) ? active_phase : phase_a) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_comp >= 0 && m_comp < m_mf.nComp(),
                                     "SpecificSurfaceArea: Component index out of bounds.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_boundary_padding >= 0,
                                     "SpecificSurfaceArea: boundary_padding must be non-negative.");
    if (m_active_mask != nullptr) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_active_mask->nGrow() >= 1,
            "SpecificSurfaceArea: active_mask must have at least 1 ghost cell.");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_active_phase == m_phase_a || m_active_phase == m_phase_b,
            "SpecificSurfaceArea: active_phase must equal phase_a or phase_b.");
    }
}

void SpecificSurfaceArea::directionFaceCounts(amrex::Array<long long, AMREX_SPACEDIM>& dir_faces,
                                              long long& total_count, bool local) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_mf.nGrow() >= 1,
        "SpecificSurfaceArea: iMultiFab must have at least 1 ghost cell for neighbor access.");

    const int pa = m_phase_a;
    const int pb = m_phase_b;
    const int act_phase = m_active_phase;
    const int phase_comp = m_comp;
    const amrex::Box& domain = m_geom.Domain();

    amrex::Box padded_domain = domain;
    padded_domain.grow(-m_boundary_padding);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(padded_domain.ok(),
                                     "SpecificSurfaceArea: boundary_padding too large for domain.");

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<long long> reduce_data(reduce_op);

        amrex::IntVect offset(amrex::IntVect::TheZeroVector());
        offset[d] = 1;
        const int od0 = offset[0], od1 = offset[1], od2 = offset[2];

        for (amrex::MFIter mfi(m_mf); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            const auto& fab = m_mf.const_array(mfi, phase_comp);

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
                if (m_active_mask == nullptr) {
                    reduce_op.eval(
                        check_bx, reduce_data,
                        [=] AMREX_GPU_DEVICE(int i, int j,
                                             int k) noexcept -> amrex::GpuTuple<long long> {
                            int val = fab(i, j, k);
                            int nbr = fab(i + od0, j + od1, k + od2);
                            long long fc =
                                ((val == pa && nbr == pb) || (val == pb && nbr == pa)) ? 1LL : 0LL;
                            return {fc};
                        });
                } else {
                    const auto& act = m_active_mask->const_array(mfi, 0);
                    reduce_op.eval(check_bx, reduce_data,
                                   [=] AMREX_GPU_DEVICE(
                                       int i, int j, int k) noexcept -> amrex::GpuTuple<long long> {
                                       const int val = fab(i, j, k);
                                       const int nbr = fab(i + od0, j + od1, k + od2);
                                       long long fc = 0LL;
                                       if (val == pa && nbr == pb) {
                                           const int a_flag = (act_phase == pa)
                                                                  ? act(i, j, k)
                                                                  : act(i + od0, j + od1, k + od2);
                                           fc = (a_flag != 0) ? 1LL : 0LL;
                                       } else if (val == pb && nbr == pa) {
                                           const int a_flag = (act_phase == pb)
                                                                  ? act(i, j, k)
                                                                  : act(i + od0, j + od1, k + od2);
                                           fc = (a_flag != 0) ? 1LL : 0LL;
                                       }
                                       return {fc};
                                   });
                }
            }
        }

        dir_faces[d] = amrex::get<0>(reduce_data.value());
    }

    long long local_total_count = 0;
    for (amrex::MFIter mfi(m_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const amrex::Box count_bx = bx & padded_domain;
        if (count_bx.ok()) {
            local_total_count += count_bx.numPts();
        }
    }

    if (!local) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::ParallelAllReduce::Sum(dir_faces[d], amrex::ParallelContext::CommunicatorSub());
        }
        amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());
    }

    total_count = local_total_count;
}

void SpecificSurfaceArea::value(long long& face_count, long long& total_count, bool local) const {
    amrex::Array<long long, AMREX_SPACEDIM> dir_faces{};
    directionFaceCounts(dir_faces, total_count, local);
    face_count = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        face_count += dir_faces[d];
    }
}

amrex::Real SpecificSurfaceArea::value_physical(bool local) const {
    amrex::Array<long long, AMREX_SPACEDIM> dir_faces{};
    long long total_count = 0;
    directionFaceCounts(dir_faces, total_count, local);
    if (total_count <= 0) {
        return 0.0;
    }

    const amrex::Real* dx = m_geom.CellSize();
    amrex::Real weighted_sum = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dx[d] > 0.0,
                                         "SpecificSurfaceArea: non-positive cell size.");
        weighted_sum += static_cast<amrex::Real>(dir_faces[d]) / dx[d];
    }
    return weighted_sum / static_cast<amrex::Real>(total_count);
}

} // namespace OpenImpala
