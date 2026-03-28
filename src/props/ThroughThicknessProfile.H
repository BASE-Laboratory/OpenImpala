/** @file ThroughThicknessProfile.H
 *  @brief Computes per-slice volume fraction profiles along a given direction.
 */
#ifndef OPENIMPALA_THROUGH_THICKNESS_PROFILE_H
#define OPENIMPALA_THROUGH_THICKNESS_PROFILE_H

#include "Tortuosity.H" // For OpenImpala::Direction

#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>

#include <vector>

namespace OpenImpala {

/**
 * @brief Computes a through-thickness volume fraction profile.
 *
 * Slices the domain along the specified direction into individual planes
 * (one voxel thick each) and computes the local volume fraction of the
 * target phase in each slice. This produces a 1D array epsilon(z).
 *
 * @note Ghost cells are not required; only valid regions are used.
 * @warning The iMultiFab must remain valid for the lifetime of this object.
 */
class ThroughThicknessProfile {
public:
    /**
     * @brief Construct and compute the through-thickness profile.
     *
     * @param geom AMReX Geometry.
     * @param mf_phase iMultiFab containing phase IDs.
     * @param phase_id Phase to compute the profile for.
     * @param dir Direction to slice along.
     * @param comp Component index within mf_phase (default 0).
     */
    ThroughThicknessProfile(const amrex::Geometry& geom, const amrex::iMultiFab& mf_phase,
                            int phase_id, OpenImpala::Direction dir, int comp = 0);

    ThroughThicknessProfile(const ThroughThicknessProfile&) = delete;
    ThroughThicknessProfile& operator=(const ThroughThicknessProfile&) = delete;
    ThroughThicknessProfile(ThroughThicknessProfile&&) = default;
    ThroughThicknessProfile& operator=(ThroughThicknessProfile&&) = default;
    ~ThroughThicknessProfile() = default;

    /** @brief Returns the volume fraction profile (one value per slice). */
    const std::vector<amrex::Real>& volumeFractionProfile() const {
        return m_vf_profile;
    }

    /** @brief Returns the number of slices. */
    int numSlices() const {
        return static_cast<int>(m_vf_profile.size());
    }

    /** @brief Compute the profile (public for CUDA __device__ lambda compatibility). */
    void compute(const amrex::Geometry& geom, const amrex::iMultiFab& mf_phase, int phase_id,
                 OpenImpala::Direction dir, int comp);

private:
    std::vector<amrex::Real> m_vf_profile;
};

} // namespace OpenImpala

#endif // OPENIMPALA_THROUGH_THICKNESS_PROFILE_H
