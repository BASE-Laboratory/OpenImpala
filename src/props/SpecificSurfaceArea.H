/** @file SpecificSurfaceArea.H
 *  @brief Computes specific surface area (interfacial area per unit volume) between two phases.
 */
#ifndef OPENIMPALA_SPECIFIC_SURFACE_AREA_H
#define OPENIMPALA_SPECIFIC_SURFACE_AREA_H

#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_REAL.H>

namespace OpenImpala {

/**
 * @brief Computes the specific surface area between two phases in an iMultiFab.
 *
 * Counts voxel faces where adjacent cells belong to different specified phases.
 * SSA = total_interface_faces / total_cells. With unit voxel spacing each face
 * has area 1 and each cell has volume 1.
 *
 * @note Ghost cells are not required; the computation uses only valid regions.
 * @warning The iMultiFab must remain valid for the lifetime of this object.
 */
class SpecificSurfaceArea {
public:
    /**
     * @brief Construct a SpecificSurfaceArea calculator.
     *
     * @param geom AMReX Geometry describing the domain.
     * @param fm iMultiFab containing phase IDs.
     * @param phase_a First phase ID (default 0).
     * @param phase_b Second phase ID (default 1).
     * @param comp Component index within fm (default 0).
     */
    SpecificSurfaceArea(const amrex::Geometry& geom, const amrex::iMultiFab& fm, int phase_a = 0,
                        int phase_b = 1, int comp = 0);

    SpecificSurfaceArea(const SpecificSurfaceArea&) = delete;
    SpecificSurfaceArea& operator=(const SpecificSurfaceArea&) = delete;
    SpecificSurfaceArea(SpecificSurfaceArea&&) = default;
    SpecificSurfaceArea& operator=(SpecificSurfaceArea&&) = default;
    virtual ~SpecificSurfaceArea() = default;

    /**
     * @brief Compute the interface face count and total cell count.
     *
     * @param[out] face_count Number of faces at the phase_a/phase_b interface.
     * @param[out] total_count Total number of cells in the domain.
     * @param[in] local If true, skip MPI reduction (default false).
     */
    void value(long long& face_count, long long& total_count, bool local = false) const;

    /**
     * @brief Convenience method returning SSA = face_count / total_count.
     * @param local If true, compute locally (default false).
     */
    amrex::Real value_ssa(bool local = false) const {
        long long fc, tc;
        value(fc, tc, local);
        return (tc > 0) ? static_cast<amrex::Real>(fc) / static_cast<amrex::Real>(tc) : 0.0;
    }

private:
    const amrex::Geometry& m_geom;
    const amrex::iMultiFab& m_mf;
    int m_phase_a;
    int m_phase_b;
    int m_comp;
};

} // namespace OpenImpala

#endif // OPENIMPALA_SPECIFIC_SURFACE_AREA_H
