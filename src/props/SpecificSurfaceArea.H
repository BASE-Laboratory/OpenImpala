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
 * Raw SSA = total_interface_faces / total_cells. With unit voxel spacing each
 * face has area 1 and each cell has volume 1.
 *
 * The raw face count systematically overestimates the true geometric surface
 * area because curved/diagonal interfaces are represented as axis-aligned
 * staircases on the voxel grid (the "laddering effect"). The corrected SSA
 * applies the Cauchy-Crofton factor of 2/3, which is exact for isotropic
 * surfaces (Ohser & Muecklich, 2000).
 *
 * An optional boundary padding parameter excludes the outermost N voxel layers
 * from the computation, mitigating artifacts from truncated features at domain
 * boundaries.
 *
 * @note Ghost cells are not required; face counting is restricted to valid regions.
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
     * @param boundary_padding Number of voxel layers to exclude from all domain
     *        faces (default 0). Both face counting and cell counting are
     *        restricted to the interior padded region.
     */
    SpecificSurfaceArea(const amrex::Geometry& geom, const amrex::iMultiFab& fm, int phase_a = 0,
                        int phase_b = 1, int comp = 0, int boundary_padding = 0);

    SpecificSurfaceArea(const SpecificSurfaceArea&) = delete;
    SpecificSurfaceArea& operator=(const SpecificSurfaceArea&) = delete;
    SpecificSurfaceArea(SpecificSurfaceArea&&) = default;
    SpecificSurfaceArea& operator=(SpecificSurfaceArea&&) = default;
    virtual ~SpecificSurfaceArea() = default;

    /**
     * @brief Compute the raw interface face count and total cell count.
     *
     * Both counts are restricted to the padded interior region if
     * boundary_padding > 0.
     *
     * @param[out] face_count Number of faces at the phase_a/phase_b interface.
     * @param[out] total_count Total number of cells in the (padded) domain.
     * @param[in] local If true, skip MPI reduction (default false).
     */
    void value(long long& face_count, long long& total_count, bool local = false) const;

    /**
     * @brief Raw SSA = face_count / total_count (no stereological correction).
     *
     * This value is exact for axis-aligned planar interfaces but overestimates
     * for curved or diagonal surfaces due to the voxel laddering effect.
     *
     * @param local If true, compute locally (default false).
     */
    amrex::Real value_ssa(bool local = false) const {
        long long fc, tc;
        value(fc, tc, local);
        return (tc > 0) ? static_cast<amrex::Real>(fc) / static_cast<amrex::Real>(tc) : 0.0;
    }

    /**
     * @brief Corrected SSA = (2/3) * raw SSA (Cauchy-Crofton correction).
     *
     * For a surface with isotropic orientation distribution, the expected
     * number of axis-aligned voxel face intersections is 3/2 times the true
     * surface area per unit volume. The 2/3 factor corrects for this bias.
     *
     * @param local If true, compute locally (default false).
     */
    amrex::Real value_corrected(bool local = false) const {
        return (2.0 / 3.0) * value_ssa(local);
    }

private:
    const amrex::Geometry& m_geom;
    const amrex::iMultiFab& m_mf;
    int m_phase_a;
    int m_phase_b;
    int m_comp;
    int m_boundary_padding;
};

} // namespace OpenImpala

#endif // OPENIMPALA_SPECIFIC_SURFACE_AREA_H
