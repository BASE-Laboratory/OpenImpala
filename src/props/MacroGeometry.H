/** @file MacroGeometry.H
 *  @brief Computes macro-geometric parameters (electrode thickness, cross-section) from Geometry.
 */
#ifndef OPENIMPALA_MACRO_GEOMETRY_H
#define OPENIMPALA_MACRO_GEOMETRY_H

#ifndef OPENIMPALA_UNIT_TEST
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#endif

namespace OpenImpala {

#ifdef OPENIMPALA_UNIT_TEST
using Real = double;
#else
using Real = amrex::Real;
#endif

/**
 * @brief Macro-geometric parameters derived from the AMReX Geometry.
 *
 * For DFN battery models, these provide the electrode thickness (Lp),
 * cross-sectional area (A), and total volume. All values are in voxel
 * units unless scaled by a user-supplied voxel_size_m.
 */
struct MacroGeometry {
    Real thickness = 0.0;     ///< Length along flow direction (Lp) in voxel units
    Real cross_section = 0.0; ///< Area perpendicular to flow in voxel units
    Real total_volume = 0.0;  ///< Lp * A
    int nx = 0;               ///< Grid dimension X
    int ny = 0;               ///< Grid dimension Y
    int nz = 0;               ///< Grid dimension Z

    /**
     * @brief Construct MacroGeometry from known dimensions and a flow direction index.
     *
     * @param nx_in Number of cells in X.
     * @param ny_in Number of cells in Y.
     * @param nz_in Number of cells in Z.
     * @param flow_dir_idx Flow direction as integer (0=X, 1=Y, 2=Z).
     */
    static MacroGeometry fromDimensions(int nx_in, int ny_in, int nz_in, int flow_dir_idx) {
        MacroGeometry mg;
        mg.nx = nx_in;
        mg.ny = ny_in;
        mg.nz = nz_in;

        int dims[3] = {nx_in, ny_in, nz_in};
        mg.thickness = static_cast<Real>(dims[flow_dir_idx]);

        // Cross-section = product of lateral dimensions
        Real cs = 1.0;
        for (int d = 0; d < 3; ++d) {
            if (d != flow_dir_idx) {
                cs *= static_cast<Real>(dims[d]);
            }
        }
        mg.cross_section = cs;
        mg.total_volume = mg.thickness * mg.cross_section;
        return mg;
    }

#ifndef OPENIMPALA_UNIT_TEST
    /**
     * @brief Construct MacroGeometry from an AMReX Geometry and flow Direction.
     *
     * @param geom AMReX Geometry object.
     * @param flow_dir Flow direction enum (X, Y, or Z).
     */
    static MacroGeometry fromGeometry(const amrex::Geometry& geom, int flow_dir_idx) {
        const amrex::Box& domain = geom.Domain();
        return fromDimensions(domain.length(0), domain.length(1), domain.length(2), flow_dir_idx);
    }
#endif
};

} // namespace OpenImpala

#endif // OPENIMPALA_MACRO_GEOMETRY_H
