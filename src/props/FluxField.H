/** @file FluxField.H
 *  @brief Cell-centred 3-D diffusive flux vector and magnitude statistics for tortuosity solvers.
 */
#ifndef OPENIMPALA_FLUX_FIELD_H
#define OPENIMPALA_FLUX_FIELD_H

#include "Tortuosity.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <vector>

namespace OpenImpala {

/** @brief Summary statistics of the local diffusive flux magnitude across active voxels. */
struct FluxStats {
    amrex::Real mean_magnitude = 0.0;
    amrex::Real max_magnitude = 0.0;
    amrex::Real std_magnitude = 0.0;
    std::vector<amrex::Real> histogram_edges;
    std::vector<long long> histogram_counts;
};

/** @brief Populate a 4-component cell-centred flux MultiFab (Jx, Jy, Jz, |J|).
 *
 *  Interior faces between active neighbours use harmonic-mean face conductances
 *  consistent with the 7-point finite-volume discretisation. Exterior inlet/outlet
 *  Dirichlet boundaries along *flow_dir* use the exact half-cell face flux
 *  ``-2 * D * (phi - v_bc) / dx[flow_dir]``, while lateral Neumann boundaries and
 *  active/inactive phase interfaces carry zero normal flux. Cell-centred components
 *  average the lower and upper face flux densities along each axis:
 *  ``J_d = 0.5 * (J_{d,-} + J_{d,+})``. Inactive voxels receive zero flux.
 *
 *  @param[out] mf_flux         Output MultiFab (defined automatically if needed, >=4 comps).
 *  @param[in]  geom            Domain Geometry providing cell sizes and index bounds.
 *  @param[in]  mf_soln         Solved potential field (>=1 ghost cell filled across patches).
 *  @param[in]  mf_active_mask  Percolating active-cell mask (1 = active, >=1 ghost cell).
 *  @param[in]  mf_diff_coeff   Cell-centred diffusivity field (>=1 ghost cell).
 *  @param[in]  flow_dir        Macroscopic Dirichlet flow direction (X, Y, or Z).
 *  @param[in]  vlo             Dirichlet potential at the low domain face along *flow_dir*.
 *  @param[in]  vhi             Dirichlet potential at the high domain face along *flow_dir*.
 *  @param[in]  soln_comp       Component index of the potential within *mf_soln* (default 0).
 */
void computeTortuosityCellFluxField(amrex::MultiFab& mf_flux, const amrex::Geometry& geom,
                                    const amrex::MultiFab& mf_soln,
                                    const amrex::iMultiFab& mf_active_mask,
                                    const amrex::MultiFab& mf_diff_coeff,
                                    OpenImpala::Direction flow_dir, amrex::Real vlo,
                                    amrex::Real vhi, int soln_comp = 0);

/** @brief Compute MPI-reduced summary statistics and histogram of ``|J|`` over active voxels. */
FluxStats computeTortuosityFluxStats(const amrex::Geometry& geom, const amrex::MultiFab& mf_soln,
                                     const amrex::iMultiFab& mf_active_mask,
                                     const amrex::MultiFab& mf_diff_coeff,
                                     OpenImpala::Direction flow_dir, amrex::Real vlo,
                                     amrex::Real vhi, int num_bins = 16, int soln_comp = 0);

} // namespace OpenImpala

#endif // OPENIMPALA_FLUX_FIELD_H
