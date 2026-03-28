/** @file TortuosityMLMG.H
 *  @brief Matrix-free tortuosity solver using AMReX's native MLMG geometric multigrid.
 *
 *  Solves the same steady-state diffusion equation as TortuosityHypre
 *  (-div(D * grad(phi)) = 0 with Dirichlet BCs) but uses AMReX's built-in
 *  MLABecLaplacian operator and MLMG V-cycle solver instead of HYPRE.
 *
 *  Advantages over HYPRE for small/medium grids:
 *    - No explicit matrix assembly (matrix-free operator application)
 *    - Lower memory footprint (~3x less than HYPRE StructMatrix)
 *    - Faster time-to-solution on shared-memory systems (no AMG setup)
 *
 *  For large distributed problems or high-contrast multi-phase media,
 *  TortuosityHypre with algebraic multigrid remains the better choice.
 */
#ifndef TortuosityMLMG_H
#define TortuosityMLMG_H

#include <string>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include "TortuositySolverBase.H"

namespace OpenImpala {

/** @brief Computes tortuosity using AMReX's native matrix-free MLMG solver.
 *
 *  Solves div(D * grad(phi)) = 0 with Dirichlet BCs at inlet/outlet
 *  and zero-flux Neumann on lateral faces.  The diffusion coefficient D
 *  is set per-cell from the phase map (harmonic-mean face averaging is
 *  handled internally by AMReX's MLABecLaplacian operator).
 *
 *  Inherits from TortuositySolverBase which provides shared infrastructure:
 *  phase preconditioning, diffusion coefficient field construction,
 *  activity mask generation, flux integration, and tortuosity calculation.
 */
class TortuosityMLMG : public TortuositySolverBase {

public:
    /** @brief Construct a new MLMG tortuosity solver.
     *
     *  @param geom            AMReX Geometry defining the domain.
     *  @param ba              AMReX BoxArray defining grid patch layout.
     *  @param dm              AMReX DistributionMapping for processor assignment.
     *  @param mf_phase_input  Input iMultiFab of phase IDs (1 component, >=1 ghost).
     *  @param vf              Total volume fraction of the target phase (reference).
     *  @param phase           Phase ID to solve for.
     *  @param dir             Flow direction (X, Y, or Z).
     *  @param resultspath     Output directory for results files.
     *  @param vlo             Dirichlet potential at inlet face. Default 0.0.
     *  @param vhi             Dirichlet potential at outlet face. Default 1.0.
     *  @param verbose         Verbosity level (0=silent). Default 0.
     *  @param write_plotfile  Write AMReX plotfile of solution. Default false.
     */
    TortuosityMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                   const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase_input,
                   const amrex::Real vf, const int phase, const OpenImpala::Direction dir,
                   const std::string& resultspath, const amrex::Real vlo = 0.0,
                   const amrex::Real vhi = 1.0, int verbose = 0, bool write_plotfile = false);

    ~TortuosityMLMG() override = default;

    // Non-copyable
    TortuosityMLMG(const TortuosityMLMG&) = delete;
    TortuosityMLMG& operator=(const TortuosityMLMG&) = delete;

    /** @brief Run the MLMG solver, populating m_mf_solution.
     *
     *  Sets up MLABecLaplacian with appropriate BCs and B-coefficients,
     *  runs MLMG V-cycle solve, and optionally writes a plotfile.
     *  Public for CUDA __device__ lambda compatibility.
     *
     *  @return true if solver converged.
     */
    bool solve() override;

private:
    // MLMG solver parameters
    amrex::Real m_eps = 1.0e-9;
    int m_maxiter = 200;
    int m_max_coarsening_level = 30;
};

} // namespace OpenImpala

#endif // TortuosityMLMG_H
