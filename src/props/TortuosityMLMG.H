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

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include "Tortuosity.H"

namespace OpenImpala {

/** @brief Computes tortuosity using AMReX's native matrix-free MLMG solver.
 *
 *  Solves div(D * grad(phi)) = 0 with Dirichlet BCs at inlet/outlet
 *  and zero-flux Neumann on lateral faces.  The diffusion coefficient D
 *  is set per-cell from the phase map (harmonic-mean face averaging is
 *  handled internally by AMReX's MLABecLaplacian operator).
 *
 *  Inherits from Tortuosity (abstract base) but NOT from HypreStructSolver,
 *  since it uses no HYPRE infrastructure.
 */
class TortuosityMLMG : public OpenImpala::Tortuosity {

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

    /** @brief Calculate or retrieve the tortuosity value.
     *
     *  Runs the MLMG solver on first call (or when refresh=true), computes
     *  flux, and returns tau = active_vf / D_eff.  Returns NaN on failure.
     */
    amrex::Real value(const bool refresh = false) override;

    // --- Public getters ---
    bool getSolverConverged() const { return m_converged; }
    amrex::Real getFinalRelativeResidualNorm() const { return m_final_res_norm; }
    int getSolverIterations() const { return m_num_iterations; }
    amrex::Real getFluxIn() const { return m_flux_in; }
    amrex::Real getFluxOut() const { return m_flux_out; }
    amrex::Real getActiveVolumeFraction() const { return m_active_vf; }
    const std::vector<amrex::Real>& getPlaneFluxes() const { return m_plane_fluxes; }
    amrex::Real getPlaneFluxMaxDeviation() const { return m_plane_flux_max_dev; }

private:
    // --- Private methods ---
    bool solve();
    void preconditionPhaseFab();
    void generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID,
                              OpenImpala::Direction dir);
    void globalFluxes();
    void computePlaneFluxes(const amrex::MultiFab& mf_soln);

    // --- Configuration ---
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    std::string m_resultspath;
    const int m_phase;
    const OpenImpala::Direction m_dir;
    amrex::Real m_vlo;
    amrex::Real m_vhi;
    const amrex::Real m_vf;
    bool m_write_plotfile;
    int m_verbose;

    // MLMG solver parameters
    amrex::Real m_eps = 1.0e-9;
    int m_maxiter = 200;
    int m_max_coarsening_level = 30;

    // --- AMReX data ---
    amrex::iMultiFab m_mf_phase;
    amrex::MultiFab m_mf_solution;     ///< Cell-centred solution phi
    amrex::iMultiFab m_mf_active_mask; ///< 1 = percolating, 0 = isolated
    amrex::MultiFab m_mf_diff_coeff;   ///< Cell-centred D(x)

    // --- State ---
    amrex::Real m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool m_first_call = true;
    amrex::Real m_active_vf = 0.0;

    // --- Solver diagnostics ---
    bool m_converged = false;
    int m_num_iterations = 0;
    amrex::Real m_final_res_norm = 0.0;

    // --- Flux results ---
    amrex::Real m_flux_in = 0.0;
    amrex::Real m_flux_out = 0.0;
    std::vector<amrex::Real> m_plane_fluxes;
    amrex::Real m_plane_flux_max_dev = 0.0;
};

} // namespace OpenImpala

#endif // TortuosityMLMG_H
