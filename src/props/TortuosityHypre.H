#ifndef TortuosityHypre_H
#define TortuosityHypre_H

#include <memory> // Needed for std::unique_ptr
#include <string> // Needed for std::string
#include <vector> // Needed for Vector<IntVect> in helper functions
#include <limits> // Needed for std::numeric_limits
#include <map>    // Needed for phase-coefficient mapping

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H> // Include Geometry explicitly
#include <AMReX_Array.H>    // Needed for return type of loV/hiV
#include <AMReX_IntVect.H>  // Needed for Vector<IntVect>

#include <HYPRE.h>           // Main HYPRE header
#include <HYPRE_struct_ls.h> // Struct interface header (should provide HYPRE_Int)

#include "BoundaryCondition.H"
#include "HypreStructSolver.H"
#include "Tortuosity.H" // Defines OpenImpala::Tortuosity and OpenImpala::Direction

/** @file TortuosityHypre.H
 * @brief Defines a class to compute tortuosity using HYPRE solvers.
 */

namespace OpenImpala {

/** Computes the tortuosity of a porous structure using HYPRE.
 *
 * Solves a diffusion equation (-Div(Grad(phi)) = 0 with Dirichlet BCs)
 * on a masked subset of the domain derived from phase data, using HYPRE's
 * structured-grid solvers (PFMG, SMG, Krylov methods) to find the potential field.
 * Tortuosity is calculated from the resulting flux and **active** volume fraction,
 * including checks for solver convergence and flux conservation.
 *
 * Inherits HYPRE grid/stencil/solver infrastructure from HypreStructSolver.
 */
class TortuosityHypre : public OpenImpala::Tortuosity, public HypreStructSolver {

public:
    /** @brief Alias for the shared HYPRE solver type enum (defined in Tortuosity.H). */
    using SolverType = OpenImpala::SolverType;

    /** Construct new Tortuosity solver using HYPRE.
     *
     * @param geom AMReX Geometry object defining the domain.
     * @param ba AMReX BoxArray defining the distribution of grid patches.
     * @param dm AMReX DistributionMapping defining processor assignments.
     * @param mf_phase_input Input iMultiFab containing phase IDs (expects 1 component, >=1 ghost
     * cell).
     * @param vf Volume fraction of the phase of interest (*total* VF, passed for potential
     * reference).
     * @param phase Phase ID to calculate tortuosity for.
     * @param dir Direction across which to apply potential difference (X, Y, or Z).
     * @param solvertype Solver type to use (from SolverType enum).
     * @param resultspath Path to directory for writing output files (e.g., plotfiles).
     * @param vlo Potential value applied at the low boundary in the specified direction. Default
     * 0.0.
     * @param vhi Potential value applied at the high boundary in the specified direction.
     * Default 1.0.
     * @param verbose Verbosity level (0=silent, 1=basic, 2=detailed HYPRE output). Default 0.
     * @param write_plotfile If true, write a plotfile of the solution potential field. Default
     * false.
     */
    TortuosityHypre(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase_input,
                    const amrex::Real vf, const int phase, const OpenImpala::Direction dir,
                    const SolverType solvertype, const std::string& resultspath,
                    const amrex::Real vlo = 0.0, const amrex::Real vhi = 1.0, int verbose = 0,
                    bool write_plotfile = false);

    /** Destructor. Base class handles HYPRE resource cleanup. */
    virtual ~TortuosityHypre() override = default;

    // Deleted Copy Operations
    TortuosityHypre(const TortuosityHypre&) = delete;
    TortuosityHypre& operator=(const TortuosityHypre&) = delete;

    /** Calculate or retrieve the tortuosity value.
     *
     * Solves the underlying diffusion equation using HYPRE if not already solved
     * or if refresh is requested. Checks solver convergence and flux conservation.
     * Calculates flux and computes tortuosity using the average flux magnitude
     * and the *active* volume fraction if checks pass.
     * Returns NaN if the solver fails to converge, if flux is not conserved,
     * or if calculation leads to invalid states (e.g., zero active VF or Deff).
     *
     * @param refresh If true, forces recalculation by running the solver again. Default false.
     * @return Tortuosity value (ActiveVF / (Deff/D0)), or NaN on failure.
     */
    amrex::Real value(const bool refresh = false) override;

    /**
     * @brief Performs mathematical checks on the assembled HYPRE matrix and RHS vector.
     * @return True if all checks pass on all ranks, False otherwise.
     */
    bool checkMatrixProperties();

    // --- Public Getters for Status and Results ---
    // Solver statistics are inherited from HypreStructSolver:
    //   getSolverConverged(), getFinalRelativeResidualNorm(), getSolverIterations()

    amrex::Real getFluxIn() const {
        return m_flux_in;
    }
    amrex::Real getFluxOut() const {
        return m_flux_out;
    }
    amrex::Real getActiveVolumeFraction() const {
        return m_active_vf;
    }

    /** @brief Returns the per-plane flux profile computed after solving. */
    const std::vector<amrex::Real>& getPlaneFluxes() const {
        return m_plane_fluxes;
    }

    /** @brief Maximum relative deviation across interior plane fluxes. */
    amrex::Real getPlaneFluxMaxDeviation() const {
        return m_plane_flux_max_dev;
    }

    /** @brief Returns true if multi-phase transport coefficients are configured. */
    bool isMultiPhase() const {
        return m_is_multi_phase;
    }

    /** @brief Returns the phase-to-coefficient mapping. */
    const std::map<int, amrex::Real>& getPhaseCoeffMap() const {
        return m_phase_coeff_map;
    }

    /** @brief Returns the inlet/outlet BC type. */
    BCType getInletOutletBCType() const {
        return m_bc_inlet_outlet_type;
    }
    /** @brief Returns the lateral (sides) BC type. */
    BCType getSidesBCType() const {
        return m_bc_sides_type;
    }

    // Static Helper Functions (delegate to base class)
    static amrex::Array<HYPRE_Int, AMREX_SPACEDIM> loV(const amrex::Box& b) {
        return HypreStructSolver::loV(b);
    }
    static amrex::Array<HYPRE_Int, AMREX_SPACEDIM> hiV(const amrex::Box& b) {
        return HypreStructSolver::hiV(b);
    }


private:
    // --- Private Methods ---
    bool solve();
    void setupMatrixEquation();
    void preconditionPhaseFab();
    void generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID,
                              OpenImpala::Direction dir);
    void getSolution(amrex::MultiFab& soln, int ncomp = 0); // Not implemented
    void getCellTypes(amrex::MultiFab& phi, int ncomp = 1); // Not implemented
    void global_fluxes();
    void computePlaneFluxes(const amrex::MultiFab& mf_soln);

    // --- Member Variables ---
    // Configuration (solver config m_solvertype/m_eps/m_maxiter/m_verbose are in base class)
    std::string m_resultspath;
    const int m_phase;
    const OpenImpala::Direction m_dir;
    amrex::Real m_vlo;
    amrex::Real m_vhi;
    const amrex::Real m_vf; // Original total VF (kept for reference if needed)
    bool m_write_plotfile;

    // Boundary condition configuration
    BCType m_bc_inlet_outlet_type = BCType::DirichletExternal;
    BCType m_bc_sides_type = BCType::Neumann;
    std::unique_ptr<BoundaryCondition> m_bc_inlet_outlet;
    std::unique_ptr<BoundaryCondition> m_bc_sides;

    // AMReX Data (m_geom/m_ba/m_dm are in base class)
    amrex::iMultiFab m_mf_phase;
    amrex::MultiFab m_mf_phi;
    amrex::iMultiFab m_mf_active_mask;
    amrex::MultiFab m_mf_diff_coeff; ///< Spatially varying transport coefficient D(x)

    // Multi-phase configuration
    std::map<int, amrex::Real> m_phase_coeff_map; ///< Maps phase ID -> transport coefficient
    bool m_is_multi_phase = false;                ///< True if multiple conducting phases configured

    // State
    amrex::Real m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool m_first_call = true;

    // Results (solver stats m_num_iterations/m_final_res_norm/m_converged are in base class)
    amrex::Real m_flux_in = 0.0;
    amrex::Real m_flux_out = 0.0;
    amrex::Real m_active_vf = 0.0; // Volume fraction of the percolating phase

    // Interior plane flux profile (one entry per inter-cell face along flow direction)
    std::vector<amrex::Real> m_plane_fluxes;
    amrex::Real m_plane_flux_max_dev = 0.0; ///< max |F_i - F_mean| / |F_mean|

    // HYPRE Data Structures are in base class (m_grid, m_stencil, m_A, m_b, m_x)
};


} // namespace OpenImpala

#endif // TortuosityHypre_H
