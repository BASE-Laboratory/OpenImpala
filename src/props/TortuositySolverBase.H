/** @file TortuositySolverBase.H
 *  @brief Shared base class for tortuosity solvers (HYPRE, MLMG, etc.).
 *
 *  Extracts the solver-independent logic that was previously duplicated:
 *    - Phase data management and preconditioning
 *    - Diffusion coefficient field construction
 *    - Activity mask generation via parallel flood fill
 *    - Post-solve flux integration and tortuosity calculation
 *
 *  Subclasses implement only solve(), which populates m_mf_solution.
 */
#ifndef TortuositySolverBase_H
#define TortuositySolverBase_H

#include <limits>
#include <map>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include "Tortuosity.H"

namespace OpenImpala {

/** @brief Shared base class for all tortuosity solvers.
 *
 *  Handles problem setup (phase data, diffusion coefficients, activity mask)
 *  and post-solve analysis (flux integration, tortuosity calculation).
 *  Subclasses implement the pure-virtual solve() method.
 */
class TortuositySolverBase : public OpenImpala::Tortuosity {

public:
    /** @brief Construct common solver infrastructure.
     *
     *  Performs all solver-independent setup: copies phase data, parses
     *  multi-phase configuration, removes isolated cells, builds the
     *  diffusion coefficient field, and generates the activity mask
     *  via boundary-seeded parallel flood fill.
     *
     *  @param geom            AMReX Geometry defining the domain.
     *  @param ba              AMReX BoxArray defining grid patch layout.
     *  @param dm              AMReX DistributionMapping for processor assignment.
     *  @param mf_phase_input  Input iMultiFab of phase IDs (1 component, >=1 ghost).
     *  @param vf              Total volume fraction of the target phase.
     *  @param phase           Phase ID to solve for.
     *  @param dir             Flow direction (X, Y, or Z).
     *  @param resultspath     Output directory for results files.
     *  @param vlo             Dirichlet potential at inlet face.
     *  @param vhi             Dirichlet potential at outlet face.
     *  @param verbose         Verbosity level (0=silent).
     *  @param write_plotfile  Write AMReX plotfile of solution.
     */
    TortuositySolverBase(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                         const amrex::DistributionMapping& dm,
                         const amrex::iMultiFab& mf_phase_input, const amrex::Real vf,
                         const int phase, const OpenImpala::Direction dir,
                         const std::string& resultspath, const amrex::Real vlo,
                         const amrex::Real vhi, int verbose, bool write_plotfile);

    ~TortuositySolverBase() override = default;

    // Non-copyable
    TortuositySolverBase(const TortuositySolverBase&) = delete;
    TortuositySolverBase& operator=(const TortuositySolverBase&) = delete;

    /** @brief Template method: solve, compute flux, return tortuosity.
     *
     *  On first call (or refresh), calls solve(), then globalFluxes(),
     *  checks flux conservation, and computes tau = active_vf / D_eff.
     *  Returns NaN on failure (non-convergence, flux non-conservation).
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
    bool isMultiPhase() const { return m_is_multi_phase; }
    const std::map<int, amrex::Real>& getPhaseCoeffMap() const { return m_phase_coeff_map; }

protected:
    /** @brief Run the linear solver, populating m_mf_solution.
     *
     *  Subclasses must:
     *  1. Solve the system and write the solution into m_mf_solution
     *  2. Call m_mf_solution.FillBoundary() after populating
     *  3. Set m_converged, m_num_iterations, m_final_res_norm
     *
     *  @return true if solver converged.
     */
    virtual bool solve() = 0;

    // --- Shared setup helpers ---
    void preconditionPhaseFab();
    void buildDiffusionCoeffField();
    void generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID,
                              OpenImpala::Direction dir);
    void parallelFloodFill(amrex::iMultiFab& reachabilityMask, const amrex::iMultiFab& phaseFab,
                           int phaseID, const amrex::Vector<amrex::IntVect>& seedPoints);

    // --- Post-solve analysis ---
    void globalFluxes();
    void computePlaneFluxes(const amrex::MultiFab& mf_soln);
    void writeSolutionPlotfile(const std::string& label);

    // --- Grid data (stored by value to avoid dangling references) ---
    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    // --- AMReX data ---
    amrex::iMultiFab m_mf_phase;       ///< Phase IDs (with ghosts)
    amrex::MultiFab m_mf_solution;     ///< Solution phi (1 comp, 1 ghost) — populated by solve()
    amrex::iMultiFab m_mf_active_mask; ///< 1 = percolating, 0 = isolated
    amrex::MultiFab m_mf_diff_coeff;   ///< Cell-centred D(x) from phase map

    // --- Problem configuration ---
    std::string m_resultspath;
    const int m_phase;
    const OpenImpala::Direction m_dir;
    amrex::Real m_vlo;
    amrex::Real m_vhi;
    const amrex::Real m_vf; ///< Original total VF (reference)
    bool m_write_plotfile;
    int m_verbose;

    // --- Multi-phase configuration ---
    std::map<int, amrex::Real> m_phase_coeff_map; ///< Phase ID -> transport coefficient
    bool m_is_multi_phase = false;

    // --- Cached results ---
    amrex::Real m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool m_first_call = true;
    amrex::Real m_active_vf = 0.0; ///< Volume fraction of percolating phase

    // --- Solver diagnostics (set by subclass solve()) ---
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

#endif // TortuositySolverBase_H
