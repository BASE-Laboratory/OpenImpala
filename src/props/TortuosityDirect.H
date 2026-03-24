/** @file TortuosityDirect.H
 *  @brief Legacy iterative tortuosity solver using Forward Euler time-stepping.
 */
#ifndef TORTUOSITYDIRECT_H
#define TORTUOSITYDIRECT_H

#include <AMReX.H> // Includes REAL, IntVect, Box, etc.
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H> // Include Geometry explicitly
#include <AMReX_DistributionMapping.H>
#include <AMReX_Array.H> // For amrex::Array (Needed for m_dxinv)
// #include <AMReX_GpuContainers.H> // <<< REMOVED: No longer needed for GpuArray >>>
#include "Tortuosity.H" // Includes base class and enums under OpenImpala namespace

#include <string> // For std::string
#include <limits> // For numeric_limits

// Forward declaration if needed, although includes seem sufficient
// namespace amrex { class MultiFab; class iMultiFab; ... }

namespace OpenImpala {

/**
 * @brief Computes tortuosity using a simple iterative Finite Volume solver for the Poisson
 * equation.
 *
 * This class implements the Tortuosity interface by solving the Laplace/Poisson equation
 * on the specified phase within the domain using a basic, parallel iterative method.
 * Boundary conditions are applied to drive a flux in the specified direction.
 *
 * Note: The implementation aims for correctness but prioritizes simplicity over
 * high performance compared to specialized solvers (like HYPRE). It serves
 * well as a baseline or for smaller problems.
 *
 * Note: This class stores references to geometry and input data. The caller must
 * ensure these referenced objects remain valid for the lifetime of any
 * TortuosityDirect instance.
 *
 * Note: This class manages internal state (cached value, solver residuals). It is
 * not designed for concurrent access from multiple threads within the same
 * MPI rank. Ensure each instance is used sequentially.
 */
class TortuosityDirect : public OpenImpala::Tortuosity {

public:
    /**
     * @brief Construct a new TortuosityDirect solver instance.
     *
     * @param geom Reference to the problem domain geometry. Must remain valid.
     * @param ba Reference to the BoxArray defining the domain decomposition. Must remain valid.
     * @param dm Reference to the DistributionMapping for parallel distribution. Must remain valid.
     * @param mf_phase Reference to the iMultiFab containing phase identifiers. Requires at least 1
     * ghost cell. Must remain valid.
     * @param phase The integer identifier of the phase for which tortuosity is calculated.
     * @param dir The principal direction (X, Y, or Z) for applying boundary conditions to drive
     * flux.
     * @param eps Convergence criterion (e.g., relative residual norm) for the iterative solver.
     * @param n_steps Maximum number of iterations allowed for the solver. <<< Changed from size_t
     * to int >>>
     * @param plot_interval Frequency for checking residual and optionally plotting (0 disables).
     * <<< ADDED >>>
     * @param plot_basename Base filename for diagnostic plotfiles. <<< ADDED >>>
     * @param vlo Potential value applied at the low boundary in the specified direction.
     * @param vhi Potential value applied at the high boundary in the specified direction.
     */
    TortuosityDirect(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm,
                     const amrex::iMultiFab& mf_phase, // Phase data input
                     const int phase, const OpenImpala::Direction dir, const amrex::Real eps,
                     // const size_t max_steps, // Original was size_t
                     const int n_steps,                // <<< Changed to int >>>
                     const int plot_interval,          // <<< ADDED >>>
                     const std::string& plot_basename, // <<< ADDED >>>
                     const amrex::Real vlo, const amrex::Real vhi);

    // Override the virtual destructor from the base class
    ~TortuosityDirect() override = default; // Default is sufficient if no raw resources owned

    // Delete copy constructor and assignment operator (managing state/references makes copying
    // complex/unsafe)
    TortuosityDirect(const TortuosityDirect&) = delete;
    TortuosityDirect& operator=(const TortuosityDirect&) = delete;

    /**
     * @brief Calculate and return the tortuosity value. Overrides base class method.
     *
     * Computes or retrieves the cached tortuosity. Calls the internal solver if needed.
     *
     * @param refresh If true, forces a recalculation. If false (default), returns cached value if
     * available.
     * @return The calculated tortuosity. Returns cached value if refresh=false and calculation was
     * already done. Returns NaN on failure.
     */
    amrex::Real value(const bool refresh = false) override;

    // --- Solver Diagnostics ---

    /**
     * @brief Returns the number of iterations performed during the last call to solve().
     * @return Number of iterations, or 0 if solve() hasn't been called.
     */
    int getNumIterations() const {
        return m_last_iterations;
    }

    /**
     * @brief Returns the final residual norm achieved during the last call to solve().
     * @return Final residual norm, or a negative value if solve() hasn't been called or failed
     * early.
     */
    amrex::Real getFinalResidual() const {
        return m_last_residual;
    }


    // --- Implementation methods (public for CUDA __device__ lambda compatibility) ---
    bool solve();
    void global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const;
    amrex::Real residual(const amrex::MultiFab& phiold, const amrex::MultiFab& phinew) const;
    void initializeBoundaryConditions();
    void initializeFluxMultiFabs();
    void fillDomainBoundary(amrex::MultiFab& phi, int comp);
    void fillCellTypes(amrex::MultiFab& phi);
    void fillInitialState(amrex::MultiFab& phi);
    void advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new, const amrex::Real& dt);


    // --- Member Variables ---

    // References to external data (caller must ensure lifetime)
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    const amrex::iMultiFab& m_mf_phase; // Input phase data (needs ghost cells)

    // Internal data structures
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_flux; // Stores face fluxes
    amrex::BCRec m_bc; // Describes boundary condition types for phi (comp 0)

    // Phase and Direction info
    const int m_phase;                 // Phase ID to compute tortuosity for
    const OpenImpala::Direction m_dir; // Principal direction for flux

    // Solver Control Parameters (set via constructor)
    const int m_n_steps;     // <<< Changed type >>> Max iterations
    const amrex::Real m_eps; // Convergence tolerance

    // Boundary Condition Values (set via constructor)
    const amrex::Real m_vlo; // Potential at low boundary face in m_dir
    const amrex::Real m_vhi; // Potential at high boundary face in m_dir

    // Caching and Diagnostics
    amrex::Real m_value;         // Cached tortuosity value
    bool m_first_call;           // Flag for caching logic
    int m_last_iterations;       // Iterations used in last solve
    amrex::Real m_last_residual; // Residual achieved in last solve

    // <<< ADDED Missing Member Declarations (from .cpp analysis) >>>
    // FIX 1: Changed GpuArray to Array for CPU compatibility
    amrex::Array<amrex::Real, AMREX_SPACEDIM>
        m_dxinv;                 // Inverse cell sizes (calculated in constructor)
    int m_plot_interval;         // Plotting frequency (from constructor)
    std::string m_plot_basename; // Base name for plotfiles (from constructor)

}; // class TortuosityDirect

} // namespace OpenImpala

#endif // TORTUOSITYDIRECT_H
