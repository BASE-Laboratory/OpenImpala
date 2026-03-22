/** @file HypreStructSolver.H
 *  @brief Base class providing shared HYPRE structured-grid solver infrastructure.
 *
 *  Encapsulates the HYPRE lifecycle (grid, stencil, matrix, vectors), solver dispatch
 *  (FlexGMRES, PCG with SMG/PFMG preconditioners), and cleanup logic that is common
 *  to all structured-grid physics solvers in OpenImpala.
 *
 *  Derived classes (e.g. TortuosityHypre, EffectiveDiffusivityHypre) inherit this base
 *  and provide their own matrix fill kernels, boundary conditions, and post-processing.
 */
#ifndef OPENIMPALA_HYPRE_STRUCT_SOLVER_H_
#define OPENIMPALA_HYPRE_STRUCT_SOLVER_H_

#include "Tortuosity.H" // For SolverType enum

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <HYPRE.h>
#include <HYPRE_struct_ls.h>

#include <limits>

namespace OpenImpala {

/** @brief Preconditioner type for HYPRE Krylov solvers. */
enum class PrecondType {
    SMG, /**< Semicoarsening multigrid — robust, good for anisotropic grids */
    PFMG /**< Parallel semicoarsening multigrid — more scalable for large problems */
};

/** @brief Base class for HYPRE structured-grid solvers.
 *
 *  Provides:
 *  - HYPRE object lifecycle (grid, stencil, matrix, vectors, cleanup)
 *  - 7-point stencil setup
 *  - Solver dispatch (FlexGMRES + preconditioner, PCG + preconditioner)
 *  - Convergence statistics (iterations, residual norm)
 *  - Box-to-HYPRE index conversion utilities
 *
 *  Derived classes call the protected setup/solve methods from their own constructors
 *  and solve routines, and provide their own matrix fill logic.
 */
class HypreStructSolver {
public:
    /** @brief Virtual destructor. Cleans up HYPRE resources safely (guards against
     * post-MPI_Finalize). */
    virtual ~HypreStructSolver();

    // Non-copyable
    HypreStructSolver(const HypreStructSolver&) = delete;
    HypreStructSolver& operator=(const HypreStructSolver&) = delete;

    // --- Solver statistics getters ---
    bool getSolverConverged() const {
        return m_converged;
    }
    amrex::Real getFinalRelativeResidualNorm() const {
        return m_final_res_norm;
    }
    int getSolverIterations() const {
        return m_num_iterations;
    }

    // --- Static utilities ---

    /** @brief Convert AMReX Box lower corner to HYPRE_Int array. */
    static amrex::Array<HYPRE_Int, AMREX_SPACEDIM> loV(const amrex::Box& b);

    /** @brief Convert AMReX Box upper corner to HYPRE_Int array. */
    static amrex::Array<HYPRE_Int, AMREX_SPACEDIM> hiV(const amrex::Box& b);

protected:
    /** @brief Construct base solver with configuration. Does NOT call any HYPRE functions.
     *
     *  Derived classes should call setupGrid(), setupStencil(), etc. from their own
     *  constructors after initializing their physics-specific members.
     */
    HypreStructSolver(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                      const amrex::DistributionMapping& dm, SolverType solvertype, amrex::Real eps,
                      int maxiter, int verbose);

    // --- HYPRE setup methods (called by derived classes) ---

    /** @brief Create HYPRE structured grid from AMReX BoxArray.
     *  @param periodic If true, sets HYPRE periodic BCs using AMReX Geometry periodicity.
     */
    void setupGrid(bool periodic = false);

    /** @brief Create 7-point stencil (center + 6 face neighbours). */
    void setupStencil();

    /** @brief Create and initialize HYPRE matrix and vectors (A, b, x).
     *  Must be called after setupGrid() and setupStencil().
     */
    void createMatrixAndVectors();

    /** @brief Assemble HYPRE matrix and vectors after box values have been set. */
    void assembleSystem();

    /** @brief Run the HYPRE solver specified by m_solvertype.
     *  @param precond_type Preconditioner to use with Krylov solvers.
     *  @return True if the solver converged within tolerance.
     *
     *  Updates m_num_iterations, m_final_res_norm, and m_converged.
     */
    bool runSolver(PrecondType precond_type = PrecondType::SMG);

    // --- HYPRE handles (owned by this base class, cleaned up in destructor) ---
    HYPRE_StructGrid m_grid = nullptr;
    HYPRE_StructStencil m_stencil = nullptr;
    HYPRE_StructMatrix m_A = nullptr;
    HYPRE_StructVector m_b = nullptr;
    HYPRE_StructVector m_x = nullptr;

    // --- Solver configuration ---
    const SolverType m_solvertype;
    amrex::Real m_eps;
    int m_maxiter;
    int m_verbose;

    // --- Solver statistics ---
    HYPRE_Int m_num_iterations = -1;
    amrex::Real m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool m_converged = false;

    // --- AMReX grid info (stored by value to avoid dangling references) ---
    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    // --- Constants ---
    static constexpr int STENCIL_SIZE = 7;
};

} // namespace OpenImpala

#endif // OPENIMPALA_HYPRE_STRUCT_SOLVER_H_
