// --- HypreStructSolver.cpp ---

#include "HypreStructSolver.H"

#include <cmath>
#include <limits>
#include <string>

#include <AMReX_BLassert.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>
#include <mpi.h>

// HYPRE error checking macro (same as in TortuosityHypre.cpp / EffectiveDiffusivityHypre.cpp)
#define HYPRE_CHECK(ierr)                                                                          \
    do {                                                                                           \
        if ((ierr) != 0) {                                                                         \
            char hypre_error_msg[256];                                                             \
            HYPRE_DescribeError(ierr, hypre_error_msg);                                            \
            amrex::Abort("HYPRE Error: " + std::string(hypre_error_msg) +                          \
                         " - Error Code: " + std::to_string(ierr) + " File: " + __FILE__ +         \
                         " Line: " + std::to_string(__LINE__));                                    \
        }                                                                                          \
    } while (0)

namespace OpenImpala {

// ---------------------------------------------------------------------------
// Static utilities
// ---------------------------------------------------------------------------
amrex::Array<HYPRE_Int, AMREX_SPACEDIM> HypreStructSolver::loV(const amrex::Box& b) {
    const int* lo_ptr = b.loVect();
    amrex::Array<HYPRE_Int, AMREX_SPACEDIM> hypre_lo;
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        hypre_lo[i] = static_cast<HYPRE_Int>(lo_ptr[i]);
    return hypre_lo;
}

amrex::Array<HYPRE_Int, AMREX_SPACEDIM> HypreStructSolver::hiV(const amrex::Box& b) {
    const int* hi_ptr = b.hiVect();
    amrex::Array<HYPRE_Int, AMREX_SPACEDIM> hypre_hi;
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
        hypre_hi[i] = static_cast<HYPRE_Int>(hi_ptr[i]);
    return hypre_hi;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
HypreStructSolver::HypreStructSolver(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                     const amrex::DistributionMapping& dm, SolverType solvertype,
                                     amrex::Real eps, int maxiter, int verbose)
    : m_solvertype(solvertype), m_eps(eps), m_maxiter(maxiter), m_verbose(verbose), m_geom(geom),
      m_ba(ba), m_dm(dm) {}

// ---------------------------------------------------------------------------
// Destructor — safely cleans up HYPRE resources
// ---------------------------------------------------------------------------
HypreStructSolver::~HypreStructSolver() {
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        if (m_x)
            HYPRE_StructVectorDestroy(m_x);
        if (m_b)
            HYPRE_StructVectorDestroy(m_b);
        if (m_A)
            HYPRE_StructMatrixDestroy(m_A);
        if (m_stencil)
            HYPRE_StructStencilDestroy(m_stencil);
        if (m_grid)
            HYPRE_StructGridDestroy(m_grid);
    }
    m_x = m_b = nullptr;
    m_A = nullptr;
    m_stencil = nullptr;
    m_grid = nullptr;
}

// ---------------------------------------------------------------------------
// setupGrid — create HYPRE structured grid from AMReX BoxArray
// ---------------------------------------------------------------------------
void HypreStructSolver::setupGrid(bool periodic) {
    HYPRE_Int ierr = 0;
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    HYPRE_CHECK(ierr);

    for (int i = 0; i < m_ba.size(); ++i) {
        if (m_dm[i] == amrex::ParallelDescriptor::MyProc()) {
            amrex::Box bx = m_ba[i];
            auto lo = HypreStructSolver::loV(bx);
            auto hi = HypreStructSolver::hiV(bx);
            if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  setupGrid: Rank " << amrex::ParallelDescriptor::MyProc()
                               << " adding box " << bx << std::endl;
            }
            ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data());
            HYPRE_CHECK(ierr);
        }
    }

    if (periodic) {
        const amrex::Box& domain_box = m_geom.Domain();
        HYPRE_Int periodic_hyp[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (m_geom.isPeriodic(d)) {
                periodic_hyp[d] = static_cast<HYPRE_Int>(domain_box.length(d));
                if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << "  setupGrid: Dim " << d << " is periodic with length "
                                   << periodic_hyp[d] << std::endl;
                }
            } else {
                periodic_hyp[d] = 0;
            }
        }
        ierr = HYPRE_StructGridSetPeriodic(m_grid, periodic_hyp);
        HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  setupGrid: HYPRE_StructGridSetPeriodic called." << std::endl;
        }
    }

    ierr = HYPRE_StructGridAssemble(m_grid);
    HYPRE_CHECK(ierr);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_grid != nullptr,
                                     "m_grid is NULL after HYPRE_StructGridAssemble!");
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupGrid: Assemble complete." << std::endl;
    }
}

// ---------------------------------------------------------------------------
// setupStencil — create 7-point structured stencil
// ---------------------------------------------------------------------------
void HypreStructSolver::setupStencil() {
    HYPRE_Int ierr = 0;
    HYPRE_Int offsets[STENCIL_SIZE][AMREX_SPACEDIM] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
                                                       {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupStencil: Creating " << STENCIL_SIZE << "-point stencil..."
                       << std::endl;
    }
    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, STENCIL_SIZE, &m_stencil);
    HYPRE_CHECK(ierr);
    for (int i = 0; i < STENCIL_SIZE; ++i) {
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]);
        HYPRE_CHECK(ierr);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_stencil != nullptr, "m_stencil is NULL after HYPRE_StructStencilCreate/SetElement!");
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupStencil: Complete." << std::endl;
    }
}

// ---------------------------------------------------------------------------
// createMatrixAndVectors — allocate HYPRE matrix A, RHS b, solution x
// ---------------------------------------------------------------------------
void HypreStructSolver::createMatrixAndVectors() {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_grid != nullptr, "m_grid is NULL in createMatrixAndVectors. Call setupGrid first.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_stencil != nullptr,
        "m_stencil is NULL in createMatrixAndVectors. Call setupStencil first.");

    HYPRE_Int ierr = 0;

    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
    HYPRE_CHECK(ierr);

    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b);
    HYPRE_CHECK(ierr);

    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    HYPRE_CHECK(ierr);

#ifdef OPENIMPALA_USE_GPU
    // When GPU acceleration is enabled, tell HYPRE to allocate matrix and
    // vector data on the device.  The SetBoxValues calls will then expect
    // device pointers (or HYPRE performs the transfer internally depending
    // on build configuration).  HYPRE_MEMORY_DEVICE is defined when HYPRE
    // is built with --with-cuda or --with-hip.
    HYPRE_StructMatrixSetMemoryLocation(m_A, HYPRE_MEMORY_DEVICE);
    HYPRE_StructVectorSetMemoryLocation(m_b, HYPRE_MEMORY_DEVICE);
    HYPRE_StructVectorSetMemoryLocation(m_x, HYPRE_MEMORY_DEVICE);

    // Use a GPU-friendly execution policy for the preconditioner
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  createMatrixAndVectors: HYPRE memory location set to DEVICE."
                       << std::endl;
    }
#endif

    ierr = HYPRE_StructMatrixInitialize(m_A);
    HYPRE_CHECK(ierr);

    ierr = HYPRE_StructVectorInitialize(m_b);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0);
    HYPRE_CHECK(ierr);

    ierr = HYPRE_StructVectorInitialize(m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0);
    HYPRE_CHECK(ierr);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  createMatrixAndVectors: HYPRE A, b, x created and initialized."
                       << std::endl;
    }
}

// ---------------------------------------------------------------------------
// assembleSystem — finalize HYPRE matrix and vectors after box values are set
// ---------------------------------------------------------------------------
void HypreStructSolver::assembleSystem() {
    HYPRE_Int ierr = 0;
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_b);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_x);
    HYPRE_CHECK(ierr);
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  assembleSystem: HYPRE Matrix/Vector assembly complete." << std::endl;
    }
}

// ---------------------------------------------------------------------------
// runSolver — dispatch to the appropriate HYPRE Krylov solver
// ---------------------------------------------------------------------------
bool HypreStructSolver::runSolver(PrecondType precond_type) {
    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = nullptr;
    m_num_iterations = -1;
    m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();
    m_converged = false;

    // Helper: create and configure preconditioner
    auto createPrecond = [&]() {
        if (precond_type == PrecondType::PFMG) {
            ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
            HYPRE_CHECK(ierr);
            HYPRE_StructPFMGSetTol(precond, 0.0);
            HYPRE_StructPFMGSetMaxIter(precond, 1);
            HYPRE_StructPFMGSetNumPreRelax(precond, 1);
            HYPRE_StructPFMGSetNumPostRelax(precond, 1);
            HYPRE_StructPFMGSetPrintLevel(precond, (m_verbose > 3) ? 1 : 0);
        } else {
            ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
            HYPRE_CHECK(ierr);
            HYPRE_StructSMGSetTol(precond, 0.0);
            HYPRE_StructSMGSetMaxIter(precond, 1);
            HYPRE_StructSMGSetNumPreRelax(precond, 1);
            HYPRE_StructSMGSetNumPostRelax(precond, 1);
            HYPRE_StructSMGSetPrintLevel(precond, 0);
        }
    };

    // Helper: destroy preconditioner
    auto destroyPrecond = [&]() {
        if (precond) {
            if (precond_type == PrecondType::PFMG) {
                HYPRE_StructPFMGDestroy(precond);
            } else {
                HYPRE_StructSMGDestroy(precond);
            }
        }
    };

    // Helper: get preconditioner function pointers
    auto getPrecondSolve = [&]() -> HYPRE_PtrToStructSolverFcn {
        return (precond_type == PrecondType::PFMG) ? HYPRE_StructPFMGSolve : HYPRE_StructSMGSolve;
    };
    auto getPrecondSetup = [&]() -> HYPRE_PtrToStructSolverFcn {
        return (precond_type == PrecondType::PFMG) ? HYPRE_StructPFMGSetup : HYPRE_StructSMGSetup;
    };

    const char* precond_name = (precond_type == PrecondType::PFMG) ? "PFMG" : "SMG";

    if (m_solvertype == SolverType::FlexGMRES) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Setting up HYPRE FlexGMRES Solver with " << precond_name
                           << " Preconditioner..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructFlexGMRESSetTol(solver, m_eps);
        HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructFlexGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);

        createPrecond();
        HYPRE_StructFlexGMRESSetPrecond(solver, getPrecondSolve(), getPrecondSetup(), precond);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Running HYPRE_StructFlexGMRESSetup..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Running HYPRE_StructFlexGMRESSolve..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
        if (ierr != 0 && ierr != HYPRE_ERROR_CONV) {
            HYPRE_CHECK(ierr);
        }

        HYPRE_StructFlexGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        m_converged = !(std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm));
        m_converged = m_converged && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);

        if (ierr == HYPRE_ERROR_CONV && !m_converged && m_verbose >= 0) {
            amrex::Warning("HYPRE FlexGMRES solver did not converge within tolerance!");
        } else if (ierr != 0 && ierr != HYPRE_ERROR_CONV && m_verbose >= 0) {
            amrex::Warning("HYPRE FlexGMRES solver returned error code: " + std::to_string(ierr));
        }

        HYPRE_StructFlexGMRESDestroy(solver);
        destroyPrecond();

    } else if (m_solvertype == SolverType::PCG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Setting up HYPRE PCG Solver with " << precond_name
                           << " Preconditioner..." << std::endl;
        }
        ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructPCGSetTol(solver, m_eps);
        HYPRE_StructPCGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 2 : 0);
        HYPRE_StructPCGSetTwoNorm(solver, 1);

        createPrecond();
        HYPRE_StructPCGSetPrecond(solver, getPrecondSolve(), getPrecondSetup(), precond);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Running HYPRE_StructPCGSetup..." << std::endl;
        }
        ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Running HYPRE_StructPCGSolve..." << std::endl;
        }
        ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
        if (ierr != 0 && ierr != HYPRE_ERROR_CONV) {
            HYPRE_CHECK(ierr);
        }

        HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        m_converged = !(std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm));
        m_converged = m_converged && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);

        if (ierr == HYPRE_ERROR_CONV && !m_converged && m_verbose >= 0) {
            amrex::Warning("HYPRE PCG solver did not converge within tolerance!");
        } else if (ierr != 0 && ierr != HYPRE_ERROR_CONV && m_verbose >= 0) {
            amrex::Warning("HYPRE PCG solver returned error code: " + std::to_string(ierr));
        }

        HYPRE_StructPCGDestroy(solver);
        destroyPrecond();

    } else {
        amrex::Abort("Unsupported solver type requested: " +
                     std::to_string(static_cast<int>(m_solvertype)));
    }

    // Post-solve diagnostics
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver iterations: " << m_num_iterations << std::endl;
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << std::scientific
                       << m_final_res_norm << std::defaultfloat << std::endl;
        amrex::Print() << "  Solver Converged: " << (m_converged ? "Yes" : "No") << std::endl;
    }

    if (std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm)) {
        amrex::Warning("HYPRE solve resulted in NaN or Inf residual norm!");
        m_converged = false;
    }

    return m_converged;
}

} // namespace OpenImpala
