/** @file tSyntheticEffectiveDiffusivity.cpp
 *  @brief Synthetic analytical test for EffectiveDiffusivityHypre.
 *
 *  Creates a uniform in-memory domain (no TIFF file required) and validates
 *  the cell-problem corrector chi_k against the known analytical result.
 *
 *  For a uniform single-phase medium with D=1.0 and periodic boundaries,
 *  the cell-problem corrector chi_k = 0 everywhere, which implies the
 *  effective diffusivity tensor equals the material diffusivity (D_eff = D*I).
 *
 *  This test exercises:
 *    - EffectiveDiffusivityHypre constructor and matrix setup
 *    - FlexGMRES solver path (solve in 3 directions)
 *    - getChiSolution() retrieval
 *    - Convergence diagnostics
 *    - Active mask generation for single-phase domains
 */

#include "EffectiveDiffusivityHypre.H"
#include "Tortuosity.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_Loop.H>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

#include <HYPRE.h>
#include <mpi.h>


int main(int argc, char* argv[]) {
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
        fprintf(stderr, "FATAL: HYPRE_Init() failed with code %d\n", hypre_ierr);
        return 1;
    }

    amrex::Initialize(argc, argv);
    {
        bool test_passed = true;
        std::string fail_reason;

        // --- Configuration via ParmParse ---
        int domain_size = 16;
        int box_size = 16;
        int verbose = 1;
        amrex::Real chi_tolerance = 1e-6;
        std::string resultsdir = "./tSyntheticEffDiff_results";

        {
            amrex::ParmParse pp;
            pp.query("domain_size", domain_size);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("chi_tolerance", chi_tolerance);
            pp.query("resultsdir", resultsdir);
        }

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Synthetic Effective Diffusivity Test ---\n";
            amrex::Print() << "  Domain Size:       " << domain_size << "^3\n";
            amrex::Print() << "  Box Size:          " << box_size << "\n";
            amrex::Print() << "  Chi Tolerance:     " << chi_tolerance << "\n";
            amrex::Print() << "--------------------------------------------\n\n";
        }

        // --- Create synthetic uniform domain ---
        amrex::Box domain_box(amrex::IntVect(0, 0, 0),
                              amrex::IntVect(domain_size - 1, domain_size - 1, domain_size - 1));
        amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                          {AMREX_D_DECL(amrex::Real(domain_size), amrex::Real(domain_size),
                                        amrex::Real(domain_size))});
        // Periodic boundaries (required for cell-problem homogenization)
        amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
        amrex::Geometry geom;
        geom.define(domain_box, &rb, 0, is_periodic.data());

        amrex::BoxArray ba(domain_box);
        ba.maxSize(box_size);
        amrex::DistributionMapping dm(ba);

        // Uniform phase field: all cells = phase 0
        amrex::iMultiFab mf_phase(ba, dm, 1, 1);
        mf_phase.setVal(0);
        mf_phase.FillBoundary(geom.periodicity());

        // Create results directory
        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        // --- Solve cell problems in all 3 directions ---
        std::vector<OpenImpala::Direction> directions = {
            OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};
        std::vector<std::string> dir_names = {"X", "Y", "Z"};

        for (int d = 0; d < 3 && test_passed; ++d) {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Solving cell problem for chi_" << dir_names[d] << "...\n";
            }

            std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> solver;
            try {
                solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                    geom, ba, dm, mf_phase, 0 /* phase_id */, directions[d],
                    OpenImpala::EffectiveDiffusivityHypre::SolverType::FlexGMRES, resultsdir,
                    verbose, false /* write_plotfile */);
            } catch (const std::exception& e) {
                test_passed = false;
                fail_reason = "EffDiffHypre construction failed for " + dir_names[d] + ": " +
                              std::string(e.what());
                break;
            }

            // Check convergence
            if (!solver->getSolverConverged()) {
                test_passed = false;
                fail_reason =
                    "Solver did not converge for chi_" + dir_names[d] + " (iterations=" +
                    std::to_string(solver->getSolverIterations()) +
                    ", residual=" + std::to_string(solver->getFinalRelativeResidualNorm()) + ")";
                break;
            }

            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "   Converged in " << solver->getSolverIterations()
                               << " iterations (residual="
                               << solver->getFinalRelativeResidualNorm() << ")\n";
            }

            // --- Validate chi_k ≈ 0 for uniform medium ---
            // For a uniform material, the corrector field should be zero everywhere.
            amrex::MultiFab mf_chi(ba, dm, 1, 1);
            mf_chi.setVal(0.0);
            solver->getChiSolution(mf_chi);

            amrex::Real chi_max = mf_chi.norm0(0);
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "   ||chi_" << dir_names[d] << "||_inf = " << chi_max
                               << " (expected ~0)\n";
            }

            if (chi_max > chi_tolerance) {
                test_passed = false;
                fail_reason = "chi_" + dir_names[d] + " is not near zero: ||chi||_inf = " +
                              std::to_string(chi_max) + " > tolerance " +
                              std::to_string(chi_tolerance);
                break;
            }

            // Check that active mask marks all cells as active
            const amrex::iMultiFab& active_mask = solver->getActiveMask();
            long long total_active = 0;
            for (amrex::MFIter mfi(active_mask); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                amrex::Array4<const int> const mask_arr = active_mask.const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    if (mask_arr(i, j, k, 0) == 1) {
                        total_active++;
                    }
                });
            }
            amrex::ParallelDescriptor::ReduceLongLongSum(total_active);
            long long expected_active = domain_box.numPts();
            if (total_active != expected_active) {
                test_passed = false;
                fail_reason = "Active mask count mismatch for " + dir_names[d] + ": got " +
                              std::to_string(total_active) + ", expected " +
                              std::to_string(expected_active);
                break;
            }

            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "   Active cells: " << total_active << "/" << expected_active
                               << " PASS\n";
            }

            // Clean up solver before next iteration
            solver.reset();
        }

        // --- Final summary ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (test_passed) {
                amrex::Print() << "\n----------------------------------------\n";
                amrex::Print() << "--- TEST RESULT: PASS ---\n";
                amrex::Print() << "----------------------------------------\n";
            } else {
                amrex::Print() << "\n-------------------------\n";
                amrex::Print() << "--- TEST RESULT: FAIL ---\n";
                amrex::Print() << "  Reason: " << fail_reason << "\n";
                amrex::Print() << "-------------------------\n";
            }
        }

        if (!test_passed) {
            amrex::Abort("SyntheticEffectiveDiffusivity Test FAILED.");
        }
    }
    amrex::Finalize();

    hypre_ierr = HYPRE_Finalize();
    if (hypre_ierr != 0) {
        fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
        return 1;
    }
    return 0;
}
