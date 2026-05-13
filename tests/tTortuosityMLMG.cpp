// tests/tTortuosityMLMG.cpp
//
// Validates the TortuosityMLMG matrix-free solver against known analytical
// solutions. Uses the same synthetic geometry approach as tMultiPhaseTransport.
//
// Test cases (selected via inputs):
//   uniform:  All cells = phase 0, tau = (N-1)/N
//   twophase: Alternating layers with equal D, tau = (N-1)/N
//
// Masked porous-media coverage lives in python/tests/test_mlmg_porespy.py,
// which runs the actual user-facing facade against a real porespy blob
// structure.

#include "TortuosityMLMG.H"
#include "Tortuosity.H"
#include "SolverConfig.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_Loop.H>

#include <cstdlib>
#include <string>
#include <cmath>
#include <limits>
#include <memory>


int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::Real strt_time = amrex::second();
        bool test_passed = true;
        std::string fail_reason;

        // --- Configuration via ParmParse ---
        int domain_size = 32;
        int box_size = 16;
        int verbose = 1;
        int num_phases_fill = 1;
        std::string direction_str = "X";
        amrex::Real expected_tau = 1.0;
        amrex::Real tau_tolerance = 1e-3;
        std::string resultsdir = "./tTortuosityMLMG_results";

        {
            amrex::ParmParse pp;
            pp.query("domain_size", domain_size);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("num_phases_fill", num_phases_fill);
            pp.query("direction", direction_str);
            pp.query("expected_tau", expected_tau);
            pp.query("tau_tolerance", tau_tolerance);
            pp.query("resultsdir", resultsdir);
        }

        OpenImpala::Direction direction = OpenImpala::parseDirection(direction_str);

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- TortuosityMLMG Test ---\n";
            amrex::Print() << "  Domain Size:       " << domain_size << "^3\n";
            amrex::Print() << "  Direction:         " << direction_str << "\n";
            amrex::Print() << "  Expected Tau:      " << expected_tau << "\n";
            amrex::Print() << "  Tau Tolerance:     " << tau_tolerance << "\n";
            amrex::Print() << "-------------------------------\n\n";
        }

        // --- Create synthetic domain ---
        amrex::Box domain_box(amrex::IntVect(0, 0, 0),
                              amrex::IntVect(domain_size - 1, domain_size - 1, domain_size - 1));
        amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                          {AMREX_D_DECL(amrex::Real(domain_size), amrex::Real(domain_size),
                                        amrex::Real(domain_size))});
        amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
        amrex::Geometry geom;
        geom.define(domain_box, &rb, 0, is_periodic.data());

        amrex::BoxArray ba(domain_box);
        ba.maxSize(box_size);
        amrex::DistributionMapping dm(ba);

        // --- Create and fill phase field ---
        amrex::iMultiFab mf_phase(ba, dm, 1, 1);

        if (num_phases_fill == 1) {
            mf_phase.setVal(0);
        } else {
            // Alternating layers along X
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                amrex::Array4<int> const phase_arr = mf_phase.array(mfi);
                int dir_idx = static_cast<int>(direction);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    int coord = (dir_idx == 0) ? i : (dir_idx == 1) ? j : k;
                    phase_arr(i, j, k, 0) = (coord % 2 == 0) ? 0 : 1;
                });
            }
        }
        mf_phase.FillBoundary(geom.periodicity());

        amrex::Real vf = 1.0;

        // --- Create results directory ---
        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        // --- Construct and run TortuosityMLMG ---
        std::unique_ptr<OpenImpala::TortuosityMLMG> tort;
        try {
            tort = std::make_unique<OpenImpala::TortuosityMLMG>(
                geom, ba, dm, mf_phase, vf, 0 /* phase_id */, direction, resultsdir, 0.0 /* vlo */,
                1.0 /* vhi */, verbose, false /* write_plotfile */);
        } catch (const std::exception& e) {
            test_passed = false;
            fail_reason = "TortuosityMLMG construction failed: " + std::string(e.what());
        }

        // --- Calculate tortuosity ---
        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        if (test_passed && tort) {
            try {
                actual_tau = tort->value();
                if (std::isnan(actual_tau) || std::isinf(actual_tau)) {
                    test_passed = false;
                    fail_reason = "Tortuosity value is NaN or Inf";
                }
            } catch (const std::exception& e) {
                test_passed = false;
                fail_reason = "Exception during solve: " + std::string(e.what());
            }
        }

        // --- Check solver convergence ---
        if (test_passed && tort) {
            if (!tort->getSolverConverged()) {
                test_passed = false;
                fail_reason = "MLMG solver did not converge (residual=" +
                              std::to_string(tort->getFinalRelativeResidualNorm()) + ")";
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Solver convergence:       PASS (" << tort->getSolverIterations()
                               << " iterations, residual=" << tort->getFinalRelativeResidualNorm()
                               << ")\n";
            }
        }

        // --- Validate tortuosity against expected value ---
        if (test_passed) {
            amrex::Real diff = std::abs(actual_tau - expected_tau);
            if (diff > tau_tolerance) {
                test_passed = false;
                fail_reason = "Tortuosity mismatch. Expected: " + std::to_string(expected_tau) +
                              ", Got: " + std::to_string(actual_tau) +
                              ", Diff: " + std::to_string(diff);
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Tortuosity value check:   PASS (tau=" << actual_tau
                               << ", expected=" << expected_tau << ", diff=" << diff << ")\n";
            }
        }

        // --- Check active volume fraction ---
        if (test_passed && tort) {
            amrex::Real active_vf = tort->getActiveVolumeFraction();
            if (active_vf < 0.99) {
                test_passed = false;
                fail_reason =
                    "Active volume fraction unexpectedly low: " + std::to_string(active_vf);
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Active VF check:          PASS (active_vf=" << active_vf
                               << ")\n";
            }
        }

        // --- Check plane flux conservation ---
        if (test_passed && tort) {
            const auto& plane_fluxes = tort->getPlaneFluxes();
            amrex::Real max_dev = tort->getPlaneFluxMaxDeviation();
            constexpr amrex::Real plane_flux_tol = 1.0e-6;

            if (!plane_fluxes.empty() && max_dev > plane_flux_tol) {
                test_passed = false;
                fail_reason =
                    "Plane flux conservation failed. Max deviation: " + std::to_string(max_dev);
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Plane flux conservation:  PASS (" << plane_fluxes.size()
                               << " faces, max_dev=" << std::scientific << max_dev
                               << std::defaultfloat << ")\n";
            }
        }

        // --- Final summary ---
        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time,
                                                 amrex::ParallelDescriptor::IOProcessorNumber());

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n Run time = " << stop_time << " sec\n";
            if (test_passed) {
                amrex::Print() << "\n--- TEST RESULT: PASS ---\n";
            } else {
                amrex::Print() << "\n--- TEST RESULT: FAIL ---\n";
                amrex::Print() << "  Reason: " << fail_reason << "\n";
            }
        }

        if (!test_passed) {
            amrex::Abort("TortuosityMLMG Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
