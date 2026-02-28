// tests/tMultiPhaseTransport.cpp
//
// Synthetic multi-phase transport coefficient test.
//
// Creates a domain in memory (no TIFF file required) and validates
// the multi-phase tortuosity calculation against known analytical results.
//
// Test scenarios (selected via inputs):
//   num_phases_fill=1: Uniform phase field (all cells = phase 0)
//   num_phases_fill=2: Alternating layers of phase 0 and phase 1
//
// With appropriate D values, both cases can yield tau = 1.0 for validation.

#include "TortuosityHypre.H"
#include "Tortuosity.H"

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
#include <algorithm>
#include <vector>

#include <HYPRE.h>
#include <mpi.h>

namespace {

OpenImpala::TortuosityHypre::SolverType stringToSolverType(const std::string& solver_str) {
    std::string s = solver_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "jacobi")
        return OpenImpala::TortuosityHypre::SolverType::Jacobi;
    if (s == "gmres")
        return OpenImpala::TortuosityHypre::SolverType::GMRES;
    if (s == "flexgmres")
        return OpenImpala::TortuosityHypre::SolverType::FlexGMRES;
    if (s == "pcg")
        return OpenImpala::TortuosityHypre::SolverType::PCG;
    if (s == "bicgstab")
        return OpenImpala::TortuosityHypre::SolverType::BiCGSTAB;
    if (s == "smg")
        return OpenImpala::TortuosityHypre::SolverType::SMG;
    if (s == "pfmg")
        return OpenImpala::TortuosityHypre::SolverType::PFMG;
    amrex::Abort("Invalid solver string: '" + solver_str + "'.");
    return OpenImpala::TortuosityHypre::SolverType::FlexGMRES;
}

OpenImpala::Direction stringToDirection(const std::string& dir_str) {
    std::string s = dir_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "x")
        return OpenImpala::Direction::X;
    if (s == "y")
        return OpenImpala::Direction::Y;
    if (s == "z")
        return OpenImpala::Direction::Z;
    amrex::Abort("Invalid direction string: " + dir_str + ". Use X, Y, or Z.");
    return OpenImpala::Direction::X;
}

} // anonymous namespace


int main(int argc, char* argv[]) {
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
        fprintf(stderr, "FATAL: HYPRE_Init() failed with code %d\n", hypre_ierr);
        return 1;
    }

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
        std::string solver_str = "FlexGMRES";
        std::string direction_str = "X";
        amrex::Real expected_tau = 1.0;
        amrex::Real tau_tolerance = 1e-3;
        std::string resultsdir = "./tMultiPhaseTransport_results";

        {
            amrex::ParmParse pp;
            pp.query("domain_size", domain_size);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("num_phases_fill", num_phases_fill);
            pp.query("solver", solver_str);
            pp.query("direction", direction_str);
            pp.query("expected_tau", expected_tau);
            pp.query("tau_tolerance", tau_tolerance);
            pp.query("resultsdir", resultsdir);
        }

        OpenImpala::Direction direction = stringToDirection(direction_str);
        OpenImpala::TortuosityHypre::SolverType solver_type = stringToSolverType(solver_str);

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Multi-Phase Transport Test ---\n";
            amrex::Print() << "  Domain Size:       " << domain_size << "^3\n";
            amrex::Print() << "  Box Size:          " << box_size << "\n";
            amrex::Print() << "  Num Phases Fill:   " << num_phases_fill << "\n";
            amrex::Print() << "  Direction:         " << direction_str << "\n";
            amrex::Print() << "  Solver:            " << solver_str << "\n";
            amrex::Print() << "  Expected Tau:      " << expected_tau << "\n";
            amrex::Print() << "  Tau Tolerance:     " << tau_tolerance << "\n";
            amrex::Print() << "---------------------------------\n\n";
        }

        // --- Validate parameters ---
        if (domain_size <= 0) {
            amrex::Abort("Error: 'domain_size' must be positive.");
        }
        if (box_size <= 0) {
            amrex::Abort("Error: 'box_size' must be positive.");
        }
        if (num_phases_fill < 1 || num_phases_fill > 2) {
            amrex::Abort("Error: 'num_phases_fill' must be 1 or 2.");
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
            // Uniform: all cells = phase 0
            mf_phase.setVal(0);
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Phase field: uniform (all cells = phase 0)\n";
            }
        } else {
            // Alternating layers along the solve direction
            int dir_idx = static_cast<int>(direction);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                amrex::Array4<int> const phase_arr = mf_phase.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    int coord = (dir_idx == 0) ? i : (dir_idx == 1) ? j : k;
                    phase_arr(i, j, k, 0) = (coord % 2 == 0) ? 0 : 1;
                });
            }
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Phase field: alternating layers along " << direction_str
                               << " (phases 0 and 1)\n";
            }
        }
        mf_phase.FillBoundary(geom.periodicity());

        // VF = 1.0 for uniform material (all cells are some active phase)
        amrex::Real vf = 1.0;

        // --- Create results directory ---
        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        // --- Construct TortuosityHypre ---
        std::unique_ptr<OpenImpala::TortuosityHypre> tort;
        try {
            tort = std::make_unique<OpenImpala::TortuosityHypre>(
                geom, ba, dm, mf_phase, vf, 0 /* phase_id */, direction, solver_type, resultsdir,
                0.0 /* vlo */, 1.0 /* vhi */, verbose, false /* write_plotfile */);
        } catch (const std::exception& e) {
            test_passed = false;
            fail_reason = "TortuosityHypre construction failed: " + std::string(e.what());
        } catch (...) {
            test_passed = false;
            fail_reason = "Unknown exception during TortuosityHypre construction.";
        }

        // --- Validate multi-phase API ---
        if (test_passed && tort) {
            amrex::ParmParse pp_tort("tortuosity");
            amrex::Vector<int> active_phases_check;
            pp_tort.queryarr("active_phases", active_phases_check);

            bool expect_multi_phase = !active_phases_check.empty();
            if (tort->isMultiPhase() != expect_multi_phase) {
                test_passed = false;
                fail_reason = "isMultiPhase() returned " +
                              std::string(tort->isMultiPhase() ? "true" : "false") +
                              " but expected " + std::string(expect_multi_phase ? "true" : "false");
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Multi-phase API check:    PASS (isMultiPhase()="
                               << (tort->isMultiPhase() ? "true" : "false") << ")\n";
            }

            // Validate coefficient map contents
            if (expect_multi_phase && test_passed) {
                const auto& coeff_map = tort->getPhaseCoeffMap();
                amrex::Vector<amrex::Real> phase_diffs_check;
                pp_tort.queryarr("phase_diffusivities", phase_diffs_check);

                if (coeff_map.size() != active_phases_check.size()) {
                    test_passed = false;
                    fail_reason = "Phase coefficient map size mismatch: got " +
                                  std::to_string(coeff_map.size()) + ", expected " +
                                  std::to_string(active_phases_check.size());
                } else {
                    for (size_t idx = 0; idx < active_phases_check.size(); ++idx) {
                        auto it = coeff_map.find(active_phases_check[idx]);
                        if (it == coeff_map.end()) {
                            test_passed = false;
                            fail_reason = "Phase " + std::to_string(active_phases_check[idx]) +
                                          " not found in coefficient map";
                            break;
                        }
                        if (std::abs(it->second - phase_diffs_check[idx]) > 1e-12) {
                            test_passed = false;
                            fail_reason = "Coefficient mismatch for phase " +
                                          std::to_string(active_phases_check[idx]) + ": got " +
                                          std::to_string(it->second) + ", expected " +
                                          std::to_string(phase_diffs_check[idx]);
                            break;
                        }
                    }
                    if (test_passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << " Coefficient map check:    PASS\n";
                    }
                }
            }
        }

        // --- Calculate tortuosity ---
        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        if (test_passed && tort) {
            try {
                actual_tau = tort->value();
                if (std::isnan(actual_tau) || std::isinf(actual_tau)) {
                    test_passed = false;
                    fail_reason =
                        "Tortuosity value is NaN or Inf (indicates solver or calculation failure)";
                }
            } catch (const std::exception& e) {
                test_passed = false;
                fail_reason = "Exception during tortuosity calculation: " + std::string(e.what());
            } catch (...) {
                test_passed = false;
                fail_reason = "Unknown exception during tortuosity calculation.";
            }
        }

        // --- Check solver convergence ---
        if (test_passed && tort) {
            if (!tort->getSolverConverged()) {
                test_passed = false;
                fail_reason = "Solver did not converge";
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
                              ", Diff: " + std::to_string(diff) +
                              ", Tolerance: " + std::to_string(tau_tolerance);
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Tortuosity value check:   PASS (tau=" << actual_tau
                               << ", expected=" << expected_tau << ", diff=" << diff << ")\n";
            }
        }

        // --- Check active volume fraction ---
        if (test_passed && tort) {
            amrex::Real active_vf = tort->getActiveVolumeFraction();
            // For uniform material, all cells should be active
            if (active_vf < 0.99) {
                test_passed = false;
                fail_reason =
                    "Active volume fraction unexpectedly low: " + std::to_string(active_vf) +
                    " (expected ~1.0)";
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Active VF check:          PASS (active_vf=" << active_vf
                               << ")\n";
            }
        }

        // --- Final summary ---
        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time,
                                                 amrex::ParallelDescriptor::IOProcessorNumber());

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n Run time = " << stop_time << " sec\n";
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
            amrex::Abort("MultiPhaseTransport Test FAILED.");
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
