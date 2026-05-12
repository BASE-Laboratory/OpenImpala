// tests/tTortuosityMLMG.cpp
//
// Validates the TortuosityMLMG matrix-free solver against known analytical
// solutions. Uses the same synthetic geometry approach as tMultiPhaseTransport.
//
// Test cases (selected via inputs, num_phases_fill):
//   1 = uniform:  All cells = phase 0, tau = (N-1)/N
//   2 = twophase: Alternating layers with equal D, tau = (N-1)/N
//   3 = masked-porous: Pseudo-random ~60% porosity mask with a guaranteed
//                       through-channel at j=k=N/2. Asserts MLMG returns a
//                       finite tortuosity and survives the boundary flux
//                       conservation guard — the regression case for the
//                       eps=1e-9 -> eps=1e-11 tightening (PR landing this
//                       comment). On masked geometry MLMG's relative tol is
//                       referenced to ||r_initial|| ~ O(1) rather than the
//                       Dirichlet-RHS norm HYPRE uses, so a residual that
//                       reads as "converged" can still leave per-cell errors
//                       large enough to fail the 1e-4 boundary flux guard.

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
        amrex::Real tau_min = 1.0;
        amrex::Real tau_max = 100.0;
        bool skip_tau_value_check = false;
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
            pp.query("tau_min", tau_min);
            pp.query("tau_max", tau_max);
            pp.query("skip_tau_value_check", skip_tau_value_check);
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
        } else if (num_phases_fill == 2) {
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
        } else {
            // Pseudo-random masked-porous geometry with a guaranteed through-
            // channel in the flow direction at the centre of the cross-section.
            // Deterministic hash so the test is bit-reproducible across runs
            // and platforms. ~60% phase 0 (active), ~40% phase 1 (masked).
            const int N = domain_size;
            const int dir_idx = static_cast<int>(direction);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                amrex::Array4<int> const phase_arr = mf_phase.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    // Through-channel: width-2 tube along the flow axis at the
                    // centre of the perpendicular plane. Guarantees percolation
                    // regardless of the random pattern.
                    int u, v;
                    if (dir_idx == 0) {
                        u = j;
                        v = k;
                    } else if (dir_idx == 1) {
                        u = i;
                        v = k;
                    } else {
                        u = i;
                        v = j;
                    }
                    const int cu = N / 2;
                    const int cv = N / 2;
                    if (std::abs(u - cu) <= 1 && std::abs(v - cv) <= 1) {
                        phase_arr(i, j, k, 0) = 0;
                        return;
                    }
                    // Cheap deterministic hash → ~60% phase 0, 40% phase 1.
                    unsigned int h = static_cast<unsigned int>(i * 73856093u
                                                               ^ j * 19349663u
                                                               ^ k * 83492791u);
                    h = (h ^ (h >> 13)) * 1274126177u;
                    h = h ^ (h >> 16);
                    phase_arr(i, j, k, 0) = ((h % 100u) < 60u) ? 0 : 1;
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
        if (test_passed && !skip_tau_value_check) {
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
        } else if (test_passed) {
            // Range check used by the masked-porous regression case where no
            // closed-form tau exists. The relevant assertion is that MLMG
            // returns a finite, plausible value rather than NaN-ing out of
            // the boundary flux guard.
            if (actual_tau < tau_min || actual_tau > tau_max) {
                test_passed = false;
                fail_reason = "Tortuosity out of range. Got: " + std::to_string(actual_tau) +
                              ", expected in [" + std::to_string(tau_min) + ", " +
                              std::to_string(tau_max) + "]";
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Tortuosity range check:   PASS (tau=" << actual_tau
                               << " in [" << tau_min << ", " << tau_max << "])\n";
            }
        }

        // --- Check active volume fraction ---
        // Only meaningful for fully-active (uniform / two-layer) geometries.
        // The masked-porous regression case deliberately has ~60% active and
        // its assertion is the flux guard survival, not the VF.
        if (test_passed && tort && num_phases_fill <= 2) {
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
        // Tightness expected scales with how heterogeneous the geometry is:
        // uniform / two-layer ⇒ near machine precision, porous mask ⇒ the
        // global boundary check (1e-4 in TortuositySolverBase::value) governs
        // and per-plane variance is informational only.
        if (test_passed && tort) {
            const auto& plane_fluxes = tort->getPlaneFluxes();
            amrex::Real max_dev = tort->getPlaneFluxMaxDeviation();
            const amrex::Real plane_flux_tol = (num_phases_fill <= 2) ? 1.0e-6 : 1.0e-2;

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
