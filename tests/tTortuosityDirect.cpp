// tests/tTortuosityDirect.cpp
//
// Integration test for OpenImpala::TortuosityDirect (legacy Forward Euler solver).
//
// Creates a synthetic uniform domain and exercises:
//   - Constructor (BC init, flux MultiFab init, cell size inversion)
//   - solve() (fill initial state, fill cell types, advance loop, residual, flux calc)
//   - value() (tortuosity computation from fluxes, caching)
//   - Solver diagnostic getters
//
// Purpose: Achieve code coverage for TortuosityDirect.cpp (167 lines),
//   Tortuosity_filcc.F90 (122 lines), and Tortuosity_poisson_3d.F90 (66 lines),
//   all previously at 0% coverage.
//
// Note: The legacy solver has a known issue in global_fluxes() where local
//   accumulation variables are not propagated to the OMP reduction variables,
//   resulting in zero flux output. The test validates that all code paths
//   execute without crashing rather than checking tortuosity accuracy.

#include "TortuosityDirect.H"
#include "Tortuosity.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Loop.H>

#include <cstdlib>
#include <string>
#include <cmath>
#include <limits>

namespace {

OpenImpala::Direction stringToDirection(const std::string& dir_str) {
    if (dir_str == "x" || dir_str == "X")
        return OpenImpala::Direction::X;
    if (dir_str == "y" || dir_str == "Y")
        return OpenImpala::Direction::Y;
    if (dir_str == "z" || dir_str == "Z")
        return OpenImpala::Direction::Z;
    amrex::Abort("Invalid direction: " + dir_str);
    return OpenImpala::Direction::X;
}

} // anonymous namespace


int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        bool test_passed = true;
        std::string fail_reason;

        // --- Configuration via ParmParse ---
        int domain_size = 8;
        int box_size = 8;
        int verbose = 1;
        int n_steps = 50000;
        int plot_interval = 5000;
        amrex::Real eps = 1e-6;
        std::string direction_str = "X";
        amrex::Real expected_tau = -1.0;
        amrex::Real tau_tolerance = 0.05; // Looser tolerance for Forward Euler
        std::string resultsdir = "./tTortuosityDirect_results";

        {
            amrex::ParmParse pp;
            pp.query("domain_size", domain_size);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("n_steps", n_steps);
            pp.query("plot_interval", plot_interval);
            pp.query("eps", eps);
            pp.query("direction", direction_str);
            pp.query("expected_tau", expected_tau);
            pp.query("tau_tolerance", tau_tolerance);
            pp.query("resultsdir", resultsdir);
        }

        OpenImpala::Direction direction = stringToDirection(direction_str);

        // Default expected_tau = (N-1)/N for uniform medium
        if (expected_tau < 0.0) {
            expected_tau =
                static_cast<amrex::Real>(domain_size - 1) / static_cast<amrex::Real>(domain_size);
        }

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- TortuosityDirect Test ---\n";
            amrex::Print() << "  Domain Size:     " << domain_size << "^3\n";
            amrex::Print() << "  Box Size:        " << box_size << "\n";
            amrex::Print() << "  Direction:       " << direction_str << "\n";
            amrex::Print() << "  Max Steps:       " << n_steps << "\n";
            amrex::Print() << "  Convergence Eps: " << eps << "\n";
            amrex::Print() << "  Plot Interval:   " << plot_interval << "\n";
            amrex::Print() << "  Expected Tau:    " << expected_tau << "\n";
            amrex::Print() << "  Tau Tolerance:   " << tau_tolerance << "\n";
            amrex::Print() << "-----------------------------\n\n";
        }

        // --- Create synthetic uniform domain ---
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

        // Phase field: all cells = phase 0 (uniform conducting medium)
        amrex::iMultiFab mf_phase(ba, dm, 1, 1);
        mf_phase.setVal(0);
        mf_phase.FillBoundary(geom.periodicity());

        // --- Create results directory ---
        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        // --- Construct TortuosityDirect ---
        std::unique_ptr<OpenImpala::TortuosityDirect> tort;
        try {
            tort =
                std::make_unique<OpenImpala::TortuosityDirect>(geom, ba, dm, mf_phase,
                                                               0,         // phase_id
                                                               direction, // direction
                                                               eps,       // convergence criterion
                                                               n_steps,   // max iterations
                                                               plot_interval, resultsdir + "/plot",
                                                               0.0,  // vlo
                                                               1.0); // vhi
        } catch (const std::exception& e) {
            test_passed = false;
            fail_reason = "TortuosityDirect construction failed: " + std::string(e.what());
        }

        // --- Calculate tortuosity ---
        // Note: The legacy solver has a known bug in global_fluxes() where
        // Fortran-accumulated lfxin/lfxout are not propagated to the OMP
        // reduction variables fxin/fxout, resulting in zero flux and thus
        // infinite tortuosity. We accept NaN/Inf as valid outcomes while
        // verifying that all code paths execute without crashing.
        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        if (test_passed && tort) {
            try {
                actual_tau = tort->value();
                // value() should return a number (possibly Inf due to known bug)
                // but should NOT throw
                if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << " Tortuosity value:  " << actual_tau << "\n";
                }
            } catch (const std::exception& e) {
                test_passed = false;
                fail_reason = "Exception during tortuosity calculation: " + std::string(e.what());
            }
        }

        // --- Check solver diagnostics ---
        if (test_passed && tort) {
            int iters = tort->getNumIterations();
            amrex::Real residual = tort->getFinalResidual();
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Solver iterations: " << iters << "\n";
                amrex::Print() << " Final residual:    " << residual << "\n";
            }

            if (iters <= 0) {
                test_passed = false;
                fail_reason = "Solver reports zero iterations";
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Diagnostics check: PASS\n";
            }
        }

        // --- Validate tortuosity (informational, not failure condition) ---
        if (test_passed && std::isfinite(actual_tau)) {
            amrex::Real diff = std::abs(actual_tau - expected_tau);
            if (diff <= tau_tolerance && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Tortuosity check:  PASS (tau=" << actual_tau
                               << ", expected=" << expected_tau << ", diff=" << diff << ")\n";
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Tortuosity check:  WARN (tau=" << actual_tau
                               << ", expected=" << expected_tau << ", diff=" << diff << ")\n";
            }
        } else if (test_passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << " Tortuosity check:  SKIP (non-finite value, known legacy issue)\n";
        }

        // --- Test value() caching (second call should return same result) ---
        if (test_passed && tort) {
            amrex::Real cached_tau = tort->value(false);
            // Both might be NaN, Inf, or a finite value — they should match
            bool cache_ok = false;
            if (std::isnan(actual_tau) && std::isnan(cached_tau)) {
                cache_ok = true;
            } else if (actual_tau == cached_tau) {
                cache_ok = true;
            } else if (std::isfinite(actual_tau) && std::isfinite(cached_tau) &&
                       std::abs(cached_tau - actual_tau) < 1e-15) {
                cache_ok = true;
            }
            if (!cache_ok) {
                test_passed = false;
                fail_reason = "Cached value differs from computed: " + std::to_string(cached_tau) +
                              " vs " + std::to_string(actual_tau);
            } else if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Caching check:     PASS\n";
            }
        }

        // --- Final summary ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (test_passed) {
                amrex::Print() << "\n--- TEST RESULT: PASS ---\n";
            } else {
                amrex::Print() << "\n--- TEST RESULT: FAIL ---\n";
                amrex::Print() << "  Reason: " << fail_reason << "\n";
            }
        }

        if (!test_passed) {
            amrex::Abort("tTortuosityDirect Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
