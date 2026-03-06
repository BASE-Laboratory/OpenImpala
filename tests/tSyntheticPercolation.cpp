// tests/tSyntheticPercolation.cpp
//
// Synthetic data test for OpenImpala::PercolationCheck.
//
// Creates phase fields in memory with known connectivity and validates:
//   - Fully connected domain: percolates = true, activeVF = 1.0
//   - Blocked domain (wall in middle): percolates = false
//   - Phase not present: percolates = false
//   - Directional tests: connected in X but not Y (or vice versa)
//   - Active volume fraction correctness

#include "PercolationCheck.H"
#include "Tortuosity.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
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

namespace {

struct TestStatus {
    bool passed = true;
    std::string fail_reason;

    void recordFail(const std::string& reason)
    {
        passed = false;
        fail_reason = reason;
    }
};

} // anonymous namespace


int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        TestStatus status;
        int verbose = 1;
        int domain_size = 16;
        int box_size = 8;

        {
            amrex::ParmParse pp;
            pp.query("verbose", verbose);
            pp.query("domain_size", domain_size);
            pp.query("box_size", box_size);
        }

        // Setup geometry and grid
        amrex::Box domain_box(amrex::IntVect(0, 0, 0),
                              amrex::IntVect(domain_size - 1, domain_size - 1,
                                             domain_size - 1));
        amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                          {AMREX_D_DECL(amrex::Real(domain_size),
                                        amrex::Real(domain_size),
                                        amrex::Real(domain_size))});
        amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
        amrex::Geometry geom;
        geom.define(domain_box, &rb, 0, is_periodic.data());

        amrex::BoxArray ba(domain_box);
        ba.maxSize(box_size);
        amrex::DistributionMapping dm(ba);

        // ================================================================
        // Test 1: Fully connected (all cells = phase 0) → percolates = true
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 1);
            mf.setVal(0);
            mf.FillBoundary(geom.periodicity());

            OpenImpala::PercolationCheck perc(geom, ba, dm, mf, 0,
                                              OpenImpala::Direction::X, verbose);

            if (!perc.percolates()) {
                status.recordFail("Test 1: uniform domain should percolate");
            }

            amrex::Real avf = perc.activeVolumeFraction();
            if (std::abs(avf - 1.0) > 0.01) {
                status.recordFail("Test 1: activeVF = " + std::to_string(avf) +
                                  ", expected ~1.0");
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 1 (fully connected): PASS (activeVF="
                               << avf << ")\n";
            }
        }

        // ================================================================
        // Test 2: Blocked — wall of phase 1 cutting through the middle
        //         Phase 0 cannot reach from X-low to X-high
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 1);
            mf.setVal(0); // All cells = phase 0

            int wall_pos = domain_size / 2;

            // Place a wall of phase 1 at x = wall_pos
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    if (i == wall_pos) {
                        arr(i, j, k, 0) = 1; // Solid wall
                    }
                });
            }
            mf.FillBoundary(geom.periodicity());

            OpenImpala::PercolationCheck perc(geom, ba, dm, mf, 0,
                                              OpenImpala::Direction::X, verbose);

            if (perc.percolates()) {
                status.recordFail("Test 2: blocked domain should NOT percolate in X");
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2 (X-blocked):       PASS\n";
            }
        }

        // ================================================================
        // Test 3: Phase not present → percolates = false
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 1);
            mf.setVal(0);
            mf.FillBoundary(geom.periodicity());

            // Check percolation for phase 5 (not in the data)
            OpenImpala::PercolationCheck perc(geom, ba, dm, mf, 5,
                                              OpenImpala::Direction::X, verbose);

            if (perc.percolates()) {
                status.recordFail("Test 3: absent phase should not percolate");
            }

            amrex::Real avf = perc.activeVolumeFraction();
            if (std::abs(avf) > 1e-12) {
                status.recordFail("Test 3: activeVF = " + std::to_string(avf) +
                                  ", expected 0.0");
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 3 (absent phase):    PASS\n";
            }
        }

        // ================================================================
        // Test 4: Percolates in Y direction but blocked in X
        //         Phase 0 forms columns along Y axis
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 1);
            mf.setVal(1); // All solid

            // Create Y-direction columns at x=0, all z
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    // Open column at i=0, all j, all k
                    if (i == 0) {
                        arr(i, j, k, 0) = 0;
                    }
                });
            }
            mf.FillBoundary(geom.periodicity());

            // Should percolate in Y
            OpenImpala::PercolationCheck perc_y(geom, ba, dm, mf, 0,
                                                OpenImpala::Direction::Y, verbose);
            if (!perc_y.percolates()) {
                status.recordFail("Test 4: Y-column should percolate in Y");
            }

            // Should NOT percolate in X (only at i=0, doesn't reach i=N-1)
            OpenImpala::PercolationCheck perc_x(geom, ba, dm, mf, 0,
                                                OpenImpala::Direction::X, verbose);
            if (perc_x.percolates()) {
                status.recordFail("Test 4: Y-column should NOT percolate in X");
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 4 (directional):     PASS\n";
            }
        }

        // ================================================================
        // Test 5: All directions — symmetric uniform domain
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 1);
            mf.setVal(0);
            mf.FillBoundary(geom.periodicity());

            for (int d = 0; d < 3; ++d) {
                OpenImpala::Direction dir = static_cast<OpenImpala::Direction>(d);
                OpenImpala::PercolationCheck perc(geom, ba, dm, mf, 0, dir,
                                                  verbose);
                if (!perc.percolates()) {
                    status.recordFail("Test 5: uniform domain should percolate in "
                                      "direction " + std::to_string(d));
                    break;
                }
                amrex::Real avf = perc.activeVolumeFraction();
                if (std::abs(avf - 1.0) > 0.01) {
                    status.recordFail("Test 5: dir " + std::to_string(d) +
                                      " activeVF = " + std::to_string(avf));
                    break;
                }
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 5 (all directions):  PASS\n";
            }
        }

        // ================================================================
        // Final summary
        // ================================================================
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (status.passed) {
                amrex::Print() << "\n--- TEST RESULT: PASS ---\n";
            } else {
                amrex::Print() << "\n--- TEST RESULT: FAIL ---\n";
                amrex::Print() << "  Reason: " << status.fail_reason << "\n";
            }
        }

        if (!status.passed) {
            amrex::Abort("tSyntheticPercolation Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
