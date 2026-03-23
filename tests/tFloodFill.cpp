// tests/tFloodFill.cpp
//
// Direct tests for the shared FloodFill utility (FloodFill.H/cpp).
//
// Validates:
//   1. Full flood on uniform domain — all cells reachable from a single seed
//   2. Partial flood — two isolated blocks, only seeded block is reached
//   3. collectBoundarySeeds — correct inlet/outlet seed collection
//   4. Multi-label flood — two separate floods with distinct labels

#include "FloodFill.H"
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

namespace {

struct TestStatus {
    bool passed = true;
    std::string fail_reason;

    void recordFail(const std::string& reason) {
        if (passed) {
            passed = false;
            fail_reason = reason;
        }
    }
};

} // anonymous namespace


int main(int argc, char* argv[]) {
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

        const int N = domain_size;

        // Setup geometry and grid
        amrex::Box domain_box(amrex::IntVect(0, 0, 0), amrex::IntVect(N - 1, N - 1, N - 1));
        amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                          {AMREX_D_DECL(amrex::Real(N), amrex::Real(N), amrex::Real(N))});
        amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
        amrex::Geometry geom;
        geom.define(domain_box, &rb, 0, is_periodic.data());

        amrex::BoxArray ba(domain_box);
        ba.maxSize(box_size);
        amrex::DistributionMapping dm(ba);

        // ================================================================
        // Test 1: Full flood on uniform domain
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf_phase(ba, dm, 1, 1);
            mf_phase.setVal(0); // all phase 0
            mf_phase.FillBoundary(geom.periodicity());

            amrex::iMultiFab mask(ba, dm, 1, 1);
            mask.setVal(OpenImpala::FLOOD_INACTIVE);

            amrex::Vector<amrex::IntVect> seeds = {amrex::IntVect(0, 0, 0)};
            OpenImpala::parallelFloodFill(mask, mf_phase, 0, seeds, geom, verbose);

            // Count reachable cells
            long long reached = 0;
            for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                const auto arr = mask.const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    if (arr(i, j, k, 0) == OpenImpala::FLOOD_ACTIVE) {
                        reached++;
                    }
                });
            }
            amrex::ParallelAllReduce::Sum(reached, amrex::ParallelContext::CommunicatorSub());

            long long expected = static_cast<long long>(N) * N * N;
            if (reached != expected) {
                status.recordFail("Test 1 (full flood): reached=" + std::to_string(reached) +
                                  ", expected=" + std::to_string(expected));
            }
            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 1 (full flood):       PASS (" << reached << "/" << expected
                               << " cells)\n";
            }
        }

        // ================================================================
        // Test 2: Partial flood — two isolated blocks
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf_phase(ba, dm, 1, 1);
            mf_phase.setVal(1); // background = phase 1

            int cube_size = 3;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                auto arr = mf_phase.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    // Block A at corner (1,1,1)
                    if (i >= 1 && i < 1 + cube_size && j >= 1 && j < 1 + cube_size && k >= 1 &&
                        k < 1 + cube_size) {
                        arr(i, j, k, 0) = 0;
                    }
                    // Block B at far corner, separated by gap
                    int lo2 = N - 1 - cube_size;
                    if (i >= lo2 && i < lo2 + cube_size && j >= lo2 && j < lo2 + cube_size &&
                        k >= lo2 && k < lo2 + cube_size) {
                        arr(i, j, k, 0) = 0;
                    }
                });
            }
            mf_phase.FillBoundary(geom.periodicity());

            // Seed only in block A
            amrex::iMultiFab mask(ba, dm, 1, 1);
            mask.setVal(OpenImpala::FLOOD_INACTIVE);

            amrex::Vector<amrex::IntVect> seeds = {amrex::IntVect(1, 1, 1)};
            OpenImpala::parallelFloodFill(mask, mf_phase, 0, seeds, geom, verbose);

            long long reached = 0;
            for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                const auto arr = mask.const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    if (arr(i, j, k, 0) == OpenImpala::FLOOD_ACTIVE) {
                        reached++;
                    }
                });
            }
            amrex::ParallelAllReduce::Sum(reached, amrex::ParallelContext::CommunicatorSub());

            long long expected_block_vol = static_cast<long long>(cube_size) * cube_size * cube_size;
            if (reached != expected_block_vol) {
                status.recordFail("Test 2 (partial flood): reached=" + std::to_string(reached) +
                                  ", expected=" + std::to_string(expected_block_vol));
            }
            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2 (partial flood):    PASS (" << reached << " cells, "
                               << "only block A)\n";
            }
        }

        // ================================================================
        // Test 3: collectBoundarySeeds
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf_phase(ba, dm, 1, 1);
            mf_phase.setVal(0); // all phase 0
            mf_phase.FillBoundary(geom.periodicity());

            amrex::Vector<amrex::IntVect> inletSeeds, outletSeeds;
            OpenImpala::collectBoundarySeeds(mf_phase, 0, 0 /* X direction */, geom, inletSeeds,
                                             outletSeeds);

            // For X direction: inlet = i=0 face, outlet = i=N-1 face
            // Each face has N*N cells
            long long expected_seeds = static_cast<long long>(N) * N;
            if (static_cast<long long>(inletSeeds.size()) != expected_seeds) {
                status.recordFail(
                    "Test 3 (boundary seeds): inlet seeds=" + std::to_string(inletSeeds.size()) +
                    ", expected=" + std::to_string(expected_seeds));
            }
            if (static_cast<long long>(outletSeeds.size()) != expected_seeds) {
                status.recordFail(
                    "Test 3 (boundary seeds): outlet seeds=" + std::to_string(outletSeeds.size()) +
                    ", expected=" + std::to_string(expected_seeds));
            }

            // Verify all inlet seeds have i=0
            for (const auto& s : inletSeeds) {
                if (s[0] != 0) {
                    status.recordFail("Test 3 (boundary seeds): inlet seed with i=" +
                                      std::to_string(s[0]) + " (expected 0)");
                    break;
                }
            }
            // Verify all outlet seeds have i=N-1
            for (const auto& s : outletSeeds) {
                if (s[0] != N - 1) {
                    status.recordFail("Test 3 (boundary seeds): outlet seed with i=" +
                                      std::to_string(s[0]) + " (expected " + std::to_string(N - 1) +
                                      ")");
                    break;
                }
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 3 (boundary seeds):   PASS (inlet=" << inletSeeds.size()
                               << ", outlet=" << outletSeeds.size() << ")\n";
            }
        }

        // ================================================================
        // Test 4: Multi-label flood (two labels on same mask)
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf_phase(ba, dm, 1, 1);
            mf_phase.setVal(1); // background

            int cube_size = 3;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                auto arr = mf_phase.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    if (i >= 1 && i < 1 + cube_size && j >= 1 && j < 1 + cube_size && k >= 1 &&
                        k < 1 + cube_size) {
                        arr(i, j, k, 0) = 0;
                    }
                    int lo2 = N - 1 - cube_size;
                    if (i >= lo2 && i < lo2 + cube_size && j >= lo2 && j < lo2 + cube_size &&
                        k >= lo2 && k < lo2 + cube_size) {
                        arr(i, j, k, 0) = 0;
                    }
                });
            }
            mf_phase.FillBoundary(geom.periodicity());

            amrex::iMultiFab mask(ba, dm, 1, 1);
            mask.setVal(OpenImpala::FLOOD_INACTIVE);

            // Flood block A with label=1
            amrex::Vector<amrex::IntVect> seeds_a = {amrex::IntVect(1, 1, 1)};
            OpenImpala::parallelFloodFill(mask, mf_phase, 0, seeds_a, geom, verbose, 1);

            // Flood block B with label=2
            int lo2 = N - 1 - cube_size;
            amrex::Vector<amrex::IntVect> seeds_b = {amrex::IntVect(lo2, lo2, lo2)};
            OpenImpala::parallelFloodFill(mask, mf_phase, 0, seeds_b, geom, verbose, 2);

            long long count_1 = 0, count_2 = 0;
            for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                const auto arr = mask.const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    if (arr(i, j, k, 0) == 1) count_1++;
                    if (arr(i, j, k, 0) == 2) count_2++;
                });
            }
            amrex::ParallelAllReduce::Sum(count_1, amrex::ParallelContext::CommunicatorSub());
            amrex::ParallelAllReduce::Sum(count_2, amrex::ParallelContext::CommunicatorSub());

            long long expected_vol = static_cast<long long>(cube_size) * cube_size * cube_size;
            if (count_1 != expected_vol) {
                status.recordFail("Test 4 (multi-label): label 1 count=" + std::to_string(count_1) +
                                  ", expected=" + std::to_string(expected_vol));
            }
            if (count_2 != expected_vol) {
                status.recordFail("Test 4 (multi-label): label 2 count=" + std::to_string(count_2) +
                                  ", expected=" + std::to_string(expected_vol));
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 4 (multi-label):      PASS (label1=" << count_1
                               << ", label2=" << count_2 << ")\n";
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
            amrex::Abort("tFloodFill Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
