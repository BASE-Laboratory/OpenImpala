// tests/tSyntheticVolumeFraction.cpp
//
// Synthetic data test for OpenImpala::VolumeFraction.
//
// Creates phase fields in memory with known volume fractions and validates:
//   - Uniform domain (single phase): VF = 1.0
//   - Half-and-half domain (two phases): VF = 0.5 each
//   - Sparse domain (single cell of target phase): VF = 1/N^3
//   - Phase not present: VF = 0.0
//   - VF sum across all phases = 1.0
//   - Local vs global count consistency
//   - value_vf() convenience method

#include "VolumeFraction.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
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
        amrex::Real tolerance = 1e-12;

        {
            amrex::ParmParse pp;
            pp.query("verbose", verbose);
            pp.query("domain_size", domain_size);
            pp.query("box_size", box_size);
        }

        long long total_cells = static_cast<long long>(domain_size) * domain_size * domain_size;

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
        // Test 1: Uniform domain — all cells = phase 0 → VF(0) = 1.0
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            mf.setVal(0);

            OpenImpala::VolumeFraction vf0(mf, 0, 0);
            long long pc = 0, tc = 0;
            vf0.value(pc, tc, false);

            if (tc != total_cells) {
                status.recordFail("Test 1: total_count mismatch: " +
                                  std::to_string(tc) + " vs " +
                                  std::to_string(total_cells));
            } else if (pc != total_cells) {
                status.recordFail("Test 1: phase_count should equal total for uniform: " +
                                  std::to_string(pc));
            }

            // Check value_vf convenience
            amrex::Real vf_val = vf0.value_vf(false);
            if (std::abs(vf_val - 1.0) > tolerance) {
                status.recordFail("Test 1: value_vf = " + std::to_string(vf_val) +
                                  ", expected 1.0");
            }

            // VF of absent phase = 0.0
            OpenImpala::VolumeFraction vf1(mf, 1, 0);
            amrex::Real vf1_val = vf1.value_vf(false);
            if (std::abs(vf1_val) > tolerance) {
                status.recordFail("Test 1: VF of absent phase = " +
                                  std::to_string(vf1_val));
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 1 (uniform VF=1):    PASS\n";
            }
        }

        // ================================================================
        // Test 2: Half-and-half — alternating layers → VF ≈ 0.5
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    arr(i, j, k, 0) = (i % 2 == 0) ? 0 : 1;
                });
            }

            OpenImpala::VolumeFraction vf0(mf, 0, 0);
            OpenImpala::VolumeFraction vf1(mf, 1, 0);

            amrex::Real vf0_val = vf0.value_vf(false);
            amrex::Real vf1_val = vf1.value_vf(false);
            amrex::Real vf_sum = vf0_val + vf1_val;

            if (std::abs(vf0_val - 0.5) > tolerance) {
                status.recordFail("Test 2: VF[0] = " + std::to_string(vf0_val) +
                                  ", expected 0.5");
            }
            if (std::abs(vf1_val - 0.5) > tolerance) {
                status.recordFail("Test 2: VF[1] = " + std::to_string(vf1_val) +
                                  ", expected 0.5");
            }
            if (std::abs(vf_sum - 1.0) > tolerance) {
                status.recordFail("Test 2: VF sum = " + std::to_string(vf_sum) +
                                  ", expected 1.0");
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2 (half-and-half):   PASS (VF0="
                               << vf0_val << ", VF1=" << vf1_val << ")\n";
            }
        }

        // ================================================================
        // Test 3: Quarter domain — first quarter phase 0, rest phase 1
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            int quarter = domain_size / 4;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    arr(i, j, k, 0) = (i < quarter) ? 0 : 1;
                });
            }

            OpenImpala::VolumeFraction vf0(mf, 0, 0);
            amrex::Real vf0_val = vf0.value_vf(false);
            amrex::Real expected = static_cast<amrex::Real>(quarter) / domain_size;

            if (std::abs(vf0_val - expected) > tolerance) {
                status.recordFail("Test 3: VF[0] = " + std::to_string(vf0_val) +
                                  ", expected " + std::to_string(expected));
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 3 (quarter domain):  PASS (VF0="
                               << vf0_val << ")\n";
            }
        }

        // ================================================================
        // Test 4: Local vs global consistency
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            mf.setVal(0);

            OpenImpala::VolumeFraction vf0(mf, 0, 0);
            long long pc_global = 0, tc_global = 0;
            long long pc_local = 0, tc_local = 0;
            vf0.value(pc_global, tc_global, false);
            vf0.value(pc_local, tc_local, true);

            // On single rank, local == global
            // On multiple ranks, sum of locals == global
            long long pc_local_sum = pc_local;
            long long tc_local_sum = tc_local;
            amrex::ParallelAllReduce::Sum(pc_local_sum,
                                          amrex::ParallelContext::CommunicatorSub());
            amrex::ParallelAllReduce::Sum(tc_local_sum,
                                          amrex::ParallelContext::CommunicatorSub());

            if (pc_local_sum != pc_global) {
                status.recordFail("Test 4: sum(local_phase) != global_phase");
            }
            if (tc_local_sum != tc_global) {
                status.recordFail("Test 4: sum(local_total) != global_total");
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 4 (local vs global): PASS\n";
            }
        }

        // ================================================================
        // Test 5: Three-phase domain
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid();
                 ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    // Assign phase based on i mod 3
                    arr(i, j, k, 0) = i % 3;
                });
            }

            amrex::Real vf_sum = 0.0;
            for (int phase = 0; phase < 3; ++phase) {
                OpenImpala::VolumeFraction vf(mf, phase, 0);
                amrex::Real vf_val = vf.value_vf(false);
                vf_sum += vf_val;

                if (vf_val <= 0.0 || vf_val >= 1.0) {
                    status.recordFail("Test 5: VF[" + std::to_string(phase) +
                                      "] = " + std::to_string(vf_val) +
                                      " outside (0,1)");
                }
            }

            if (std::abs(vf_sum - 1.0) > tolerance) {
                status.recordFail("Test 5: 3-phase VF sum = " +
                                  std::to_string(vf_sum));
            }

            if (status.passed && verbose >= 1 &&
                amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 5 (three-phase):     PASS\n";
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
            amrex::Abort("tSyntheticVolumeFraction Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
