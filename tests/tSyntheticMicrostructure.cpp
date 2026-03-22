// tests/tSyntheticMicrostructure.cpp
//
// Synthetic data tests for the Microstructural Parameterization Engine (Issue #170).
//
// Validates:
//   1. SpecificSurfaceArea: uniform (0), half-split (1/N), alternating layers
//   2. MacroGeometry: dimension extraction
//   3. ThroughThicknessProfile: step-function profile on half-split domain
//   4. ConnectedComponents: known number of components
//   5. ParticleSizeDistribution: known radii from cube volumes

#include "SpecificSurfaceArea.H"
#include "MacroGeometry.H"
#include "ThroughThicknessProfile.H"
#include "ConnectedComponents.H"
#include "ParticleSizeDistribution.H"
#include "VolumeFraction.H"
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
        amrex::Real tolerance = 1e-10;

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
        // Test 1: SSA — Uniform domain → SSA = 0
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            mf.setVal(0);

            OpenImpala::SpecificSurfaceArea ssa(geom, mf, 0, 1);
            amrex::Real ssa_val = ssa.value_ssa(false);

            if (std::abs(ssa_val) > tolerance) {
                status.recordFail("Test 1 (SSA uniform): expected 0, got " +
                                  std::to_string(ssa_val));
            }
            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 1 (SSA uniform=0):   PASS\n";
            }
        }

        // ================================================================
        // Test 2: SSA — Half-and-half split along X → SSA = N^2 / N^3 = 1/N
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            int half = N / 2;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(
                    bx, [&](int i, int j, int k) { arr(i, j, k, 0) = (i < half) ? 0 : 1; });
            }

            OpenImpala::SpecificSurfaceArea ssa(geom, mf, 0, 1);
            long long face_count = 0, total_count = 0;
            ssa.value(face_count, total_count, false);

            // There should be exactly N*N interface faces (one plane at i=half-1/half)
            long long expected_faces = static_cast<long long>(N) * N;
            if (face_count != expected_faces) {
                status.recordFail(
                    "Test 2 (SSA half-split): face_count=" + std::to_string(face_count) +
                    ", expected=" + std::to_string(expected_faces));
            }

            amrex::Real expected_ssa =
                static_cast<amrex::Real>(expected_faces) / static_cast<amrex::Real>(N * N * N);
            amrex::Real ssa_val = ssa.value_ssa(false);
            if (std::abs(ssa_val - expected_ssa) > tolerance) {
                status.recordFail("Test 2 (SSA half-split): ssa=" + std::to_string(ssa_val) +
                                  ", expected=" + std::to_string(expected_ssa));
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2 (SSA half=1/N):    PASS (SSA=" << ssa_val << ")\n";
            }
        }

        // ================================================================
        // Test 2b: SSA — Corrected SSA = (2/3) * raw for half-split
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            int half = N / 2;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(
                    bx, [&](int i, int j, int k) { arr(i, j, k, 0) = (i < half) ? 0 : 1; });
            }

            OpenImpala::SpecificSurfaceArea ssa(geom, mf, 0, 1);
            amrex::Real raw_ssa = ssa.value_ssa(false);
            amrex::Real corrected_ssa = ssa.value_corrected(false);
            amrex::Real expected_corrected = (2.0 / 3.0) * raw_ssa;

            if (std::abs(corrected_ssa - expected_corrected) > tolerance) {
                status.recordFail("Test 2b (SSA corrected): got " + std::to_string(corrected_ssa) +
                                  ", expected " + std::to_string(expected_corrected));
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2b (SSA corrected): PASS (raw=" << raw_ssa
                               << " corrected=" << corrected_ssa << ")\n";
            }
        }

        // ================================================================
        // Test 2c: SSA boundary padding — cube touching domain face
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            mf.setVal(0); // all phase 0

            // Place a small cube of phase 1 touching the low-X domain face
            int cube_sz = 3;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    // Cube at (0, N/2-1, N/2-1) touching low-X face
                    if (i < cube_sz && j >= N / 2 - 1 && j < N / 2 - 1 + cube_sz &&
                        k >= N / 2 - 1 && k < N / 2 - 1 + cube_sz) {
                        arr(i, j, k, 0) = 1;
                    }
                });
            }

            // With padding=0, the cube touching the boundary has interface faces
            OpenImpala::SpecificSurfaceArea ssa_no_pad(geom, mf, 0, 1, 0, 0);
            long long fc_no_pad = 0, tc_no_pad = 0;
            ssa_no_pad.value(fc_no_pad, tc_no_pad, false);

            // With padding=1, the outermost layer is excluded.
            // The cube at x=0..2 loses its x=0 layer from counting.
            OpenImpala::SpecificSurfaceArea ssa_pad(geom, mf, 0, 1, 0, 1);
            long long fc_pad = 0, tc_pad = 0;
            ssa_pad.value(fc_pad, tc_pad, false);

            // Padding should reduce both face count and total count
            if (fc_pad >= fc_no_pad) {
                status.recordFail(
                    "Test 2c (SSA padding): padded face_count=" + std::to_string(fc_pad) +
                    " should be < unpadded=" + std::to_string(fc_no_pad));
            }
            if (tc_pad >= tc_no_pad) {
                status.recordFail(
                    "Test 2c (SSA padding): padded total_count=" + std::to_string(tc_pad) +
                    " should be < unpadded=" + std::to_string(tc_no_pad));
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2c (SSA padding):   PASS (faces: " << fc_no_pad << " -> "
                               << fc_pad << ", cells: " << tc_no_pad << " -> " << tc_pad << ")\n";
            }
        }

        // ================================================================
        // Test 2d: SSA — Diagonal plane (i+j+k < N) demonstrates laddering
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(
                    bx, [&](int i, int j, int k) { arr(i, j, k, 0) = (i + j + k < N) ? 0 : 1; });
            }

            OpenImpala::SpecificSurfaceArea ssa(geom, mf, 0, 1);
            amrex::Real raw_ssa = ssa.value_ssa(false);
            amrex::Real corrected_ssa = ssa.value_corrected(false);

            // For a diagonal plane, raw SSA overestimates. The Cauchy-Crofton
            // correction (2/3 factor) should bring it closer to the true value.
            // We verify: corrected < raw, and both are positive.
            if (corrected_ssa >= raw_ssa || raw_ssa <= 0.0) {
                status.recordFail(
                    "Test 2d (SSA diagonal): corrected=" + std::to_string(corrected_ssa) +
                    " should be < raw=" + std::to_string(raw_ssa));
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2d (SSA diagonal):  PASS (raw=" << std::fixed
                               << std::setprecision(6) << raw_ssa << " corrected=" << corrected_ssa
                               << ")\n";
            }
        }

        // ================================================================
        // Test 3: MacroGeometry — verify dimensions
        // ================================================================
        if (status.passed) {
            auto mg = OpenImpala::MacroGeometry::fromGeometry(geom, 2); // Z direction
            if (mg.nx != N || mg.ny != N || mg.nz != N) {
                status.recordFail("Test 3 (MacroGeometry): dimension mismatch");
            }
            if (std::abs(mg.thickness - N) > tolerance) {
                status.recordFail("Test 3 (MacroGeometry): thickness=" +
                                  std::to_string(mg.thickness) + ", expected=" + std::to_string(N));
            }
            amrex::Real expected_cs = static_cast<amrex::Real>(N) * N;
            if (std::abs(mg.cross_section - expected_cs) > tolerance) {
                status.recordFail("Test 3 (MacroGeometry): cross_section mismatch");
            }
            amrex::Real expected_vol = static_cast<amrex::Real>(N) * N * N;
            if (std::abs(mg.total_volume - expected_vol) > tolerance) {
                status.recordFail("Test 3 (MacroGeometry): volume mismatch");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 3 (MacroGeometry):   PASS (Lp=" << mg.thickness
                               << " A=" << mg.cross_section << ")\n";
            }
        }

        // ================================================================
        // Test 4: ThroughThicknessProfile — half-split along Z
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 0);
            int half = N / 2;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(
                    bx, [&](int i, int j, int k) { arr(i, j, k, 0) = (k < half) ? 0 : 1; });
            }

            OpenImpala::ThroughThicknessProfile profile(geom, mf, 0, OpenImpala::Direction::Z);
            const auto& vf = profile.volumeFractionProfile();

            if (profile.numSlices() != N) {
                status.recordFail("Test 4 (TTP): numSlices=" + std::to_string(profile.numSlices()) +
                                  ", expected=" + std::to_string(N));
            }

            // First half should be VF=1.0, second half VF=0.0
            for (int s = 0; s < N && status.passed; ++s) {
                amrex::Real expected = (s < half) ? 1.0 : 0.0;
                if (std::abs(vf[s] - expected) > tolerance) {
                    status.recordFail("Test 4 (TTP): slice " + std::to_string(s) +
                                      " VF=" + std::to_string(vf[s]) +
                                      ", expected=" + std::to_string(expected));
                }
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 4 (ThroughThickness): PASS (" << profile.numSlices()
                               << " slices, step-function verified)\n";
            }
        }

        // ================================================================
        // Test 5: ConnectedComponents — two separated blocks
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 1);
            mf.setVal(0); // background = phase 0

            // Place two small cubes of phase 1, separated by at least one cell gap
            int cube_size = 3;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    // Cube 1: corner at (1,1,1) with size cube_size
                    if (i >= 1 && i < 1 + cube_size && j >= 1 && j < 1 + cube_size && k >= 1 &&
                        k < 1 + cube_size) {
                        arr(i, j, k, 0) = 1;
                    }
                    // Cube 2: corner at (N-1-cube_size, N-1-cube_size, N-1-cube_size)
                    int lo2 = N - 1 - cube_size;
                    if (i >= lo2 && i < lo2 + cube_size && j >= lo2 && j < lo2 + cube_size &&
                        k >= lo2 && k < lo2 + cube_size) {
                        arr(i, j, k, 0) = 1;
                    }
                });
            }
            mf.FillBoundary(geom.periodicity());

            OpenImpala::ConnectedComponents ccl(geom, ba, dm, mf, 1, verbose);

            if (ccl.numComponents() != 2) {
                status.recordFail("Test 5 (CCL): numComponents=" +
                                  std::to_string(ccl.numComponents()) + ", expected=2");
            }

            // Each cube should have volume cube_size^3
            long long expected_vol = static_cast<long long>(cube_size) * cube_size * cube_size;
            const auto& vols = ccl.componentVolumes();
            for (int c = 0; c < ccl.numComponents() && status.passed; ++c) {
                if (vols[c] != expected_vol) {
                    status.recordFail("Test 5 (CCL): component " + std::to_string(c) +
                                      " volume=" + std::to_string(vols[c]) +
                                      ", expected=" + std::to_string(expected_vol));
                }
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 5 (CCL 2 cubes):     PASS (components="
                               << ccl.numComponents() << ", vol=" << vols[0] << ")\n";
            }

            // ============================================================
            // Test 6: PSD — equivalent radius from known cube volumes
            // ============================================================
            if (status.passed) {
                OpenImpala::ParticleSizeDistribution psd(ccl);

                if (psd.numParticles() != 2) {
                    status.recordFail("Test 6 (PSD): numParticles=" +
                                      std::to_string(psd.numParticles()) + ", expected=2");
                }

                amrex::Real expected_radius =
                    std::cbrt(3.0 * static_cast<double>(expected_vol) / (4.0 * M_PI));

                for (int p = 0; p < psd.numParticles() && status.passed; ++p) {
                    if (std::abs(psd.radii()[p] - expected_radius) > 1e-6) {
                        status.recordFail("Test 6 (PSD): radius[" + std::to_string(p) +
                                          "]=" + std::to_string(psd.radii()[p]) +
                                          ", expected=" + std::to_string(expected_radius));
                    }
                }

                if (std::abs(psd.meanRadius() - expected_radius) > 1e-6) {
                    status.recordFail(
                        "Test 6 (PSD): meanRadius=" + std::to_string(psd.meanRadius()) +
                        ", expected=" + std::to_string(expected_radius));
                }

                if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << " Test 6 (PSD radii):       PASS (mean_R=" << std::fixed
                                   << std::setprecision(4) << psd.meanRadius() << ")\n";
                }
            }
        }

        // ================================================================
        // Test 7: ConnectedComponents — single connected block
        // ================================================================
        if (status.passed) {
            amrex::iMultiFab mf(ba, dm, 1, 1);
            mf.setVal(0);

            // L-shaped block: two adjacent cubes sharing a face
            int cs = 3;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.growntilebox();
                auto arr = mf.array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    // Cube A: (1..cs, 1..cs, 1..cs)
                    if (i >= 1 && i < 1 + cs && j >= 1 && j < 1 + cs && k >= 1 && k < 1 + cs) {
                        arr(i, j, k, 0) = 1;
                    }
                    // Cube B: adjacent to A in X direction: (1+cs..1+2*cs, 1..cs, 1..cs)
                    if (i >= 1 + cs && i < 1 + 2 * cs && j >= 1 && j < 1 + cs && k >= 1 &&
                        k < 1 + cs) {
                        arr(i, j, k, 0) = 1;
                    }
                });
            }
            mf.FillBoundary(geom.periodicity());

            OpenImpala::ConnectedComponents ccl(geom, ba, dm, mf, 1, verbose);

            if (ccl.numComponents() != 1) {
                status.recordFail("Test 7 (CCL touching): numComponents=" +
                                  std::to_string(ccl.numComponents()) + ", expected=1");
            }

            long long expected_vol = 2 * static_cast<long long>(cs) * cs * cs;
            if (!ccl.componentVolumes().empty() && ccl.componentVolumes()[0] != expected_vol) {
                status.recordFail(
                    "Test 7 (CCL touching): volume=" + std::to_string(ccl.componentVolumes()[0]) +
                    ", expected=" + std::to_string(expected_vol));
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 7 (CCL touching):    PASS (1 component, vol="
                               << ccl.componentVolumes()[0] << ")\n";
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
            amrex::Abort("tSyntheticMicrostructure Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
