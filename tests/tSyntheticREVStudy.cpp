/** @file tSyntheticREVStudy.cpp
 *  @brief Synthetic test for the REV convergence study module.
 *
 *  Creates a uniform in-memory domain and runs an REV study with small
 *  sub-volumes.  For a uniform single-phase medium the effective diffusivity
 *  tensor should be close to the identity (D_eff ≈ I) at all sub-volume sizes,
 *  so the diagonal components should be near 1.0 and off-diagonals near 0.0.
 *
 *  This test exercises:
 *    - REVConfig construction and parameter passing
 *    - runREVStudy() full pipeline (sub-volume extraction, solve, CSV output)
 *    - Empty sizes early-exit path
 *    - CSV file creation and content validation
 *    - Tensor symmetry and magnitude checks
 */

#include "REVStudy.H"
#include "Tortuosity.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Loop.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_iMultiFab.H>

#include <HYPRE.h>
#include <mpi.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

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

int main(int argc, char* argv[]) {
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
        fprintf(stderr, "FATAL: HYPRE_Init() failed with code %d\n", hypre_ierr);
        return 1;
    }

    amrex::Initialize(argc, argv);
    {
        TestStatus status;

        // --- Configuration ---
        int domain_size = 16;
        int box_size = 16;
        int verbose = 1;
        amrex::Real diag_tolerance = 0.15;
        amrex::Real offdiag_tolerance = 0.1;
        std::string resultsdir = "./tSyntheticREVStudy_results";

        {
            amrex::ParmParse pp;
            pp.query("domain_size", domain_size);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("diag_tolerance", diag_tolerance);
            pp.query("offdiag_tolerance", offdiag_tolerance);
            pp.query("resultsdir", resultsdir);
        }

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Synthetic REV Study Test ---\n";
            amrex::Print() << "  Domain Size:       " << domain_size << "^3\n";
            amrex::Print() << "  Box Size:          " << box_size << "\n";
            amrex::Print() << "  Diag Tolerance:    " << diag_tolerance << "\n";
            amrex::Print() << "  OffDiag Tolerance: " << offdiag_tolerance << "\n";
            amrex::Print() << "-------------------------------\n\n";
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

        // Uniform phase field: all cells = phase 0 (active)
        amrex::iMultiFab mf_phase(ba, dm, 1, 1);
        mf_phase.setVal(0);
        mf_phase.FillBoundary(geom.periodicity());

        // Create results directory
        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        // =====================================================================
        // Test 1: Empty sizes — should return without error
        // =====================================================================
        if (status.passed) {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 1: Empty REV sizes (early exit)...\n";
            }

            OpenImpala::REVConfig config_empty;
            config_empty.sizes = {};
            config_empty.num_samples = 1;
            config_empty.phase_id = 0;
            config_empty.box_size = box_size;
            config_empty.verbose = verbose;
            config_empty.results_path = resultsdir;
            config_empty.csv_filename = "rev_empty.csv";

            try {
                OpenImpala::runREVStudy(geom, ba, dm, mf_phase, config_empty);
            } catch (const std::exception& e) {
                status.recordFail(std::string("Empty sizes threw exception: ") + e.what());
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "   PASS\n";
            }
        }

        // =====================================================================
        // Test 2: Single sub-volume on uniform domain
        // =====================================================================
        if (status.passed) {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2: REV study on uniform domain...\n";
            }

            OpenImpala::REVConfig config;
            config.sizes = {12};
            config.num_samples = 1;
            config.phase_id = 0;
            config.box_size = box_size;
            config.verbose = verbose;
            config.write_plotfiles = false;
            config.solver_type = OpenImpala::SolverType::FlexGMRES;
            config.results_path = resultsdir;
            config.csv_filename = "rev_uniform.csv";

            try {
                OpenImpala::runREVStudy(geom, ba, dm, mf_phase, config);
            } catch (const std::exception& e) {
                status.recordFail(std::string("REV study threw exception: ") + e.what());
            }

            // Validate CSV output
            if (status.passed && amrex::ParallelDescriptor::IOProcessor()) {
                std::filesystem::path csv_path =
                    std::filesystem::path(resultsdir) / "rev_uniform.csv";

                if (!std::filesystem::exists(csv_path)) {
                    status.recordFail("CSV file not created: " + csv_path.string());
                } else {
                    std::ifstream csv(csv_path.string());
                    std::string header;
                    std::getline(csv, header);

                    // Check header
                    if (header.find("SampleNo") == std::string::npos ||
                        header.find("D_xx") == std::string::npos) {
                        status.recordFail("CSV header malformed: " + header);
                    }

                    // Read first data row
                    std::string row;
                    if (!std::getline(csv, row) || row.empty()) {
                        status.recordFail("CSV has no data rows");
                    } else {
                        // Parse CSV: SampleNo,SeedX,SeedY,SeedZ,Target,ActX,ActY,ActZ,
                        //            D_xx,D_yy,D_zz,D_xy,D_xz,D_yz
                        std::istringstream ss(row);
                        std::string token;
                        std::vector<std::string> fields;
                        while (std::getline(ss, token, ',')) {
                            fields.push_back(token);
                        }

                        if (fields.size() < 14) {
                            status.recordFail("CSV row has " + std::to_string(fields.size()) +
                                              " fields, expected 14");
                        } else {
                            amrex::Real Dxx = std::stod(fields[8]);
                            amrex::Real Dyy = std::stod(fields[9]);
                            amrex::Real Dzz = std::stod(fields[10]);
                            amrex::Real Dxy = std::stod(fields[11]);
                            amrex::Real Dxz = std::stod(fields[12]);
                            amrex::Real Dyz = std::stod(fields[13]);

                            if (verbose >= 1) {
                                amrex::Print() << "   D_eff diagonal:     (" << Dxx << ", " << Dyy
                                               << ", " << Dzz << ")\n";
                                amrex::Print() << "   D_eff off-diagonal: (" << Dxy << ", " << Dxz
                                               << ", " << Dyz << ")\n";
                            }

                            // For uniform medium, D_eff ≈ I
                            // Sub-volumes with Dirichlet BCs give D_eff = N/(N-1),
                            // but REV uses periodic BCs, so D_eff ≈ 1.0
                            if (std::abs(Dxx - 1.0) > diag_tolerance) {
                                status.recordFail(
                                    "D_xx = " + std::to_string(Dxx) +
                                    ", expected ~1.0 (tol=" + std::to_string(diag_tolerance) + ")");
                            }
                            if (status.passed && std::abs(Dyy - 1.0) > diag_tolerance) {
                                status.recordFail("D_yy = " + std::to_string(Dyy) +
                                                  ", expected ~1.0");
                            }
                            if (status.passed && std::abs(Dzz - 1.0) > diag_tolerance) {
                                status.recordFail("D_zz = " + std::to_string(Dzz) +
                                                  ", expected ~1.0");
                            }

                            // Off-diagonals should be near zero
                            if (status.passed && std::abs(Dxy) > offdiag_tolerance) {
                                status.recordFail("D_xy = " + std::to_string(Dxy) +
                                                  ", expected ~0.0");
                            }
                            if (status.passed && std::abs(Dxz) > offdiag_tolerance) {
                                status.recordFail("D_xz = " + std::to_string(Dxz) +
                                                  ", expected ~0.0");
                            }
                            if (status.passed && std::abs(Dyz) > offdiag_tolerance) {
                                status.recordFail("D_yz = " + std::to_string(Dyz) +
                                                  ", expected ~0.0");
                            }
                        }
                    }
                }
            }

            // Broadcast pass/fail from IO processor
            int pass_int = status.passed ? 1 : 0;
            amrex::ParallelDescriptor::Bcast(&pass_int, 1,
                                             amrex::ParallelDescriptor::IOProcessorNumber());
            if (pass_int == 0 && status.passed) {
                status.recordFail("Failed on IO processor (see log above)");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "   PASS\n";
            }
        }

        // --- Final summary ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (status.passed) {
                amrex::Print() << "\n--- TEST RESULT: PASS ---\n";
            } else {
                amrex::Print() << "\n--- TEST RESULT: FAIL ---\n";
                amrex::Print() << "  Reason: " << status.fail_reason << "\n";
            }
        }

        if (!status.passed) {
            amrex::Abort("SyntheticREVStudy Test FAILED.");
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
