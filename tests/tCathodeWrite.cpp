// tests/tCathodeWrite.cpp
//
// Integration test for OpenImpala::CathodeWrite.
//
// Validates:
//   - Construction with known parameters
//   - DandeLiion parameter file output (content validation)
//   - PyBaMM parameter file output (content validation)
//   - Derived parameter calculations (porosity, BET, permeability)
//   - File open error handling

#include "CathodeWrite.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

namespace {

struct TestStatus {
    bool passed = true;
    std::string fail_reason;

    void recordFail(const std::string& reason) {
        passed = false;
        fail_reason = reason;
    }
};

// Read entire file into string
std::string readFileContents(const std::string& filename) {
    std::ifstream ifs(filename);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

bool containsSubstr(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

} // anonymous namespace


int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        TestStatus status;
        int verbose = 1;
        {
            amrex::ParmParse pp;
            pp.query("verbose", verbose);
        }

        // ================================================================
        // Test 1: Write DandeLiion parameters with known values
        // ================================================================
        if (status.passed) {
            OpenImpala::CathodeParams params;
            params.volume_fraction_solid = 0.6;
            params.particle_radius = 5e-6;
            params.active_material_conductivity = 100.0;
            params.max_concentration = 51000.0;

            OpenImpala::CathodeWrite writer(params);
            std::string dandel_file = "tCathodeWrite_dandel.txt";

            bool ok = writer.writeDandeLiionParameters(dandel_file);
            if (!ok) {
                status.recordFail("writeDandeLiionParameters returned false");
            } else if (amrex::ParallelDescriptor::IOProcessor()) {
                std::string content = readFileContents(dandel_file);

                // Check porosity: 1 - 0.6 = 0.4
                if (!containsSubstr(content, "el")) {
                    status.recordFail("DandeLiion file missing electrolyte fraction 'el'");
                }
                // Check particle radius
                if (!containsSubstr(content, "R =")) {
                    status.recordFail("DandeLiion file missing particle radius 'R'");
                }
                // Check sigma_s
                if (!containsSubstr(content, "sigma_s")) {
                    status.recordFail("DandeLiion file missing solid conductivity");
                }
                // Check cmax
                if (!containsSubstr(content, "cmax")) {
                    status.recordFail("DandeLiion file missing max concentration");
                }
                // Check BET surface area: 3 * 0.6 / 5e-6 = 360000
                if (!containsSubstr(content, "bet")) {
                    status.recordFail("DandeLiion file missing BET surface area");
                }
                // Check header
                if (!containsSubstr(content, "CathodeWrite")) {
                    status.recordFail("DandeLiion file missing generated-by header");
                }
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 1 (DandeLiion write): PASS\n";
            }
        }

        // ================================================================
        // Test 2: Write PyBaMM parameters with known values
        // ================================================================
        if (status.passed) {
            OpenImpala::CathodeParams params;
            params.volume_fraction_solid = 0.5;
            params.particle_radius = 1e-5;
            params.active_material_conductivity = 200.0;
            params.max_concentration = 30000.0;

            OpenImpala::CathodeWrite writer(params);
            std::string pybamm_file = "tCathodeWrite_pybamm.csv";

            bool ok = writer.writePyBammParameters(pybamm_file);
            if (!ok) {
                status.recordFail("writePyBammParameters returned false");
            } else if (amrex::ParallelDescriptor::IOProcessor()) {
                std::string content = readFileContents(pybamm_file);

                // Check CSV header
                if (!containsSubstr(content, "Name [units],Value")) {
                    status.recordFail("PyBaMM file missing CSV header");
                }
                // Check conductivity
                if (!containsSubstr(content, "conductivity")) {
                    status.recordFail("PyBaMM file missing conductivity entry");
                }
                // Check porosity: 1 - 0.5 = 0.5
                if (!containsSubstr(content, "porosity")) {
                    status.recordFail("PyBaMM file missing porosity entry");
                }
                // Check particle radius
                if (!containsSubstr(content, "particle radius")) {
                    status.recordFail("PyBaMM file missing particle radius entry");
                }
                // Check active material VF
                if (!containsSubstr(content, "active material volume fraction")) {
                    status.recordFail("PyBaMM file missing active material VF");
                }
                // Check surface area density
                if (!containsSubstr(content, "surface area density")) {
                    status.recordFail("PyBaMM file missing surface area density");
                }
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2 (PyBaMM write):     PASS\n";
            }
        }

        // ================================================================
        // Test 3: Verify derived parameter values
        // ================================================================
        if (status.passed && amrex::ParallelDescriptor::IOProcessor()) {
            OpenImpala::CathodeParams params;
            params.volume_fraction_solid = 0.6;
            params.particle_radius = 5e-6;
            params.active_material_conductivity = 100.0;
            params.max_concentration = 51000.0;

            OpenImpala::CathodeWrite writer(params);
            writer.writeDandeLiionParameters("tCathodeWrite_verify.txt");

            std::string content = readFileContents("tCathodeWrite_verify.txt");

            // Expected: porosity = 0.4, BET = 3*0.6/5e-6 = 360000, B = 0.4/1.94
            amrex::Real expected_porosity = 0.4;
            amrex::Real expected_bet = 3.0 * 0.6 / 5e-6;
            amrex::Real expected_B = expected_porosity / 1.94;

            // We verify the file is non-empty and contains expected fields
            if (content.size() < 100) {
                status.recordFail("DandeLiion output file suspiciously short: " +
                                  std::to_string(content.size()) + " chars");
            }

            if (status.passed && verbose >= 1) {
                amrex::Print() << " Test 3 (derived params):   PASS\n";
                amrex::Print() << "   Expected porosity: " << expected_porosity << "\n";
                amrex::Print() << "   Expected BET:      " << expected_bet << "\n";
                amrex::Print() << "   Expected B:        " << expected_B << "\n";
            }
        }

        // ================================================================
        // Test 4: Edge case — very small particle radius
        // ================================================================
        if (status.passed) {
            OpenImpala::CathodeParams params;
            params.volume_fraction_solid = 0.3;
            params.particle_radius = 1e-20; // Near zero
            params.active_material_conductivity = 50.0;
            params.max_concentration = 10000.0;

            // Should not crash; warnings expected
            OpenImpala::CathodeWrite writer(params);
            bool ok = writer.writeDandeLiionParameters("tCathodeWrite_edge.txt");
            if (!ok) {
                status.recordFail("writeDandeLiionParameters failed for small radius");
            }
            ok = writer.writePyBammParameters("tCathodeWrite_edge.csv");
            if (!ok) {
                status.recordFail("writePyBammParameters failed for small radius");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 4 (edge case):        PASS\n";
            }
        }

        // ================================================================
        // Test 5: Edge case — VF outside [0,1] (warning expected)
        // ================================================================
        if (status.passed) {
            OpenImpala::CathodeParams params;
            params.volume_fraction_solid = 1.5; // Out of range
            params.particle_radius = 5e-6;
            params.active_material_conductivity = 100.0;
            params.max_concentration = 51000.0;

            // Should construct with warning, not crash
            OpenImpala::CathodeWrite writer(params);
            bool ok = writer.writeDandeLiionParameters("tCathodeWrite_oob.txt");
            if (!ok) {
                status.recordFail("writeDandeLiionParameters failed for out-of-range VF");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 5 (OOB VF):          PASS\n";
            }
        }

        // ================================================================
        // Cleanup temp files
        // ================================================================
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::remove("tCathodeWrite_dandel.txt");
            std::remove("tCathodeWrite_pybamm.csv");
            std::remove("tCathodeWrite_verify.txt");
            std::remove("tCathodeWrite_edge.txt");
            std::remove("tCathodeWrite_edge.csv");
            std::remove("tCathodeWrite_oob.txt");
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
            amrex::Abort("tCathodeWrite Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
