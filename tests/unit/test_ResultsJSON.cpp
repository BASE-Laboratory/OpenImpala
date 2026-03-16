// Unit tests for ResultsJSON — runs without AMReX, MPI, or HYPRE.
// OPENIMPALA_UNIT_TEST is defined via target_compile_definitions in CMakeLists.txt.
#include "ResultsJSON.H"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

using Catch::Approx;
using namespace OpenImpala;

// ============================================================================
// Helper: build a PhysicsConfig for testing
// ============================================================================
static PhysicsConfig makeConfig(const std::string& type_str, double bulk = 1.0) {
    PhysicsConfig cfg;
    cfg.bulk_property = bulk;
    REQUIRE(PhysicsConfig::fromTypeString(type_str, cfg));
    return cfg;
}

// ============================================================================
// Basic JSON structure
// ============================================================================
TEST_CASE("ResultsJSON produces valid JSON with openimpala block", "[ResultsJSON][structure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setInputFile("test.tif");
    writer.setPhaseId(1);
    writer.setGridInfo(100, 100, 100, 32);
    writer.setSolverInfo("FlexGMRES", true);
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);

    auto j = writer.buildJSON();

    REQUIRE(j.contains("openimpala"));
    auto& oi = j["openimpala"];
    CHECK(oi["version"].get<std::string>() == "0.1.0");
    CHECK(oi["physics_type"].get<std::string>() == "Diffusion");
    CHECK(oi["input_file"].get<std::string>() == "test.tif");
    CHECK(oi["phase_id"].get<int>() == 1);
    CHECK(oi["solver"].get<std::string>() == "FlexGMRES");
    CHECK(oi["converged"].get<bool>() == true);
    CHECK(oi["volume_fraction"].get<double>() == Approx(0.35));
}

TEST_CASE("ResultsJSON grid info is correct", "[ResultsJSON][structure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setGridInfo(128, 64, 256, 16);

    auto j = writer.buildJSON();
    auto& grid = j["openimpala"]["grid"];

    CHECK(grid["nx"].get<int>() == 128);
    CHECK(grid["ny"].get<int>() == 64);
    CHECK(grid["nz"].get<int>() == 256);
    CHECK(grid["box_size"].get<int>() == 16);
}

// ============================================================================
// Direction results
// ============================================================================
TEST_CASE("ResultsJSON includes per-direction results", "[ResultsJSON][results]") {
    ResultsJSON writer;
    auto cfg = makeConfig("diffusion");
    writer.setPhysicsConfig(cfg);
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);
    writer.addDirectionResult("Y", 0.11);
    writer.addDirectionResult("Z", 0.13);

    auto j = writer.buildJSON();
    auto& results = j["openimpala"]["results"];

    REQUIRE(results.contains("X"));
    REQUIRE(results.contains("Y"));
    REQUIRE(results.contains("Z"));

    CHECK(results["X"]["Deff_ratio"].get<double>() == Approx(0.12));
    CHECK(results["Y"]["Deff_ratio"].get<double>() == Approx(0.11));
    CHECK(results["Z"]["Deff_ratio"].get<double>() == Approx(0.13));

    // Check tortuosity is computed correctly: vf / D_eff_ratio
    CHECK(results["X"]["tortuosity"].get<double>() == Approx(0.35 / 0.12));
    CHECK(results["Y"]["tortuosity"].get<double>() == Approx(0.35 / 0.11));
}

TEST_CASE("ResultsJSON omits effective property when bulk == 1.0", "[ResultsJSON][results]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion", 1.0));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);

    auto j = writer.buildJSON();
    auto& x_result = j["openimpala"]["results"]["X"];

    CHECK_FALSE(x_result.contains("EffectiveDiffusivity"));
    CHECK_FALSE(j["openimpala"].contains("bulk_property"));
}

TEST_CASE("ResultsJSON includes effective property when bulk != 1.0", "[ResultsJSON][results]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("electrical_conductivity", 5.96e7));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);

    auto j = writer.buildJSON();
    auto& oi = j["openimpala"];
    auto& x_result = oi["results"]["X"];

    CHECK(oi["bulk_property"].get<double>() == Approx(5.96e7));
    CHECK(oi["bulk_property_label"].get<std::string>() == "sigma_bulk");
    CHECK(x_result["EffectiveConductivity"].get<double>() == Approx(0.12 * 5.96e7));
    CHECK(x_result["formation_factor"].get<double>() == Approx(1.0 / 0.12));
}

TEST_CASE("ResultsJSON formation_factor only for electrical conductivity",
          "[ResultsJSON][results]") {
    for (const auto& type : {"diffusion", "thermal_conductivity"}) {
        DYNAMIC_SECTION("type=" << type) {
            ResultsJSON writer;
            writer.setPhysicsConfig(makeConfig(type, 2.0));
            writer.setVolumeFraction(0.35);
            writer.addDirectionResult("X", 0.12);

            auto j = writer.buildJSON();
            CHECK_FALSE(j["openimpala"]["results"]["X"].contains("formation_factor"));
        }
    }
}

// ============================================================================
// Provenance passthrough
// ============================================================================
TEST_CASE("ResultsJSON includes provenance when provided", "[ResultsJSON][provenance]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setProvenance("CT-2024-0042", "https://faraday.ac.uk/samples/CT-2024-0042");

    auto j = writer.buildJSON();

    REQUIRE(j.contains("provenance"));
    CHECK(j["provenance"]["sample_id"].get<std::string>() == "CT-2024-0042");
    CHECK(j["provenance"]["provenance_uri"].get<std::string>() ==
          "https://faraday.ac.uk/samples/CT-2024-0042");
}

TEST_CASE("ResultsJSON omits provenance when not provided", "[ResultsJSON][provenance]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));

    auto j = writer.buildJSON();
    CHECK_FALSE(j.contains("provenance"));
}

TEST_CASE("ResultsJSON includes partial provenance", "[ResultsJSON][provenance]") {
    SECTION("sample_id only") {
        ResultsJSON writer;
        writer.setPhysicsConfig(makeConfig("diffusion"));
        writer.setProvenance("CT-2024-0042", "");

        auto j = writer.buildJSON();
        REQUIRE(j.contains("provenance"));
        CHECK(j["provenance"].contains("sample_id"));
        CHECK_FALSE(j["provenance"].contains("provenance_uri"));
    }

    SECTION("uri only") {
        ResultsJSON writer;
        writer.setPhysicsConfig(makeConfig("diffusion"));
        writer.setProvenance("", "https://example.com/sample/123");

        auto j = writer.buildJSON();
        REQUIRE(j.contains("provenance"));
        CHECK_FALSE(j["provenance"].contains("sample_id"));
        CHECK(j["provenance"].contains("provenance_uri"));
    }
}

// ============================================================================
// All physics types produce valid JSON
// ============================================================================
TEST_CASE("ResultsJSON works for all physics types", "[ResultsJSON][physics]") {
    const std::vector<std::pair<std::string, std::string>> types = {
        {"diffusion", "Diffusion"},
        {"electrical_conductivity", "Electrical Conductivity"},
        {"thermal_conductivity", "Thermal Conductivity"},
        {"dielectric_permittivity", "Dielectric Permittivity"},
        {"magnetic_permeability", "Magnetic Permeability"}};

    for (const auto& [type_str, expected_name] : types) {
        DYNAMIC_SECTION("type=" << type_str) {
            ResultsJSON writer;
            writer.setPhysicsConfig(makeConfig(type_str, 2.5));
            writer.setInputFile("sample.tif");
            writer.setPhaseId(0);
            writer.setGridInfo(32, 32, 32, 16);
            writer.setSolverInfo("FlexGMRES", true);
            writer.setVolumeFraction(0.4);
            writer.addDirectionResult("X", 0.15);
            writer.addDirectionResult("Y", 0.14);
            writer.addDirectionResult("Z", 0.16);

            auto j = writer.buildJSON();
            CHECK(j["openimpala"]["physics_type"].get<std::string>() == expected_name);
            CHECK(j["openimpala"]["results"].size() == 3);

            // Verify JSON serializes without errors
            std::string serialized = j.dump(2);
            CHECK(serialized.size() > 0);

            // Verify it round-trips
            auto parsed = nlohmann::json::parse(serialized);
            CHECK(parsed["openimpala"]["physics_type"] == expected_name);
        }
    }
}

// ============================================================================
// File writing
// ============================================================================
TEST_CASE("ResultsJSON write() creates valid JSON file", "[ResultsJSON][file]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setInputFile("test.tif");
    writer.setPhaseId(1);
    writer.setGridInfo(100, 100, 100, 32);
    writer.setSolverInfo("FlexGMRES", true);
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);
    writer.setProvenance("SAMPLE-001", "");

    std::string test_path = "/tmp/openimpala_test_results.json";
    REQUIRE(writer.write(test_path));

    // Read it back and parse
    std::ifstream ifs(test_path);
    REQUIRE(ifs.is_open());
    auto j = nlohmann::json::parse(ifs);

    CHECK(j["openimpala"]["input_file"] == "test.tif");
    CHECK(j["openimpala"]["results"]["X"]["Deff_ratio"].get<double>() == Approx(0.12));
    CHECK(j["provenance"]["sample_id"] == "SAMPLE-001");

    // Cleanup
    std::filesystem::remove(test_path);
}

TEST_CASE("ResultsJSON write() returns false for invalid path", "[ResultsJSON][file]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));

    CHECK_FALSE(writer.write("/nonexistent/directory/results.json"));
}

// ============================================================================
// Numerical precision
// ============================================================================
TEST_CASE("ResultsJSON preserves numerical precision", "[ResultsJSON][precision]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.352481697);
    writer.addDirectionResult("X", 4.12345678901e-01);

    auto j = writer.buildJSON();

    // JSON should preserve at least 9 significant digits
    double vf = j["openimpala"]["volume_fraction"].get<double>();
    double ratio = j["openimpala"]["results"]["X"]["Deff_ratio"].get<double>();

    CHECK(vf == Approx(0.352481697).epsilon(1e-9));
    CHECK(ratio == Approx(4.12345678901e-01).epsilon(1e-9));
}

// ============================================================================
// SI unit annotations (BattINFO-aligned)
// ============================================================================
TEST_CASE("ResultsJSON always includes units block", "[ResultsJSON][units]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);

    auto j = writer.buildJSON();
    REQUIRE(j.contains("units"));
    auto& u = j["units"];

    CHECK(u["volume_fraction"].get<std::string>() == "1");
    CHECK(u["tortuosity"].get<std::string>() == "1");
    CHECK(u["Deff_ratio"].get<std::string>() == "1");
}

TEST_CASE("ResultsJSON units block includes SI units for conductivity", "[ResultsJSON][units]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("electrical_conductivity", 5.96e7));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);

    auto j = writer.buildJSON();
    auto& u = j["units"];

    CHECK(u["sigma_eff_ratio"].get<std::string>() == "1");
    CHECK(u["formation_factor"].get<std::string>() == "1");
    CHECK(u["EffectiveConductivity"].get<std::string>() == "S.m-1");
}

TEST_CASE("ResultsJSON units block includes SI units for thermal", "[ResultsJSON][units]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("thermal_conductivity", 2.0));

    auto j = writer.buildJSON();
    CHECK(j["units"]["EffectiveThermalConductivity"].get<std::string>() == "W.m-1.K-1");
}

TEST_CASE("ResultsJSON units omits effective property unit when bulk == 1",
          "[ResultsJSON][units]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("thermal_conductivity", 1.0));

    auto j = writer.buildJSON();
    CHECK_FALSE(j["units"].contains("EffectiveThermalConductivity"));
}

// ============================================================================
// BPX electrode fragment
// ============================================================================
TEST_CASE("ResultsJSON omits bpx block when no electrode set", "[ResultsJSON][bpx]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);

    auto j = writer.buildJSON();
    CHECK_FALSE(j.contains("bpx"));
}

TEST_CASE("ResultsJSON emits BPX fragment for negative electrode", "[ResultsJSON][bpx]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.254);
    writer.addDirectionResult("X", 0.128);
    writer.addDirectionResult("Y", 0.130);
    writer.addDirectionResult("Z", 0.126);
    writer.setBPXElectrode("negative");

    auto j = writer.buildJSON();

    REQUIRE(j.contains("bpx"));
    auto& bpx = j["bpx"];
    CHECK(bpx["electrode"].get<std::string>() == "negative");

    // Check BPX Parameterisation structure
    REQUIRE(bpx.contains("Parameterisation"));
    REQUIRE(bpx["Parameterisation"].contains("Negative electrode"));
    auto& params = bpx["Parameterisation"]["Negative electrode"];

    // Porosity = volume_fraction
    CHECK(params["Porosity"].get<double>() == Approx(0.254));

    // Transport efficiency = mean D_eff_ratio
    double mean_te = (0.128 + 0.130 + 0.126) / 3.0;
    CHECK(params["Transport efficiency"].get<double>() == Approx(mean_te));

    // Bruggeman coefficient = log(mean_te) / log(porosity)
    double expected_b = std::log(mean_te) / std::log(0.254);
    CHECK(params["Bruggeman coefficient (electrolyte)"].get<double>() == Approx(expected_b));
}

TEST_CASE("ResultsJSON BPX works for positive electrode", "[ResultsJSON][bpx]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.15);
    writer.setBPXElectrode("positive");

    auto j = writer.buildJSON();
    REQUIRE(j["bpx"]["Parameterisation"].contains("Positive electrode"));

    auto& params = j["bpx"]["Parameterisation"]["Positive electrode"];
    CHECK(params["Porosity"].get<double>() == Approx(0.35));
    CHECK(params["Transport efficiency"].get<double>() == Approx(0.15));
}

TEST_CASE("ResultsJSON BPX includes anisotropic data for multi-direction", "[ResultsJSON][bpx]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.30);
    writer.addDirectionResult("X", 0.10);
    writer.addDirectionResult("Y", 0.08);
    writer.addDirectionResult("Z", 0.12);
    writer.setBPXElectrode("negative");

    auto j = writer.buildJSON();

    REQUIRE(j["bpx"].contains("anisotropic"));
    auto& aniso = j["bpx"]["anisotropic"];
    CHECK(aniso["transport_efficiency"]["X"].get<double>() == Approx(0.10));
    CHECK(aniso["transport_efficiency"]["Y"].get<double>() == Approx(0.08));
    CHECK(aniso["transport_efficiency"]["Z"].get<double>() == Approx(0.12));

    // Tortuosity per direction = vf / D_eff_ratio
    CHECK(aniso["tortuosity"]["X"].get<double>() == Approx(0.30 / 0.10));
    CHECK(aniso["tortuosity"]["Y"].get<double>() == Approx(0.30 / 0.08));
    CHECK(aniso["tortuosity"]["Z"].get<double>() == Approx(0.30 / 0.12));

    // Mean tortuosity
    double mean_te = (0.10 + 0.08 + 0.12) / 3.0;
    CHECK(aniso["mean_tortuosity"].get<double>() == Approx(0.30 / mean_te));
}

TEST_CASE("ResultsJSON BPX omits anisotropic for single direction", "[ResultsJSON][bpx]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.30);
    writer.addDirectionResult("X", 0.10);
    writer.setBPXElectrode("negative");

    auto j = writer.buildJSON();
    CHECK_FALSE(j["bpx"].contains("anisotropic"));
}

TEST_CASE("ResultsJSON BPX Bruggeman handles edge cases", "[ResultsJSON][bpx]") {
    SECTION("zero volume fraction yields Bruggeman = 0") {
        ResultsJSON writer;
        writer.setPhysicsConfig(makeConfig("diffusion"));
        writer.setVolumeFraction(0.0);
        writer.addDirectionResult("X", 0.12);
        writer.setBPXElectrode("negative");

        auto j = writer.buildJSON();
        CHECK(j["bpx"]["Parameterisation"]["Negative electrode"]
               ["Bruggeman coefficient (electrolyte)"]
                   .get<double>() == Approx(0.0));
    }

    SECTION("volume fraction = 1 yields Bruggeman = 0") {
        ResultsJSON writer;
        writer.setPhysicsConfig(makeConfig("diffusion"));
        writer.setVolumeFraction(1.0);
        writer.addDirectionResult("X", 0.12);
        writer.setBPXElectrode("negative");

        auto j = writer.buildJSON();
        CHECK(j["bpx"]["Parameterisation"]["Negative electrode"]
               ["Bruggeman coefficient (electrolyte)"]
                   .get<double>() == Approx(0.0));
    }
}

// ============================================================================
// Microstructure parameters
// ============================================================================
TEST_CASE("ResultsJSON includes SSA in microstructure block", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setSpecificSurfaceArea(0.0625);

    auto j = writer.buildJSON();
    REQUIRE(j["openimpala"].contains("microstructure"));
    CHECK(j["openimpala"]["microstructure"]["specific_surface_area"].get<double>() ==
          Approx(0.0625));
}

TEST_CASE("ResultsJSON includes both raw and corrected SSA", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setSpecificSurfaceArea(0.042);
    writer.setSpecificSurfaceAreaRaw(0.0625);

    auto j = writer.buildJSON();
    REQUIRE(j["openimpala"].contains("microstructure"));
    CHECK(j["openimpala"]["microstructure"]["specific_surface_area"].get<double>() ==
          Approx(0.042));
    CHECK(j["openimpala"]["microstructure"]["specific_surface_area_raw"].get<double>() ==
          Approx(0.0625));
}

TEST_CASE("ResultsJSON includes multi-phase volume fractions", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setMultiPhaseVolumeFractions({{0, 0.35}, {1, 0.50}, {2, 0.15}});

    auto j = writer.buildJSON();
    auto& pvf = j["openimpala"]["microstructure"]["phase_volume_fractions"];
    CHECK(pvf["0"].get<double>() == Approx(0.35));
    CHECK(pvf["1"].get<double>() == Approx(0.50));
    CHECK(pvf["2"].get<double>() == Approx(0.15));
}

TEST_CASE("ResultsJSON includes macro geometry", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setMacroGeometry(300.0, 20000.0, 6000000.0);

    auto j = writer.buildJSON();
    auto& mg = j["openimpala"]["microstructure"]["macro_geometry"];
    CHECK(mg["thickness_voxels"].get<double>() == Approx(300.0));
    CHECK(mg["cross_section_voxels"].get<double>() == Approx(20000.0));
    CHECK(mg["total_volume_voxels"].get<double>() == Approx(6000000.0));
}

TEST_CASE("ResultsJSON includes through-thickness profile", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    std::vector<double> profile = {0.3, 0.35, 0.32, 0.28};
    writer.setThroughThicknessProfile("Z", profile);

    auto j = writer.buildJSON();
    auto& ttp = j["openimpala"]["microstructure"]["through_thickness_profiles"]["Z"];
    auto vf = ttp["volume_fraction"].get<std::vector<double>>();
    REQUIRE(vf.size() == 4);
    CHECK(vf[0] == Approx(0.3));
    CHECK(vf[3] == Approx(0.28));
}

TEST_CASE("ResultsJSON includes solid-phase Bruggeman", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setSolidPhaseBruggeman(2.5);

    auto j = writer.buildJSON();
    CHECK(j["openimpala"]["microstructure"]["solid_phase_bruggeman"].get<double>() == Approx(2.5));
}

TEST_CASE("ResultsJSON includes PSD", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setParticleSizeDistribution(12.3, 147, {8.1, 9.3, 10.2});

    auto j = writer.buildJSON();
    auto& psd = j["openimpala"]["microstructure"]["particle_size"];
    CHECK(psd["mean_radius_voxels"].get<double>() == Approx(12.3));
    CHECK(psd["num_particles"].get<int>() == 147);
    auto radii = psd["radii_voxels"].get<std::vector<double>>();
    REQUIRE(radii.size() == 3);
    CHECK(radii[0] == Approx(8.1));
}

TEST_CASE("ResultsJSON omits microstructure block when no params set",
          "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);

    auto j = writer.buildJSON();
    CHECK_FALSE(j["openimpala"].contains("microstructure"));
}

TEST_CASE("ResultsJSON BPX includes microstructure params with voxel size",
          "[ResultsJSON][bpx][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVolumeFraction(0.35);
    writer.addDirectionResult("X", 0.12);
    writer.setBPXElectrode("negative");
    writer.setSolidPhaseBruggeman(2.5);
    writer.setParticleSizeDistribution(12.3, 147, {8.1, 9.3, 10.2});
    writer.setMacroGeometry(300.0, 20000.0, 6000000.0);
    writer.setVoxelSize(0.5e-6);

    auto j = writer.buildJSON();
    auto& params = j["bpx"]["Parameterisation"]["Negative electrode"];
    CHECK(params["Bruggeman coefficient (electrode)"].get<double>() == Approx(2.5));
    CHECK(params["Particle radius [m]"].get<double>() == Approx(12.3 * 0.5e-6));
    CHECK(params["Electrode thickness [m]"].get<double>() == Approx(300.0 * 0.5e-6));
}

TEST_CASE("ResultsJSON includes voxel_size_m in microstructure", "[ResultsJSON][microstructure]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("diffusion"));
    writer.setVoxelSize(0.5e-6);
    writer.setSpecificSurfaceArea(0.0625);

    auto j = writer.buildJSON();
    CHECK(j["openimpala"]["microstructure"]["voxel_size_m"].get<double>() == Approx(0.5e-6));
}

// ============================================================================
// Full round-trip with all features
// ============================================================================
TEST_CASE("ResultsJSON full round-trip with BPX and provenance", "[ResultsJSON][integration]") {
    ResultsJSON writer;
    writer.setPhysicsConfig(makeConfig("electrical_conductivity", 5.96e7));
    writer.setInputFile("electrode_scan.h5");
    writer.setPhaseId(1);
    writer.setGridInfo(256, 256, 256, 64);
    writer.setSolverInfo("FlexGMRES", true);
    writer.setVolumeFraction(0.254);
    writer.addDirectionResult("X", 0.128);
    writer.addDirectionResult("Y", 0.130);
    writer.addDirectionResult("Z", 0.126);
    writer.setProvenance("CT-2024-0042", "https://faraday.ac.uk/samples/CT-2024-0042");
    writer.setBPXElectrode("negative");

    std::string test_path = "/tmp/openimpala_test_full.json";
    REQUIRE(writer.write(test_path));

    // Parse it back
    std::ifstream ifs(test_path);
    auto j = nlohmann::json::parse(ifs);

    // All top-level blocks present
    CHECK(j.contains("openimpala"));
    CHECK(j.contains("units"));
    CHECK(j.contains("bpx"));
    CHECK(j.contains("provenance"));

    // Spot-check values
    CHECK(j["openimpala"]["physics_type"] == "Electrical Conductivity");
    CHECK(j["units"]["EffectiveConductivity"] == "S.m-1");
    CHECK(j["bpx"]["electrode"] == "negative");
    CHECK(j["provenance"]["sample_id"] == "CT-2024-0042");

    std::filesystem::remove(test_path);
}
