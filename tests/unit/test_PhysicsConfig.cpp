// Unit tests for PhysicsConfig — runs without AMReX, MPI, or HYPRE.
// OPENIMPALA_UNIT_TEST is defined via target_compile_definitions in CMakeLists.txt.
#include "PhysicsConfig.H"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <limits>
#include <sstream>

using Catch::Approx;
using namespace OpenImpala;

// ============================================================================
// Helper: construct a PhysicsConfig from a type string, asserting success
// ============================================================================
static PhysicsConfig makeConfig(const std::string& type_str, double bulk = 1.0) {
    PhysicsConfig cfg;
    cfg.bulk_property = bulk;
    REQUIRE(PhysicsConfig::fromTypeString(type_str, cfg));
    return cfg;
}

// ============================================================================
// fromTypeString — type parsing
// ============================================================================
TEST_CASE("fromTypeString parses all valid physics types", "[PhysicsConfig][parsing]") {
    SECTION("diffusion") {
        auto cfg = makeConfig("diffusion");
        CHECK(cfg.type == PhysicsType::Diffusion);
        CHECK(cfg.name == "Diffusion");
        CHECK(cfg.coeff_label == "D");
        CHECK(cfg.field_label == "concentration");
        CHECK(cfg.eff_property_label == "EffectiveDiffusivity");
        CHECK(cfg.ratio_label == "Deff_ratio");
    }

    SECTION("electrical_conductivity") {
        auto cfg = makeConfig("electrical_conductivity");
        CHECK(cfg.type == PhysicsType::ElectricalConductivity);
        CHECK(cfg.name == "Electrical Conductivity");
        CHECK(cfg.coeff_label == "sigma");
        CHECK(cfg.field_label == "voltage");
        CHECK(cfg.eff_property_label == "EffectiveConductivity");
        CHECK(cfg.ratio_label == "sigma_eff_ratio");
    }

    SECTION("thermal_conductivity") {
        auto cfg = makeConfig("thermal_conductivity");
        CHECK(cfg.type == PhysicsType::ThermalConductivity);
        CHECK(cfg.name == "Thermal Conductivity");
        CHECK(cfg.coeff_label == "k");
        CHECK(cfg.field_label == "temperature");
        CHECK(cfg.eff_property_label == "EffectiveThermalConductivity");
        CHECK(cfg.ratio_label == "k_eff_ratio");
    }

    SECTION("dielectric_permittivity") {
        auto cfg = makeConfig("dielectric_permittivity");
        CHECK(cfg.type == PhysicsType::DielectricPermittivity);
        CHECK(cfg.name == "Dielectric Permittivity");
        CHECK(cfg.coeff_label == "epsilon");
        CHECK(cfg.field_label == "potential");
        CHECK(cfg.eff_property_label == "EffectivePermittivity");
        CHECK(cfg.ratio_label == "eps_eff_ratio");
    }

    SECTION("magnetic_permeability") {
        auto cfg = makeConfig("magnetic_permeability");
        CHECK(cfg.type == PhysicsType::MagneticPermeability);
        CHECK(cfg.name == "Magnetic Permeability");
        CHECK(cfg.coeff_label == "mu");
        CHECK(cfg.field_label == "potential");
        CHECK(cfg.eff_property_label == "EffectivePermeability");
        CHECK(cfg.ratio_label == "mu_eff_ratio");
    }
}

TEST_CASE("fromTypeString is case-insensitive", "[PhysicsConfig][parsing]") {
    PhysicsConfig cfg;

    SECTION("UPPERCASE") {
        CHECK(PhysicsConfig::fromTypeString("DIFFUSION", cfg));
        CHECK(cfg.type == PhysicsType::Diffusion);
    }

    SECTION("MixedCase") {
        CHECK(PhysicsConfig::fromTypeString("Electrical_Conductivity", cfg));
        CHECK(cfg.type == PhysicsType::ElectricalConductivity);
    }

    SECTION("Thermal_CONDUCTIVITY") {
        CHECK(PhysicsConfig::fromTypeString("THERMAL_CONDUCTIVITY", cfg));
        CHECK(cfg.type == PhysicsType::ThermalConductivity);
    }
}

TEST_CASE("fromTypeString returns false for invalid types", "[PhysicsConfig][parsing]") {
    PhysicsConfig cfg;
    CHECK_FALSE(PhysicsConfig::fromTypeString("invalid_type", cfg));
    CHECK_FALSE(PhysicsConfig::fromTypeString("", cfg));
    CHECK_FALSE(PhysicsConfig::fromTypeString("conductivity", cfg));
    CHECK_FALSE(PhysicsConfig::fromTypeString("electric", cfg));
}

TEST_CASE("fromTypeString preserves bulk_property set before call", "[PhysicsConfig][parsing]") {
    PhysicsConfig cfg;
    cfg.bulk_property = 42.0;
    REQUIRE(PhysicsConfig::fromTypeString("diffusion", cfg));
    CHECK(cfg.bulk_property == 42.0);
}

// ============================================================================
// effectiveProperty
// ============================================================================
TEST_CASE("effectiveProperty scales D_eff_ratio by bulk_property", "[PhysicsConfig][computation]") {
    SECTION("unit bulk property") {
        auto cfg = makeConfig("diffusion", 1.0);
        CHECK(cfg.effectiveProperty(0.5) == Approx(0.5));
        CHECK(cfg.effectiveProperty(1.0) == Approx(1.0));
        CHECK(cfg.effectiveProperty(0.0) == Approx(0.0));
    }

    SECTION("non-unit bulk property (copper conductivity)") {
        auto cfg = makeConfig("electrical_conductivity", 5.96e7);
        CHECK(cfg.effectiveProperty(0.5) == Approx(2.98e7));
        CHECK(cfg.effectiveProperty(0.0) == Approx(0.0));
        CHECK(cfg.effectiveProperty(1.0) == Approx(5.96e7));
    }

    SECTION("negative D_eff_ratio") {
        auto cfg = makeConfig("diffusion", 2.0);
        CHECK(cfg.effectiveProperty(-0.1) == Approx(-0.2));
    }
}

// ============================================================================
// tortuosityFactor
// ============================================================================
TEST_CASE("tortuosityFactor computes vf / D_eff_ratio", "[PhysicsConfig][computation]") {
    auto cfg = makeConfig("diffusion");

    SECTION("typical values") {
        CHECK(cfg.tortuosityFactor(0.5, 0.4) == Approx(0.8));
        CHECK(cfg.tortuosityFactor(0.25, 0.5) == Approx(2.0));
        CHECK(cfg.tortuosityFactor(1.0, 1.0) == Approx(1.0));
    }

    SECTION("round-trip: tau -> D_eff -> tau") {
        double vf = 0.35;
        double tau_original = 1.42;
        double D_eff = vf / tau_original;
        double tau_recovered = cfg.tortuosityFactor(D_eff, vf);
        CHECK(tau_recovered == Approx(tau_original));
    }

    SECTION("zero D_eff_ratio returns infinity") {
        double result = cfg.tortuosityFactor(0.0, 0.5);
        CHECK(std::isinf(result));
    }

    SECTION("negative D_eff_ratio returns infinity") {
        double result = cfg.tortuosityFactor(-0.1, 0.5);
        CHECK(std::isinf(result));
    }
}

// ============================================================================
// formationFactor
// ============================================================================
TEST_CASE("formationFactor computes 1 / D_eff_ratio", "[PhysicsConfig][computation]") {
    auto cfg = makeConfig("electrical_conductivity");

    SECTION("typical values") {
        CHECK(cfg.formationFactor(0.5) == Approx(2.0));
        CHECK(cfg.formationFactor(0.25) == Approx(4.0));
        CHECK(cfg.formationFactor(1.0) == Approx(1.0));
    }

    SECTION("very small D_eff_ratio gives large formation factor") {
        CHECK(cfg.formationFactor(1e-6) == Approx(1e6));
    }

    SECTION("zero D_eff_ratio returns infinity") {
        double result = cfg.formationFactor(0.0);
        CHECK(std::isinf(result));
    }
}

// ============================================================================
// writeDirectionResults — output formatting
// ============================================================================
TEST_CASE("writeDirectionResults produces correct output", "[PhysicsConfig][output]") {
    SECTION("diffusion with unit bulk writes ratio and tortuosity only") {
        auto cfg = makeConfig("diffusion", 1.0);
        std::ostringstream oss;
        cfg.writeDirectionResults(oss, "X", 0.5, 0.4);
        std::string output = oss.str();

        CHECK(output.find("Deff_ratio_X:") != std::string::npos);
        CHECK(output.find("Tortuosity_X:") != std::string::npos);
        // No EffectiveDiffusivity line when bulk_property == 1.0
        CHECK(output.find("EffectiveDiffusivity_X:") == std::string::npos);
        // No FormationFactor for diffusion type
        CHECK(output.find("FormationFactor_X:") == std::string::npos);
    }

    SECTION("electrical_conductivity with non-unit bulk writes all fields") {
        auto cfg = makeConfig("electrical_conductivity", 5.96e7);
        std::ostringstream oss;
        cfg.writeDirectionResults(oss, "Z", 0.3, 0.5);
        std::string output = oss.str();

        CHECK(output.find("sigma_eff_ratio_Z:") != std::string::npos);
        CHECK(output.find("Tortuosity_Z:") != std::string::npos);
        CHECK(output.find("EffectiveConductivity_Z:") != std::string::npos);
        CHECK(output.find("FormationFactor_Z:") != std::string::npos);
    }

    SECTION("thermal_conductivity with non-unit bulk — no formation factor") {
        auto cfg = makeConfig("thermal_conductivity", 401.0);
        std::ostringstream oss;
        cfg.writeDirectionResults(oss, "Y", 0.5, 0.4);
        std::string output = oss.str();

        CHECK(output.find("k_eff_ratio_Y:") != std::string::npos);
        CHECK(output.find("Tortuosity_Y:") != std::string::npos);
        CHECK(output.find("EffectiveThermalConductivity_Y:") != std::string::npos);
        CHECK(output.find("FormationFactor_Y:") == std::string::npos);
    }
}

// ============================================================================
// writeHeader — header formatting
// ============================================================================
TEST_CASE("writeHeader produces correct header", "[PhysicsConfig][output]") {
    SECTION("diffusion with unit bulk — no bulk property line") {
        auto cfg = makeConfig("diffusion", 1.0);
        std::ostringstream oss;
        cfg.writeHeader(oss, "test.tif", 0);
        std::string output = oss.str();

        CHECK(output.find("# Physics Type: Diffusion") != std::string::npos);
        CHECK(output.find("# Input File: test.tif") != std::string::npos);
        CHECK(output.find("# Analysis Phase ID: 0") != std::string::npos);
        CHECK(output.find("# Bulk Property") == std::string::npos);
    }

    SECTION("conductivity with bulk — includes bulk property line") {
        auto cfg = makeConfig("electrical_conductivity", 5.96e7);
        std::ostringstream oss;
        cfg.writeHeader(oss, "sample.h5", 1);
        std::string output = oss.str();

        CHECK(output.find("# Physics Type: Electrical Conductivity") != std::string::npos);
        CHECK(output.find("# Bulk Property (sigma_bulk):") != std::string::npos);
    }
}

// ============================================================================
// Default construction
// ============================================================================
TEST_CASE("Default PhysicsConfig has sensible defaults", "[PhysicsConfig]") {
    PhysicsConfig cfg;
    CHECK(cfg.type == PhysicsType::Diffusion);
    CHECK(cfg.bulk_property == 1.0);
    CHECK(cfg.name.empty());
    CHECK(cfg.coeff_label.empty());
}

// ============================================================================
// Cross-physics consistency: all types produce finite results
// ============================================================================
TEST_CASE("All physics types produce finite results for valid inputs",
          "[PhysicsConfig][computation]") {
    const std::vector<std::string> types = {"diffusion", "electrical_conductivity",
                                            "thermal_conductivity", "dielectric_permittivity",
                                            "magnetic_permeability"};

    for (const auto& type_str : types) {
        DYNAMIC_SECTION("type=" << type_str) {
            auto cfg = makeConfig(type_str, 100.0);
            double D_eff_ratio = 0.42;
            double vf = 0.35;

            CHECK(std::isfinite(cfg.effectiveProperty(D_eff_ratio)));
            CHECK(std::isfinite(cfg.tortuosityFactor(D_eff_ratio, vf)));
            CHECK(std::isfinite(cfg.formationFactor(D_eff_ratio)));
            CHECK(cfg.effectiveProperty(D_eff_ratio) == Approx(D_eff_ratio * 100.0));
        }
    }
}
