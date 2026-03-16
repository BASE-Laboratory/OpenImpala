// Unit tests for MacroGeometry — runs without AMReX, MPI, or HYPRE.
// OPENIMPALA_UNIT_TEST is defined via target_compile_definitions in CMakeLists.txt.
#include "MacroGeometry.H"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;
using namespace OpenImpala;

TEST_CASE("MacroGeometry fromDimensions with Z flow direction", "[MacroGeometry]") {
    auto mg = MacroGeometry::fromDimensions(100, 200, 300, 2); // Z direction

    CHECK(mg.nx == 100);
    CHECK(mg.ny == 200);
    CHECK(mg.nz == 300);
    CHECK(mg.thickness == Approx(300.0));
    CHECK(mg.cross_section == Approx(100.0 * 200.0));
    CHECK(mg.total_volume == Approx(300.0 * 100.0 * 200.0));
}

TEST_CASE("MacroGeometry fromDimensions with X flow direction", "[MacroGeometry]") {
    auto mg = MacroGeometry::fromDimensions(64, 128, 256, 0); // X direction

    CHECK(mg.thickness == Approx(64.0));
    CHECK(mg.cross_section == Approx(128.0 * 256.0));
    CHECK(mg.total_volume == Approx(64.0 * 128.0 * 256.0));
}

TEST_CASE("MacroGeometry fromDimensions with Y flow direction", "[MacroGeometry]") {
    auto mg = MacroGeometry::fromDimensions(10, 20, 30, 1); // Y direction

    CHECK(mg.thickness == Approx(20.0));
    CHECK(mg.cross_section == Approx(10.0 * 30.0));
    CHECK(mg.total_volume == Approx(10.0 * 20.0 * 30.0));
}

TEST_CASE("MacroGeometry cubic domain", "[MacroGeometry]") {
    auto mg = MacroGeometry::fromDimensions(32, 32, 32, 0);

    CHECK(mg.thickness == Approx(32.0));
    CHECK(mg.cross_section == Approx(32.0 * 32.0));
    CHECK(mg.total_volume == Approx(32.0 * 32.0 * 32.0));
}
