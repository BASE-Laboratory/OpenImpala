/** @file enums.cpp
 *  @brief pybind11 bindings for OpenImpala enums.
 */

#include <pybind11/pybind11.h>

#include "Tortuosity.H"
#include "TortuosityHypre.H"
#include "EffectiveDiffusivityHypre.H"
#include "RawReader.H"
#include "PhysicsConfig.H"

namespace py = pybind11;

void init_enums(py::module_& m) {
    // --- Direction (X / Y / Z) ---
    py::enum_<OpenImpala::Direction>(m, "Direction",
                                     "Cardinal direction for transport calculations.")
        .value("X", OpenImpala::Direction::X)
        .value("Y", OpenImpala::Direction::Y)
        .value("Z", OpenImpala::Direction::Z);

    // --- CellType ---
    py::enum_<OpenImpala::CellType>(m, "CellType", "Cell role within the computational domain.")
        .value("BLOCKED", OpenImpala::CellType::BLOCKED)
        .value("FREE", OpenImpala::CellType::FREE)
        .value("BOUNDARY_X_LO", OpenImpala::CellType::BOUNDARY_X_LO)
        .value("BOUNDARY_X_HI", OpenImpala::CellType::BOUNDARY_X_HI)
        .value("BOUNDARY_Y_LO", OpenImpala::CellType::BOUNDARY_Y_LO)
        .value("BOUNDARY_Y_HI", OpenImpala::CellType::BOUNDARY_Y_HI)
        .value("BOUNDARY_Z_LO", OpenImpala::CellType::BOUNDARY_Z_LO)
        .value("BOUNDARY_Z_HI", OpenImpala::CellType::BOUNDARY_Z_HI);

    // --- RawDataType ---
    py::enum_<OpenImpala::RawDataType>(
        m, "RawDataType", "Primitive data type and endianness for raw binary voxel files.")
        .value("UNKNOWN", OpenImpala::RawDataType::UNKNOWN)
        .value("UINT8", OpenImpala::RawDataType::UINT8)
        .value("INT8", OpenImpala::RawDataType::INT8)
        .value("INT16_LE", OpenImpala::RawDataType::INT16_LE)
        .value("INT16_BE", OpenImpala::RawDataType::INT16_BE)
        .value("UINT16_LE", OpenImpala::RawDataType::UINT16_LE)
        .value("UINT16_BE", OpenImpala::RawDataType::UINT16_BE)
        .value("INT32_LE", OpenImpala::RawDataType::INT32_LE)
        .value("INT32_BE", OpenImpala::RawDataType::INT32_BE)
        .value("UINT32_LE", OpenImpala::RawDataType::UINT32_LE)
        .value("UINT32_BE", OpenImpala::RawDataType::UINT32_BE)
        .value("FLOAT32_LE", OpenImpala::RawDataType::FLOAT32_LE)
        .value("FLOAT32_BE", OpenImpala::RawDataType::FLOAT32_BE)
        .value("FLOAT64_LE", OpenImpala::RawDataType::FLOAT64_LE)
        .value("FLOAT64_BE", OpenImpala::RawDataType::FLOAT64_BE);

    // --- SolverType (HYPRE structured solvers — shared by Tortuosity and EffDiff) ---
    py::enum_<OpenImpala::TortuosityHypre::SolverType>(m, "SolverType",
                                                       "HYPRE structured-grid solver algorithm.")
        .value("Jacobi", OpenImpala::TortuosityHypre::SolverType::Jacobi)
        .value("GMRES", OpenImpala::TortuosityHypre::SolverType::GMRES)
        .value("FlexGMRES", OpenImpala::TortuosityHypre::SolverType::FlexGMRES)
        .value("PCG", OpenImpala::TortuosityHypre::SolverType::PCG)
        .value("BiCGSTAB", OpenImpala::TortuosityHypre::SolverType::BiCGSTAB)
        .value("SMG", OpenImpala::TortuosityHypre::SolverType::SMG)
        .value("PFMG", OpenImpala::TortuosityHypre::SolverType::PFMG);

    // --- EffDiffSolverType (separate enum in EffectiveDiffusivityHypre) ---
    py::enum_<OpenImpala::EffectiveDiffusivityHypre::SolverType>(
        m, "EffDiffSolverType",
        "HYPRE solver algorithm for the effective-diffusivity cell problem.")
        .value("Jacobi", OpenImpala::EffectiveDiffusivityHypre::SolverType::Jacobi)
        .value("GMRES", OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES)
        .value("FlexGMRES", OpenImpala::EffectiveDiffusivityHypre::SolverType::FlexGMRES)
        .value("PCG", OpenImpala::EffectiveDiffusivityHypre::SolverType::PCG)
        .value("BiCGSTAB", OpenImpala::EffectiveDiffusivityHypre::SolverType::BiCGSTAB)
        .value("SMG", OpenImpala::EffectiveDiffusivityHypre::SolverType::SMG)
        .value("PFMG", OpenImpala::EffectiveDiffusivityHypre::SolverType::PFMG);

    // --- PhysicsType ---
    py::enum_<OpenImpala::PhysicsType>(m, "PhysicsType", "Physical quantity being computed.")
        .value("Diffusion", OpenImpala::PhysicsType::Diffusion)
        .value("ElectricalConductivity", OpenImpala::PhysicsType::ElectricalConductivity)
        .value("ThermalConductivity", OpenImpala::PhysicsType::ThermalConductivity)
        .value("DielectricPermittivity", OpenImpala::PhysicsType::DielectricPermittivity)
        .value("MagneticPermeability", OpenImpala::PhysicsType::MagneticPermeability);
}
