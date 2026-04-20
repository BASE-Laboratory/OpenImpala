/** @file enums.cpp
 *  @brief pybind11 bindings for OpenImpala enums.
 */

#include <pybind11/pybind11.h>

#include "BoundaryCondition.H"
#include "HypreStructSolver.H"
#include "Tortuosity.H"
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
    py::enum_<OpenImpala::SolverType>(m, "SolverType", "HYPRE structured-grid solver algorithm.")
        .value("Jacobi", OpenImpala::SolverType::Jacobi)
        .value("GMRES", OpenImpala::SolverType::GMRES)
        .value("FlexGMRES", OpenImpala::SolverType::FlexGMRES)
        .value("PCG", OpenImpala::SolverType::PCG)
        .value("BiCGSTAB", OpenImpala::SolverType::BiCGSTAB)
        .value("SMG", OpenImpala::SolverType::SMG)
        .value("PFMG", OpenImpala::SolverType::PFMG);

    // --- EffDiffSolverType: backward-compatible alias for SolverType ---
    m.attr("EffDiffSolverType") = m.attr("SolverType");

    // --- PrecondType: multigrid preconditioner for Krylov solvers ---
    // Used when ``solver_type`` is PCG/GMRES/FlexGMRES/BiCGSTAB; ignored for
    // standalone SMG/PFMG/Jacobi. This is the asymptotic-scaling lever —
    // plain PCG is O(N^p) with p>1; PCG+PFMG restores closer to O(N).
    py::enum_<OpenImpala::PrecondType>(m, "PrecondType",
                                       "Multigrid preconditioner for HYPRE Krylov solvers.")
        .value("SMG", OpenImpala::PrecondType::SMG,
               "Semicoarsening multigrid — robust for anisotropic grids")
        .value("PFMG", OpenImpala::PrecondType::PFMG,
               "Parallel semicoarsening multigrid — more scalable for large problems");

    // --- BCType (boundary condition strategy for transport solvers) ---
    py::enum_<OpenImpala::BCType>(m, "BCType", "Boundary condition type for transport solvers.")
        .value("DirichletExternal", OpenImpala::BCType::DirichletExternal,
               "Dirichlet at domain faces (standard default)")
        .value("DirichletPhaseBoundary", OpenImpala::BCType::DirichletPhaseBoundary,
               "Dirichlet at active cells adjacent to inactive cells near boundary")
        .value("Neumann", OpenImpala::BCType::Neumann, "Zero-flux Neumann (implicit)")
        .value("Periodic", OpenImpala::BCType::Periodic,
               "Periodic wrapping (requires periodic grid)");

    // --- PhysicsType ---
    py::enum_<OpenImpala::PhysicsType>(m, "PhysicsType", "Physical quantity being computed.")
        .value("Diffusion", OpenImpala::PhysicsType::Diffusion)
        .value("ElectricalConductivity", OpenImpala::PhysicsType::ElectricalConductivity)
        .value("ThermalConductivity", OpenImpala::PhysicsType::ThermalConductivity)
        .value("DielectricPermittivity", OpenImpala::PhysicsType::DielectricPermittivity)
        .value("MagneticPermeability", OpenImpala::PhysicsType::MagneticPermeability);
}
