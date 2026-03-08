/** @file module.cpp
 *  @brief Top-level pybind11 module definition for OpenImpala Python bindings.
 *
 *  Defines the PYBIND11_MODULE entry point and delegates to per-subsystem
 *  init functions declared in the other binding translation units.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations — each implemented in its own .cpp
void init_enums(py::module_& m);
void init_io(py::module_& m);
void init_props(py::module_& m);
void init_solvers(py::module_& m);
void init_config(py::module_& m);

PYBIND11_MODULE(_core, m) {
    // Import pyAMReX's 3D module to merge the pybind11 type registries.
    // This allows Python to pass an amrex.iMultiFab into OpenImpala's C++ functions
    py::module_::import("amrex.space3d");

    m.doc() = "OpenImpala C++ backend — low-level bindings for transport property "
              "computation on 3-D voxel images of porous microstructures.";

    // Register enums first (used by everything else)
    init_enums(m);

    // I/O readers
    init_io(m);

    // Lightweight property calculators (VolumeFraction, PercolationCheck)
    init_props(m);

    // Heavy solvers (TortuosityHypre, TortuosityDirect, EffectiveDiffusivityHypre)
    init_solvers(m);

    // Configuration and output helpers (PhysicsConfig, ResultsJSON, CathodeWrite)
    init_config(m);
}
