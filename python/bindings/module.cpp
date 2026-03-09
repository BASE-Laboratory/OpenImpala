/** @file module.cpp
 *  @brief Top-level pybind11 module definition for OpenImpala Python bindings.
 *
 *  Defines the PYBIND11_MODULE entry point and delegates to per-subsystem
 *  init functions declared in the other binding translation units.
 */

#include <pybind11/pybind11.h>

#include <HYPRE.h>
#include <stdexcept>

namespace py = pybind11;

// Forward declarations — each implemented in its own .cpp
void init_enums(py::module_& m);
void init_io(py::module_& m);
void init_props(py::module_& m);
void init_solvers(py::module_& m);
void init_config(py::module_& m);

PYBIND11_MODULE(_core, m) {
    // Import pyAMReX's exact C++ extension to merge the pybind11 type registries.
    py::module_::import("amrex.space3d.amrex_3d_pybind");

    m.doc() = "OpenImpala C++ backend — low-level bindings for transport property "
              "computation on 3-D voxel images of porous microstructures.";

    // --- HYPRE lifecycle functions ---
    m.def(
        "hypre_init",
        []() {
            int ierr = HYPRE_Init();
            if (ierr != 0) {
                throw std::runtime_error("HYPRE_Init() failed with error code " +
                                         std::to_string(ierr));
            }
        },
        "Initialise the HYPRE library.  Must be called before using any HYPRE-based solver.");

    m.def(
        "hypre_finalize",
        []() {
            int ierr = HYPRE_Finalize();
            if (ierr != 0) {
                throw std::runtime_error("HYPRE_Finalize() failed with error code " +
                                         std::to_string(ierr));
            }
        },
        "Shut down the HYPRE library.  Call after all HYPRE solvers have been destroyed.");

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
