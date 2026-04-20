/** @file solvers.cpp
 *  @brief pybind11 bindings for OpenImpala transport-property solvers.
 *
 *  Accepts VoxelImage handles — no pyamrex dependency.
 */

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "HypreStructSolver.H"
#include "TortuosityHypre.H"
#include "TortuosityMLMG.H"
#include "TortuosityDirect.H"
#include "EffectiveDiffusivityHypre.H"
#include "VoxelImage.H"

namespace py = pybind11;
using namespace OpenImpala;

void init_solvers(py::module_& m) {
    // =========================================================================
    // TortuosityHypre
    // =========================================================================
    py::class_<TortuosityHypre>(m, "TortuosityHypre",
                                "HYPRE-based tortuosity solver.  Solves the steady-state diffusion "
                                "equation on a masked phase and computes tortuosity from the "
                                "resulting flux.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, amrex::Real vf, int phase,
                         OpenImpala::Direction dir, TortuosityHypre::SolverType solver_type,
                         const std::string& results_path, amrex::Real vlo, amrex::Real vhi,
                         int verbose, bool write_plotfile,
                         OpenImpala::PrecondType preconditioner) {
                 return new TortuosityHypre(img->geom, img->ba, img->dm, *(img->mf), vf, phase, dir,
                                            solver_type, results_path, vlo, vhi, verbose,
                                            write_plotfile, preconditioner);
             }),
             py::arg("img"), py::arg("vf"), py::arg("phase"), py::arg("dir"),
             py::arg("solver_type"), py::arg("results_path"), py::arg("vlo") = 0.0,
             py::arg("vhi") = 1.0, py::arg("verbose") = 0, py::arg("write_plotfile") = false,
             py::arg("preconditioner") = OpenImpala::PrecondType::SMG,
             // keep VoxelImage alive while this object lives
             py::keep_alive<1, 2>())

        // value() — translate NaN-on-failure into a Python exception
        .def(
            "value",
            [](TortuosityHypre& self, bool refresh) {
                amrex::Real val = self.value(refresh);
                if (std::isnan(val)) {
                    std::string reason = self.getSolverConverged()
                                             ? "converged but produced an invalid result"
                                             : "solver did not converge";
                    throw std::runtime_error("TortuosityHypre.value() failed: " + reason);
                }
                return val;
            },
            py::arg("refresh") = false,
            "Compute (or return cached) tortuosity.  Raises RuntimeError on failure.")

        // Raw value without exception wrapping for power users
        .def(
            "value_raw", [](TortuosityHypre& self, bool refresh) { return self.value(refresh); },
            py::arg("refresh") = false, "Return tortuosity (NaN on failure, no exception).")

        .def("check_matrix_properties", &TortuosityHypre::checkMatrixProperties,
             "Verify HYPRE matrix / RHS properties (for debugging).")

        // Read-only diagnostics
        .def_property_readonly("solver_converged", &TortuosityHypre::getSolverConverged)
        .def_property_readonly("residual_norm", &TortuosityHypre::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &TortuosityHypre::getSolverIterations)
        .def_property_readonly("flux_in", &TortuosityHypre::getFluxIn)
        .def_property_readonly("flux_out", &TortuosityHypre::getFluxOut)
        .def_property_readonly("active_volume_fraction", &TortuosityHypre::getActiveVolumeFraction)
        .def_property_readonly("plane_fluxes", &TortuosityHypre::getPlaneFluxes)
        .def_property_readonly("plane_flux_max_deviation",
                               &TortuosityHypre::getPlaneFluxMaxDeviation)
        .def_property_readonly("is_multi_phase", &TortuosityHypre::isMultiPhase)
        .def_property_readonly("inlet_outlet_bc_type", &TortuosityHypre::getInletOutletBCType)
        .def_property_readonly("sides_bc_type", &TortuosityHypre::getSidesBCType);

    // =========================================================================
    // TortuosityMLMG
    // =========================================================================
    py::class_<TortuosityMLMG>(m, "TortuosityMLMG",
                               "Matrix-free MLMG tortuosity solver using AMReX's native geometric "
                               "multigrid.  Lower memory and faster setup than HYPRE for "
                               "small/medium grids.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, amrex::Real vf, int phase,
                         OpenImpala::Direction dir, const std::string& results_path,
                         amrex::Real vlo, amrex::Real vhi, int verbose, bool write_plotfile,
                         amrex::Real eps, int maxiter, int max_coarsening_level) {
                 return new TortuosityMLMG(img->geom, img->ba, img->dm, *(img->mf), vf, phase, dir,
                                           results_path, vlo, vhi, verbose, write_plotfile,
                                           eps, maxiter, max_coarsening_level);
             }),
             py::arg("img"), py::arg("vf"), py::arg("phase"), py::arg("dir"),
             py::arg("results_path"), py::arg("vlo") = 0.0, py::arg("vhi") = 1.0,
             py::arg("verbose") = 0, py::arg("write_plotfile") = false,
             py::arg("eps") = 1.0e-9, py::arg("maxiter") = 200,
             py::arg("max_coarsening_level") = 30, py::keep_alive<1, 2>())

        .def(
            "value",
            [](TortuosityMLMG& self, bool refresh) {
                amrex::Real val = self.value(refresh);
                if (std::isnan(val)) {
                    std::string reason = self.getSolverConverged()
                                             ? "converged but produced an invalid result"
                                             : "solver did not converge";
                    throw std::runtime_error("TortuosityMLMG.value() failed: " + reason);
                }
                return val;
            },
            py::arg("refresh") = false,
            "Compute (or return cached) tortuosity.  Raises RuntimeError on failure.")

        .def(
            "value_raw", [](TortuosityMLMG& self, bool refresh) { return self.value(refresh); },
            py::arg("refresh") = false, "Return tortuosity (NaN on failure, no exception).")

        .def_property_readonly("solver_converged", &TortuosityMLMG::getSolverConverged)
        .def_property_readonly("residual_norm", &TortuosityMLMG::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &TortuosityMLMG::getSolverIterations)
        .def_property_readonly("flux_in", &TortuosityMLMG::getFluxIn)
        .def_property_readonly("flux_out", &TortuosityMLMG::getFluxOut)
        .def_property_readonly("active_volume_fraction", &TortuosityMLMG::getActiveVolumeFraction)
        .def_property_readonly("plane_fluxes", &TortuosityMLMG::getPlaneFluxes)
        .def_property_readonly("plane_flux_max_deviation",
                               &TortuosityMLMG::getPlaneFluxMaxDeviation);

    // =========================================================================
    // TortuosityDirect
    // =========================================================================
    py::class_<TortuosityDirect>(
        m, "TortuosityDirect",
        "Legacy iterative tortuosity solver using Forward-Euler time-stepping.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase, OpenImpala::Direction dir,
                         amrex::Real eps, int n_steps, int plot_interval,
                         const std::string& plot_basename, amrex::Real vlo, amrex::Real vhi) {
                 return new TortuosityDirect(img->geom, img->ba, img->dm, *(img->mf), phase, dir,
                                             eps, n_steps, plot_interval, plot_basename, vlo, vhi);
             }),
             py::arg("img"), py::arg("phase"), py::arg("dir"), py::arg("eps"), py::arg("n_steps"),
             py::arg("plot_interval"), py::arg("plot_basename"), py::arg("vlo"), py::arg("vhi"),
             // keep VoxelImage alive while this object lives
             py::keep_alive<1, 2>())

        .def(
            "value",
            [](TortuosityDirect& self, bool refresh) {
                amrex::Real val = self.value(refresh);
                if (std::isnan(val)) {
                    throw std::runtime_error(
                        "TortuosityDirect.value() failed: solver did not converge");
                }
                return val;
            },
            py::arg("refresh") = false,
            "Compute (or return cached) tortuosity.  Raises RuntimeError on failure.")

        .def(
            "value_raw", [](TortuosityDirect& self, bool refresh) { return self.value(refresh); },
            py::arg("refresh") = false)

        .def_property_readonly("num_iterations", &TortuosityDirect::getNumIterations)
        .def_property_readonly("final_residual", &TortuosityDirect::getFinalResidual);

    // =========================================================================
    // EffectiveDiffusivityHypre
    // =========================================================================
    py::class_<EffectiveDiffusivityHypre>(
        m, "EffectiveDiffusivityHypre",
        "Solves the cell problem for effective diffusivity via HYPRE.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_id, OpenImpala::Direction dir,
                         EffectiveDiffusivityHypre::SolverType solver_type,
                         const std::string& results_path, int verbose, bool write_plotfile) {
                 return new EffectiveDiffusivityHypre(img->geom, img->ba, img->dm, *(img->mf),
                                                      phase_id, dir, solver_type, results_path,
                                                      verbose, write_plotfile);
             }),
             py::arg("img"), py::arg("phase_id"), py::arg("dir"), py::arg("solver_type"),
             py::arg("results_path"), py::arg("verbose") = 1, py::arg("write_plotfile") = false,
             // keep VoxelImage alive while this object lives
             py::keep_alive<1, 2>())

        .def("solve", &EffectiveDiffusivityHypre::solve,
             "Solve the cell problem.  Returns True if the solver converged.")

        .def("get_chi_solution", &EffectiveDiffusivityHypre::getChiSolution, py::arg("chi_field"),
             "Copy the solved corrector field into *chi_field*.")

        .def_property_readonly("solver_converged", &EffectiveDiffusivityHypre::getSolverConverged)
        .def_property_readonly("residual_norm",
                               &EffectiveDiffusivityHypre::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &EffectiveDiffusivityHypre::getSolverIterations);
}
