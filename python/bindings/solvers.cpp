/** @file solvers.cpp
 *  @brief pybind11 bindings for OpenImpala transport-property solvers.
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "TortuosityHypre.H"
#include "TortuosityDirect.H"
#include "EffectiveDiffusivityHypre.H"

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

        .def(py::init([](py::object geom_obj, py::object ba_obj, py::object dm_obj,
                         py::object mf_obj, amrex::Real vf, int phase, OpenImpala::Direction dir,
                         TortuosityHypre::SolverType solver_type, const std::string& results_path,
                         amrex::Real vlo, amrex::Real vhi, int verbose, bool write_plotfile) {
                 auto& geom = py::cast<const amrex::Geometry&>(geom_obj);
                 auto& ba = py::cast<const amrex::BoxArray&>(ba_obj);
                 auto& dm = py::cast<const amrex::DistributionMapping&>(dm_obj);
                 auto& mf = py::cast<const amrex::iMultiFab&>(mf_obj);
                 return new TortuosityHypre(geom, ba, dm, mf, vf, phase, dir, solver_type,
                                            results_path, vlo, vhi, verbose, write_plotfile);
             }),
             py::arg("geom"), py::arg("ba"), py::arg("dm"), py::arg("mf_phase"), py::arg("vf"),
             py::arg("phase"), py::arg("dir"), py::arg("solver_type"), py::arg("results_path"),
             py::arg("vlo") = 0.0, py::arg("vhi") = 1.0, py::arg("verbose") = 0,
             py::arg("write_plotfile") = false,
             // prevent GC of referenced AMReX objects
             py::keep_alive<1, 2>(), // geom
             py::keep_alive<1, 3>(), // ba
             py::keep_alive<1, 4>(), // dm
             py::keep_alive<1, 5>()) // mf_phase

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
        .def_property_readonly("is_multi_phase", &TortuosityHypre::isMultiPhase);

    // =========================================================================
    // TortuosityDirect
    // =========================================================================
    py::class_<TortuosityDirect>(
        m, "TortuosityDirect",
        "Legacy iterative tortuosity solver using Forward-Euler time-stepping.")

        .def(py::init([](py::object geom_obj, py::object ba_obj, py::object dm_obj,
                         py::object mf_obj, int phase, OpenImpala::Direction dir, amrex::Real eps,
                         int n_steps, int plot_interval, const std::string& plot_basename,
                         amrex::Real vlo, amrex::Real vhi) {
                 auto& geom = py::cast<const amrex::Geometry&>(geom_obj);
                 auto& ba = py::cast<const amrex::BoxArray&>(ba_obj);
                 auto& dm = py::cast<const amrex::DistributionMapping&>(dm_obj);
                 auto& mf = py::cast<const amrex::iMultiFab&>(mf_obj);
                 return new TortuosityDirect(geom, ba, dm, mf, phase, dir, eps, n_steps,
                                             plot_interval, plot_basename, vlo, vhi);
             }),
             py::arg("geom"), py::arg("ba"), py::arg("dm"), py::arg("mf_phase"), py::arg("phase"),
             py::arg("dir"), py::arg("eps"), py::arg("n_steps"), py::arg("plot_interval"),
             py::arg("plot_basename"), py::arg("vlo"), py::arg("vhi"),
             py::keep_alive<1, 2>(), // geom
             py::keep_alive<1, 3>(), // ba
             py::keep_alive<1, 4>(), // dm
             py::keep_alive<1, 5>()) // mf_phase

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

        .def(py::init([](py::object geom_obj, py::object ba_obj, py::object dm_obj,
                         py::object mf_obj, int phase_id, OpenImpala::Direction dir,
                         EffectiveDiffusivityHypre::SolverType solver_type,
                         const std::string& results_path, int verbose, bool write_plotfile) {
                 auto& geom = py::cast<const amrex::Geometry&>(geom_obj);
                 auto& ba = py::cast<const amrex::BoxArray&>(ba_obj);
                 auto& dm = py::cast<const amrex::DistributionMapping&>(dm_obj);
                 auto& mf = py::cast<const amrex::iMultiFab&>(mf_obj);
                 return new EffectiveDiffusivityHypre(geom, ba, dm, mf, phase_id, dir, solver_type,
                                                      results_path, verbose, write_plotfile);
             }),
             py::arg("geom"), py::arg("ba"), py::arg("dm"), py::arg("mf_phase"),
             py::arg("phase_id"), py::arg("dir"), py::arg("solver_type"), py::arg("results_path"),
             py::arg("verbose") = 1, py::arg("write_plotfile") = false,
             py::keep_alive<1, 2>(), // geom
             py::keep_alive<1, 3>(), // ba
             py::keep_alive<1, 4>(), // dm
             py::keep_alive<1, 5>()) // mf_phase

        .def("solve", &EffectiveDiffusivityHypre::solve,
             "Solve the cell problem.  Returns True if the solver converged.")

        .def("get_chi_solution", &EffectiveDiffusivityHypre::getChiSolution, py::arg("chi_field"),
             "Copy the solved corrector field into *chi_field*.")

        .def_property_readonly("solver_converged", &EffectiveDiffusivityHypre::getSolverConverged)
        .def_property_readonly("residual_norm",
                               &EffectiveDiffusivityHypre::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &EffectiveDiffusivityHypre::getSolverIterations);
}
