/** @file solvers.cpp
 *  @brief pybind11 bindings for OpenImpala transport-property solvers.
 *
 *  Accepts VoxelImage handles — no pyamrex dependency.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_CoordSys.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_iMultiFab.H>

#include "DeffTensor.H"
#include "EffectiveDiffusivityHypre.H"
#include "EffectiveDiffusivityMLMG.H"
#include "HypreStructSolver.H"
#include "TortuosityDirect.H"
#include "TortuosityHypre.H"
#include "TortuosityMLMG.H"
#include "VolumeFraction.H"
#include "VoxelImage.H"

namespace py = pybind11;
using namespace OpenImpala;

namespace {

py::tuple export_flux_numpy_pair(const amrex::MultiFab& mf_flux, const amrex::Geometry& geom) {
    const amrex::Box& domain = geom.Domain();
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    const std::size_t n_vox =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    auto jx_arr = OpenImpala::multifab_to_numpy(mf_flux, geom, 0);
    auto jy_arr = OpenImpala::multifab_to_numpy(mf_flux, geom, 1);
    auto jz_arr = OpenImpala::multifab_to_numpy(mf_flux, geom, 2);
    auto jmag_arr = OpenImpala::multifab_to_numpy(mf_flux, geom, 3);

    py::array_t<double> flux_zyx({static_cast<py::ssize_t>(nz), static_cast<py::ssize_t>(ny),
                                  static_cast<py::ssize_t>(nx), static_cast<py::ssize_t>(3)});
    auto* dst = static_cast<double*>(flux_zyx.request().ptr);
    const auto* jx = static_cast<const double*>(jx_arr.request().ptr);
    const auto* jy = static_cast<const double*>(jy_arr.request().ptr);
    const auto* jz = static_cast<const double*>(jz_arr.request().ptr);

    for (std::size_t idx = 0; idx < n_vox; ++idx) {
        dst[idx * 3 + 0] = jz[idx];
        dst[idx * 3 + 1] = jy[idx];
        dst[idx * 3 + 2] = jx[idx];
    }

    return py::make_tuple(flux_zyx, jmag_arr);
}

py::dict compute_effective_diffusivity_tensor(std::shared_ptr<VoxelImage> img, int phase_id,
                                              OpenImpala::SolverType solver_type,
                                              const std::string& results_path, int verbose,
                                              bool write_plotfile,
                                              const std::map<int, amrex::Real>& phase_coeffs,
                                              bool return_chi, bool use_mlmg, amrex::Real mlmg_eps,
                                              int mlmg_maxiter, int mlmg_max_coarsening_level,
                                              OpenImpala::MLMGPrecision mlmg_precision) {
    if (!img || !img->mf) {
        throw std::runtime_error("Cannot compute effective diffusivity on an empty VoxelImage.");
    }

    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom_per;
    geom_per.define(img->geom.Domain(), &(img->geom.ProbDomain()), amrex::CoordSys::cartesian,
                    is_periodic.data());

    amrex::iMultiFab mf_phase_per(img->ba, img->dm, 1, 1);
    amrex::Copy(mf_phase_per, *(img->mf), 0, 0, 1, 0);
    mf_phase_per.FillBoundary(geom_per.periodicity());

    amrex::MultiFab mf_chi_x(img->ba, img->dm, 1, 1);
    amrex::MultiFab mf_chi_y(img->ba, img->dm, 1, 1);
    amrex::MultiFab mf_chi_z(img->ba, img->dm, 1, 1);
    mf_chi_x.setVal(0.0);
    mf_chi_y.setVal(0.0);
    mf_chi_z.setVal(0.0);

    amrex::iMultiFab active_mask_0g(img->ba, img->dm, 1, 0);
    amrex::MultiFab diff_coeff_0g(img->ba, img->dm, 1, 0);
    active_mask_0g.setVal(0);
    diff_coeff_0g.setVal(0.0);

    const std::array<OpenImpala::Direction, 3> dirs = {
        OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};
    amrex::MultiFab* chi_ptrs[3] = {&mf_chi_x, &mf_chi_y, &mf_chi_z};

    std::array<bool, 3> dir_converged = {false, false, false};
    std::array<int, 3> dir_iterations = {0, 0, 0};
    std::array<double, 3> dir_residuals = {0.0, 0.0, 0.0};
    long active_cells = 0;
    bool empty_active_phase = false;
    bool all_converged = true;

    amrex::iMultiFab shared_cell_type;
    bool has_shared_cell_type = false;

    for (int d = 0; d < 3; ++d) {
        if (use_mlmg) {
            const amrex::iMultiFab* cell_type_ptr =
                has_shared_cell_type ? &shared_cell_type : nullptr;
            EffectiveDiffusivityMLMG solver(geom_per, img->ba, img->dm, mf_phase_per, phase_id,
                                            dirs[d], results_path, verbose, write_plotfile,
                                            mlmg_eps, mlmg_maxiter, mlmg_max_coarsening_level,
                                            phase_coeffs, cell_type_ptr, mlmg_precision);

            if (d == 0) {
                amrex::Copy(active_mask_0g, solver.getActiveMask(), 0, 0, 1, 0);
                amrex::Copy(diff_coeff_0g, solver.getDiffCoeff(), 0, 0, 1, 0);
                VolumeFraction vf_calc(active_mask_0g, 1, 0);
                long long active_ll = 0;
                long long total_ll = 0;
                vf_calc.value(active_ll, total_ll, false);
                active_cells = static_cast<long>(active_ll);
                if (active_cells == 0) {
                    empty_active_phase = true;
                    dir_converged = {true, true, true};
                    break;
                }
                shared_cell_type.define(img->ba, img->dm, 1, 1);
                amrex::Copy(shared_cell_type, solver.getCellTypeMask(), 0, 0, 1, 0);
                shared_cell_type.FillBoundary(geom_per.periodicity());
                has_shared_cell_type = true;
            }

            const bool ok = solver.solve();
            dir_converged[d] = ok && solver.getSolverConverged();
            dir_iterations[d] = solver.getSolverIterations();
            dir_residuals[d] = static_cast<double>(solver.getFinalRelativeResidualNorm());

            if (!dir_converged[d]) {
                all_converged = false;
                break;
            }

            solver.getChiSolution(*chi_ptrs[d]);
            chi_ptrs[d]->FillBoundary(geom_per.periodicity());
        } else {
            EffectiveDiffusivityHypre solver(geom_per, img->ba, img->dm, mf_phase_per, phase_id,
                                             dirs[d], solver_type, results_path, verbose,
                                             write_plotfile, phase_coeffs);

            if (d == 0) {
                amrex::Copy(active_mask_0g, solver.getActiveMask(), 0, 0, 1, 0);
                amrex::Copy(diff_coeff_0g, solver.getDiffCoeff(), 0, 0, 1, 0);
                VolumeFraction vf_calc(active_mask_0g, 1, 0);
                long long active_ll = 0;
                long long total_ll = 0;
                vf_calc.value(active_ll, total_ll, false);
                active_cells = static_cast<long>(active_ll);
                if (active_cells == 0) {
                    empty_active_phase = true;
                    dir_converged = {true, true, true};
                    break;
                }
            }

            const bool ok = solver.solve();
            dir_converged[d] = ok && solver.getSolverConverged();
            dir_iterations[d] = solver.getSolverIterations();
            dir_residuals[d] = static_cast<double>(solver.getFinalRelativeResidualNorm());

            if (!dir_converged[d]) {
                all_converged = false;
                break;
            }

            solver.getChiSolution(*chi_ptrs[d]);
            chi_ptrs[d]->FillBoundary(geom_per.periodicity());
        }
    }

    const long total_cells = geom_per.Domain().numPts();
    const double active_vf =
        (total_cells > 0) ? static_cast<double>(active_cells) / static_cast<double>(total_cells)
                          : 0.0;

    py::array_t<double> tensor_arr({static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)});
    auto* tensor_ptr = static_cast<double*>(tensor_arr.request().ptr);

    if (empty_active_phase) {
        for (int i = 0; i < 9; ++i) {
            tensor_ptr[i] = 0.0;
        }
        if (write_plotfile) {
            amrex::MultiFab mf_zero_plot(img->ba, img->dm, 2, 0);
            mf_zero_plot.setVal(0.0);
            const amrex::Vector<std::string> varnames = {"chi_k", "active_mask_from_solver"};
            for (int d = 0; d < 3; ++d) {
                const std::string full_plot_path =
                    results_path + "/effdiff_chi_dir" + std::to_string(d);
                amrex::WriteSingleLevelPlotfile(full_plot_path, mf_zero_plot, varnames, geom_per,
                                                0.0, 0);
            }
        }
    } else if (all_converged) {
        amrex::Real Deff_raw[AMREX_SPACEDIM][AMREX_SPACEDIM];
        OpenImpala::calculateDeffTensor(Deff_raw, mf_chi_x, mf_chi_y, mf_chi_z, active_mask_0g,
                                        geom_per, verbose, &diff_coeff_0g);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                tensor_ptr[r * 3 + c] = static_cast<double>(Deff_raw[r][c]);
            }
        }
    } else {
        for (int i = 0; i < 9; ++i) {
            tensor_ptr[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    py::dict out;
    out["tensor"] = tensor_arr;
    out["active_volume_fraction"] = active_vf;
    out["converged"] = all_converged;
    out["dir_converged"] = py::make_tuple(dir_converged[0], dir_converged[1], dir_converged[2]);
    out["iterations"] = py::make_tuple(dir_iterations[0], dir_iterations[1], dir_iterations[2]);
    out["residuals"] = py::make_tuple(dir_residuals[0], dir_residuals[1], dir_residuals[2]);

    if (return_chi) {
        py::dict chi_dict;
        chi_dict["x"] = OpenImpala::multifab_to_numpy(mf_chi_x, geom_per, 0);
        chi_dict["y"] = OpenImpala::multifab_to_numpy(mf_chi_y, geom_per, 0);
        chi_dict["z"] = OpenImpala::multifab_to_numpy(mf_chi_z, geom_per, 0);
        out["chi"] = chi_dict;
    }

    return out;
}

} // namespace

void init_solvers(py::module_& m) {
    // =========================================================================
    // TortuosityHypre
    // =========================================================================
    py::class_<TortuosityHypre>(m, "TortuosityHypre",
                                "HYPRE-based tortuosity solver.  Solves the steady-state diffusion "
                                "equation on a masked or multi-phase domain and computes "
                                "tortuosity and effective diffusivity from the resulting flux.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, amrex::Real vf, int phase,
                         OpenImpala::Direction dir, TortuosityHypre::SolverType solver_type,
                         const std::string& results_path, amrex::Real vlo, amrex::Real vhi,
                         int verbose, bool write_plotfile, OpenImpala::PrecondType preconditioner,
                         const std::map<int, amrex::Real>& phase_coeffs) {
                 return new TortuosityHypre(img->geom, img->ba, img->dm, *(img->mf), vf, phase, dir,
                                            solver_type, results_path, vlo, vhi, verbose,
                                            write_plotfile, preconditioner, phase_coeffs);
             }),
             py::arg("img"), py::arg("vf"), py::arg("phase"), py::arg("dir"),
             py::arg("solver_type"), py::arg("results_path"), py::arg("vlo") = 0.0,
             py::arg("vhi") = 1.0, py::arg("verbose") = 0, py::arg("write_plotfile") = false,
             py::arg("preconditioner") = OpenImpala::PrecondType::SMG,
             py::arg("phase_coeffs") = std::map<int, amrex::Real>{}, py::keep_alive<1, 2>())

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

        .def(
            "value_raw", [](TortuosityHypre& self, bool refresh) { return self.value(refresh); },
            py::arg("refresh") = false, "Return tortuosity (NaN on failure, no exception).")

        .def(
            "solution_field",
            [](const TortuosityHypre& self) {
                return OpenImpala::multifab_to_numpy(self.getSolutionField(), self.getGeometry(),
                                                     0);
            },
            "Return the solved 3-D steady-state potential field as a float64 NumPy array "
            "of shape (nz, ny, nx).")

        .def(
            "flux_field",
            [](const TortuosityHypre& self) {
                amrex::MultiFab mf_flux;
                self.computeCellFluxField(mf_flux);
                return export_flux_numpy_pair(mf_flux, self.getGeometry());
            },
            "Return (flux_zyx, flux_magnitude) where flux_zyx has shape (nz, ny, nx, 3) ordered "
            "(Jz, Jy, Jx) and flux_magnitude has shape (nz, ny, nx).")

        .def("set_write_flux_field", &TortuosityHypre::setWriteFluxField, py::arg("enable") = true,
             "Include 3-D flux components in the AMReX solution plotfile when write_plotfile=True.")

        .def("check_matrix_properties", &TortuosityHypre::checkMatrixProperties,
             "Verify HYPRE matrix / RHS properties (for debugging).")

        .def_property_readonly("solver_converged", &TortuosityHypre::getSolverConverged)
        .def_property_readonly("residual_norm", &TortuosityHypre::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &TortuosityHypre::getSolverIterations)
        .def_property_readonly("flux_in", &TortuosityHypre::getFluxIn)
        .def_property_readonly("flux_out", &TortuosityHypre::getFluxOut)
        .def_property_readonly("effective_diffusivity", &TortuosityHypre::getEffectiveDiffusivity)
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
                               "multigrid (supports float64, float32, and mixed precision).")

        .def(py::init([](std::shared_ptr<VoxelImage> img, amrex::Real vf, int phase,
                         OpenImpala::Direction dir, const std::string& results_path,
                         amrex::Real vlo, amrex::Real vhi, int verbose, bool write_plotfile,
                         amrex::Real eps, int maxiter, int max_coarsening_level,
                         const std::map<int, amrex::Real>& phase_coeffs,
                         OpenImpala::MLMGPrecision precision) {
                 return new TortuosityMLMG(img->geom, img->ba, img->dm, *(img->mf), vf, phase, dir,
                                           results_path, vlo, vhi, verbose, write_plotfile, eps,
                                           maxiter, max_coarsening_level, phase_coeffs, precision);
             }),
             py::arg("img"), py::arg("vf"), py::arg("phase"), py::arg("dir"),
             py::arg("results_path"), py::arg("vlo") = 0.0, py::arg("vhi") = 1.0,
             py::arg("verbose") = 0, py::arg("write_plotfile") = false, py::arg("eps") = 1.0e-8,
             py::arg("maxiter") = 200, py::arg("max_coarsening_level") = 30,
             py::arg("phase_coeffs") = std::map<int, amrex::Real>{},
             py::arg("precision") = OpenImpala::MLMGPrecision::Float64, py::keep_alive<1, 2>())

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

        .def(
            "solution_field",
            [](const TortuosityMLMG& self) {
                return OpenImpala::multifab_to_numpy(self.getSolutionField(), self.getGeometry(),
                                                     0);
            },
            "Return the solved 3-D steady-state potential field as a float64 NumPy array "
            "of shape (nz, ny, nx).")

        .def(
            "flux_field",
            [](const TortuosityMLMG& self) {
                amrex::MultiFab mf_flux;
                self.computeCellFluxField(mf_flux);
                return export_flux_numpy_pair(mf_flux, self.getGeometry());
            },
            "Return (flux_zyx, flux_magnitude) where flux_zyx has shape (nz, ny, nx, 3) ordered "
            "(Jz, Jy, Jx) and flux_magnitude has shape (nz, ny, nx).")

        .def("set_write_flux_field", &TortuosityMLMG::setWriteFluxField, py::arg("enable") = true,
             "Include 3-D flux components in the AMReX solution plotfile when write_plotfile=True.")

        .def_property_readonly("solver_converged", &TortuosityMLMG::getSolverConverged)
        .def_property_readonly("residual_norm", &TortuosityMLMG::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &TortuosityMLMG::getSolverIterations)
        .def_property_readonly("flux_in", &TortuosityMLMG::getFluxIn)
        .def_property_readonly("flux_out", &TortuosityMLMG::getFluxOut)
        .def_property_readonly("effective_diffusivity", &TortuosityMLMG::getEffectiveDiffusivity)
        .def_property_readonly("active_volume_fraction", &TortuosityMLMG::getActiveVolumeFraction)
        .def_property_readonly("plane_fluxes", &TortuosityMLMG::getPlaneFluxes)
        .def_property_readonly("plane_flux_max_deviation",
                               &TortuosityMLMG::getPlaneFluxMaxDeviation)
        .def_property_readonly("is_multi_phase", &TortuosityMLMG::isMultiPhase)
        .def_property_readonly("precision", &TortuosityMLMG::getPrecision);

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
                         const std::string& results_path, int verbose, bool write_plotfile,
                         const std::map<int, amrex::Real>& phase_coeffs) {
                 return new EffectiveDiffusivityHypre(img->geom, img->ba, img->dm, *(img->mf),
                                                      phase_id, dir, solver_type, results_path,
                                                      verbose, write_plotfile, phase_coeffs);
             }),
             py::arg("img"), py::arg("phase_id"), py::arg("dir"), py::arg("solver_type"),
             py::arg("results_path"), py::arg("verbose") = 1, py::arg("write_plotfile") = false,
             py::arg("phase_coeffs") = std::map<int, amrex::Real>{}, py::keep_alive<1, 2>())

        .def("solve", &EffectiveDiffusivityHypre::solve,
             "Solve the cell problem.  Returns True if the solver converged.")

        .def("get_chi_solution", &EffectiveDiffusivityHypre::getChiSolution, py::arg("chi_field"),
             "Copy the solved corrector field into *chi_field*.")

        .def_property_readonly("solver_converged", &EffectiveDiffusivityHypre::getSolverConverged)
        .def_property_readonly("residual_norm",
                               &EffectiveDiffusivityHypre::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &EffectiveDiffusivityHypre::getSolverIterations);

    // =========================================================================
    // EffectiveDiffusivityMLMG
    // =========================================================================
    py::class_<EffectiveDiffusivityMLMG>(
        m, "EffectiveDiffusivityMLMG",
        "Matrix-free AMReX MLMG solver for 3-D periodic homogenisation cell problems.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_id, OpenImpala::Direction dir,
                         const std::string& results_path, int verbose, bool write_plotfile,
                         amrex::Real eps, int maxiter, int max_coarsening_level,
                         const std::map<int, amrex::Real>& phase_coeffs,
                         OpenImpala::MLMGPrecision precision) {
                 return new EffectiveDiffusivityMLMG(
                     img->geom, img->ba, img->dm, *(img->mf), phase_id, dir, results_path, verbose,
                     write_plotfile, eps, maxiter, max_coarsening_level, phase_coeffs, nullptr,
                     precision);
             }),
             py::arg("img"), py::arg("phase_id"), py::arg("dir"), py::arg("results_path") = ".",
             py::arg("verbose") = 0, py::arg("write_plotfile") = false, py::arg("eps") = 1.0e-8,
             py::arg("maxiter") = 200, py::arg("max_coarsening_level") = 30,
             py::arg("phase_coeffs") = std::map<int, amrex::Real>{},
             py::arg("precision") = OpenImpala::MLMGPrecision::Float64, py::keep_alive<1, 2>())

        .def("solve", &EffectiveDiffusivityMLMG::solve,
             "Solve the periodic cell problem.  Returns True if the solver converged.")

        .def("get_chi_solution", &EffectiveDiffusivityMLMG::getChiSolution, py::arg("chi_field"),
             "Copy the solved corrector field into *chi_field*.")

        .def_property_readonly("solver_converged", &EffectiveDiffusivityMLMG::getSolverConverged)
        .def_property_readonly("residual_norm",
                               &EffectiveDiffusivityMLMG::getFinalRelativeResidualNorm)
        .def_property_readonly("iterations", &EffectiveDiffusivityMLMG::getSolverIterations)
        .def_property_readonly("precision", &EffectiveDiffusivityMLMG::getPrecision);

    // =========================================================================
    // Full 3x3 D_eff tensor homogenization helper
    // =========================================================================
    m.def("compute_effective_diffusivity_tensor", &compute_effective_diffusivity_tensor,
          py::arg("img"), py::arg("phase_id") = 1,
          py::arg("solver_type") = OpenImpala::SolverType::FlexGMRES, py::arg("results_path") = ".",
          py::arg("verbose") = 0, py::arg("write_plotfile") = false,
          py::arg("phase_coeffs") = std::map<int, amrex::Real>{}, py::arg("return_chi") = false,
          py::arg("use_mlmg") = false, py::arg("mlmg_eps") = 1.0e-8, py::arg("mlmg_maxiter") = 200,
          py::arg("mlmg_max_coarsening_level") = 30,
          py::arg("mlmg_precision") = OpenImpala::MLMGPrecision::Float64,
          "Solve the 3 periodic homogenization cell problems and assemble the 3x3 effective "
          "diffusivity tensor D_eff (in X=0, Y=1, Z=2 order).");
}
