/** @file props.cpp
 *  @brief pybind11 bindings for microstructural property calculators and REV studies.
 *
 *  Accepts VoxelImage handles — no pyamrex dependency.
 */

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_ParallelDescriptor.H>

#include "ConnectedComponents.H"
#include "ParticleSizeDistribution.H"
#include "PercolationCheck.H"
#include "REVStudy.H"
#include "SpecificSurfaceArea.H"
#include "ThroughThicknessProfile.H"
#include "VolumeFraction.H"
#include "VoxelImage.H"

namespace py = pybind11;
using namespace OpenImpala;

void init_props(py::module_& m) {
    // =========================================================================
    // VolumeFraction
    // =========================================================================
    py::class_<VolumeFraction>(m, "VolumeFraction",
                               "Computes volume fraction of a phase within a VoxelImage.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase, int comp) {
                 return new VolumeFraction(*(img->mf), phase, comp);
             }),
             py::arg("img"), py::arg("phase") = 0, py::arg("comp") = 0,
             // keep VoxelImage alive while this object lives
             py::keep_alive<1, 2>(),
             "Create a volume-fraction calculator for *phase* in component *comp* of *img*.")

        // C++ signature: void value(long long&, long long&, bool) const
        // Python: returns (phase_count, total_count) tuple
        .def(
            "value",
            [](const VolumeFraction& self, bool local) {
                long long phase_count = 0;
                long long total_count = 0;
                self.value(phase_count, total_count, local);
                return py::make_tuple(phase_count, total_count);
            },
            py::arg("local") = false,
            "Return (phase_count, total_count).  Set local=True to skip MPI reduction.")

        .def("value_vf", &VolumeFraction::value_vf, py::arg("local") = false,
             "Return the volume fraction as a float (phase_count / total_count).");

    // =========================================================================
    // PercolationCheck
    // =========================================================================
    py::class_<PercolationCheck, std::shared_ptr<PercolationCheck>>(
        m, "PercolationCheck",
        "Parallel flood-fill connectivity check for a phase in a given direction.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_id, OpenImpala::Direction dir,
                         int verbose) {
                 return std::make_shared<PercolationCheck>(img->geom, img->ba, img->dm, *(img->mf),
                                                           phase_id, dir, verbose);
             }),
             py::arg("img"), py::arg("phase_id"), py::arg("dir"), py::arg("verbose") = 0,
             // keep VoxelImage alive while this object lives
             py::keep_alive<1, 2>(),
             "Run a percolation check on construction.  Query results via properties.")

        .def_property_readonly(
            "percolates", &PercolationCheck::percolates,
            "True if the phase connects inlet to outlet in the specified direction.")

        .def_property_readonly(
            "active_volume_fraction", &PercolationCheck::activeVolumeFraction,
            "Volume fraction of the percolating (connected) subset of the phase.")

        .def(
            "active_mask",
            [](const PercolationCheck& self) {
                return OpenImpala::imultifab_to_numpy(self.activeMask(), self.getGeometry(), 0);
            },
            "Export the 3-D percolating activity mask (1 = active, 0 = inactive) as a "
            "C-contiguous int32 NumPy array of shape (nz, ny, nx).")

        .def_static("direction_string", &PercolationCheck::directionString, py::arg("dir"),
                    "Convert a Direction enum value to its string label (\"X\", \"Y\", or \"Z\").");

    // =========================================================================
    // SpecificSurfaceArea
    // =========================================================================
    py::class_<SpecificSurfaceArea>(
        m, "SpecificSurfaceArea",
        "Computes interfacial specific surface area between two phases in a VoxelImage.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_a, int phase_b, int comp,
                         int boundary_padding, std::shared_ptr<PercolationCheck> percolation,
                         int active_phase) {
                 const amrex::iMultiFab* act_ptr =
                     percolation ? &(percolation->activeMask()) : nullptr;
                 return new SpecificSurfaceArea(img->geom, *(img->mf), phase_a, phase_b, comp,
                                                boundary_padding, act_ptr, active_phase);
             }),
             py::arg("img"), py::arg("phase_a") = 0, py::arg("phase_b") = 1, py::arg("comp") = 0,
             py::arg("boundary_padding") = 0, py::arg("percolation") = nullptr,
             py::arg("active_phase") = -1, py::keep_alive<1, 2>(), py::keep_alive<1, 7>(),
             "Create a specific surface area calculator between *phase_a* and *phase_b*.")

        .def(
            "value",
            [](const SpecificSurfaceArea& self, bool local) {
                long long face_count = 0;
                long long total_count = 0;
                self.value(face_count, total_count, local);
                return py::make_tuple(face_count, total_count);
            },
            py::arg("local") = false, "Return (face_count, total_count).")

        .def(
            "direction_face_counts",
            [](const SpecificSurfaceArea& self, bool local) {
                amrex::Array<long long, AMREX_SPACEDIM> dir_faces{};
                long long total_count = 0;
                self.directionFaceCounts(dir_faces, total_count, local);
                return py::make_tuple(py::make_tuple(dir_faces[0], dir_faces[1], dir_faces[2]),
                                      total_count);
            },
            py::arg("local") = false, "Return ((faces_x, faces_y, faces_z), total_count).")

        .def("value_ssa", &SpecificSurfaceArea::value_ssa, py::arg("local") = false,
             "Raw voxel-unit SSA = face_count / total_count.")

        .def("value_corrected", &SpecificSurfaceArea::value_corrected, py::arg("local") = false,
             "Cauchy-Crofton (2/3) corrected voxel-unit SSA.")

        .def("value_physical", &SpecificSurfaceArea::value_physical, py::arg("local") = false,
             "Raw physical SSA accounting for (dx, dy, dz) cell sizes [1/length].")

        .def("value_physical_corrected", &SpecificSurfaceArea::value_physical_corrected,
             py::arg("local") = false,
             "Cauchy-Crofton (2/3) corrected physical SSA accounting for (dx, dy, dz) [1/length].");

    // =========================================================================
    // ConnectedComponents
    // =========================================================================
    py::class_<ConnectedComponents>(
        m, "ConnectedComponents",
        "Labels 6-connected components of a target phase via parallel flood-fill.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_id, int verbose) {
                 return new ConnectedComponents(img->geom, img->ba, img->dm, *(img->mf), phase_id,
                                                verbose);
             }),
             py::arg("img"), py::arg("phase_id") = 1, py::arg("verbose") = 0,
             py::keep_alive<1, 2>(), "Run 6-connected component labeling for *phase_id*.")

        .def_property_readonly("num_components", &ConnectedComponents::numComponents,
                               "Number of connected components found.")

        .def_property_readonly("component_volumes", &ConnectedComponents::componentVolumes,
                               "Voxel count of each connected component (1-indexed labels 1..N).")

        .def(
            "label_field",
            [](const ConnectedComponents& self) {
                return OpenImpala::imultifab_to_numpy(self.labelField(), self.getGeometry(), 0);
            },
            "Export the 3-D component label field (0 = background, 1..N = component ID) as a "
            "C-contiguous int32 NumPy array of shape (nz, ny, nx).");

    // =========================================================================
    // ParticleSizeDistribution
    // =========================================================================
    py::class_<ParticleSizeDistribution>(
        m, "ParticleSizeDistribution",
        "Computes equivalent-sphere radii R = (3V / (4*pi))^(1/3) from connected components.")

        .def(py::init<const ConnectedComponents&>(), py::arg("ccl"),
             "Compute particle size distribution from an existing ConnectedComponents object.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_id, int verbose) {
                 ConnectedComponents ccl(img->geom, img->ba, img->dm, *(img->mf), phase_id,
                                         verbose);
                 return new ParticleSizeDistribution(ccl);
             }),
             py::arg("img"), py::arg("phase_id") = 1, py::arg("verbose") = 0,
             "Run connected component labeling and compute equivalent-sphere radii directly.")

        .def_property_readonly("num_particles", &ParticleSizeDistribution::numParticles,
                               "Number of discrete particles/components.")

        .def_property_readonly("mean_radius", &ParticleSizeDistribution::meanRadius,
                               "Mean equivalent-sphere radius in voxel units.")

        .def_property_readonly("radii", &ParticleSizeDistribution::radii,
                               "Equivalent-sphere radius of each component in voxel units.")

        .def_property_readonly("volumes", &ParticleSizeDistribution::volumes,
                               "Voxel count of each component.");

    // =========================================================================
    // ThroughThicknessProfile
    // =========================================================================
    py::class_<ThroughThicknessProfile>(
        m, "ThroughThicknessProfile",
        "Computes per-slice volume fraction profile along a coordinate direction.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_id, OpenImpala::Direction dir,
                         int comp) {
                 return new ThroughThicknessProfile(img->geom, *(img->mf), phase_id, dir, comp);
             }),
             py::arg("img"), py::arg("phase_id") = 1, py::arg("dir") = OpenImpala::Direction::X,
             py::arg("comp") = 0, py::keep_alive<1, 2>(),
             "Compute slice-by-slice volume fraction of *phase_id* along *dir*.")

        .def_property_readonly("num_slices", &ThroughThicknessProfile::numSlices,
                               "Number of 1-voxel slices along the chosen direction.")

        .def_property_readonly("volume_fraction_profile",
                               &ThroughThicknessProfile::volumeFractionProfile,
                               "List of local volume fractions (one float per slice).");

    // =========================================================================
    // REV convergence study
    // =========================================================================
    m.def(
        "run_rev_study",
        [](std::shared_ptr<VoxelImage> img, const std::vector<int>& sizes, int num_samples,
           int phase_id, int box_size, OpenImpala::SolverType solver_type,
           const std::string& results_path, const std::string& csv_filename, int verbose,
           bool write_plotfiles, bool use_mlmg, amrex::Real mlmg_eps, int mlmg_maxiter,
           int mlmg_max_coarsening_level, OpenImpala::MLMGPrecision mlmg_precision, bool resume) {
            REVConfig cfg;
            cfg.sizes = sizes;
            cfg.num_samples = num_samples;
            cfg.phase_id = phase_id;
            cfg.box_size = box_size;
            cfg.solver_type = solver_type;
            cfg.use_mlmg = use_mlmg;
            cfg.mlmg_eps = mlmg_eps;
            cfg.mlmg_maxiter = mlmg_maxiter;
            cfg.mlmg_max_coarsening_level = mlmg_max_coarsening_level;
            cfg.mlmg_precision = mlmg_precision;
            cfg.resume = resume;
            cfg.results_path = std::filesystem::path(results_path);
            cfg.csv_filename = csv_filename;
            cfg.verbose = verbose;
            cfg.write_plotfiles = write_plotfiles;
            runREVStudy(img->geom, img->ba, img->dm, *(img->mf), cfg);
            amrex::ParallelDescriptor::Barrier();

            const std::string out_path = (cfg.results_path / cfg.csv_filename).string();
            std::string csv_content;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ifstream ifs(out_path);
                if (ifs) {
                    std::ostringstream oss;
                    oss << ifs.rdbuf();
                    csv_content = oss.str();
                }
            }
            if (amrex::ParallelDescriptor::NProcs() > 1) {
                int len = static_cast<int>(csv_content.size());
                amrex::ParallelDescriptor::Bcast(&len, 1,
                                                 amrex::ParallelDescriptor::IOProcessorNumber());
                if (!amrex::ParallelDescriptor::IOProcessor()) {
                    csv_content.resize(static_cast<std::size_t>(len));
                }
                if (len > 0) {
                    amrex::ParallelDescriptor::Bcast(
                        csv_content.data(), len, amrex::ParallelDescriptor::IOProcessorNumber());
                }
            }
            return py::make_tuple(out_path, csv_content);
        },
        py::arg("img"), py::arg("sizes"), py::arg("num_samples") = 3, py::arg("phase_id") = 1,
        py::arg("box_size") = 32, py::arg("solver_type") = OpenImpala::SolverType::FlexGMRES,
        py::arg("results_path") = ".", py::arg("csv_filename") = "rev_study_Deff.csv",
        py::arg("verbose") = 0, py::arg("write_plotfiles") = false, py::arg("use_mlmg") = false,
        py::arg("mlmg_eps") = 1.0e-8, py::arg("mlmg_maxiter") = 200,
        py::arg("mlmg_max_coarsening_level") = 30,
        py::arg("mlmg_precision") = OpenImpala::MLMGPrecision::Float64, py::arg("resume") = false,
        "Run a Representative Elementary Volume (REV) convergence study and return "
        "(csv_path, csv_content).");
}
