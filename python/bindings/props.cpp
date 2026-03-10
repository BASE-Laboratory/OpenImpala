/** @file props.cpp
 *  @brief pybind11 bindings for lightweight property calculators.
 *
 *  Accepts VoxelImage handles — no pyamrex dependency.
 */

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "VolumeFraction.H"
#include "PercolationCheck.H"
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
    py::class_<PercolationCheck>(
        m, "PercolationCheck",
        "Parallel flood-fill connectivity check for a phase in a given direction.")

        .def(py::init([](std::shared_ptr<VoxelImage> img, int phase_id,
                         OpenImpala::Direction dir, int verbose) {
                 return new PercolationCheck(img->geom, img->ba, img->dm, *(img->mf),
                                            phase_id, dir, verbose);
             }),
             py::arg("img"), py::arg("phase_id"), py::arg("dir"),
             py::arg("verbose") = 0,
             // keep VoxelImage alive while this object lives
             py::keep_alive<1, 2>(),
             "Run a percolation check on construction.  Query results via properties.")

        .def_property_readonly(
            "percolates", &PercolationCheck::percolates,
            "True if the phase connects inlet to outlet in the specified direction.")

        .def_property_readonly(
            "active_volume_fraction", &PercolationCheck::activeVolumeFraction,
            "Volume fraction of the percolating (connected) subset of the phase.")

        .def_static("direction_string", &PercolationCheck::directionString, py::arg("dir"),
                    "Convert a Direction enum value to its string label (\"X\", \"Y\", or \"Z\").");
}
