/** @file config.cpp
 *  @brief pybind11 bindings for PhysicsConfig, ResultsJSON, and CathodeWrite.
 */

#include <stdexcept>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PhysicsConfig.H"
#include "ResultsJSON.H"
#include "CathodeWrite.H"

namespace py = pybind11;
using namespace OpenImpala;

void init_config(py::module_& m)
{
    // =========================================================================
    // PhysicsConfig
    // =========================================================================
    py::class_<PhysicsConfig>(m, "PhysicsConfig",
        "Maps solver output to physical quantities (diffusion, conductivity, etc.).")

        .def(py::init<>())

        .def_readwrite("type",               &PhysicsConfig::type)
        .def_readwrite("name",               &PhysicsConfig::name)
        .def_readwrite("coeff_label",        &PhysicsConfig::coeff_label)
        .def_readwrite("field_label",        &PhysicsConfig::field_label)
        .def_readwrite("eff_property_label", &PhysicsConfig::eff_property_label)
        .def_readwrite("ratio_label",        &PhysicsConfig::ratio_label)
        .def_readwrite("bulk_property",      &PhysicsConfig::bulk_property)

        .def("effective_property", &PhysicsConfig::effectiveProperty,
             py::arg("D_eff_ratio"))

        .def("tortuosity_factor", &PhysicsConfig::tortuosityFactor,
             py::arg("D_eff_ratio"), py::arg("vf"))

        .def("formation_factor", &PhysicsConfig::formationFactor,
             py::arg("D_eff_ratio"))

        .def_static("from_type_string", [](const std::string& type_str) {
                PhysicsConfig cfg;
                if (!PhysicsConfig::fromTypeString(type_str, cfg)) {
                    throw py::value_error("Unknown physics type: '" + type_str + "'");
                }
                return cfg;
             },
             py::arg("type_str"),
             "Create a PhysicsConfig from a type string (e.g. 'diffusion').")

        .def("__repr__", [](const PhysicsConfig& c) {
            return "<PhysicsConfig type='" + c.name + "'>";
        });

    // =========================================================================
    // ResultsJSON
    // =========================================================================
    py::class_<ResultsJSON>(m, "ResultsJSON",
        "Builder for structured JSON output (BPX / BattINFO compatible).")

        .def(py::init<>())

        .def("set_physics_config", &ResultsJSON::setPhysicsConfig,
             py::arg("config"))
        .def("set_input_file", &ResultsJSON::setInputFile,
             py::arg("filename"))
        .def("set_phase_id", &ResultsJSON::setPhaseId,
             py::arg("phase_id"))
        .def("set_grid_info", &ResultsJSON::setGridInfo,
             py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("box_size"))
        .def("set_solver_info", &ResultsJSON::setSolverInfo,
             py::arg("solver"), py::arg("converged"))
        .def("set_provenance", &ResultsJSON::setProvenance,
             py::arg("sample_id"), py::arg("provenance_uri"))
        .def("set_volume_fraction", &ResultsJSON::setVolumeFraction,
             py::arg("vf"))
        .def("add_direction_result", &ResultsJSON::addDirectionResult,
             py::arg("dir"), py::arg("D_eff_ratio"))
        .def("set_bpx_electrode", &ResultsJSON::setBPXElectrode,
             py::arg("electrode"))

        // Return JSON as a Python string (user can json.loads() it)
        .def("build_json_string", [](const ResultsJSON& self) {
                return self.buildJSON().dump(2);
             },
             "Build the JSON output and return it as a formatted string.")

        .def("write", &ResultsJSON::write,
             py::arg("filepath"),
             "Write the JSON output to a file.  Returns True on success.");

    // =========================================================================
    // CathodeParams
    // =========================================================================
    py::class_<CathodeParams>(m, "CathodeParams",
        "Parameter set for cathode electrode output.")

        .def(py::init<>())

        .def_readwrite("volume_fraction_solid",
             &CathodeParams::volume_fraction_solid)
        .def_readwrite("particle_radius",
             &CathodeParams::particle_radius)
        .def_readwrite("active_material_conductivity",
             &CathodeParams::active_material_conductivity)
        .def_readwrite("max_concentration",
             &CathodeParams::max_concentration);

    // =========================================================================
    // CathodeWrite
    // =========================================================================
    py::class_<CathodeWrite>(m, "CathodeWrite",
        "Writes cathode parameter files for battery-model integration.")

        .def(py::init<const CathodeParams&>(),
             py::arg("params"))

        .def("write_dandeliion_parameters",
             &CathodeWrite::writeDandeLiionParameters,
             py::arg("filename"),
             "Write DandeLiion-compatible parameter file.")

        .def("write_pybamm_parameters",
             &CathodeWrite::writePyBammParameters,
             py::arg("filename"),
             "Write PyBaMM-compatible parameter file.");
}
