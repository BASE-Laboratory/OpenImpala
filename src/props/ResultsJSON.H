// --- ResultsJSON.H ---
//
// Structured JSON output for OpenImpala transport property results.
//
// Implements the "Mimic and Passthrough" strategy for interoperability
// with battery modelling standards (BPX, BattINFO, BDF):
//
//   - "openimpala" block: native results with full solver provenance
//   - "provenance" block: passthrough metadata (sample_id, provenance_uri)
//     for linking to external digital-twin databases (Faraday Engine, BDF)
//
// The JSON output is written alongside the existing results.txt file.
// Variable naming follows BattINFO conventions where applicable.
//
// Usage:
//   OpenImpala::ResultsJSON json;
//   json.setPhysicsConfig(physics_config);
//   json.setGridInfo(nx, ny, nz, box_size);
//   json.setSolverInfo("FlexGMRES", converged);
//   json.setProvenance(sample_id, provenance_uri);
//   json.setVolumeFraction(vf);
//   json.addDirectionResult("X", D_eff_ratio);
//   json.write("/path/to/results.json");

#ifndef RESULTSJSON_H
#define RESULTSJSON_H

#include "PhysicsConfig.H"

#include <nlohmann/json.hpp>

#include <fstream>
#include <map>
#include <string>

namespace OpenImpala {

class ResultsJSON {
public:
    // --- Setters for each data category ---

    void setPhysicsConfig(const PhysicsConfig& cfg) {
        m_physics = cfg;
    }

    void setInputFile(const std::string& filename) {
        m_input_file = filename;
    }

    void setPhaseId(int phase_id) {
        m_phase_id = phase_id;
    }

    void setGridInfo(int nx, int ny, int nz, int box_size) {
        m_nx = nx;
        m_ny = ny;
        m_nz = nz;
        m_box_size = box_size;
    }

    void setSolverInfo(const std::string& solver, bool converged) {
        m_solver = solver;
        m_converged = converged;
    }

    void setProvenance(const std::string& sample_id, const std::string& provenance_uri) {
        m_sample_id = sample_id;
        m_provenance_uri = provenance_uri;
    }

    void setVolumeFraction(Real vf) {
        m_volume_fraction = vf;
    }

    void addDirectionResult(const std::string& dir, Real D_eff_ratio) {
        m_deff_ratios[dir] = D_eff_ratio;
    }

    // --- Build the JSON object ---
    nlohmann::ordered_json buildJSON() const {
        nlohmann::ordered_json j;

        // --- openimpala block: native results with full provenance ---
        auto& oi = j["openimpala"];
        oi["version"] = OPENIMPALA_VERSION;
        oi["physics_type"] = m_physics.name;
        oi["input_file"] = m_input_file;
        oi["phase_id"] = m_phase_id;

        if (m_physics.bulk_property != 1.0) {
            oi["bulk_property"] = m_physics.bulk_property;
            oi["bulk_property_label"] = m_physics.coeff_label + "_bulk";
        }

        oi["solver"] = m_solver;
        oi["converged"] = m_converged;

        auto& grid = oi["grid"];
        grid["nx"] = m_nx;
        grid["ny"] = m_ny;
        grid["nz"] = m_nz;
        grid["box_size"] = m_box_size;

        oi["volume_fraction"] = m_volume_fraction;

        // Per-direction results
        auto& results = oi["results"];
        for (const auto& [dir, ratio] : m_deff_ratios) {
            auto& dir_obj = results[dir];
            dir_obj[m_physics.ratio_label] = ratio;
            dir_obj["tortuosity"] = m_physics.tortuosityFactor(ratio, m_volume_fraction);

            if (m_physics.bulk_property != 1.0) {
                dir_obj[m_physics.eff_property_label] = m_physics.effectiveProperty(ratio);
            }
            if (m_physics.type == PhysicsType::ElectricalConductivity) {
                dir_obj["formation_factor"] = m_physics.formationFactor(ratio);
            }
        }

        // --- provenance block: metadata passthrough for BDF/Faraday Engine ---
        if (!m_sample_id.empty() || !m_provenance_uri.empty()) {
            auto& prov = j["provenance"];
            if (!m_sample_id.empty()) {
                prov["sample_id"] = m_sample_id;
            }
            if (!m_provenance_uri.empty()) {
                prov["provenance_uri"] = m_provenance_uri;
            }
        }

        return j;
    }

    // --- Write JSON to file ---
    bool write(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            return false;
        }
        ofs << buildJSON().dump(2) << "\n";
        ofs.close();
        return true;
    }

private:
    static constexpr const char* OPENIMPALA_VERSION = "0.1.0";

    // Physics configuration
    PhysicsConfig m_physics;
    std::string m_input_file;
    int m_phase_id = 0;

    // Grid info
    int m_nx = 0;
    int m_ny = 0;
    int m_nz = 0;
    int m_box_size = 0;

    // Solver info
    std::string m_solver;
    bool m_converged = false;

    // Provenance passthrough
    std::string m_sample_id;
    std::string m_provenance_uri;

    // Results
    Real m_volume_fraction = 0.0;
    std::map<std::string, Real> m_deff_ratios;
};

} // namespace OpenImpala

#endif // RESULTSJSON_H
