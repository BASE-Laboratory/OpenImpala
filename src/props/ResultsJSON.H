/** @file ResultsJSON.H
 *  @brief Structured JSON output for transport results (BPX/BattINFO compatible).
 */
// --- ResultsJSON.H ---
//
// Structured JSON output for OpenImpala transport property results.
//
// Implements the "Mimic and Passthrough" strategy for interoperability
// with battery modelling standards (BPX, BattINFO, BDF):
//
//   - "openimpala" block: native results with full solver provenance
//   - "units" block: SI unit annotations (BattINFO-aligned)
//   - "bpx" block: BPX-compatible electrode fragment for PyBaMM import
//   - "provenance" block: passthrough metadata (sample_id, provenance_uri)
//     for linking to external digital-twin databases (Faraday Engine, BDF)
//
// The JSON output is written alongside the existing results.txt file.
//
// Usage:
//   OpenImpala::ResultsJSON json;
//   json.setPhysicsConfig(physics_config);
//   json.setGridInfo(nx, ny, nz, box_size);
//   json.setSolverInfo("FlexGMRES", converged);
//   json.setProvenance(sample_id, provenance_uri);
//   json.setVolumeFraction(vf);
//   json.addDirectionResult("X", D_eff_ratio);
//   json.setBPXElectrode("negative");            // optional: enable BPX fragment
//   json.write("/path/to/results.json");

#ifndef RESULTSJSON_H
#define RESULTSJSON_H

#include "PhysicsConfig.H"

#ifndef OPENIMPALA_UNIT_TEST
#include <AMReX_SPACE.H>
#else
#ifndef AMREX_SPACEDIM
#define AMREX_SPACEDIM 3
#endif
#ifndef AMREX_D_DECL
#define AMREX_D_DECL(a, b, c) a, b, c
#endif
#endif

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

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

    /// Set BPX electrode type to enable BPX-compatible output fragment.
    /// Valid values: "negative", "positive", "separator".
    void setBPXElectrode(const std::string& electrode) {
        m_bpx_electrode = electrode;
    }

    // --- Microstructure parameter setters ---

    /// Set corrected specific surface area (Cauchy-Crofton corrected, voxel units).
    void setSpecificSurfaceArea(Real ssa) {
        m_ssa = ssa;
        m_has_ssa = true;
    }

    /// Set raw specific surface area (uncorrected voxel face count per volume).
    void setSpecificSurfaceAreaRaw(Real ssa_raw) {
        m_ssa_raw = ssa_raw;
        m_has_ssa_raw = true;
    }

    /// Set multi-phase volume fractions (phase_id -> VF).
    void setMultiPhaseVolumeFractions(const std::map<int, Real>& phase_vfs) {
        m_phase_vfs = phase_vfs;
    }

    /// Set macro-geometric parameters (thickness, cross-section, volume).
    void setMacroGeometry(Real thickness, Real cross_section, Real total_volume) {
        m_macro_thickness = thickness;
        m_macro_cross_section = cross_section;
        m_macro_volume = total_volume;
        m_has_macro = true;
    }

    /// Set through-thickness volume fraction profile for a direction.
    void setThroughThicknessProfile(const std::string& dir, const std::vector<Real>& vf_profile) {
        m_tt_profiles[dir] = vf_profile;
    }

    /// Set solid-phase Bruggeman exponent.
    void setSolidPhaseBruggeman(Real b_s) {
        m_solid_bruggeman = b_s;
        m_has_solid_bruggeman = true;
    }

    /// Set particle size distribution results.
    void setParticleSizeDistribution(Real mean_radius, int num_particles,
                                     const std::vector<Real>& radii) {
        m_psd_mean_radius = mean_radius;
        m_psd_num_particles = num_particles;
        m_psd_radii = radii;
        m_has_psd = true;
    }

    /// Set the full effective diffusivity tensor (from homogenization).
    void setDeffTensor(const Real tensor[AMREX_SPACEDIM][AMREX_SPACEDIM]) {
        for (int r = 0; r < AMREX_SPACEDIM; ++r) {
            for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                m_deff_tensor[r][c] = tensor[r][c];
            }
        }
        m_has_deff_tensor = true;
    }

    /// Set voxel size in meters for unit conversion in BPX output.
    void setVoxelSize(Real voxel_size_m) {
        m_voxel_size_m = voxel_size_m;
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

        // D_eff tensor (from homogenization)
        if (m_has_deff_tensor) {
            auto& tensor_obj = oi["deff_tensor"];
            const char* labels[AMREX_SPACEDIM] = {AMREX_D_DECL("X", "Y", "Z")};
            for (int r = 0; r < AMREX_SPACEDIM; ++r) {
                for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                    std::string key = std::string(labels[r]) + std::string(labels[c]);
                    tensor_obj[key] = m_deff_tensor[r][c];
                }
            }
        }

        // --- microstructure block: additional geometric parameters ---
        buildMicrostructureBlock(oi);

        // --- units block: SI unit annotations (BattINFO-aligned) ---
        buildUnitsBlock(j);

        // --- bpx block: BPX-compatible electrode fragment ---
        if (!m_bpx_electrode.empty()) {
            buildBPXBlock(j);
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

    // BPX electrode type (empty = no BPX fragment)
    std::string m_bpx_electrode;

    // Microstructure parameters
    Real m_ssa = 0.0;
    bool m_has_ssa = false;
    Real m_ssa_raw = 0.0;
    bool m_has_ssa_raw = false;
    std::map<int, Real> m_phase_vfs;
    Real m_macro_thickness = 0.0;
    Real m_macro_cross_section = 0.0;
    Real m_macro_volume = 0.0;
    bool m_has_macro = false;
    std::map<std::string, std::vector<Real>> m_tt_profiles;
    Real m_solid_bruggeman = 0.0;
    bool m_has_solid_bruggeman = false;
    Real m_psd_mean_radius = 0.0;
    int m_psd_num_particles = 0;
    std::vector<Real> m_psd_radii;
    bool m_has_psd = false;
    Real m_voxel_size_m = 0.0;

    // D_eff tensor (from homogenization)
    Real m_deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM] = {};
    bool m_has_deff_tensor = false;

    // --- Build microstructure parameters block ---
    void buildMicrostructureBlock(nlohmann::ordered_json& oi) const {
        if (!m_has_ssa && !m_has_ssa_raw && m_phase_vfs.empty() && !m_has_macro &&
            m_tt_profiles.empty() && !m_has_solid_bruggeman && !m_has_psd) {
            return;
        }

        auto& ms = oi["microstructure"];

        if (m_has_ssa) {
            ms["specific_surface_area"] = m_ssa;
        }
        if (m_has_ssa_raw) {
            ms["specific_surface_area_raw"] = m_ssa_raw;
        }

        if (!m_phase_vfs.empty()) {
            auto& pvf = ms["phase_volume_fractions"];
            for (const auto& [pid, vf] : m_phase_vfs) {
                pvf[std::to_string(pid)] = vf;
            }
        }

        if (m_has_macro) {
            auto& mg = ms["macro_geometry"];
            mg["thickness_voxels"] = m_macro_thickness;
            mg["cross_section_voxels"] = m_macro_cross_section;
            mg["total_volume_voxels"] = m_macro_volume;
        }

        if (!m_tt_profiles.empty()) {
            auto& ttp = ms["through_thickness_profiles"];
            for (const auto& [dir, profile] : m_tt_profiles) {
                ttp[dir]["volume_fraction"] = profile;
            }
        }

        if (m_has_solid_bruggeman) {
            ms["solid_phase_bruggeman"] = m_solid_bruggeman;
        }

        if (m_has_psd) {
            auto& psd = ms["particle_size"];
            psd["mean_radius_voxels"] = m_psd_mean_radius;
            psd["num_particles"] = m_psd_num_particles;
            psd["radii_voxels"] = m_psd_radii;
        }

        if (m_voxel_size_m > 0.0) {
            ms["voxel_size_m"] = m_voxel_size_m;
        }
    }

    // --- Build BattINFO-aligned SI unit annotations ---
    void buildUnitsBlock(nlohmann::ordered_json& j) const {
        auto& u = j["units"];
        u["volume_fraction"] = "1";
        u["tortuosity"] = "1";

        if (m_physics.type == PhysicsType::Diffusion) {
            u[m_physics.ratio_label] = "1";
        } else if (m_physics.type == PhysicsType::ElectricalConductivity) {
            u[m_physics.ratio_label] = "1";
            u["formation_factor"] = "1";
            if (m_physics.bulk_property != 1.0) {
                u[m_physics.eff_property_label] = "S.m-1";
            }
        } else if (m_physics.type == PhysicsType::ThermalConductivity) {
            u[m_physics.ratio_label] = "1";
            if (m_physics.bulk_property != 1.0) {
                u[m_physics.eff_property_label] = "W.m-1.K-1";
            }
        } else if (m_physics.type == PhysicsType::DielectricPermittivity) {
            u[m_physics.ratio_label] = "1";
            if (m_physics.bulk_property != 1.0) {
                u[m_physics.eff_property_label] = "F.m-1";
            }
        } else if (m_physics.type == PhysicsType::MagneticPermeability) {
            u[m_physics.ratio_label] = "1";
            if (m_physics.bulk_property != 1.0) {
                u[m_physics.eff_property_label] = "H.m-1";
            }
        }
    }

    /// Compute mean D_eff_ratio across all computed directions.
    Real meanDeffRatio() const {
        if (m_deff_ratios.empty())
            return 0.0;
        Real sum = 0.0;
        for (const auto& entry : m_deff_ratios) {
            sum += entry.second;
        }
        return sum / static_cast<Real>(m_deff_ratios.size());
    }

    /// Compute Bruggeman exponent from transport efficiency and porosity:
    ///   transport_efficiency = porosity^b  =>  b = log(TE) / log(porosity)
    /// Returns 0.0 if porosity or transport efficiency is non-positive (undefined).
    static Real bruggeman(Real transport_efficiency, Real porosity) {
        if (porosity <= 0.0 || porosity >= 1.0 || transport_efficiency <= 0.0) {
            return 0.0;
        }
        return std::log(transport_efficiency) / std::log(porosity);
    }

    /// Capitalise first letter: "negative" -> "Negative"
    static std::string capitalise(const std::string& s) {
        if (s.empty())
            return s;
        std::string out = s;
        out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
        return out;
    }

    // --- Build BPX-compatible electrode fragment ---
    //
    // BPX key names follow the Faraday Institution BPX standard:
    //   https://github.com/FaradayInstitution/BPX
    //
    // Key relationships:
    //   transport_efficiency = D_eff / D_bulk = D_eff_ratio  (OpenImpala output)
    //   Bruggeman exponent b: transport_efficiency = porosity^b
    //   porosity = volume_fraction (of the conducting/electrolyte phase)
    //
    // The fragment uses the *mean* transport efficiency across computed
    // directions, since BPX is inherently isotropic. Per-direction values
    // are available in the openimpala block for anisotropic workflows.
    void buildBPXBlock(nlohmann::ordered_json& j) const {
        std::string prefix = capitalise(m_bpx_electrode) + " electrode";

        Real mean_te = meanDeffRatio();
        Real mean_tau = m_physics.tortuosityFactor(mean_te, m_volume_fraction);
        Real b = bruggeman(mean_te, m_volume_fraction);

        auto& bpx = j["bpx"];
        bpx["_comment"] = "BPX-compatible fragment. Merge into a full BPX file for PyBaMM import.";
        bpx["electrode"] = m_bpx_electrode;

        auto& params = bpx["Parameterisation"][prefix];
        params["Porosity"] = m_volume_fraction;
        params["Transport efficiency"] = mean_te;
        params["Bruggeman coefficient (electrolyte)"] = b;

        // Microstructure parameters in BPX
        if (m_has_solid_bruggeman) {
            params["Bruggeman coefficient (electrode)"] = m_solid_bruggeman;
        }
        if (m_has_psd && m_voxel_size_m > 0.0) {
            params["Particle radius [m]"] = m_psd_mean_radius * m_voxel_size_m;
        }
        if (m_has_macro && m_voxel_size_m > 0.0) {
            params["Electrode thickness [m]"] = m_macro_thickness * m_voxel_size_m;
        }

        // Per-direction transport efficiencies (BPX extension — not in base spec)
        if (m_deff_ratios.size() > 1) {
            auto& aniso = bpx["anisotropic"];
            aniso["_comment"] = "Per-direction values (extension, not in base BPX spec)";
            for (const auto& [dir, ratio] : m_deff_ratios) {
                aniso["transport_efficiency"][dir] = ratio;
                aniso["tortuosity"][dir] = m_physics.tortuosityFactor(ratio, m_volume_fraction);
            }
            aniso["mean_tortuosity"] = mean_tau;
        }
    }
};

} // namespace OpenImpala

#endif // RESULTSJSON_H
