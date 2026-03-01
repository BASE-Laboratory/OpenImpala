// --- PhysicsConfig.H ---
//
// Lightweight configuration for different Laplacian-type physics problems.
//
// The TortuosityHypre / EffectiveDiffusivityHypre solvers compute an effective
// transport ratio (D_eff / D_phase).  This struct maps that raw output to
// the appropriate physical quantity: effective conductivity, formation factor,
// thermal conductivity, etc.
//
// Usage:
//   auto physics = OpenImpala::PhysicsConfig::fromParmParse();
//   amrex::Real sigma_eff = physics.effectiveProperty(D_eff_ratio);
//
// Input file (all optional — defaults to Diffusion with unit bulk property):
//   physics.type = electrical_conductivity
//   physics.bulk_property = 1.5e7
//
// Unit testing:
//   Define OPENIMPALA_UNIT_TEST before including this header to compile
//   without AMReX dependencies.  In that mode, fromParmParse() and print()
//   are unavailable; use fromTypeString() and direct construction instead.

#ifndef PHYSICSCONFIG_H
#define PHYSICSCONFIG_H

#ifndef OPENIMPALA_UNIT_TEST
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#endif

#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>

namespace OpenImpala {

// When compiling for unit tests without AMReX, use double directly
#ifdef OPENIMPALA_UNIT_TEST
using Real = double;
#else
using Real = amrex::Real;
#endif

enum class PhysicsType {
    Diffusion,
    ElectricalConductivity,
    ThermalConductivity,
    DielectricPermittivity,
    MagneticPermeability
};

struct PhysicsConfig {
    PhysicsType type = PhysicsType::Diffusion;

    // Human-readable labels for output
    std::string name;
    std::string coeff_label;
    std::string field_label;
    std::string eff_property_label;
    std::string ratio_label;

    // User-supplied bulk/reference property value (e.g. sigma_bulk)
    Real bulk_property = 1.0;

    // --- Interpret raw solver output ---

    Real effectiveProperty(Real D_eff_ratio) const {
        return D_eff_ratio * bulk_property;
    }

    Real tortuosityFactor(Real D_eff_ratio, Real vf) const {
        if (D_eff_ratio > 0.0)
            return vf / D_eff_ratio;
        return std::numeric_limits<Real>::infinity();
    }

    Real formationFactor(Real D_eff_ratio) const {
        if (D_eff_ratio > 0.0)
            return 1.0 / D_eff_ratio;
        return std::numeric_limits<Real>::infinity();
    }

    // --- Factory: build from a type string (no AMReX dependency) ---
    // Returns true on success, false if type_str is unrecognised.
    static bool fromTypeString(const std::string& type_str_in, PhysicsConfig& cfg) {
        std::string type_str = type_str_in;
        std::transform(type_str.begin(), type_str.end(), type_str.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        if (type_str == "diffusion") {
            cfg.type = PhysicsType::Diffusion;
            cfg.name = "Diffusion";
            cfg.coeff_label = "D";
            cfg.field_label = "concentration";
            cfg.eff_property_label = "EffectiveDiffusivity";
            cfg.ratio_label = "Deff_ratio";
        } else if (type_str == "electrical_conductivity") {
            cfg.type = PhysicsType::ElectricalConductivity;
            cfg.name = "Electrical Conductivity";
            cfg.coeff_label = "sigma";
            cfg.field_label = "voltage";
            cfg.eff_property_label = "EffectiveConductivity";
            cfg.ratio_label = "sigma_eff_ratio";
        } else if (type_str == "thermal_conductivity") {
            cfg.type = PhysicsType::ThermalConductivity;
            cfg.name = "Thermal Conductivity";
            cfg.coeff_label = "k";
            cfg.field_label = "temperature";
            cfg.eff_property_label = "EffectiveThermalConductivity";
            cfg.ratio_label = "k_eff_ratio";
        } else if (type_str == "dielectric_permittivity") {
            cfg.type = PhysicsType::DielectricPermittivity;
            cfg.name = "Dielectric Permittivity";
            cfg.coeff_label = "epsilon";
            cfg.field_label = "potential";
            cfg.eff_property_label = "EffectivePermittivity";
            cfg.ratio_label = "eps_eff_ratio";
        } else if (type_str == "magnetic_permeability") {
            cfg.type = PhysicsType::MagneticPermeability;
            cfg.name = "Magnetic Permeability";
            cfg.coeff_label = "mu";
            cfg.field_label = "potential";
            cfg.eff_property_label = "EffectivePermeability";
            cfg.ratio_label = "mu_eff_ratio";
        } else {
            return false;
        }
        return true;
    }

#ifndef OPENIMPALA_UNIT_TEST
    // --- Factory: build from input file parameters ---
    static PhysicsConfig fromParmParse() {
        PhysicsConfig cfg;

        amrex::ParmParse pp("physics");
        std::string type_str = "diffusion";
        pp.query("type", type_str);
        pp.query("bulk_property", cfg.bulk_property);

        if (!fromTypeString(type_str, cfg)) {
            amrex::Abort("Unknown physics.type: '" + type_str +
                         "'. Valid options: diffusion, electrical_conductivity, "
                         "thermal_conductivity, dielectric_permittivity, magnetic_permeability");
        }

        return cfg;
    }

    // --- Print configuration summary ---
    void print(int verbose = 1) const {
        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Physics Type:      " << name << "\n";
            amrex::Print() << "  Coefficient:       " << coeff_label << "\n";
            amrex::Print() << "  Field:             " << field_label << "\n";
            amrex::Print() << "  Bulk Property:     " << bulk_property << "\n";
        }
    }
#endif

    // --- Write results for a single direction ---
    void writeDirectionResults(std::ostream& os, const std::string& dir_label, Real D_eff_ratio,
                               Real vf) const {
        os << ratio_label << "_" << dir_label << ": " << std::scientific << D_eff_ratio << "\n";
        os << "Tortuosity_" << dir_label << ": " << std::fixed << std::setprecision(9)
           << tortuosityFactor(D_eff_ratio, vf) << "\n";
        if (bulk_property != 1.0) {
            os << eff_property_label << "_" << dir_label << ": " << std::scientific
               << effectiveProperty(D_eff_ratio) << "\n";
        }
        if (type == PhysicsType::ElectricalConductivity) {
            os << "FormationFactor_" << dir_label << ": " << std::fixed << std::setprecision(9)
               << formationFactor(D_eff_ratio) << "\n";
        }
    }

    // --- Write file header ---
    void writeHeader(std::ostream& os, const std::string& input_file, int phase_id) const {
        os << "# Transport Property Calculation Results\n";
        os << "# Physics Type: " << name << "\n";
        os << "# Input File: " << input_file << "\n";
        os << "# Analysis Phase ID: " << phase_id << "\n";
        if (bulk_property != 1.0) {
            os << "# Bulk Property (" << coeff_label << "_bulk): " << std::scientific
               << bulk_property << "\n";
        }
        os << "# -----------------------------\n";
    }
};

} // namespace OpenImpala

#endif // PHYSICSCONFIG_H
