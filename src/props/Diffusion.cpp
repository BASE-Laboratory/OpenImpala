/** @file Diffusion.cpp
 *  @brief Main application driver for effective transport property calculations.
 *
 *  Orchestrates the full pipeline: image loading, pre-processing checks,
 *  solver execution (homogenization or flow-through), and results output.
 *  Supports full-domain analysis and REV convergence studies.
 *
 *  This file acts as a thin orchestrator, delegating to:
 *    - ImageLoader    — file I/O and phase thresholding
 *    - DeffTensor     — effective diffusivity tensor assembly
 *    - REVStudy       — representative elementary volume analysis
 *    - TortuosityHypre / EffectiveDiffusivityHypre — solvers
 *    - PhysicsConfig  — physics-type interpretation
 *    - ResultsJSON    — structured output
 */

#include "ImageLoader.H"

#include "DeffTensor.H"
#include "REVStudy.H"
#include "SolverConfig.H"
#include "EffectiveDiffusivityHypre.H"
#include "TortuosityHypre.H"
#include "VolumeFraction.H"
#include "PercolationCheck.H"
#include "Tortuosity.H"
#include "PhysicsConfig.H"
#include "ResultsJSON.H"
#include "SpecificSurfaceArea.H"
#include "MacroGeometry.H"
#include "ThroughThicknessProfile.H"
#include "ConnectedComponents.H"
#include "ParticleSizeDistribution.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_ParmParse.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Loop.H>

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>

// Anonymous namespace for helpers local to this translation unit
namespace {

// ---------------------------------------------------------------------------
// Microstructure parameterization: SSA, macro geometry, profiles, PSD.
// ---------------------------------------------------------------------------
void runMicrostructureParams(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                             const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                             const amrex::Box& domain_box, int phase_id, int verbose,
                             OpenImpala::ResultsJSON& json_writer) {
    amrex::ParmParse pp_ms("microstructure");

    bool compute_ssa = false;
    bool compute_profiles = false;
    bool compute_psd = false;
    int solid_phase_id = -1;
    amrex::Real voxel_size_m = 0.0;
    std::string profile_dir_str = "Z";

    pp_ms.query("compute_ssa", compute_ssa);
    pp_ms.query("compute_profiles", compute_profiles);
    pp_ms.query("compute_psd", compute_psd);
    pp_ms.query("solid_phase_id", solid_phase_id);
    pp_ms.query("voxel_size_m", voxel_size_m);
    pp_ms.query("profile_direction", profile_dir_str);

    if (!compute_ssa && !compute_profiles && !compute_psd) {
        return;
    }

    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Microstructure Parameterization ---\n";
    }

    if (voxel_size_m > 0.0) {
        json_writer.setVoxelSize(voxel_size_m);
    }

    // Always compute macro geometry (trivial)
    {
        auto mg = OpenImpala::MacroGeometry::fromGeometry(geom, 2); // default Z
        json_writer.setMacroGeometry(mg.thickness, mg.cross_section, mg.total_volume);
        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Macro Geometry: thickness=" << mg.thickness
                           << " cross_section=" << mg.cross_section << " volume=" << mg.total_volume
                           << "\n";
        }
    }

    // Multi-phase volume fractions
    {
        std::map<int, OpenImpala::Real> phase_vfs;
        for (int pid = 0; pid <= 1; ++pid) {
            OpenImpala::VolumeFraction vf_calc(mf_phase, pid);
            amrex::Real vf_val = vf_calc.value_vf(false);
            phase_vfs[pid] = vf_val;
        }
        json_writer.setMultiPhaseVolumeFractions(phase_vfs);
    }

    // SSA
    if (compute_ssa) {
        int ssa_phase_a = 0;
        int ssa_phase_b = 1;
        pp_ms.query("ssa_phase_a", ssa_phase_a);
        pp_ms.query("ssa_phase_b", ssa_phase_b);

        OpenImpala::SpecificSurfaceArea ssa_calc(geom, mf_phase, ssa_phase_a, ssa_phase_b);
        amrex::Real ssa = ssa_calc.value_ssa(false);
        json_writer.setSpecificSurfaceArea(ssa);

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Specific Surface Area: " << std::fixed << std::setprecision(6)
                           << ssa << " (faces/voxel)\n";
        }
    }

    // Through-thickness profiles
    if (compute_profiles) {
        auto profile_dir = OpenImpala::parseDirection(profile_dir_str);
        OpenImpala::ThroughThicknessProfile profile(geom, mf_phase, phase_id, profile_dir);
        std::string dir_upper = profile_dir_str;
        std::transform(dir_upper.begin(), dir_upper.end(), dir_upper.begin(), ::toupper);
        json_writer.setThroughThicknessProfile(dir_upper, profile.volumeFractionProfile());

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Through-thickness profile (" << dir_upper
                           << "): " << profile.numSlices() << " slices\n";
        }
    }

    // Particle size distribution via CCL
    if (compute_psd) {
        int psd_phase = (solid_phase_id >= 0) ? solid_phase_id : 1;
        pp_ms.query("psd_phase_id", psd_phase);

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Computing PSD for phase " << psd_phase << "...\n";
        }

        OpenImpala::ConnectedComponents ccl(geom, ba, dm, mf_phase, psd_phase, verbose);
        OpenImpala::ParticleSizeDistribution psd(ccl);

        json_writer.setParticleSizeDistribution(psd.meanRadius(), psd.numParticles(), psd.radii());

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  PSD: " << psd.numParticles()
                           << " particles, mean radius = " << std::fixed << std::setprecision(4)
                           << psd.meanRadius() << " voxels\n";
        }
    }

    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "--- Microstructure Parameterization Complete ---\n\n";
    }
}

// ---------------------------------------------------------------------------
// Dry-run mode: domain summary, volume fractions, and percolation checks.
// ---------------------------------------------------------------------------
void runDryRunChecks(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                     const amrex::Box& domain_box, const std::string& filename, int phase_id,
                     int box_size, amrex::Real threshold_val, const std::string& calc_method,
                     int verbose) {
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n"
                       << "============================================================\n"
                       << "                    DRY RUN REPORT\n"
                       << "============================================================\n";
        amrex::Print() << "\n--- Domain Summary ---\n";
        amrex::Print() << "  Input file:    " << filename << "\n";
        amrex::Print() << "  Dimensions:    " << domain_box.length(0) << " x "
                       << domain_box.length(1) << " x " << domain_box.length(2) << "\n";
        amrex::Print() << "  Total voxels:  " << domain_box.numPts() << "\n";
        amrex::Print() << "  Box size:      " << box_size << "\n";
        amrex::Print() << "  Num boxes:     " << ba.size() << "\n";
        amrex::Print() << "  MPI ranks:     " << amrex::ParallelDescriptor::NProcs() << "\n";
        amrex::Print() << "  Threshold:     " << threshold_val << "\n";
        amrex::Print() << "  Analysis phase:" << phase_id << "\n";
        amrex::Print() << "  Calc method:   " << calc_method << "\n";
    }

    // Volume fractions for phases 0 and 1
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Volume Fractions ---\n";
    }
    for (int pid = 0; pid <= 1; ++pid) {
        OpenImpala::VolumeFraction vf_calc(mf_phase, pid);
        long long phase_count = 0, total_count = 0;
        vf_calc.value(phase_count, total_count, false);
        amrex::Real vf = (total_count > 0) ? static_cast<amrex::Real>(phase_count) /
                                                 static_cast<amrex::Real>(total_count)
                                           : 0.0;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::string marker = (pid == phase_id) ? " <-- analysis phase" : "";
            amrex::Print() << "  Phase " << pid << ": " << std::fixed << std::setprecision(6) << vf
                           << " (" << phase_count << " / " << total_count << " voxels)" << marker
                           << "\n";
        }
    }

    // Percolation checks in all 3 directions
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Percolation Check (Phase " << phase_id << ") ---\n";
    }

    amrex::Geometry geom_np;
    {
        amrex::RealBox rb_np = geom.ProbDomain();
        amrex::Array<int, AMREX_SPACEDIM> is_periodic_np = {AMREX_D_DECL(0, 0, 0)};
        geom_np.define(domain_box, &rb_np, 0, is_periodic_np.data());
    }

    bool any_failure = false;
    std::vector<OpenImpala::Direction> check_dirs = {
        OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};

    for (const auto& dir : check_dirs) {
        OpenImpala::PercolationCheck pc(geom_np, ba, dm, mf_phase, phase_id, dir, verbose);
        std::string dir_str = OpenImpala::PercolationCheck::directionString(dir);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (pc.percolates()) {
                amrex::Print() << "  " << dir_str
                               << "-direction: SUCCESS (percolates, active VF = " << std::fixed
                               << std::setprecision(6) << pc.activeVolumeFraction() << ")\n";
            } else {
                amrex::Print() << "  " << dir_str << "-direction: FAILURE (does not percolate)\n";
                any_failure = true;
            }
        }
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Recommendation ---\n";
        if (any_failure && calc_method == "flow_through") {
            amrex::Print() << "  WARNING: One or more directions do not percolate.\n"
                           << "  A flow-through tortuosity calculation in a non-percolating\n"
                           << "  direction will produce meaningless results (infinite "
                              "tortuosity).\n";
        } else {
            amrex::Print() << "  All checks passed. The dataset looks suitable for a "
                           << calc_method << " calculation.\n";
        }
        amrex::Print() << "\n"
                       << "============================================================\n"
                       << "  Dry run complete. No solver was executed.\n"
                       << "============================================================\n\n";
    }
}

// ---------------------------------------------------------------------------
// Full-domain homogenization: solve cell problems + assemble D_eff tensor.
// ---------------------------------------------------------------------------
void runHomogenization(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                       const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                       int phase_id, OpenImpala::SolverType solver_type,
                       const std::filesystem::path& results_path,
                       const OpenImpala::PhysicsConfig& physics_config, int verbose,
                       bool write_plotfiles) {
    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Effective Diffusivity via Homogenization (Full Domain) ---\n";
    }

    amrex::MultiFab mf_chi_x(ba, dm, 1, 1);
    amrex::MultiFab mf_chi_y(ba, dm, 1, 1);
    amrex::MultiFab mf_chi_z;
    if (AMREX_SPACEDIM == 3) {
        mf_chi_z.define(ba, dm, 1, 1);
    }

    bool all_converged = true;
    std::vector<OpenImpala::Direction> dirs = {OpenImpala::Direction::X, OpenImpala::Direction::Y};
    if (AMREX_SPACEDIM == 3) {
        dirs.push_back(OpenImpala::Direction::Z);
    }

    for (const auto& dir : dirs) {
        std::string dir_str = (dir == OpenImpala::Direction::X)   ? "X"
                              : (dir == OpenImpala::Direction::Y) ? "Y"
                                                                  : "Z";
        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Solving for Full Domain chi_" << dir_str << " ---\n";
        }

        std::filesystem::path plot_dir = results_path / ("FullDomain_chi_" + dir_str);
        if (write_plotfiles && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(plot_dir.string(), 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> solver;
        try {
            solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                geom, ba, dm, mf_phase, phase_id, dir, solver_type, plot_dir.string(), verbose,
                write_plotfiles);
            if (!solver->solve()) {
                all_converged = false;
                break;
            }
            if (dir == OpenImpala::Direction::X)
                solver->getChiSolution(mf_chi_x);
            else if (dir == OpenImpala::Direction::Y)
                solver->getChiSolution(mf_chi_y);
            else if (AMREX_SPACEDIM == 3 && dir == OpenImpala::Direction::Z)
                solver->getChiSolution(mf_chi_z);
        } catch (const std::exception& e) {
            if (verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "    Full Domain Chi solve EXCEPTION for dir "
                               << static_cast<int>(dir) << ": " << e.what() << std::endl;
            }
            all_converged = false;
            break;
        }
    }

    if (all_converged) {
        amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM];
        amrex::iMultiFab active_mask(ba, dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tb = mfi.tilebox();
            auto const& mask_arr = active_mask.array(mfi);
            auto const& phase_arr = mf_phase.const_array(mfi);
            amrex::LoopOnCpu(tb, [=](int i, int j, int k) {
                mask_arr(i, j, k, 0) = (phase_arr(i, j, k, 0) == phase_id) ? 1 : 0;
            });
        }
        OpenImpala::calculateDeffTensor(Deff_tensor, mf_chi_x, mf_chi_y, mf_chi_z, active_mask,
                                        geom, verbose);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Full Domain Effective " << physics_config.name << " Tensor ("
                           << physics_config.ratio_label << "):\n";
            for (int r = 0; r < AMREX_SPACEDIM; ++r) {
                amrex::Print() << "  [";
                for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                    amrex::Print() << std::scientific << std::setprecision(8) << Deff_tensor[r][c]
                                   << (c == AMREX_SPACEDIM - 1 ? "" : ", ");
                }
                amrex::Print() << "]\n";
            }
            if (physics_config.bulk_property != 1.0) {
                amrex::Print() << "Full Domain Absolute " << physics_config.eff_property_label
                               << " Tensor (scaled by " << physics_config.coeff_label
                               << "_bulk=" << std::scientific << physics_config.bulk_property
                               << "):\n";
                for (int r = 0; r < AMREX_SPACEDIM; ++r) {
                    amrex::Print() << "  [";
                    for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                        amrex::Print() << std::scientific << std::setprecision(8)
                                       << physics_config.effectiveProperty(Deff_tensor[r][c])
                                       << (c == AMREX_SPACEDIM - 1 ? "" : ", ");
                    }
                    amrex::Print() << "]\n";
                }
            }
        }
    } else {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
                << "Full domain D_eff calculation skipped due to chi_k non-convergence.\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Full-domain flow-through tortuosity.
// ---------------------------------------------------------------------------
void runFlowThrough(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                    const amrex::Box& domain_box, int phase_id, const std::string& solver_str,
                    const std::filesystem::path& results_path,
                    const OpenImpala::PhysicsConfig& physics_config, const std::string& filename,
                    const std::string& provenance_sample_id, const std::string& provenance_uri,
                    const std::string& bpx_electrode, int verbose, bool write_plotfiles) {
    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Full Domain Calculation: Tortuosity via Flow-Through ---\n";
    }

    // Boundary condition values
    amrex::Real vlo = -1.0;
    amrex::Real vhi = 1.0;
    {
        amrex::ParmParse pp_tort("tortuosity");
        pp_tort.query("vlo", vlo);
        pp_tort.query("vhi", vhi);
    }

    // Volume fraction
    if (verbose > 0) {
        amrex::Print() << "Calculating Volume Fraction for Phase ID: " << phase_id << "\n";
    }
    OpenImpala::VolumeFraction vf_calc(mf_phase, phase_id);
    long long phase_voxels = 0, total_voxels = 0;
    vf_calc.value(phase_voxels, total_voxels, false);
    amrex::Real volume_fraction =
        (total_voxels > 0)
            ? (static_cast<amrex::Real>(phase_voxels) / static_cast<amrex::Real>(total_voxels))
            : 0.0;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Volume Fraction = " << std::fixed << std::setprecision(8)
                       << volume_fraction << "\n";
    }

    // Parse directions
    std::map<std::string, amrex::Real> tortuosity_results;
    std::map<std::string, amrex::Real> deff_ratio_results;
    std::string direction_str;
    {
        amrex::ParmParse pp;
        pp.get("direction", direction_str);
    }
    std::string upper_dir = direction_str;
    std::transform(upper_dir.begin(), upper_dir.end(), upper_dir.begin(), ::toupper);

    std::vector<OpenImpala::Direction> directions_to_run;
    if (upper_dir.find("ALL") != std::string::npos) {
        directions_to_run = {OpenImpala::Direction::X, OpenImpala::Direction::Y,
                             OpenImpala::Direction::Z};
    } else {
        std::stringstream ss(upper_dir);
        std::string single_dir;
        while (ss >> single_dir) {
            if (single_dir == "X")
                directions_to_run.push_back(OpenImpala::Direction::X);
            else if (single_dir == "Y")
                directions_to_run.push_back(OpenImpala::Direction::Y);
            else if (single_dir == "Z")
                directions_to_run.push_back(OpenImpala::Direction::Z);
        }
    }

    if (directions_to_run.empty()) {
        amrex::Warning("No valid directions specified in 'direction' input. Skipping tortuosity.");
        return;
    }

    // Non-periodic geometry for tortuosity (Dirichlet inlet/outlet)
    amrex::Geometry geom_tort;
    {
        amrex::RealBox rb_np = geom.ProbDomain();
        amrex::Array<int, AMREX_SPACEDIM> is_periodic_np = {AMREX_D_DECL(0, 0, 0)};
        geom_tort.define(domain_box, &rb_np, 0, is_periodic_np.data());
    }

    // Solve tortuosity in each direction
    for (const auto& dir : directions_to_run) {
        std::string dir_char = (dir == OpenImpala::Direction::X)   ? "X"
                               : (dir == OpenImpala::Direction::Y) ? "Y"
                                                                   : "Z";
        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Solving for Tortuosity in Direction: " << dir_char << " ---\n";
        }

        auto solver_type_enum = OpenImpala::parseSolverType(solver_str);

        OpenImpala::TortuosityHypre tort_solver(
            geom_tort, ba, dm, mf_phase, volume_fraction, phase_id, dir, solver_type_enum,
            results_path.string(), vlo, vhi, verbose, write_plotfiles);

        amrex::Real tau = tort_solver.value();
        amrex::Real D_eff_ratio =
            (tau > 0.0 && !std::isnan(tau) && !std::isinf(tau)) ? volume_fraction / tau : 0.0;

        tortuosity_results["Tortuosity_" + dir_char] = tau;
        deff_ratio_results[dir_char] = D_eff_ratio;

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  >>> Calculated Tortuosity (" << dir_char << "): " << std::fixed
                           << std::setprecision(8) << tau << " <<<\n";
            if (physics_config.type != OpenImpala::PhysicsType::Diffusion) {
                amrex::Print() << "  >>> " << physics_config.eff_property_label << " (" << dir_char
                               << "): " << std::scientific
                               << physics_config.effectiveProperty(D_eff_ratio) << std::defaultfloat
                               << " <<<\n";
            }
        }
    }

    // --- Microstructure parameterization (runs on all ranks) ---
    OpenImpala::ResultsJSON json_writer;
    json_writer.setPhysicsConfig(physics_config);
    json_writer.setInputFile(filename);
    json_writer.setPhaseId(phase_id);
    json_writer.setGridInfo(domain_box.length(0), domain_box.length(1), domain_box.length(2),
                            0); // box_size not needed here
    json_writer.setSolverInfo(solver_str, true);
    json_writer.setProvenance(provenance_sample_id, provenance_uri);
    json_writer.setVolumeFraction(volume_fraction);
    for (const auto& pair : deff_ratio_results) {
        json_writer.addDirectionResult(pair.first, pair.second);
    }
    if (!bpx_electrode.empty()) {
        json_writer.setBPXElectrode(bpx_electrode);
    }

    // Run microstructure parameterization (SSA, PSD, profiles, etc.)
    runMicrostructureParams(geom, ba, dm, mf_phase, domain_box, phase_id, verbose, json_writer);

    // Write results files
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::string output_filename = "results.txt";
        {
            amrex::ParmParse pp;
            pp.query("output_filename", output_filename);
        }

        std::filesystem::path output_filepath = results_path / output_filename;
        amrex::Print() << "\nWriting final results to: " << output_filepath << "\n";

        std::ofstream outfile(output_filepath);
        if (outfile.is_open()) {
            physics_config.writeHeader(outfile, filename, phase_id);
            outfile << "VolumeFraction: " << std::fixed << std::setprecision(9) << volume_fraction
                    << "\n";
            for (const auto& pair : deff_ratio_results) {
                physics_config.writeDirectionResults(outfile, pair.first, pair.second,
                                                     volume_fraction);
            }
            outfile.close();
        } else {
            amrex::Warning("Could not open output file for writing: " + output_filepath.string());
        }

        // Structured JSON output
        std::filesystem::path json_filepath = results_path / "results.json";
        if (json_writer.write(json_filepath.string())) {
            amrex::Print() << "Writing JSON results to:  " << json_filepath << "\n";
        } else {
            amrex::Warning("Could not write JSON results to: " + json_filepath.string());
        }
    }
}

} // end anonymous namespace


// ===========================================================================
// Main — thin orchestrator
// ===========================================================================
int main(int argc, char* argv[]) {
    HYPRE_Init();
    amrex::Initialize(argc, argv);
    {
        amrex::Real master_strt_time = amrex::second();

        // ===================================================================
        // 1. Parse all configuration
        // ===================================================================
        std::string main_filename;
        std::string main_data_path_str = "./data/";
        std::string main_results_path_str = "./results_diffusion/";
        std::string main_hdf5_dataset = "image";
        amrex::Real main_threshold_val = 0.5;
        int main_phase_id = 1;
        std::string main_solver_str = "FlexGMRES";
        int main_box_size = 32;
        int main_verbose = 1;
        int main_write_plotfile = 0;
        std::string main_calc_method = "homogenization";
        bool dry_run = false;

        bool rev_do_study = false;
        int rev_num_samples = 3;
        std::string rev_sizes_str = "32 64 96";
        std::string rev_solver_str = "FlexGMRES";
        std::string rev_results_filename = "rev_study_Deff.csv";
        int rev_write_plotfiles = 0;
        int rev_verbose = 1;

        {
            amrex::ParmParse pp;
            pp.get("filename", main_filename);
            pp.query("data_path", main_data_path_str);
            pp.query("results_path", main_results_path_str);
            pp.query("hdf5_dataset", main_hdf5_dataset);
            pp.query("threshold_val", main_threshold_val);
            pp.query("phase_id", main_phase_id);
            pp.query("solver_type", main_solver_str);
            pp.query("box_size", main_box_size);
            pp.query("verbose", main_verbose);
            pp.query("write_plotfile", main_write_plotfile);
            pp.query("calculation_method", main_calc_method);
            pp.query("dry_run", dry_run);

            amrex::ParmParse ppr("rev");
            ppr.query("do_study", rev_do_study);
            ppr.query("num_samples", rev_num_samples);
            ppr.query("sizes", rev_sizes_str);
            ppr.query("solver_type", rev_solver_str);
            ppr.query("results_file", rev_results_filename);
            ppr.query("write_plotfiles", rev_write_plotfiles);
            ppr.query("verbose", rev_verbose);
        }

        auto physics_config = OpenImpala::PhysicsConfig::fromParmParse();
        if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Physics Configuration ---\n";
            physics_config.print(main_verbose);
            amrex::Print() << "-----------------------------\n\n";
        }

        std::string provenance_sample_id;
        std::string provenance_uri;
        {
            amrex::ParmParse pp_prov("provenance");
            pp_prov.query("sample_id", provenance_sample_id);
            pp_prov.query("uri", provenance_uri);
        }

        std::string bpx_electrode;
        {
            amrex::ParmParse pp_bpx("bpx");
            pp_bpx.query("electrode", bpx_electrode);
        }

        // Validate
        if (main_box_size <= 0) {
            amrex::Abort("Error: 'box_size' must be positive (got " +
                         std::to_string(main_box_size) + ").");
        }
        if (main_threshold_val < 0.0) {
            amrex::Abort("Error: 'threshold_val' must be non-negative (got " +
                         std::to_string(main_threshold_val) + ").");
        }

        std::filesystem::path main_data_path(main_data_path_str);
        std::filesystem::path main_results_path(main_results_path_str);
        std::filesystem::path full_input_path = main_data_path / main_filename;

        // Create results directory
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (!std::filesystem::exists(main_results_path)) {
                std::filesystem::create_directories(main_results_path);
                if (main_verbose >= 1) {
                    amrex::Print()
                        << "Created results directory: " << main_results_path.string() << "\n";
                }
            }
        }
        amrex::ParallelDescriptor::Barrier();

        // ===================================================================
        // 2. Load image data
        // ===================================================================
        OpenImpala::ImageData img;
        try {
            img = OpenImpala::loadImage(full_input_path, main_hdf5_dataset, main_threshold_val,
                                        main_box_size, main_verbose);
        } catch (const std::exception& e) {
            amrex::Print() << "Error loading data: " << e.what() << std::endl;
            amrex::Abort("Data loading failed.");
        }

        // ===================================================================
        // 3. Dispatch to appropriate calculation mode
        // ===================================================================
        if (dry_run) {
            runDryRunChecks(img.geom, img.ba, img.dm, img.mf_phase, img.domain_box, main_filename,
                            main_phase_id, main_box_size, main_threshold_val, main_calc_method,
                            main_verbose);

        } else if (rev_do_study) {
            // Parse REV sizes
            OpenImpala::REVConfig rev_config;
            rev_config.num_samples = rev_num_samples;
            {
                std::stringstream ss(rev_sizes_str);
                int sz;
                while (ss >> sz) {
                    rev_config.sizes.push_back(sz);
                }
            }
            rev_config.phase_id = main_phase_id;
            rev_config.box_size = main_box_size;
            rev_config.verbose = rev_verbose;
            rev_config.write_plotfiles = (rev_write_plotfiles != 0);
            rev_config.solver_type = OpenImpala::parseSolverType(rev_solver_str);
            rev_config.results_path = main_results_path;
            rev_config.csv_filename = rev_results_filename;

            OpenImpala::runREVStudy(img.geom, img.ba, img.dm, img.mf_phase, rev_config);
        }

        // Full domain calculation (runs unless dry_run, can run alongside REV)
        if (!dry_run && (!rev_do_study || main_calc_method != "skip_if_rev")) {
            if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Full Domain Calculation (" << main_calc_method
                               << ") using phase " << main_phase_id << " ---\n";
            }

            if (main_calc_method == "homogenization") {
                runHomogenization(img.geom, img.ba, img.dm, img.mf_phase, main_phase_id,
                                  OpenImpala::parseSolverType(main_solver_str), main_results_path,
                                  physics_config, main_verbose, (main_write_plotfile != 0));
            } else if (main_calc_method == "flow_through") {
                runFlowThrough(img.geom, img.ba, img.dm, img.mf_phase, img.domain_box,
                               main_phase_id, main_solver_str, main_results_path, physics_config,
                               main_filename, provenance_sample_id, provenance_uri, bpx_electrode,
                               main_verbose, (main_write_plotfile != 0));
            }
        }

        // ===================================================================
        // 4. Finish
        // ===================================================================
        if (amrex::ParmParse::QueryUnusedInputs() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Warning(
                "There are unused parameters in the inputs file (see list above). "
                "These may be typos. Set amrex.abort_on_unused_inputs=1 to treat as error.");
        }

        amrex::Real master_stop_time = amrex::second() - master_strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(master_stop_time,
                                                 amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << std::endl
                           << "Total run time (seconds) = " << master_stop_time << std::endl;
        }
    }
    amrex::Finalize();
    HYPRE_Finalize();
    return 0;
}
