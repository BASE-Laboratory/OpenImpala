/** @file Diffusion.cpp
 *  @brief Main application driver for effective transport property calculations.
 *
 *  Orchestrates the full pipeline: image loading, pre-processing checks,
 *  solver execution (homogenization or flow-through), checkpoint/restart,
 *  and results output. Supports full-domain analysis and REV convergence studies.
 */

#include "ImageLoader.H"

#include "ConnectedComponents.H"
#include "DeffTensor.H"
#include "EffectiveDiffusivityHypre.H"
#include "EffectiveDiffusivityMLMG.H"
#include "FluxField.H"
#include "MacroGeometry.H"
#include "ParticleSizeDistribution.H"
#include "PercolationCheck.H"
#include "PhysicsConfig.H"
#include "REVStudy.H"
#include "ResultsJSON.H"
#include "SolverConfig.H"
#include "SpecificSurfaceArea.H"
#include "ThroughThicknessProfile.H"
#include "Tortuosity.H"
#include "TortuosityHypre.H"
#include "TortuosityMLMG.H"
#include "VolumeFraction.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Loop.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <AMReX_iMultiFab.H>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool broadcastFileExists(const std::filesystem::path& p) {
    int exists_flag = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        exists_flag = std::filesystem::exists(p) ? 1 : 0;
    }
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        amrex::ParallelDescriptor::Bcast(&exists_flag, 1,
                                         amrex::ParallelDescriptor::IOProcessorNumber());
    }
    return (exists_flag != 0);
}

void ensureDirectoryOnIORank(const std::filesystem::path& dir_path) {
    if (dir_path.empty()) {
        return;
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::filesystem::create_directories(dir_path);
    }
    amrex::ParallelDescriptor::Barrier();
}

// ---------------------------------------------------------------------------
// Microstructure parameterization: SSA, active SSA, macro geometry, profiles, PSD.
// ---------------------------------------------------------------------------
void runMicrostructureParams(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                             const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                             const amrex::Box& /*domain_box*/, int phase_id, int verbose,
                             OpenImpala::ResultsJSON& json_writer) {
    amrex::ParmParse pp_ms("microstructure");

    bool compute_ssa = false;
    bool compute_active_ssa = false;
    bool compute_profiles = false;
    bool compute_psd = false;
    int solid_phase_id = -1;
    amrex::Real voxel_size_m = 0.0;
    std::string profile_dir_str = "Z";

    pp_ms.query("compute_ssa", compute_ssa);
    pp_ms.query("compute_active_ssa", compute_active_ssa);
    pp_ms.query("compute_profiles", compute_profiles);
    pp_ms.query("compute_psd", compute_psd);
    pp_ms.query("solid_phase_id", solid_phase_id);
    pp_ms.query("voxel_size_m", voxel_size_m);
    pp_ms.query("profile_direction", profile_dir_str);

    if (!compute_ssa && !compute_active_ssa && !compute_profiles && !compute_psd) {
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
        std::string upper_profile_dir = profile_dir_str;
        std::transform(upper_profile_dir.begin(), upper_profile_dir.end(),
                       upper_profile_dir.begin(), ::toupper);
        int flow_dir_idx = 2; // default Z
        if (upper_profile_dir == "X")
            flow_dir_idx = 0;
        else if (upper_profile_dir == "Y")
            flow_dir_idx = 1;
        auto mg = OpenImpala::MacroGeometry::fromGeometry(geom, flow_dir_idx);
        json_writer.setMacroGeometry(mg.thickness, mg.cross_section, mg.total_volume);
        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Macro Geometry: thickness=" << mg.thickness
                           << " cross_section=" << mg.cross_section << " volume=" << mg.total_volume
                           << "\n";
        }
    }

    // Multi-phase volume fractions
    {
        int local_max_phase = 0;
        for (amrex::MFIter mfi(mf_phase); mfi.isValid(); ++mfi) {
            const auto& fab = mf_phase.const_array(mfi);
            const amrex::Box& bx = mfi.validbox();
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                local_max_phase = std::max(local_max_phase, fab(i, j, k, 0));
            });
        }
        int max_phase = local_max_phase;
        amrex::ParallelDescriptor::ReduceIntMax(max_phase);

        std::map<int, OpenImpala::Real> phase_vfs;
        for (int pid = 0; pid <= max_phase; ++pid) {
            OpenImpala::VolumeFraction vf_calc(mf_phase, pid);
            amrex::Real vf_val = vf_calc.value_vf(false);
            phase_vfs[pid] = vf_val;
        }
        json_writer.setMultiPhaseVolumeFractions(phase_vfs);
    }

    // SSA & Active SSA
    if (compute_ssa || compute_active_ssa) {
        int ssa_phase_a = 0;
        int ssa_phase_b = 1;
        int ssa_boundary_padding = 0;
        pp_ms.query("ssa_phase_a", ssa_phase_a);
        pp_ms.query("ssa_phase_b", ssa_phase_b);
        pp_ms.query("ssa_boundary_padding", ssa_boundary_padding);

        if (compute_ssa) {
            OpenImpala::SpecificSurfaceArea ssa_calc(geom, mf_phase, ssa_phase_a, ssa_phase_b, 0,
                                                     ssa_boundary_padding);
            amrex::Real ssa_raw = ssa_calc.value_ssa(false);
            amrex::Real ssa_corrected = ssa_calc.value_corrected(false);
            json_writer.setSpecificSurfaceArea(ssa_corrected);
            json_writer.setSpecificSurfaceAreaRaw(ssa_raw);

            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Specific Surface Area (raw): " << std::fixed
                               << std::setprecision(6) << ssa_raw << " (faces/voxel)\n";
                amrex::Print() << "  Specific Surface Area (corrected): " << std::fixed
                               << std::setprecision(6) << ssa_corrected << " (Cauchy-Crofton)\n";
            }
        }

        if (compute_active_ssa) {
            std::string active_dir_str = profile_dir_str;
            int active_phase = phase_id;
            pp_ms.query("ssa_active_direction", active_dir_str);
            pp_ms.query("ssa_active_phase", active_phase);
            if (active_phase != ssa_phase_a && active_phase != ssa_phase_b) {
                active_phase = ssa_phase_a;
            }
            const auto act_dir = OpenImpala::parseDirection(active_dir_str);
            OpenImpala::PercolationCheck pc(geom, ba, dm, mf_phase, active_phase, act_dir, verbose);
            OpenImpala::SpecificSurfaceArea active_ssa_calc(geom, mf_phase, ssa_phase_a,
                                                            ssa_phase_b, 0, ssa_boundary_padding,
                                                            &(pc.activeMask()), active_phase);
            const amrex::Real act_raw = active_ssa_calc.value_ssa(false);
            const amrex::Real act_corr = active_ssa_calc.value_corrected(false);
            const std::string dir_label = OpenImpala::PercolationCheck::directionString(act_dir);
            json_writer.setActiveSpecificSurfaceArea(act_corr, act_raw, dir_label);

            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Active Specific Surface Area (" << dir_label
                               << ", corrected): " << std::fixed << std::setprecision(6) << act_corr
                               << " (Cauchy-Crofton)\n";
            }
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

    int local_max_phase = 0;
    for (amrex::MFIter mfi(mf_phase); mfi.isValid(); ++mfi) {
        const auto& fab = mf_phase.const_array(mfi);
        const amrex::Box& bx = mfi.validbox();
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
            local_max_phase = std::max(local_max_phase, fab(i, j, k, 0));
        });
    }
    int max_phase = local_max_phase;
    amrex::ParallelDescriptor::ReduceIntMax(max_phase);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Volume Fractions ---\n";
    }
    for (int pid = 0; pid <= max_phase; ++pid) {
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
                       const amrex::Box& domain_box, int phase_id,
                       OpenImpala::SolverType solver_type, const std::string& solver_str,
                       const std::filesystem::path& results_path,
                       const OpenImpala::PhysicsConfig& physics_config, const std::string& filename,
                       int verbose, bool write_plotfiles, const std::filesystem::path& chk_dir,
                       bool chk_restart) {
    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Effective Diffusivity via Homogenization (Full Domain) ---\n";
    }

    if (!chk_dir.empty()) {
        ensureDirectoryOnIORank(chk_dir);
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

    const bool use_mlmg = OpenImpala::isMLMGSolverString(solver_str);
    amrex::iMultiFab shared_cell_type;
    bool has_shared_cell_type = false;

    for (const auto& dir : dirs) {
        std::string dir_str = (dir == OpenImpala::Direction::X)   ? "X"
                              : (dir == OpenImpala::Direction::Y) ? "Y"
                                                                  : "Z";
        amrex::MultiFab* target_chi = (dir == OpenImpala::Direction::X)   ? &mf_chi_x
                                      : (dir == OpenImpala::Direction::Y) ? &mf_chi_y
                                                                          : &mf_chi_z;

        const std::filesystem::path chk_prefix = chk_dir / ("chi_" + dir_str);
        const std::filesystem::path chk_done = chk_dir / ("chi_" + dir_str + ".done");
        if (!chk_dir.empty() && chk_restart && broadcastFileExists(chk_done)) {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Restoring chi_" << dir_str
                               << " from checkpoint: " << chk_prefix.string() << "\n";
            }
            amrex::MultiFab mf_loaded;
            amrex::VisMF::Read(mf_loaded, chk_prefix.string());
            target_chi->ParallelCopy(mf_loaded, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                                     amrex::IntVect::TheZeroVector(), geom.periodicity());
            target_chi->FillBoundary(geom.periodicity());
            continue;
        }

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Solving for Full Domain chi_" << dir_str << " ---\n";
        }

        std::filesystem::path plot_dir = results_path / ("FullDomain_chi_" + dir_str);
        if (write_plotfiles && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(plot_dir.string(), 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        try {
            if (use_mlmg) {
                const amrex::iMultiFab* cell_type_ptr =
                    has_shared_cell_type ? &shared_cell_type : nullptr;
                OpenImpala::EffectiveDiffusivityMLMG mlmg_solver(
                    geom, ba, dm, mf_phase, phase_id, dir, plot_dir.string(), verbose,
                    write_plotfiles, 1.0e-8, 200, 30, {}, cell_type_ptr);
                if (!has_shared_cell_type) {
                    shared_cell_type.define(ba, dm, 1, 1);
                    amrex::Copy(shared_cell_type, mlmg_solver.getCellTypeMask(), 0, 0, 1, 0);
                    shared_cell_type.FillBoundary(geom.periodicity());
                    has_shared_cell_type = true;
                }
                if (!mlmg_solver.solve()) {
                    all_converged = false;
                    break;
                }
                mlmg_solver.getChiSolution(*target_chi);
            } else {
                auto solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                    geom, ba, dm, mf_phase, phase_id, dir, solver_type, plot_dir.string(), verbose,
                    write_plotfiles);
                if (!solver->solve()) {
                    all_converged = false;
                    break;
                }
                solver->getChiSolution(*target_chi);
            }

            if (!chk_dir.empty()) {
                amrex::MultiFab mf_save(ba, dm, 1, 0);
                amrex::Copy(mf_save, *target_chi, 0, 0, 1, 0);
                amrex::VisMF::Write(mf_save, chk_prefix.string());
                amrex::ParallelDescriptor::Barrier();
                if (amrex::ParallelDescriptor::IOProcessor()) {
                    std::ofstream done_ofs(chk_done);
                    done_ofs << "ok\n";
                }
                amrex::ParallelDescriptor::Barrier();
            }
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
        mf_chi_x.FillBoundary(geom.periodicity());
        mf_chi_y.FillBoundary(geom.periodicity());
        if (AMREX_SPACEDIM == 3) {
            mf_chi_z.FillBoundary(geom.periodicity());
        }
        amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM];
        amrex::iMultiFab active_mask(ba, dm, 1, 0);
        for (amrex::MFIter mfi(active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tb = mfi.tilebox();
            auto const& mask_arr = active_mask.array(mfi);
            auto const& phase_arr = mf_phase.const_array(mfi);
            amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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

            OpenImpala::ResultsJSON json_writer;
            json_writer.setPhysicsConfig(physics_config);
            json_writer.setInputFile(filename);
            json_writer.setPhaseId(phase_id);
            json_writer.setGridInfo(domain_box.length(0), domain_box.length(1),
                                    domain_box.length(2), 0);
            json_writer.setSolverInfo(solver_str, all_converged);
            json_writer.setDeffTensor(Deff_tensor);

            const char* dir_labels[AMREX_SPACEDIM] = {AMREX_D_DECL("X", "Y", "Z")};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                json_writer.addDirectionResult(dir_labels[d], Deff_tensor[d][d]);
            }

            std::filesystem::path output_filepath = results_path / "results.txt";
            amrex::Print() << "\nWriting final results to: " << output_filepath << "\n";
            std::ofstream outfile(output_filepath);
            if (outfile.is_open()) {
                physics_config.writeHeader(outfile, filename, phase_id);
                outfile << "CalculationMethod: homogenization\n";
                outfile << "DeffTensor:\n";
                for (int r = 0; r < AMREX_SPACEDIM; ++r) {
                    outfile << "  [";
                    for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                        outfile << std::scientific << std::setprecision(9) << Deff_tensor[r][c]
                                << (c == AMREX_SPACEDIM - 1 ? "" : ", ");
                    }
                    outfile << "]\n";
                }
                outfile.close();
            } else {
                amrex::Warning("Could not open output file: " + output_filepath.string());
            }

            std::filesystem::path json_filepath = results_path / "results.json";
            if (json_writer.write(json_filepath.string())) {
                amrex::Print() << "Writing JSON results to:  " << json_filepath << "\n";
            } else {
                amrex::Warning("Could not write JSON results to: " + json_filepath.string());
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
                    const std::string& bpx_electrode, int verbose, bool write_plotfiles,
                    const std::filesystem::path& chk_dir, bool chk_restart) {
    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Full Domain Calculation: Tortuosity via Flow-Through ---\n";
    }

    if (!chk_dir.empty()) {
        ensureDirectoryOnIORank(chk_dir);
    }

    amrex::Real vlo = -1.0;
    amrex::Real vhi = 1.0;
    {
        amrex::ParmParse pp_tort("tortuosity");
        pp_tort.query("vlo", vlo);
        pp_tort.query("vhi", vhi);
    }

    bool calc_flux_stats = false;
    int flux_hist_bins = 16;
    {
        amrex::ParmParse pp_post("post_process");
        pp_post.query("calc_flux_stats", calc_flux_stats);
        pp_post.query("flux_hist_bins", flux_hist_bins);
    }

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

    bool all_converged = true;
    std::map<std::string, amrex::Real> tortuosity_results;
    std::map<std::string, amrex::Real> deff_ratio_results;
    std::map<std::string, OpenImpala::FluxStats> dir_flux_stats;
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
            else {
                amrex::Warning("Unrecognized direction token: '" + single_dir +
                               "'. Valid: X, Y, Z.");
            }
        }
    }

    if (directions_to_run.empty()) {
        amrex::Warning("No valid directions specified in 'direction' input. Skipping tortuosity.");
        return;
    }

    amrex::Geometry geom_tort;
    {
        amrex::RealBox rb_np = geom.ProbDomain();
        amrex::Array<int, AMREX_SPACEDIM> is_periodic_np = {AMREX_D_DECL(0, 0, 0)};
        geom_tort.define(domain_box, &rb_np, 0, is_periodic_np.data());
    }

    for (const auto& dir : directions_to_run) {
        std::string dir_char = (dir == OpenImpala::Direction::X)   ? "X"
                               : (dir == OpenImpala::Direction::Y) ? "Y"
                                                                   : "Z";

        const std::filesystem::path chk_meta = chk_dir / ("flow_" + dir_char + ".json");
        if (!chk_dir.empty() && chk_restart && broadcastFileExists(chk_meta)) {
            amrex::Real loaded_vals[2] = {0.0, 0.0};
            int loaded_conv = 0;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ifstream ifs(chk_meta);
                nlohmann::json cj;
                ifs >> cj;
                loaded_vals[0] = cj.value("tortuosity", 0.0);
                loaded_vals[1] = cj.value("deff_ratio", 0.0);
                loaded_conv = cj.value("converged", false) ? 1 : 0;
                if (cj.contains("flux_stats")) {
                    OpenImpala::FluxStats fs;
                    fs.mean_magnitude = cj["flux_stats"].value("mean_magnitude", 0.0);
                    fs.max_magnitude = cj["flux_stats"].value("max_magnitude", 0.0);
                    fs.std_magnitude = cj["flux_stats"].value("std_magnitude", 0.0);
                    fs.histogram_edges =
                        cj["flux_stats"].value("histogram_edges", std::vector<amrex::Real>{});
                    fs.histogram_counts =
                        cj["flux_stats"].value("histogram_counts", std::vector<long long>{});
                    dir_flux_stats[dir_char] = std::move(fs);
                }
            }
            if (amrex::ParallelDescriptor::NProcs() > 1) {
                const int io_rank = amrex::ParallelDescriptor::IOProcessorNumber();
                amrex::ParallelDescriptor::Bcast(loaded_vals, 2, io_rank);
                amrex::ParallelDescriptor::Bcast(&loaded_conv, 1, io_rank);
            }
            tortuosity_results["Tortuosity_" + dir_char] = loaded_vals[0];
            deff_ratio_results[dir_char] = loaded_vals[1];
            if (loaded_conv == 0) {
                all_converged = false;
            }
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Restored flow-through direction " << dir_char
                               << " from checkpoint (tau=" << loaded_vals[0] << ")\n";
            }
            continue;
        }

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Solving for Tortuosity in Direction: " << dir_char << " ---\n";
        }

        amrex::Real tau = 0.0;
        bool dir_converged = false;
        amrex::MultiFab mf_sol_save;
        if (OpenImpala::isMLMGSolverString(solver_str)) {
            OpenImpala::TortuosityMLMG tort_solver(geom_tort, ba, dm, mf_phase, volume_fraction,
                                                   phase_id, dir, results_path.string(), vlo, vhi,
                                                   verbose, write_plotfiles);
            tau = tort_solver.value();
            dir_converged = tort_solver.getSolverConverged();
            if (dir_converged && calc_flux_stats) {
                dir_flux_stats[dir_char] = tort_solver.computeFluxStats(flux_hist_bins);
            }
            if (!chk_dir.empty() && dir_converged) {
                mf_sol_save.define(ba, dm, 1, 0);
                amrex::Copy(mf_sol_save, tort_solver.getSolutionField(), 0, 0, 1, 0);
            }
        } else {
            auto solver_type_enum = OpenImpala::parseSolverType(solver_str);
            OpenImpala::TortuosityHypre tort_solver(
                geom_tort, ba, dm, mf_phase, volume_fraction, phase_id, dir, solver_type_enum,
                results_path.string(), vlo, vhi, verbose, write_plotfiles);
            tau = tort_solver.value();
            dir_converged = tort_solver.getSolverConverged();
            if (dir_converged && calc_flux_stats) {
                dir_flux_stats[dir_char] = tort_solver.computeFluxStats(flux_hist_bins);
            }
            if (!chk_dir.empty() && dir_converged) {
                mf_sol_save.define(ba, dm, 1, 0);
                amrex::Copy(mf_sol_save, tort_solver.getSolutionField(), 0, 0, 1, 0);
            }
        }
        amrex::Real D_eff_ratio =
            (tau > 0.0 && !std::isnan(tau) && !std::isinf(tau)) ? volume_fraction / tau : 0.0;

        tortuosity_results["Tortuosity_" + dir_char] = tau;
        deff_ratio_results[dir_char] = D_eff_ratio;
        if (!dir_converged) {
            all_converged = false;
        }

        if (!chk_dir.empty() && dir_converged) {
            const std::filesystem::path sol_prefix = chk_dir / ("flow_sol_" + dir_char);
            amrex::VisMF::Write(mf_sol_save, sol_prefix.string());
            amrex::ParallelDescriptor::Barrier();
            if (amrex::ParallelDescriptor::IOProcessor()) {
                nlohmann::ordered_json cj;
                cj["direction"] = dir_char;
                cj["tortuosity"] = tau;
                cj["deff_ratio"] = D_eff_ratio;
                cj["converged"] = dir_converged;
                auto fs_it = dir_flux_stats.find(dir_char);
                if (fs_it != dir_flux_stats.end()) {
                    cj["flux_stats"]["mean_magnitude"] = fs_it->second.mean_magnitude;
                    cj["flux_stats"]["max_magnitude"] = fs_it->second.max_magnitude;
                    cj["flux_stats"]["std_magnitude"] = fs_it->second.std_magnitude;
                    cj["flux_stats"]["histogram_edges"] = fs_it->second.histogram_edges;
                    cj["flux_stats"]["histogram_counts"] = fs_it->second.histogram_counts;
                }
                std::ofstream ofs(chk_meta);
                ofs << cj.dump(2) << "\n";
            }
            amrex::ParallelDescriptor::Barrier();
        }

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

    OpenImpala::ResultsJSON json_writer;
    json_writer.setPhysicsConfig(physics_config);
    json_writer.setInputFile(filename);
    json_writer.setPhaseId(phase_id);
    json_writer.setGridInfo(domain_box.length(0), domain_box.length(1), domain_box.length(2), 0);
    json_writer.setSolverInfo(solver_str, all_converged);
    json_writer.setProvenance(provenance_sample_id, provenance_uri);
    json_writer.setVolumeFraction(volume_fraction);
    for (const auto& pair : deff_ratio_results) {
        json_writer.addDirectionResult(pair.first, pair.second);
    }
    for (const auto& [dir_k, fs] : dir_flux_stats) {
        json_writer.setDirectionFluxStats(dir_k, fs.mean_magnitude, fs.max_magnitude,
                                          fs.std_magnitude, fs.histogram_edges,
                                          fs.histogram_counts);
    }
    if (!bpx_electrode.empty()) {
        json_writer.setBPXElectrode(bpx_electrode);
    }

    runMicrostructureParams(geom, ba, dm, mf_phase, domain_box, phase_id, verbose, json_writer);

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
    amrex::Initialize(argc, argv);
    HYPRE_Init();
    {
        amrex::Real master_strt_time = amrex::second();

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
        bool rev_resume = false;

        std::string checkpoint_dir_str;
        bool checkpoint_restart = false;
        int plotfile_nfiles = -1;

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
            ppr.query("resume", rev_resume);

            amrex::ParmParse pp_chk("checkpoint");
            pp_chk.query("dir", checkpoint_dir_str);
            pp_chk.query("restart", checkpoint_restart);

            amrex::ParmParse pp_plt("plotfile");
            if (pp_plt.query("nfiles", plotfile_nfiles) && plotfile_nfiles > 0) {
                amrex::VisMF::SetNOutFiles(plotfile_nfiles);
            }
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
        std::filesystem::path checkpoint_path = checkpoint_dir_str.empty()
                                                    ? std::filesystem::path{}
                                                    : std::filesystem::path(checkpoint_dir_str);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                if (!std::filesystem::exists(main_results_path)) {
                    std::filesystem::create_directories(main_results_path);
                    if (main_verbose >= 1) {
                        amrex::Print()
                            << "Created results directory: " << main_results_path.string() << "\n";
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                amrex::Abort("Failed to create results directory '" + main_results_path.string() +
                             "': " + e.what());
            }
        }
        amrex::ParallelDescriptor::Barrier();

        OpenImpala::ImageData img;
        try {
            img = OpenImpala::loadImage(full_input_path, main_hdf5_dataset, main_threshold_val,
                                        main_box_size, main_verbose);
        } catch (const std::exception& e) {
            amrex::Print() << "Error loading data: " << e.what() << std::endl;
            amrex::Abort("Data loading failed.");
        }

        if (dry_run) {
            runDryRunChecks(img.geom, img.ba, img.dm, img.mf_phase, img.domain_box, main_filename,
                            main_phase_id, main_box_size, main_threshold_val, main_calc_method,
                            main_verbose);

        } else if (rev_do_study) {
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
            rev_config.resume = rev_resume;
            if (OpenImpala::isMLMGSolverString(rev_solver_str)) {
                rev_config.use_mlmg = true;
                amrex::ParmParse pp_mlmg("mlmg");
                std::string prec_s;
                if (pp_mlmg.query("precision", prec_s)) {
                    rev_config.mlmg_precision = OpenImpala::parseMLMGPrecision(prec_s);
                }
            } else {
                rev_config.solver_type = OpenImpala::parseSolverType(rev_solver_str);
            }
            rev_config.results_path = main_results_path;
            rev_config.csv_filename = rev_results_filename;

            OpenImpala::runREVStudy(img.geom, img.ba, img.dm, img.mf_phase, rev_config);
        }

        if (!dry_run && (!rev_do_study || main_calc_method != "skip_if_rev")) {
            if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Full Domain Calculation (" << main_calc_method
                               << ") using phase " << main_phase_id << " ---\n";
            }

            if (main_calc_method == "homogenization") {
                const auto homog_solver_type = OpenImpala::isMLMGSolverString(main_solver_str)
                                                   ? OpenImpala::SolverType::FlexGMRES
                                                   : OpenImpala::parseSolverType(main_solver_str);
                runHomogenization(img.geom, img.ba, img.dm, img.mf_phase, img.domain_box,
                                  main_phase_id, homog_solver_type, main_solver_str,
                                  main_results_path, physics_config, main_filename, main_verbose,
                                  (main_write_plotfile != 0), checkpoint_path, checkpoint_restart);
            } else if (main_calc_method == "flow_through") {
                runFlowThrough(img.geom, img.ba, img.dm, img.mf_phase, img.domain_box,
                               main_phase_id, main_solver_str, main_results_path, physics_config,
                               main_filename, provenance_sample_id, provenance_uri, bpx_electrode,
                               main_verbose, (main_write_plotfile != 0), checkpoint_path,
                               checkpoint_restart);
            } else {
                amrex::Abort("Unrecognized calculation_method: '" + main_calc_method +
                             "'. Valid options: 'homogenization', 'flow_through'.");
            }
        }

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
    HYPRE_Finalize();
    amrex::Finalize();
    return 0;
}
