/** @file REVStudy.cpp
 *  @brief Implementation of the REV convergence study module.
 */

#include "REVStudy.H"
#include "DeffTensor.H"
#include "EffectiveDiffusivityHypre.H"
#include "Tortuosity.H"

#include <AMReX_Loop.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace OpenImpala {

void runREVStudy(const amrex::Geometry& geom_full, const amrex::BoxArray& ba_full,
                 const amrex::DistributionMapping& dm_full, const amrex::iMultiFab& mf_phase_full,
                 const REVConfig& config) {
    BL_PROFILE("OpenImpala::runREVStudy");

    const amrex::Box& domain_box_full = geom_full.Domain();

    if (config.verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Starting REV Study (Homogenization Method) for Phase ID "
                       << config.phase_id << " ---\n";
        amrex::Print() << "  Number of samples per size: " << config.num_samples << std::endl;
        amrex::Print() << "  Target REV sizes:";
        for (int s : config.sizes) {
            amrex::Print() << " " << s;
        }
        amrex::Print() << std::endl;
        amrex::Print() << "  REV Plotfiles: " << (config.write_plotfiles ? "Yes" : "No")
                       << std::endl;
    }

    if (config.sizes.empty()) {
        amrex::Warning("REV sizes are empty. Skipping REV study.");
        return;
    }

    // Open CSV file
    std::ofstream csv_file;
    std::filesystem::path csv_path = config.results_path / config.csv_filename;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        csv_file.open(csv_path.string());
        csv_file << "SampleNo,SeedX,SeedY,SeedZ,REV_Size_Target,ActualSizeX,"
                    "ActualSizeY,ActualSizeZ,D_xx,D_yy,D_zz,D_xy,D_xz,D_yz\n";
    }

    // Use rank-independent seed so all MPI ranks agree on sub-volume coordinates
    std::mt19937 gen(12345 + config.num_samples);

    for (int s_idx = 0; s_idx < config.num_samples; ++s_idx) {
        for (int target_size : config.sizes) {
            // --- Generate random sub-volume seed ---
            amrex::IntVect seed_lo;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int min_coord = domain_box_full.smallEnd(d);
                int max_coord = domain_box_full.bigEnd(d) - (target_size - 1);
                if (min_coord > max_coord || target_size > domain_box_full.length(d)) {
                    seed_lo[d] = domain_box_full.smallEnd(d);
                } else {
                    std::uniform_int_distribution<> distr(min_coord, max_coord);
                    seed_lo[d] = distr(gen);
                }
            }

            amrex::Box bx_rev = amrex::Box(seed_lo, seed_lo + amrex::IntVect(target_size - 1));
            bx_rev &= domain_box_full;

            if (bx_rev.isEmpty() || bx_rev.longside() < 8) {
                if (config.verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                    std::stringstream ss;
                    ss << bx_rev;
                    amrex::Warning("Skipping REV for sample " + std::to_string(s_idx + 1) +
                                   " target size " + std::to_string(target_size) +
                                   " due to small/empty box: " + ss.str());
                }
                continue;
            }

            if (config.verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " REV Sample " << s_idx + 1 << ", Target Size " << target_size
                               << ", Seed Lo: " << seed_lo << ", Actual Box: " << bx_rev
                               << std::endl;
            }

            // --- Extract sub-volume phase data ---
            amrex::Box domain_rev = bx_rev;
            domain_rev.shift(-bx_rev.smallEnd());

            amrex::Geometry geom_rev;
            amrex::RealBox rb_rev(
                {AMREX_D_DECL(0.0, 0.0, 0.0)},
                {AMREX_D_DECL(amrex::Real(domain_rev.length(0)), amrex::Real(domain_rev.length(1)),
                              amrex::Real(domain_rev.length(2)))});
            amrex::Array<int, AMREX_SPACEDIM> is_periodic = {AMREX_D_DECL(1, 1, 1)};
            geom_rev.define(domain_rev, &rb_rev, 0, is_periodic.data());

            amrex::BoxArray ba_rev(domain_rev);
            ba_rev.maxSize(config.box_size);
            amrex::DistributionMapping dm_rev(ba_rev);
            amrex::iMultiFab mf_phase_rev(ba_rev, dm_rev, 1, 1);

            // Copy phase data from full domain to sub-volume.
            // Create a temporary with the same BoxArray/DM as mf_phase_rev but
            // shifted to bx_rev coordinates, so ParallelCopy from the full domain
            // works correctly across MPI ranks.
            amrex::BoxArray ba_shifted = ba_rev;
            ba_shifted.shift(bx_rev.smallEnd());
            amrex::iMultiFab mf_temp(ba_shifted, dm_rev, 1, 0);
            mf_temp.ParallelCopy(mf_phase_full, 0, 0, 1, amrex::IntVect::TheZeroVector(),
                                 amrex::IntVect::TheZeroVector(), geom_full.periodicity());

            // Now copy from shifted coords to REV coords (same rank, same FAB order)
            mf_phase_rev.setVal(0);
            for (amrex::MFIter mfi(mf_phase_rev); mfi.isValid(); ++mfi) {
                amrex::IArrayBox& dest_fab = mf_phase_rev[mfi];
                const amrex::IArrayBox& src_fab = mf_temp[mfi];
                const amrex::Box& dest_box = mfi.validbox();
                amrex::Box src_box = dest_box;
                src_box.shift(bx_rev.smallEnd());
                dest_fab.template copy<amrex::RunOn::Host>(src_fab, src_box, 0, dest_box, 0, 1);
            }
            mf_phase_rev.FillBoundary(geom_rev.periodicity());

            // --- Solve cell problems in all directions ---
            amrex::MultiFab mf_chi_x(ba_rev, dm_rev, 1, 1);
            amrex::MultiFab mf_chi_y(ba_rev, dm_rev, 1, 1);
            amrex::MultiFab mf_chi_z;
            if (AMREX_SPACEDIM == 3) {
                mf_chi_z.define(ba_rev, dm_rev, 1, 1);
            }

            bool all_converged = true;
            std::vector<Direction> solve_dirs = {Direction::X, Direction::Y};
            if (AMREX_SPACEDIM == 3) {
                solve_dirs.push_back(Direction::Z);
            }

            for (const auto& dir : solve_dirs) {
                std::string subdir = "REV_Sample" + std::to_string(s_idx + 1) + "_Size" +
                                     std::to_string(bx_rev.length(0)) + "_Dir" +
                                     std::to_string(static_cast<int>(dir));
                std::filesystem::path plot_path = config.results_path / subdir;

                if (config.write_plotfiles && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::UtilCreateDirectory(plot_path.string(), 0755);
                }
                amrex::ParallelDescriptor::Barrier();

                std::unique_ptr<EffectiveDiffusivityHypre> solver;
                try {
                    solver = std::make_unique<EffectiveDiffusivityHypre>(
                        geom_rev, ba_rev, dm_rev, mf_phase_rev, config.phase_id, dir,
                        config.solver_type, plot_path.string(), config.verbose,
                        config.write_plotfiles);
                    if (!solver->solve()) {
                        all_converged = false;
                        if (config.verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                            amrex::Print() << "    REV Chi solve FAILED for dir "
                                           << static_cast<int>(dir) << std::endl;
                        }
                        break;
                    }
                    if (dir == Direction::X)
                        solver->getChiSolution(mf_chi_x);
                    else if (dir == Direction::Y)
                        solver->getChiSolution(mf_chi_y);
                    else if (AMREX_SPACEDIM == 3 && dir == Direction::Z)
                        solver->getChiSolution(mf_chi_z);
                } catch (const std::exception& e) {
                    if (config.verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "    REV Chi solve EXCEPTION for dir "
                                       << static_cast<int>(dir) << ": " << e.what() << std::endl;
                    }
                    all_converged = false;
                    break;
                }
            }

            // --- Assemble D_eff tensor ---
            amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM];
            for (int r = 0; r < AMREX_SPACEDIM; ++r) {
                for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                    Deff_tensor[r][c] = std::numeric_limits<amrex::Real>::quiet_NaN();
                }
            }

            if (all_converged) {
                // Fill ghost cells for chi fields before computing gradients
                mf_chi_x.FillBoundary(geom_rev.periodicity());
                mf_chi_y.FillBoundary(geom_rev.periodicity());
                if (AMREX_SPACEDIM == 3) {
                    mf_chi_z.FillBoundary(geom_rev.periodicity());
                }

                amrex::iMultiFab active_mask(ba_rev, dm_rev, 1, 0);
                for (amrex::MFIter mfi(active_mask, amrex::TilingIfNotGPU()); mfi.isValid();
                     ++mfi) {
                    const amrex::Box& tb = mfi.tilebox();
                    auto const& mask_arr = active_mask.array(mfi);
                    auto const& phase_arr = mf_phase_rev.const_array(mfi);
                    int pid = config.phase_id;
                    amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        mask_arr(i, j, k, 0) = (phase_arr(i, j, k, 0) == pid) ? 1 : 0;
                    });
                }
                calculateDeffTensor(Deff_tensor, mf_chi_x, mf_chi_y, mf_chi_z, active_mask,
                                    geom_rev, config.verbose);
            }

            // --- Write CSV row ---
            if (amrex::ParallelDescriptor::IOProcessor()) {
                csv_file << s_idx + 1 << "," << seed_lo[0] << "," << seed_lo[1] << ","
                         << (AMREX_SPACEDIM == 3 ? seed_lo[2] : 0) << "," << target_size << ","
                         << bx_rev.length(0) << "," << bx_rev.length(1) << ","
                         << (AMREX_SPACEDIM == 3 ? bx_rev.length(2) : 1) << "," << std::fixed
                         << std::setprecision(8) << Deff_tensor[0][0] << "," << Deff_tensor[1][1]
                         << ","
                         << (AMREX_SPACEDIM == 3 ? Deff_tensor[2][2]
                                                 : std::numeric_limits<amrex::Real>::quiet_NaN())
                         << "," << Deff_tensor[0][1] << ","
                         << (AMREX_SPACEDIM == 3 ? Deff_tensor[0][2]
                                                 : std::numeric_limits<amrex::Real>::quiet_NaN())
                         << ","
                         << (AMREX_SPACEDIM == 3 ? Deff_tensor[1][2]
                                                 : std::numeric_limits<amrex::Real>::quiet_NaN())
                         << "\n";
                csv_file.flush();
            }
        }
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        csv_file.close();
    }
}

} // namespace OpenImpala
