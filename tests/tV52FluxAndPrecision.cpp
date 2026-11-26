/** @file tV52FluxAndPrecision.cpp
 *  @brief Integration and MPI verification suite for OpenImpala v5.2.0 features.
 *
 *  Covers:
 *    1. 3-D cell-centred diffusive flux vector/magnitude conservation, flux_stats
 *       reduction, and 7-component plotfile output (TortuosityMLMG & TortuosityHypre).
 *    2. Participating (percolation-filtered) interfacial SpecificSurfaceArea vs
 *       total geometric interfacial area in the presence of isolated pockets.
 *    3. Float64, Float32, and Mixed-precision MLMG solves at nanometre SI cell
 *       spacing (50 nm) for both TortuosityMLMG and EffectiveDiffusivityMLMG.
 *    4. Out-of-core collective MPI-IO RawReader streaming without full-volume RAM
 *       allocation (including uneven patch counts across MPI ranks) and deterministic
 *       REVStudy CSV checkpoint/resume.
 */

#include "DeffTensor.H"
#include "EffectiveDiffusivityMLMG.H"
#include "FluxField.H"
#include "PercolationCheck.H"
#include "REVStudy.H"
#include "RawReader.H"
#include "SpecificSurfaceArea.H"
#include "TortuosityHypre.H"
#include "TortuosityMLMG.H"
#include "VolumeFraction.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FileSystem.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace OpenImpalaV52Test {

void fillPorousWithPocket(amrex::iMultiFab& mf_phase, int n) {
    for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto arr = mf_phase.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Solid shell in [3, 12]^3 enclosing an isolated 2^3 pore pocket at [7, 8]^3,
            // plus staggered obstacles so the conducting phase (1) is genuinely tortuous.
            const bool in_shell = (i >= 3 && i <= 12 && j >= 3 && j <= 12 && k >= 3 && k <= 12);
            const bool in_pocket = (i >= 7 && i <= 8 && j >= 7 && j <= 8 && k >= 7 && k <= 8);
            const bool in_rib = ((i + 2 * j) % n < 3) && (k % 4 == 1);
            arr(i, j, k, 0) = (in_pocket || (!in_shell && !in_rib)) ? 1 : 0;
        });
    }
}

void fillChannelWithPocket(amrex::iMultiFab& mf_phase) {
    for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto arr = mf_phase.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::ignore_unused(i);
            // 4x4 X-spanning channel at j in [2, 5], k in [2, 5] (256 lateral faces on N=16)
            const bool in_channel = (j >= 2 && j <= 5 && k >= 2 && k <= 5);
            // Isolated 2x2x2 pocket at [10, 11]^3 (24 faces, non-percolating)
            const bool in_pocket = (i >= 10 && i <= 11 && j >= 10 && j <= 11 && k >= 10 && k <= 11);
            arr(i, j, k, 0) = (in_channel || in_pocket) ? 1 : 0;
        });
    }
}

bool solveDeffDiagMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                       const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                       int phase_id, OpenImpala::MLMGPrecision prec,
                       amrex::Real tensor_out[AMREX_SPACEDIM][AMREX_SPACEDIM]) {
    amrex::MultiFab chi_x(ba, dm, 1, 1);
    amrex::MultiFab chi_y(ba, dm, 1, 1);
    amrex::MultiFab chi_z(ba, dm, 1, 1);
    chi_x.setVal(0.0);
    chi_y.setVal(0.0);
    chi_z.setVal(0.0);

    const std::array<OpenImpala::Direction, 3> dirs = {
        OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};
    amrex::MultiFab* chi_ptrs[3] = {&chi_x, &chi_y, &chi_z};

    amrex::iMultiFab active_0g(ba, dm, 1, 0);
    amrex::MultiFab diff_0g(ba, dm, 1, 0);
    amrex::iMultiFab shared_ct(ba, dm, 1, 1);
    const std::map<int, amrex::Real> empty_coeffs;

    for (int d = 0; d < 3; ++d) {
        const amrex::iMultiFab* ct_ptr = (d > 0) ? &shared_ct : nullptr;
        OpenImpala::EffectiveDiffusivityMLMG solver(geom, ba, dm, mf_phase, phase_id, dirs[d],
                                                    "./tmp_v52_deff", 0, false, 1.0e-9, 200, 30,
                                                    empty_coeffs, ct_ptr, prec);
        if (d == 0) {
            amrex::Copy(active_0g, solver.getActiveMask(), 0, 0, 1, 0);
            amrex::Copy(diff_0g, solver.getDiffCoeff(), 0, 0, 1, 0);
            amrex::Copy(shared_ct, solver.getCellTypeMask(), 0, 0, 1, 0);
            shared_ct.FillBoundary(geom.periodicity());
        }
        if (!solver.solve() || !solver.getSolverConverged()) {
            return false;
        }
        solver.getChiSolution(*chi_ptrs[d]);
        chi_ptrs[d]->FillBoundary(geom.periodicity());
    }

    OpenImpala::calculateDeffTensor(tensor_out, chi_x, chi_y, chi_z, active_0g, geom, 0, &diff_0g);
    return true;
}

} // namespace OpenImpalaV52Test

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    int exit_code = 0;
    {
        bool ok = true;
        std::string fail_msg;
        auto check = [&](bool cond, const std::string& msg) {
            if (!cond && ok) {
                ok = false;
                fail_msg = msg;
            }
        };

        const int n = 16;
        const amrex::Real dx_si = 50.0e-9; // 50 nm SI spacing
        const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(n - 1, n - 1, n - 1));
        const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                {AMREX_D_DECL(n * dx_si, n * dx_si, n * dx_si)});
        amrex::Array<int, AMREX_SPACEDIM> non_per{AMREX_D_DECL(0, 0, 0)};
        amrex::Array<int, AMREX_SPACEDIM> all_per{AMREX_D_DECL(1, 1, 1)};
        amrex::Geometry geom_flow(domain, rb, 0, non_per);
        amrex::Geometry geom_per(domain, rb, 0, all_per);

        amrex::BoxArray ba(domain);
        ba.maxSize(8);
        amrex::DistributionMapping dm(ba);

        // =====================================================================
        // 1. Participating / active interfacial SSA vs total geometric SSA
        // =====================================================================
        {
            amrex::iMultiFab mf_chan(ba, dm, 1, 1);
            OpenImpalaV52Test::fillChannelWithPocket(mf_chan);
            mf_chan.FillBoundary(geom_flow.periodicity());

            OpenImpala::SpecificSurfaceArea ssa_total(geom_flow, mf_chan, 1, 0);
            long long fc_total = 0;
            long long tc_total = 0;
            ssa_total.value(fc_total, tc_total);
            check(fc_total == 280LL,
                  "Expected 280 total pore-solid faces (256 channel + 24 pocket), got " +
                      std::to_string(fc_total));

            OpenImpala::PercolationCheck perc_x(geom_flow, ba, dm, mf_chan, 1,
                                                OpenImpala::Direction::X, 0);
            check(perc_x.percolates(), "Channel must percolate along X");

            OpenImpala::SpecificSurfaceArea ssa_active(geom_flow, mf_chan, 1, 0, 0, 0,
                                                       &perc_x.activeMask(), 1);
            long long fc_active = 0;
            long long tc_active = 0;
            ssa_active.value(fc_active, tc_active);
            check(fc_active == 256LL,
                  "Expected 256 participating faces along X (pocket excluded), got " +
                      std::to_string(fc_active));
            check(ssa_active.value_ssa() < ssa_total.value_ssa(),
                  "Active SSA must be strictly smaller than total SSA when isolated pockets exist");
        }

        // =====================================================================
        // 2. Float64 / Float32 / Mixed MLMG + 3-D Flux Field & Plotfile
        // =====================================================================
        const std::filesystem::path tmp_root =
            std::filesystem::temp_directory_path() / "openimpala_tV52_test";
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::remove_all(tmp_root);
            std::filesystem::create_directories(tmp_root);
        }
        amrex::ParallelDescriptor::Barrier();

        {
            amrex::iMultiFab mf_porous(ba, dm, 1, 1);
            OpenImpalaV52Test::fillPorousWithPocket(mf_porous, n);
            mf_porous.FillBoundary(geom_flow.periodicity());

            OpenImpala::VolumeFraction vf_calc(mf_porous, 1);
            long long vf_pc = 0;
            long long vf_tc = 0;
            vf_calc.value(vf_pc, vf_tc);
            const amrex::Real vf =
                static_cast<amrex::Real>(vf_pc) / static_cast<amrex::Real>(vf_tc);

            OpenImpala::TortuosityMLMG s64(geom_flow, ba, dm, mf_porous, vf, 1,
                                           OpenImpala::Direction::X, (tmp_root / "plt64").string(),
                                           0.0, 1.0, 0, true, 1.0e-9, 200, 30, {},
                                           OpenImpala::MLMGPrecision::Float64);
            s64.setWriteFluxField(true);
            const amrex::Real tau64 = s64.value();
            check(s64.getSolverConverged() && std::isfinite(tau64) && tau64 > 1.0,
                  "TortuosityMLMG(float64) failed to converge");

            OpenImpala::TortuosityMLMG s32(geom_flow, ba, dm, mf_porous, vf, 1,
                                           OpenImpala::Direction::X, "", 0.0, 1.0, 0, false, 1.0e-6,
                                           200, 30, {}, OpenImpala::MLMGPrecision::Float32);
            const amrex::Real tau32 = s32.value();
            check(s32.getSolverConverged() && std::isfinite(tau32),
                  "TortuosityMLMG(float32) failed to converge at 50 nm spacing");
            check(std::abs(tau32 - tau64) / tau64 < 2.0e-3,
                  "TortuosityMLMG(float32) disagreed with float64 beyond 2e-3");

            OpenImpala::TortuosityMLMG smix(geom_flow, ba, dm, mf_porous, vf, 1,
                                            OpenImpala::Direction::X, "", 0.0, 1.0, 0, false,
                                            1.0e-9, 200, 30, {}, OpenImpala::MLMGPrecision::Mixed);
            const amrex::Real taumix = smix.value();
            check(smix.getSolverConverged() && std::isfinite(taumix),
                  "TortuosityMLMG(mixed) failed to converge at 50 nm spacing");
            check(std::abs(taumix - tau64) / tau64 < 1.0e-6,
                  "TortuosityMLMG(mixed) relative error vs float64 exceeded 1e-6: " +
                      std::to_string(std::abs(taumix - tau64) / tau64));

            // Verify cell-centred flux conservation & inactive/pocket zeroing
            amrex::MultiFab mf_flux;
            smix.computeCellFluxField(mf_flux);
            check(mf_flux.nComp() == 4, "Flux MultiFab must have 4 components (Jx, Jy, Jz, |J|)");

            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax> reduce_ops;
            amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_ops);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            for (amrex::MFIter mfi(mf_flux); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                const auto farr = mf_flux.const_array(mfi);
                reduce_ops.eval(
                    bx, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                        const bool in_pocket =
                            (i >= 7 && i <= 8 && j >= 7 && j <= 8 && k >= 7 && k <= 8);
                        const amrex::Real pocket_mag = in_pocket ? farr(i, j, k, 3) : 0.0;
                        return {-farr(i, j, k, 0), pocket_mag};
                    });
            }
            auto rv = reduce_data.value(reduce_ops);
            amrex::Real sum_neg_jx = amrex::get<0>(rv);
            amrex::Real max_pocket_j = amrex::get<1>(rv);
            amrex::ParallelDescriptor::ReduceRealSum(sum_neg_jx);
            amrex::ParallelDescriptor::ReduceRealMax(max_pocket_j);

            const amrex::Real deff_from_flux =
                (sum_neg_jx / static_cast<amrex::Real>(domain.numPts())) * (n * dx_si);
            const amrex::Real deff_expected = smix.getEffectiveDiffusivity();
            check(std::abs(deff_from_flux - deff_expected) / deff_expected < 1.0e-6,
                  "Volume-integrated -Jx * Lx did not match D_eff");
            check(max_pocket_j == 0.0, "Flux inside isolated pocket must be identically zero");

            const OpenImpala::FluxStats fstats = smix.computeFluxStats(16);
            check(fstats.mean_magnitude > 0.0 && fstats.max_magnitude >= fstats.mean_magnitude,
                  "Invalid FluxStats mean/max magnitude");
            const long long hist_sum = std::accumulate(fstats.histogram_counts.begin(),
                                                       fstats.histogram_counts.end(), 0LL);
            check(hist_sum > 0LL, "FluxStats histogram counts must be positive");

            // Check 7-component plotfile Header on I/O rank
            amrex::ParallelDescriptor::Barrier();
            if (amrex::ParallelDescriptor::IOProcessor()) {
                const auto hdr_path = tmp_root / "plt64" / "tortuosity_mlmg_0" / "Header";
                std::ifstream hdr(hdr_path);
                check(hdr.is_open(),
                      "Expected 7-component plotfile Header at " + hdr_path.string());
                if (hdr.is_open()) {
                    std::ostringstream oss;
                    oss << hdr.rdbuf();
                    const std::string content = oss.str();
                    check(content.find("flux_magnitude") != std::string::npos,
                          "Plotfile Header missing flux_magnitude component");
                }
            }

#ifndef AMREX_USE_GPU
            // Cross-check TortuosityHypre flux field against MLMG on CPU builds
            OpenImpala::TortuosityHypre shypre(
                geom_flow, ba, dm, mf_porous, vf, 1, OpenImpala::Direction::X,
                OpenImpala::SolverType::FlexGMRES, "", 0.0, 1.0, 0, false);
            const amrex::Real tau_hypre = shypre.value();
            check(shypre.getSolverConverged() && std::abs(tau_hypre - tau64) / tau64 < 1.0e-5,
                  "TortuosityHypre disagreed with TortuosityMLMG");
            amrex::MultiFab mf_flux_hypre;
            shypre.computeCellFluxField(mf_flux_hypre);
            amrex::MultiFab::Subtract(mf_flux_hypre, mf_flux, 0, 0, 4, 0);
            check(mf_flux_hypre.norm0(3) / fstats.max_magnitude < 1.0e-4,
                  "TortuosityHypre flux_magnitude disagreed with TortuosityMLMG");
#endif

            // Periodic EffectiveDiffusivityMLMG precision parity at 50 nm
            mf_porous.FillBoundary(geom_per.periodicity());
            amrex::Real d64[AMREX_SPACEDIM][AMREX_SPACEDIM] = {};
            amrex::Real d32[AMREX_SPACEDIM][AMREX_SPACEDIM] = {};
            amrex::Real dmix[AMREX_SPACEDIM][AMREX_SPACEDIM] = {};
            check(OpenImpalaV52Test::solveDeffDiagMLMG(geom_per, ba, dm, mf_porous, 1,
                                                       OpenImpala::MLMGPrecision::Float64, d64),
                  "EffectiveDiffusivityMLMG(float64) failed");
            check(OpenImpalaV52Test::solveDeffDiagMLMG(geom_per, ba, dm, mf_porous, 1,
                                                       OpenImpala::MLMGPrecision::Float32, d32),
                  "EffectiveDiffusivityMLMG(float32) failed");
            check(OpenImpalaV52Test::solveDeffDiagMLMG(geom_per, ba, dm, mf_porous, 1,
                                                       OpenImpala::MLMGPrecision::Mixed, dmix),
                  "EffectiveDiffusivityMLMG(mixed) failed");
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                check(std::abs(dmix[d][d] - d64[d][d]) / d64[d][d] < 1.0e-6,
                      "EffectiveDiffusivityMLMG(mixed) diagonal disagreed with float64");
                check(std::abs(d32[d][d] - d64[d][d]) / d64[d][d] < 2.0e-3,
                      "EffectiveDiffusivityMLMG(float32) diagonal disagreed with float64");
            }

            // Deterministic REVStudy checkpoint/resume verification
            OpenImpala::REVConfig rev_cfg;
            rev_cfg.sizes = {8};
            rev_cfg.num_samples = 2;
            rev_cfg.phase_id = 1;
            rev_cfg.box_size = 8;
            rev_cfg.verbose = 0;
            rev_cfg.use_mlmg = true;
            rev_cfg.mlmg_precision = OpenImpala::MLMGPrecision::Mixed;
            rev_cfg.resume = false;
            rev_cfg.results_path = tmp_root / "rev";
            rev_cfg.csv_filename = "rev_resume.csv";
            OpenImpala::runREVStudy(geom_per, ba, dm, mf_porous, rev_cfg);

            std::string first_sample_line;
            std::string second_sample_line;
            const auto csv_path = rev_cfg.results_path / rev_cfg.csv_filename;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ifstream ifs(csv_path);
                std::vector<std::string> lines;
                std::string ln;
                while (std::getline(ifs, ln)) {
                    if (!ln.empty()) {
                        lines.push_back(ln);
                    }
                }
                check(lines.size() == 3U, "Expected header + 2 sample rows before resume");
                if (lines.size() >= 3U) {
                    first_sample_line = lines[1];
                    second_sample_line = lines[2];
                    // Keep only header + sample 1, then resume to recompute sample 2 identically
                    std::ofstream ofs(csv_path, std::ios::trunc);
                    ofs << lines[0] << "\n" << lines[1] << "\n";
                }
            }
            amrex::ParallelDescriptor::Barrier();

            rev_cfg.resume = true;
            OpenImpala::runREVStudy(geom_per, ba, dm, mf_porous, rev_cfg);
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ifstream ifs(csv_path);
                std::vector<std::string> lines;
                std::string ln;
                while (std::getline(ifs, ln)) {
                    if (!ln.empty()) {
                        lines.push_back(ln);
                    }
                }
                check(lines.size() == 3U, "Expected header + 2 sample rows after resume, got " +
                                              std::to_string(lines.size()));
                if (lines.size() >= 3U) {
                    check(lines[1] == first_sample_line,
                          "Resume overwrote already-completed sample 1");
                    check(lines[2] == second_sample_line,
                          "Resume did not reproduce identical deterministic sample 2");
                }
            }
        }

        // =====================================================================
        // 3. Out-of-core RawReader MPI-IO streaming with uneven box counts
        // =====================================================================
        {
            const int rx = 14;
            const int ry = 8;
            const int rz = 8;
            const auto raw_path = tmp_root / "uneven_stream.raw";
            long long expected_ones = 0;
            for (int k = 0; k < rz; ++k) {
                for (int j = 0; j < ry; ++j) {
                    for (int i = 0; i < rx; ++i) {
                        const auto v = static_cast<std::uint8_t>((i + 3 * j + 5 * k) % 20);
                        if (v > 9U) {
                            ++expected_ones;
                        }
                    }
                }
            }
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::vector<std::uint8_t> buf(static_cast<std::size_t>(rx) *
                                              static_cast<std::size_t>(ry) *
                                              static_cast<std::size_t>(rz));
                for (int k = 0; k < rz; ++k) {
                    for (int j = 0; j < ry; ++j) {
                        for (int i = 0; i < rx; ++i) {
                            const auto idx =
                                (static_cast<std::size_t>(k) * static_cast<std::size_t>(ry) +
                                 static_cast<std::size_t>(j)) *
                                    static_cast<std::size_t>(rx) +
                                static_cast<std::size_t>(i);
                            buf[idx] = static_cast<std::uint8_t>((i + 3 * j + 5 * k) % 20);
                        }
                    }
                }
                std::ofstream ofs(raw_path, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(buf.data()),
                          static_cast<std::streamsize>(buf.size()));
            }
            amrex::ParallelDescriptor::Barrier();

            OpenImpala::RawReader reader;
            check(reader.readFile(raw_path.string(), rx, ry, rz, OpenImpala::RawDataType::UINT8),
                  "RawReader::readFile() failed");
            check(!reader.hasFullBufferAllocated(),
                  "RawReader::readFile() must not allocate full host buffer eagerly");

            // Decompose 14x8x8 with maxSize(6, 8, 8) -> 3 boxes (uneven on 2 MPI ranks!)
            const amrex::Box rdom(amrex::IntVect(0, 0, 0), amrex::IntVect(rx - 1, ry - 1, rz - 1));
            amrex::BoxArray rba(rdom);
            rba.maxSize(amrex::IntVect(6, 8, 8));
            check(rba.size() == 3, "Expected 3 boxes for uneven MPI-IO rank padding test");
            amrex::DistributionMapping rdm(rba);
            amrex::iMultiFab mf_raw(rba, rdm, 1, 0);

            reader.threshold(9.0, 1, 0, mf_raw);
            check(!reader.hasFullBufferAllocated(),
                  "RawReader::threshold() must stream subarrays without full-volume buffer");

            OpenImpala::VolumeFraction rvf(mf_raw, 1);
            long long phase_cnt = 0;
            long long total_cnt = 0;
            rvf.value(phase_cnt, total_cnt, false);
            const long long expected_total = static_cast<long long>(rx) *
                                             static_cast<long long>(ry) *
                                             static_cast<long long>(rz);
            check(total_cnt == expected_total && phase_cnt == expected_ones,
                  "Distributed RawReader threshold count mismatch: got " +
                      std::to_string(phase_cnt) + ", expected " + std::to_string(expected_ones));
        }

        amrex::ParallelDescriptor::Barrier();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::remove_all(tmp_root);
        }

        int local_fail = ok ? 0 : 1;
        amrex::ParallelDescriptor::ReduceIntMax(local_fail);
        if (local_fail != 0) {
            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "tV52FluxAndPrecision FAILED: " << fail_msg << "\n";
            }
            exit_code = 1;
        } else if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "tV52FluxAndPrecision PASSED.\n";
        }
    }
    amrex::Finalize();
    return exit_code;
}
