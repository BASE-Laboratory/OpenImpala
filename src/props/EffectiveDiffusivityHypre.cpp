// --- EffectiveDiffusivityHypre.cpp ---

#include "EffectiveDiffusivityHypre.H"
#include "EffDiffFillMtx.H" // For effDiffFillMatrix (replaces Fortran effdiff_fillmtx)
#include "HypreCheck.H"

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <atomic>

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Gpu.H>
#ifdef OPENIMPALA_USE_GPU
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuContainers.H>
#endif
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_VisMF.H>
#include <AMReX_Loop.H>
#include <AMReX_Reduce.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>

// HYPRE includes
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

// MPI include
#include <mpi.h>

// Constants namespace
namespace {
constexpr int ChiComp = 0;
constexpr int MaskComp = 0;
constexpr int numComponentsChi = 1;
constexpr int stencil_size = 7;
constexpr int cell_inactive = 0;
constexpr int cell_active = 1;
constexpr int istn_c = 0;
constexpr int istn_mx = 1;
constexpr int istn_px = 2;
constexpr int istn_my = 3;
constexpr int istn_py = 4;
constexpr int istn_mz = 5;
constexpr int istn_pz = 6;

// Helper function to manually sum active cells in an iMultiFab (GPU-compatible)
long ManualSumActiveCells(const amrex::iMultiFab& imf, int component, int active_value) {
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<long> reduce_data(reduce_op);
    for (amrex::MFIter mfi(imf); mfi.isValid(); ++mfi) {
        const amrex::Box& vbox = mfi.validbox();
        amrex::Array4<const int> const arr = imf.const_array(mfi);
        reduce_op.eval(vbox, reduce_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<long> {
                           return {(arr(i, j, k, component) == active_value) ? 1L : 0L};
                       });
    }
    long total_active_cells = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceLongSum(total_active_cells);
    return total_active_cells;
}
} // namespace

namespace OpenImpala {

EffectiveDiffusivityHypre::EffectiveDiffusivityHypre(
    const amrex::Geometry& geom, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
    const amrex::iMultiFab& mf_phase_input, const int phase_id_arg,
    const OpenImpala::Direction dir_of_chi_k, const SolverType solver_type,
    const std::string& resultspath, int verbose_level, bool write_plotfile_flag,
    const std::map<int, amrex::Real>& phase_coeffs)
    : HypreStructSolver(geom, ba, dm, solver_type, 1e-9, 1000, verbose_level),
      m_mf_phase_original(ba, dm, mf_phase_input.nComp(), mf_phase_input.nGrow()),
      m_phase_id(phase_id_arg), m_dir_solve(dir_of_chi_k), m_resultspath(resultspath),
      m_write_plotfile(write_plotfile_flag), m_mf_chi(ba, dm, numComponentsChi, 1),
      m_mf_active_mask(ba, dm, 1, 1), m_mf_diff_coeff(ba, dm, 1, 1) {
    // Ensure HYPRE is initialised exactly once (thread-safe via std::call_once).
    static std::once_flag hypre_once;
    std::call_once(hypre_once, []() { HYPRE_Init(); });

    BL_PROFILE("EffectiveDiffusivityHypre::Constructor");

    amrex::Copy(m_mf_phase_original, mf_phase_input, 0, 0, m_mf_phase_original.nComp(),
                m_mf_phase_original.nGrow());
    if (m_mf_phase_original.nGrow() > 0) {
        m_mf_phase_original.FillBoundary(m_geom.periodicity());
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Initializing for chi_k in direction "
                       << static_cast<int>(m_dir_solve) << "..." << std::endl;
        amrex::Print() << "  DEBUG HYPRE: Constructor received m_phase_id (member) = " << m_phase_id
                       << std::endl;
    }

    if (m_verbose > 1 &&
        amrex::ParallelDescriptor::IOProcessor()) { // Keep this detailed check for high verbosity
        long initial_phase_count_debug = ManualSumActiveCells(m_mf_phase_original, 0, m_phase_id);
        amrex::Print()
            << "  DEBUG HYPRE: Number of cells in input m_mf_phase_original matching m_phase_id ("
            << m_phase_id
            << ") before generateActiveMask (manual sum): " << initial_phase_count_debug
            << std::endl;
        amrex::Print() << "  DEBUG HYPRE: m_mf_phase_original nComp: "
                       << m_mf_phase_original.nComp() << ", nGrow: " << m_mf_phase_original.nGrow()
                       << std::endl;
    }

    amrex::ParmParse pp_hypre("hypre");
    pp_hypre.query("eps", m_eps);
    pp_hypre.query("maxiter", m_maxiter);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");

    const amrex::Real* dx_tmp = m_geom.CellSize();
    for (int i_dim = 0; i_dim < AMREX_SPACEDIM; ++i_dim) {
        m_dx[i_dim] = dx_tmp[i_dim];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_dx[i_dim] > 0.0, "Cell size must be positive.");
    }

    // --- Multi-phase transport coefficient parsing ---
    {
        if (!phase_coeffs.empty()) {
            for (const auto& kv : phase_coeffs) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(kv.second >= 0.0,
                                                 "Phase diffusivities must be non-negative");
                m_phase_coeff_map[kv.first] = kv.second;
            }
            m_is_multi_phase = true;
        } else {
            amrex::ParmParse pp_tort("tortuosity");
            amrex::Vector<int> active_phases_vec;
            amrex::Vector<amrex::Real> phase_diffs_vec;
            pp_tort.queryarr("active_phases", active_phases_vec);
            pp_tort.queryarr("phase_diffusivities", phase_diffs_vec);

            if (!active_phases_vec.empty() && !phase_diffs_vec.empty()) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    active_phases_vec.size() == phase_diffs_vec.size(),
                    "tortuosity.active_phases and tortuosity.phase_diffusivities must have the "
                    "same length");
                for (size_t idx = 0; idx < active_phases_vec.size(); ++idx) {
                    m_phase_coeff_map[active_phases_vec[idx]] = phase_diffs_vec[idx];
                }
                m_is_multi_phase = true;
            } else {
                m_phase_coeff_map[m_phase_id] = 1.0;
                m_is_multi_phase = false;
            }
        }
    }

    // Build coefficient MultiFab using a device-accessible lookup table
    m_mf_diff_coeff.setVal(0.0);
    initializeDiffCoeff();
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    m_mf_active_mask.setVal(cell_inactive);
    generateActiveMask();

    long num_active_cells = ManualSumActiveCells(m_mf_active_mask, MaskComp, cell_active);

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Active mask generated. Number of active cells (Manually summed): "
                       << num_active_cells << std::endl;
        if (m_verbose >
            1) { // If very verbose, also show the iMultiFab::sum() for comparison/awareness
            long num_active_cells_imfsum = m_mf_active_mask.sum(MaskComp, true);
            amrex::ParallelDescriptor::ReduceLongSum(num_active_cells_imfsum);
            amrex::Print()
                << "  Active mask generated. Number of active cells (from m_mf_active_mask.sum()): "
                << num_active_cells_imfsum << std::endl;
        }
    }

    if (num_active_cells == 0) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "WARNING: No active cells found (manual sum) for phase_id "
                           << m_phase_id << ". HYPRE setup will be skipped." << std::endl;
        }
        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        m_mf_chi.setVal(0.0);
        return;
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Setting up HYPRE structures..." << std::endl;
    }
    setupGrid(true); // Periodic BCs for the cell problem
    HypreStructSolver::setupStencil();
    setupMatrixEquation();

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Initialization complete." << std::endl;
    }
}

// Destructor is defaulted in the header — base class handles HYPRE cleanup.

void EffectiveDiffusivityHypre::initializeDiffCoeff() {
    int max_pid = 0;
    for (const auto& kv : m_phase_coeff_map) {
        max_pid = std::max(max_pid, kv.first);
    }
    amrex::Gpu::DeviceVector<amrex::Real> d_coeff_lut(max_pid + 1, 0.0);
    amrex::Gpu::HostVector<amrex::Real> h_coeff_lut(max_pid + 1, 0.0);
    for (const auto& kv : m_phase_coeff_map) {
        h_coeff_lut[kv.first] = kv.second;
    }
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_coeff_lut.begin(), h_coeff_lut.end(),
                     d_coeff_lut.begin());
    const amrex::Real* lut_ptr = d_coeff_lut.data();
    const int lut_size = max_pid + 1;

    for (amrex::MFIter mfi(m_mf_diff_coeff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<amrex::Real> const dc_arr = m_mf_diff_coeff.array(mfi);
        amrex::Array4<const int> const phase_arr = m_mf_phase_original.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int pid = phase_arr(i, j, k, 0);
            dc_arr(i, j, k, 0) = (pid >= 0 && pid < lut_size) ? lut_ptr[pid] : 0.0;
        });
    }
}

void EffectiveDiffusivityHypre::generateActiveMask() {
    BL_PROFILE("EffectiveDiffusivityHypre::generateActiveMask");

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        long phase_original_sum_debug_again =
            ManualSumActiveCells(m_mf_phase_original, 0, m_phase_id);
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Manual sum of m_mf_phase_original for "
                          "m_phase_id ("
                       << m_phase_id << ") *immediately before* mask generation loop: "
                       << phase_original_sum_debug_again << std::endl;
    }

    if (m_mf_phase_original.nGrow() > 0) {
        m_mf_phase_original.FillBoundary(m_geom.periodicity());
    }

    // Phase 1: write the active mask via a GPU-compatible ParallelFor. The
    // mask_arr Array4 view points at device memory on CUDA builds, so the
    // assignment must run on the device that owns the iMultiFab.
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& current_tile_box = mfi.tilebox();
        amrex::Array4<int> const mask_arr = m_mf_active_mask.array(mfi);
        amrex::Array4<const amrex::Real> const dc_arr = m_mf_diff_coeff.const_array(mfi);
        amrex::ParallelFor(current_tile_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            mask_arr(i, j, k, MaskComp) = (dc_arr(i, j, k, 0) > 0.0) ? cell_active : cell_inactive;
        });
    }

    // Phase 2: collect debug counters via GPU-compatible reductions over the
    // valid region only. Three independent ReduceOps because the counters
    // mean different things; merging them into one tuple-reduce would be
    // marginally faster but obscures intent.
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> debug_reduce_op;
    amrex::ReduceData<long, long, long> debug_reduce_data(debug_reduce_op);
    const int local_m_phase_id = m_phase_id;
    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        amrex::Array4<const int> const mask_arr = m_mf_active_mask.const_array(mfi);
        amrex::Array4<const int> const phase_arr = m_mf_phase_original.const_array(mfi);
        amrex::Array4<const amrex::Real> const dc_arr = m_mf_diff_coeff.const_array(mfi);
        debug_reduce_op.eval(
            vbx, debug_reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j,
                                 int k) noexcept -> amrex::GpuTuple<long, long, long> {
                const bool should_be_active = (dc_arr(i, j, k, 0) > 0.0);
                const int classified = mask_arr(i, j, k, MaskComp);
                const bool is_target_phase = (phase_arr(i, j, k, 0) == local_m_phase_id);
                long n_target_active = (should_be_active && is_target_phase) ? 1L : 0L;
                long n_not_target_became_active =
                    (!is_target_phase && classified == cell_active) ? 1L : 0L;
                long n_target_became_inactive =
                    (is_target_phase && classified == cell_inactive) ? 1L : 0L;
                return {n_not_target_became_active, n_target_active, n_target_became_inactive};
            });
    }
    const auto debug_tuple = debug_reduce_data.value();
    long cells_not_target_became_active_val = amrex::get<0>(debug_tuple);
    long cells_target_became_active_val = amrex::get<1>(debug_tuple);
    long cells_target_became_inactive_val = amrex::get<2>(debug_tuple);

    amrex::ParallelDescriptor::ReduceLongSum(cells_not_target_became_active_val);
    amrex::ParallelDescriptor::ReduceLongSum(cells_target_became_active_val);
    amrex::ParallelDescriptor::ReduceLongSum(cells_target_became_inactive_val);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { // Changed to verbose > 1
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Count of (valid) cells originally NOT "
                          "m_phase_id that BECAME ACTIVE: "
                       << cells_not_target_became_active_val << std::endl;
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Count of (valid) cells originally "
                          "m_phase_id that BECAME ACTIVE: "
                       << cells_target_became_active_val << std::endl;
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Count of (valid) cells originally "
                          "m_phase_id that BECAME INACTIVE: "
                       << cells_target_became_inactive_val << std::endl;
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        long active_mask_sum_manual_before_fb =
            ManualSumActiveCells(m_mf_active_mask, MaskComp, cell_active);
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Manual sum of m_mf_active_mask (valid "
                          "cells) *after* loop, *before* FillBoundary: "
                       << active_mask_sum_manual_before_fb << std::endl;
    }

    if (m_mf_active_mask.nGrow() > 0) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
                << "  DEBUG HYPRE generateActiveMask: Calling m_mf_active_mask.FillBoundary()..."
                << std::endl;
        }
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  DEBUG HYPRE generateActiveMask: Returned from "
                              "m_mf_active_mask.FillBoundary()."
                           << std::endl;
        }

        if (m_verbose > 1 &&
            amrex::ParallelDescriptor::IOProcessor()) { // Changed to verbose > 1 for these detailed
                                                        // checks
            long active_mask_sum_manual_after_fb =
                ManualSumActiveCells(m_mf_active_mask, MaskComp, cell_active);
            amrex::Print() << "  DEBUG HYPRE generateActiveMask: Manual sum of m_mf_active_mask "
                              "(valid cells) *immediately after* FillBoundary: "
                           << active_mask_sum_manual_after_fb << std::endl;
            long active_mask_sum_imf_after_fb = m_mf_active_mask.sum(MaskComp, true);
            amrex::ParallelDescriptor::ReduceLongSum(active_mask_sum_imf_after_fb);
            amrex::Print() << "  DEBUG HYPRE generateActiveMask: m_mf_active_mask.sum() (valid "
                              "cells) *immediately after* FillBoundary: "
                           << active_mask_sum_imf_after_fb << std::endl;
        }
    } else {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  DEBUG HYPRE generateActiveMask: Skipped "
                              "m_mf_active_mask.FillBoundary() as nGrow is 0."
                           << std::endl;
        }
    }
}

// setupGrids() and setupStencil() are now provided by the HypreStructSolver base class.


int EffectiveDiffusivityHypre::buildCellTypeMask(amrex::iMultiFab& mf_cell_type) const {
    BL_PROFILE("EffectiveDiffusivityHypre::buildCellTypeMask");

    const amrex::Box& domain = m_geom.Domain();
    const int dom_lo_x = domain.smallEnd(0);
    const int dom_lo_y = domain.smallEnd(1);
    const int dom_lo_z = domain.smallEnd(2);
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    const long total_pts_long = domain.numPts();

    mf_cell_type.setVal(CELL_INACTIVE);

#ifndef AMREX_USE_GPU
    // Fast single-rank O(N) periodic BFS on CPU: visits every active cell once and
    // marks the lexicographically smallest (k, j, i) cell of each periodic 6-connected
    // component as CELL_PINNED (chi = 0).
    if (amrex::ParallelDescriptor::NProcs() == 1) {
        const auto total_pts = static_cast<std::size_t>(total_pts_long);
        const auto snx = static_cast<std::size_t>(nx);
        const auto sny = static_cast<std::size_t>(ny);
        auto flat_idx = [=](int ix, int iy, int iz) noexcept -> std::size_t {
            return (static_cast<std::size_t>(iz) * sny + static_cast<std::size_t>(iy)) * snx +
                   static_cast<std::size_t>(ix);
        };

        // 0 = inactive, 1 = unvisited active, 2 = visited active, 3 = pinned active
        std::vector<std::uint8_t> state(total_pts, 0u);
        for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
            const amrex::Box& vbox = mfi.validbox();
            const auto mask_arr = m_mf_active_mask.const_array(mfi);
            amrex::LoopOnCpu(vbox, [&](int i, int j, int k) {
                if (mask_arr(i, j, k, MaskComp) == cell_active) {
                    state[flat_idx(i - dom_lo_x, j - dom_lo_y, k - dom_lo_z)] = 1u;
                }
            });
        }

        const bool per_x = m_geom.isPeriodic(0);
        const bool per_y = m_geom.isPeriodic(1);
        const bool per_z = m_geom.isPeriodic(2);

        struct Coord {
            int x;
            int y;
            int z;
        };
        std::vector<Coord> queue;
        queue.reserve(total_pts / 8 + 16);

        auto try_visit = [&](int nbx, int nby, int nbz) {
            const std::size_t nb_idx = flat_idx(nbx, nby, nbz);
            if (state[nb_idx] == 1u) {
                state[nb_idx] = 2u;
                queue.push_back({nbx, nby, nbz});
            }
        };

        int num_components = 0;
        for (int kz = 0; kz < nz; ++kz) {
            for (int jy = 0; jy < ny; ++jy) {
                for (int ix = 0; ix < nx; ++ix) {
                    const std::size_t seed_idx = flat_idx(ix, jy, kz);
                    if (state[seed_idx] != 1u) {
                        continue;
                    }
                    state[seed_idx] = 3u;
                    ++num_components;
                    queue.clear();
                    queue.push_back({ix, jy, kz});

                    std::size_t head = 0;
                    while (head < queue.size()) {
                        const Coord c = queue[head++];
                        if (c.x > 0) {
                            try_visit(c.x - 1, c.y, c.z);
                        } else if (per_x) {
                            try_visit(nx - 1, c.y, c.z);
                        }
                        if (c.x + 1 < nx) {
                            try_visit(c.x + 1, c.y, c.z);
                        } else if (per_x) {
                            try_visit(0, c.y, c.z);
                        }
                        if (c.y > 0) {
                            try_visit(c.x, c.y - 1, c.z);
                        } else if (per_y) {
                            try_visit(c.x, ny - 1, c.z);
                        }
                        if (c.y + 1 < ny) {
                            try_visit(c.x, c.y + 1, c.z);
                        } else if (per_y) {
                            try_visit(c.x, 0, c.z);
                        }
                        if (c.z > 0) {
                            try_visit(c.x, c.y, c.z - 1);
                        } else if (per_z) {
                            try_visit(c.x, c.y, nz - 1);
                        }
                        if (c.z + 1 < nz) {
                            try_visit(c.x, c.y, c.z + 1);
                        } else if (per_z) {
                            try_visit(c.x, c.y, 0);
                        }
                    }
                }
            }
        }

        for (amrex::MFIter mfi(mf_cell_type); mfi.isValid(); ++mfi) {
            const amrex::Box& vbox = mfi.validbox();
            auto ct_arr = mf_cell_type.array(mfi);
            amrex::LoopOnCpu(vbox, [&](int i, int j, int k) {
                const std::uint8_t s = state[flat_idx(i - dom_lo_x, j - dom_lo_y, k - dom_lo_z)];
                if (s == 3u) {
                    ct_arr(i, j, k, 0) = CELL_PINNED;
                } else if (s == 2u) {
                    ct_arr(i, j, k, 0) = CELL_ACTIVE;
                } else {
                    ct_arr(i, j, k, 0) = CELL_INACTIVE;
                }
            });
        }

        if (mf_cell_type.nGrow() > 0) {
            mf_cell_type.FillBoundary(m_geom.periodicity());
        }
        return num_components;
    }
#endif

    // General MPI + CUDA GPU path: double-buffered parallel min-label propagation.
    // Every active cell starts with its 1-based global linear index; propagating the
    // minimum label across 6-connected periodic neighbors converges all components
    // simultaneously, and each component's unique minimum-index cell satisfies
    // label == self_label -> CELL_PINNED.
    AMREX_ALWAYS_ASSERT(total_pts_long < static_cast<long>(std::numeric_limits<int>::max()));
    const int sentinel = std::numeric_limits<int>::max();

    amrex::iMultiFab mf_curr(m_ba, m_dm, 1, 1);
    amrex::iMultiFab mf_next(m_ba, m_dm, 1, 1);
    mf_curr.setVal(sentinel);
    mf_next.setVal(sentinel);

    for (amrex::MFIter mfi(mf_curr, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto curr_arr = mf_curr.array(mfi);
        const auto mask_arr = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (mask_arr(i, j, k, MaskComp) == cell_active) {
                curr_arr(i, j, k, 0) =
                    (i - dom_lo_x) + nx * ((j - dom_lo_y) + ny * (k - dom_lo_z)) + 1;
            } else {
                curr_arr(i, j, k, 0) = sentinel;
            }
        });
    }

    const int max_label_iter = static_cast<int>(std::min<long>(total_pts_long + 2L, 100000L));
    int iter = 0;
    bool changed_globally = true;

    int h_changed = 0;
#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceScalar<int> d_changed(0);
#endif

    while (changed_globally && iter < max_label_iter) {
        ++iter;
        mf_curr.FillBoundary(m_geom.periodicity());

        int* d_flag_ptr = nullptr;
#ifdef AMREX_USE_GPU
        int zero = 0;
        amrex::Gpu::htod_memcpy(d_changed.dataPtr(), &zero, sizeof(int));
        d_flag_ptr = d_changed.dataPtr();
#else
        h_changed = 0;
        d_flag_ptr = &h_changed;
#endif

        for (amrex::MFIter mfi(mf_curr, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            const auto curr_arr = mf_curr.const_array(mfi);
            auto next_arr = mf_next.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int cur = curr_arr(i, j, k, 0);
                if (cur == sentinel) {
                    next_arr(i, j, k, 0) = sentinel;
                    return;
                }
                int best = cur;
                best = amrex::min(best, curr_arr(i - 1, j, k, 0));
                best = amrex::min(best, curr_arr(i + 1, j, k, 0));
                best = amrex::min(best, curr_arr(i, j - 1, k, 0));
                best = amrex::min(best, curr_arr(i, j + 1, k, 0));
                best = amrex::min(best, curr_arr(i, j, k - 1, 0));
                best = amrex::min(best, curr_arr(i, j, k + 1, 0));
                next_arr(i, j, k, 0) = best;
                if (best < cur) {
                    amrex::Gpu::Atomic::Max(d_flag_ptr, 1);
                }
            });
        }

#ifdef AMREX_USE_GPU
        changed_globally = (d_changed.dataValue() != 0);
#else
        changed_globally = (h_changed != 0);
#endif
        amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);
        std::swap(mf_curr, mf_next);
    }

    amrex::ReduceOps<amrex::ReduceOpSum> pin_count_op;
    amrex::ReduceData<long> pin_count_data(pin_count_op);
    for (amrex::MFIter mfi(mf_cell_type); mfi.isValid(); ++mfi) {
        const amrex::Box& vbox = mfi.validbox();
        auto ct_arr = mf_cell_type.array(mfi);
        const auto curr_arr = mf_curr.const_array(mfi);
        const auto mask_arr = m_mf_active_mask.const_array(mfi);
        pin_count_op.eval(
            vbox, pin_count_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<long> {
                if (mask_arr(i, j, k, MaskComp) != cell_active) {
                    ct_arr(i, j, k, 0) = CELL_INACTIVE;
                    return {0L};
                }
                const int self_lbl =
                    (i - dom_lo_x) + nx * ((j - dom_lo_y) + ny * (k - dom_lo_z)) + 1;
                if (curr_arr(i, j, k, 0) == self_lbl) {
                    ct_arr(i, j, k, 0) = CELL_PINNED;
                    return {1L};
                }
                ct_arr(i, j, k, 0) = CELL_ACTIVE;
                return {0L};
            });
    }

    long total_components = amrex::get<0>(pin_count_data.value());
    amrex::ParallelDescriptor::ReduceLongSum(total_components);

    if (mf_cell_type.nGrow() > 0) {
        mf_cell_type.FillBoundary(m_geom.periodicity());
    }
    return static_cast<int>(total_components);
}


void EffectiveDiffusivityHypre::setupMatrixEquation() {
    BL_PROFILE("EffectiveDiffusivityHypre::setupMatrixEquation");
    HYPRE_Int ierr = 0;

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Creating HYPRE Matrix and Vectors..."
                       << std::endl;
    }

    // Create and initialize HYPRE matrix and vectors via base class
    createMatrixAndVectors();

    const amrex::Box& domain_for_kernel = m_geom.Domain();
    int stencil_indices_hypre[stencil_size];
    for (int i_loop = 0; i_loop < stencil_size; ++i_loop) {
        stencil_indices_hypre[i_loop] = i_loop;
    }
    const int current_dir_int = static_cast<int>(m_dir_solve);

    if (m_mf_active_mask.nGrow() > 0) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  DEBUG HYPRE setupMatrixEquation: Calling "
                              "m_mf_active_mask.FillBoundary()..."
                           << std::endl;
        }
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            long sum_check = ManualSumActiveCells(m_mf_active_mask, MaskComp, cell_active);
            amrex::Print() << "  DEBUG HYPRE setupMatrixEquation: Manual sum of m_mf_active_mask "
                              "after FillBoundary = "
                           << sum_check << std::endl;
        }
    }
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    amrex::Real max_d = 0.0;
    for (const auto& kv : m_phase_coeff_map) {
        max_d = std::max(max_d, kv.second);
    }
    const amrex::Real d_ref = (max_d > 0.0) ? max_d : 1.0;
    const amrex::Real pin_scale =
        d_ref * (1.0 / (m_dx[0] * m_dx[0]) + 1.0 / (m_dx[1] * m_dx[1]) + 1.0 / (m_dx[2] * m_dx[2]));

    // Build cell-state mask with one cell symmetrically pinned (chi = 0) in
    // every periodic 6-connected active component to eliminate all constant
    // nullspace modes without artificial diagonal regularisation.
    amrex::iMultiFab mf_cell_type(m_ba, m_dm, 1, 1);
    const int num_pinned_components = buildCellTypeMask(mf_cell_type);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Symmetrically pinned " << num_pinned_components
                       << " periodic connected component(s) to chi=0." << std::endl;
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print()
            << "  setupMatrixEquation: Calling C++ kernel 'effDiffFillMatrix' and SetBoxValues..."
            << std::endl;
    }

    std::vector<amrex::Real> matrix_values_buffer;
    std::vector<amrex::Real> rhs_values_buffer;
    std::vector<amrex::Real> initial_guess_buffer;

#ifdef OPENIMPALA_USE_GPU
    // GPU path: keep matrix data device-resident end-to-end. With
    // HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE) configured in the base
    // class's createMatrixAndVectors, HYPRE expects device pointers in
    // SetBoxValues — pass d_matrix.data() / d_rhs.data() / d_xinit.data()
    // directly with no host roundtrip.
    for (amrex::MFIter mfi(mf_cell_type); mfi.isValid(); ++mfi) {
        const amrex::Box& valid_bx = mfi.validbox();
        const int npts_valid = static_cast<int>(valid_bx.numPts());
        if (npts_valid == 0)
            continue;

        const size_t mtx_size = static_cast<size_t>(npts_valid) * stencil_size;
        amrex::Gpu::DeviceVector<amrex::Real> d_matrix(mtx_size);
        amrex::Gpu::DeviceVector<amrex::Real> d_rhs(npts_valid);
        amrex::Gpu::DeviceVector<amrex::Real> d_xinit(npts_valid);

        const auto mask_arr = mf_cell_type.const_array(mfi, 0);
        const auto dc_arr = m_mf_diff_coeff.const_array(mfi, 0);

        OpenImpala::effDiffFillMatrixGpu(valid_bx, d_matrix.data(), d_rhs.data(), d_xinit.data(),
                                         mask_arr, dc_arr, m_dx.dataPtr(), current_dir_int,
                                         pin_scale);
        amrex::Gpu::streamSynchronize();

        auto hypre_lo_valid = EffectiveDiffusivityHypre::loV(valid_bx);
        auto hypre_hi_valid = EffectiveDiffusivityHypre::hiV(valid_bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              stencil_size, stencil_indices_hypre, d_matrix.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              d_rhs.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              d_xinit.data());
        HYPRE_CHECK(ierr);
    }
#else
    // CPU path: no OMP parallelism here because HYPRE_StructMatrixSetBoxValues
    // and HYPRE_StructVectorSetBoxValues are not thread-safe for the same object.
    for (amrex::MFIter mfi(mf_cell_type, false); mfi.isValid(); ++mfi) {
        const amrex::Box& valid_bx = mfi.validbox();
        const long npts_long = valid_bx.numPts();
        AMREX_ALWAYS_ASSERT(npts_long <= static_cast<long>(std::numeric_limits<int>::max()));
        const int npts_valid = static_cast<int>(npts_long);
        if (npts_valid == 0)
            continue;

        matrix_values_buffer.resize(static_cast<size_t>(npts_valid) * stencil_size);
        rhs_values_buffer.resize(npts_valid);
        initial_guess_buffer.resize(npts_valid);

        const amrex::IArrayBox& mask_fab = mf_cell_type[mfi];
        const int* mask_ptr = mask_fab.dataPtr(0);
        const auto* mask_fab_lo = mask_fab.loVect();
        const auto* mask_fab_hi = mask_fab.hiVect();

        const amrex::FArrayBox& dc_fab = m_mf_diff_coeff[mfi];
        const amrex::Real* dc_ptr = dc_fab.dataPtr(0);
        const auto* dc_fab_lo = dc_fab.loVect();
        const auto* dc_fab_hi = dc_fab.hiVect();

        OpenImpala::effDiffFillMatrix(
            matrix_values_buffer.data(), rhs_values_buffer.data(), initial_guess_buffer.data(),
            npts_valid, mask_ptr, mask_fab_lo, mask_fab_hi, dc_ptr, dc_fab_lo, dc_fab_hi,
            valid_bx.loVect(), valid_bx.hiVect(), domain_for_kernel.loVect(),
            domain_for_kernel.hiVect(), m_dx.dataPtr(), current_dir_int, m_verbose, pin_scale);

        auto hypre_lo_valid = EffectiveDiffusivityHypre::loV(valid_bx);
        auto hypre_hi_valid = EffectiveDiffusivityHypre::hiV(valid_bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              stencil_size, stencil_indices_hypre,
                                              matrix_values_buffer.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              rhs_values_buffer.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              initial_guess_buffer.data());
        HYPRE_CHECK(ierr);
    }
#endif

    // Assemble via base class
    assembleSystem();

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Setup complete." << std::endl;
    }
}

bool EffectiveDiffusivityHypre::solve() {
    BL_PROFILE("EffectiveDiffusivityHypre::solve");

    long num_active_cells_in_solve = ManualSumActiveCells(m_mf_active_mask, MaskComp, cell_active);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG HYPRE solve: num_active_cells_in_solve (Manually summed) = "
                       << num_active_cells_in_solve << std::endl;
        // For comparison, also print the iMultiFab::sum() result
        long num_active_cells_imfsum_solve = m_mf_active_mask.sum(MaskComp, true);
        amrex::ParallelDescriptor::ReduceLongSum(num_active_cells_imfsum_solve);
        amrex::Print()
            << "  DEBUG HYPRE solve: num_active_cells_in_solve (from m_mf_active_mask.sum()) = "
            << num_active_cells_imfsum_solve << std::endl;
    }

    if (num_active_cells_in_solve == 0) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "EffectiveDiffusivityHypre::solve: Skipping HYPRE solve as no active "
                              "cells were found (manual sum) for phase "
                           << m_phase_id << std::endl;
        }
        m_mf_chi.setVal(0.0);
        if (m_mf_chi.nGrow() > 0)
            m_mf_chi.FillBoundary(m_geom.periodicity());

        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        return m_converged;
    }

    // Delegate solver dispatch to base class (PFMG preconditioner for periodic problems).
    runSolver(PrecondType::PFMG);

    if (m_converged) {
        getChiSolution(m_mf_chi);
    } else {
        m_mf_chi.setVal(0.0);
        if (m_mf_chi.nGrow() > 0)
            m_mf_chi.FillBoundary(m_geom.periodicity());
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Warning("Solver did not converge. Chi solution (m_mf_chi) set to 0.");
        }
    }

    if (m_write_plotfile &&
        (m_converged ||
         num_active_cells_in_solve == 0)) { // Use logic based on num_active_cells_in_solve
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Writing solution plotfile for chi_k in direction "
                           << static_cast<int>(m_dir_solve) << "..." << std::endl;
        }
        amrex::MultiFab mf_plot(m_ba, m_dm, 2, 0);
        amrex::Copy(mf_plot, m_mf_chi, ChiComp, 0, 1, 0);

        for (amrex::MFIter mfi(mf_plot, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx_plot = mfi.tilebox();
            amrex::Array4<amrex::Real> const plot_arr = mf_plot.array(mfi);
            amrex::Array4<const int> const mask_arr_plot = m_mf_active_mask.const_array(mfi);

            amrex::ParallelFor(bx_plot, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                plot_arr(i, j, k, 1) = static_cast<amrex::Real>(mask_arr_plot(i, j, k, MaskComp));
            });
        }

        std::string plot_filename_str =
            "effdiff_chi_dir" + std::to_string(static_cast<int>(m_dir_solve));
        std::string full_plot_path = m_resultspath + "/" + plot_filename_str;

        amrex::Vector<std::string> varnames = {"chi_k", "active_mask_from_solver"}; // Clarify name
        amrex::WriteSingleLevelPlotfile(full_plot_path, mf_plot, varnames, m_geom, 0.0, 0);

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Plotfile written to " << full_plot_path << std::endl;
        }
    } else if (m_write_plotfile && !m_converged) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Warning("Skipping plotfile write for chi_k because solver did not converge and "
                           "had active cells.");
        }
    }
    return m_converged;
}

void EffectiveDiffusivityHypre::getChiSolution(amrex::MultiFab& chi_field) {
    BL_PROFILE("EffectiveDiffusivityHypre::getChiSolution");

    if (!m_x) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
                << "  getChiSolution: HYPRE solution vector m_x is NULL. Setting chi_field to 0."
                << std::endl;
        }
        chi_field.setVal(0.0);
        if (chi_field.nGrow() > 0) {
            chi_field.FillBoundary(m_geom.periodicity());
        }
        return;
    }
    AMREX_ALWAYS_ASSERT(chi_field.nComp() >= numComponentsChi);
    AMREX_ALWAYS_ASSERT(chi_field.boxArray() == m_ba);
    AMREX_ALWAYS_ASSERT(chi_field.DistributionMap() == m_dm);

    // HYPRE_MEMORY_DEVICE on CUDA → GetBoxValues writes to a device pointer,
    // and chi_arr below is also device-resident, so on GPU we want HYPRE to
    // drop the solution straight into a device buffer with no host bounce.
    // No OMP: HYPRE_StructVectorGetBoxValues is not thread-safe for the same vector.
    std::vector<amrex::Real> soln_buffer_host;
    for (amrex::MFIter mfi(chi_field, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx_getsol = mfi.validbox();
        const int npts = static_cast<int>(bx_getsol.numPts());
        if (npts == 0)
            continue;

        auto hypre_lo = EffectiveDiffusivityHypre::loV(bx_getsol);
        auto hypre_hi = EffectiveDiffusivityHypre::hiV(bx_getsol);

#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<HYPRE_Real> d_soln_buffer(npts);
        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(),
                                                            d_soln_buffer.data());
        const HYPRE_Real* src_ptr = d_soln_buffer.data();
#else
        soln_buffer_host.resize(npts);
        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(),
                                                            soln_buffer_host.data());
        const HYPRE_Real* src_ptr = soln_buffer_host.data();
#endif
        if (get_ierr != 0) {
            amrex::Warning("HYPRE_StructVectorGetBoxValues failed during getChiSolution!");
            chi_field[mfi].template setVal<amrex::RunOn::Host>(0.0, bx_getsol, ChiComp,
                                                               numComponentsChi);
            continue;
        }
        amrex::Array4<amrex::Real> const chi_arr = chi_field.array(mfi);
        const int chi_comp = ChiComp;
        const auto lo = amrex::lbound(bx_getsol);
        const int nx_box = bx_getsol.length(0);
        const int ny_box = bx_getsol.length(1);
        amrex::ParallelFor(bx_getsol, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const long long lin = static_cast<long long>(k - lo.z) * nx_box * ny_box +
                                  static_cast<long long>(j - lo.y) * nx_box +
                                  static_cast<long long>(i - lo.x);
            chi_arr(i, j, k, chi_comp) = static_cast<amrex::Real>(src_ptr[lin]);
        });
    }
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif

    if (chi_field.nGrow() > 0) {
        chi_field.FillBoundary(m_geom.periodicity());
    }
}

} // End namespace OpenImpala
