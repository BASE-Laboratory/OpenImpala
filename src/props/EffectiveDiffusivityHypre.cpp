// --- EffectiveDiffusivityHypre.cpp ---

#include "EffectiveDiffusivityHypre.H"
#include "EffDiffFillMtx.H" // For effDiffFillMatrix (replaces Fortran effdiff_fillmtx)

#include <cstdlib>
#include <mutex>
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

// HYPRE_CHECK macro
#define HYPRE_CHECK(ierr)                                                                          \
    do {                                                                                           \
        if ((ierr) != 0) {                                                                         \
            char hypre_error_msg[256];                                                             \
            HYPRE_DescribeError(ierr, hypre_error_msg);                                            \
            amrex::Abort("HYPRE Error: " + std::string(hypre_error_msg) +                          \
                         " - Error Code: " + std::to_string(ierr) + " File: " + __FILE__ +         \
                         " Line: " + std::to_string(__LINE__));                                    \
        }                                                                                          \
    } while (0)

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
    const std::string& resultspath, int verbose_level, bool write_plotfile_flag)
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
        amrex::ParmParse pp_tort("tortuosity");
        amrex::Vector<int> active_phases_vec;
        amrex::Vector<amrex::Real> phase_diffs_vec;
        pp_tort.queryarr("active_phases", active_phases_vec);
        pp_tort.queryarr("phase_diffusivities", phase_diffs_vec);

        if (!active_phases_vec.empty() && !phase_diffs_vec.empty()) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                active_phases_vec.size() == phase_diffs_vec.size(),
                "tortuosity.active_phases and tortuosity.phase_diffusivities must have the same "
                "length");
            for (size_t idx = 0; idx < active_phases_vec.size(); ++idx) {
                m_phase_coeff_map[active_phases_vec[idx]] = phase_diffs_vec[idx];
            }
            m_is_multi_phase = true;
        } else {
            m_phase_coeff_map[m_phase_id] = 1.0;
            m_is_multi_phase = false;
        }
    }

    // Build coefficient MultiFab using a device-accessible lookup table
    m_mf_diff_coeff.setVal(0.0);
    {
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

    std::atomic<long> cells_not_target_became_active(0); // Renamed for clarity
    std::atomic<long> cells_target_became_active(0);     // Renamed
    std::atomic<long> cells_target_became_inactive(0);   // Renamed

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& current_tile_box = mfi.tilebox();
        amrex::Array4<int> const mask_arr = m_mf_active_mask.array(mfi);
        amrex::Array4<const int> const phase_arr = m_mf_phase_original.const_array(mfi);

        const int local_m_phase_id = m_phase_id;

        if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor() && mfi.LocalIndex() == 0 &&
            amrex::ParallelDescriptor::MyProc() ==
                amrex::ParallelDescriptor::IOProcessorNumber()) { // Changed to verbose > 2
            amrex::Print() << "  DEBUG HYPRE: generateActiveMask MFIter loop (verbose > 2) is "
                              "using local_m_phase_id = "
                           << local_m_phase_id << " (from m_phase_id = " << m_phase_id
                           << ") for comparison." << std::endl;
        }

        const amrex::Box& valid_bx_for_debug = mfi.validbox();

        amrex::Array4<const amrex::Real> const dc_arr = m_mf_diff_coeff.const_array(mfi);

        amrex::LoopOnCpu(
            current_tile_box, [=, &cells_not_target_became_active, &cells_target_became_active,
                               &cells_target_became_inactive](int i, int j, int k) noexcept {
                // In multi-phase mode, mark active if D > 0.
                // In single-phase mode, mark active if phase matches target.
                bool should_be_active = (dc_arr(i, j, k, 0) > 0.0);

                if (should_be_active) {
                    mask_arr(i, j, k, MaskComp) = cell_active;
                    if (valid_bx_for_debug.contains(i, j, k)) {
                        cells_target_became_active++;
                    }
                } else {
                    mask_arr(i, j, k, MaskComp) = cell_inactive;
                }
                // Separate check for errors in valid region
                int original_phase_val = phase_arr(i, j, k, 0);
                bool is_target_phase = (original_phase_val == local_m_phase_id);
                if (valid_bx_for_debug.contains(i, j, k)) {
                    if (!is_target_phase && mask_arr(i, j, k, MaskComp) == cell_active) {
                        cells_not_target_became_active++;
                    } else if (is_target_phase && mask_arr(i, j, k, MaskComp) == cell_inactive) {
                        cells_target_became_inactive++;
                    }
                }
            });
    }

    long cells_not_target_became_active_val = cells_not_target_became_active.load();
    long cells_target_became_active_val = cells_target_became_active.load();
    long cells_target_became_inactive_val = cells_target_became_inactive.load();

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
        // This FillBoundary is crucial for the Fortran kernel if it accesses ghost cells
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  DEBUG HYPRE setupMatrixEquation: Calling "
                              "m_mf_active_mask.FillBoundary() before Fortran..."
                           << std::endl;
        }
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            long sum_check = ManualSumActiveCells(m_mf_active_mask, MaskComp, cell_active);
            amrex::Print() << "  DEBUG HYPRE setupMatrixEquation: Manual sum of m_mf_active_mask "
                              "after FillBoundary (before Fortran) = "
                           << sum_check << std::endl;
        }
    }
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());


    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print()
            << "  setupMatrixEquation: Calling C++ kernel 'effDiffFillMatrix' and SetBoxValues..."
            << std::endl;
    }

    std::vector<amrex::Real> matrix_values_buffer;
    std::vector<amrex::Real> rhs_values_buffer;
    std::vector<amrex::Real> initial_guess_buffer;

#ifdef OPENIMPALA_USE_GPU
    // GPU path: use device-resident buffers and ParallelFor kernel
    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& valid_bx = mfi.validbox();
        const int npts_valid = static_cast<int>(valid_bx.numPts());
        if (npts_valid == 0)
            continue;

        const size_t mtx_size = static_cast<size_t>(npts_valid) * stencil_size;
        amrex::Gpu::DeviceVector<amrex::Real> d_matrix(mtx_size);
        amrex::Gpu::DeviceVector<amrex::Real> d_rhs(npts_valid);
        amrex::Gpu::DeviceVector<amrex::Real> d_xinit(npts_valid);

        const auto mask_arr = m_mf_active_mask.const_array(mfi, MaskComp);
        const auto dc_arr = m_mf_diff_coeff.const_array(mfi, 0);

        OpenImpala::effDiffFillMatrixGpu(valid_bx, d_matrix.data(), d_rhs.data(), d_xinit.data(),
                                         mask_arr, dc_arr, m_dx.dataPtr(), current_dir_int);
        amrex::Gpu::streamSynchronize();

        // Copy device buffers to host for HYPRE SetBoxValues
        matrix_values_buffer.resize(mtx_size);
        rhs_values_buffer.resize(npts_valid);
        initial_guess_buffer.resize(npts_valid);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_matrix.begin(), d_matrix.end(),
                         matrix_values_buffer.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_rhs.begin(), d_rhs.end(),
                         rhs_values_buffer.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_xinit.begin(), d_xinit.end(),
                         initial_guess_buffer.begin());

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
#else
    // CPU path: original implementation with OMP tiling
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(                                 \
        matrix_values_buffer, rhs_values_buffer, initial_guess_buffer)
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& valid_bx = mfi.validbox();
        const int npts_valid = static_cast<int>(valid_bx.numPts());
        if (npts_valid == 0)
            continue;

        matrix_values_buffer.resize(static_cast<size_t>(npts_valid) * stencil_size);
        rhs_values_buffer.resize(npts_valid);
        initial_guess_buffer.resize(npts_valid);

        const amrex::IArrayBox& mask_fab = m_mf_active_mask[mfi];
        const int* mask_ptr = mask_fab.dataPtr(MaskComp);
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
            domain_for_kernel.hiVect(), m_dx.dataPtr(), current_dir_int, m_verbose);

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

    // Delegate solver dispatch to base class (PFMG preconditioner for periodic problems)
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

    std::vector<amrex::Real> soln_buffer;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(chi_field, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx_getsol = mfi.validbox();
        const int npts = static_cast<int>(bx_getsol.numPts());
        if (npts == 0)
            continue;

        soln_buffer.resize(npts);

        auto hypre_lo = EffectiveDiffusivityHypre::loV(bx_getsol);
        auto hypre_hi = EffectiveDiffusivityHypre::hiV(bx_getsol);

        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(),
                                                            soln_buffer.data());
        if (get_ierr != 0) {
            amrex::Warning("HYPRE_StructVectorGetBoxValues failed during getChiSolution!");
            chi_field[mfi].setVal(0.0, bx_getsol, ChiComp, numComponentsChi);
            continue;
        }

        amrex::Array4<amrex::Real> const chi_arr = chi_field.array(mfi);
        long long k_lin_idx = 0;

        amrex::LoopOnCpu(bx_getsol, [=, &k_lin_idx](int i, int j, int k) noexcept {
            if (k_lin_idx < npts) {
                chi_arr(i, j, k, ChiComp) = soln_buffer[k_lin_idx];
            }
            k_lin_idx++;
        });
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            k_lin_idx == npts,
            "Point count mismatch during HYPRE GetBoxValues copy in getChiSolution");
    }

    if (chi_field.nGrow() > 0) {
        chi_field.FillBoundary(m_geom.periodicity());
    }
}

} // End namespace OpenImpala
