/** @file EffectiveDiffusivityMLMG.cpp
 *  @brief Matrix-free AMReX MLMG implementation for 3-D periodic homogenisation cell problems.
 */

#include "EffectiveDiffusivityMLMG.H"
#include "PeriodicComponentPin.H"
#include "SingleLevelGMRESMLMG.H"
#include "SolverConfig.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <string>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_CoordSys.H>
#include <AMReX_GMRES_MLMG.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_Reduce.H>
#include <AMReX_Vector.H>

namespace OpenImpala {

EffectiveDiffusivityMLMG::EffectiveDiffusivityMLMG(
    const amrex::Geometry& geom, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
    const amrex::iMultiFab& mf_phase_input, int phase_id, OpenImpala::Direction dir_of_chi_k,
    const std::string& resultspath, int verbose_level, bool write_plotfile_flag, amrex::Real eps,
    int maxiter, int max_coarsening_level, const std::map<int, amrex::Real>& phase_coeffs,
    const amrex::iMultiFab* precomputed_cell_type, MLMGPrecision precision)
    : m_ba(ba), m_dm(dm), m_resultspath(resultspath), m_phase_id(phase_id),
      m_dir_solve(dir_of_chi_k), m_verbose(verbose_level), m_write_plotfile(write_plotfile_flag),
      m_eps(eps), m_maxiter(maxiter), m_max_coarsening_level(max_coarsening_level),
      m_precision(precision), m_mf_phase_original(ba, dm, 1, 1), m_mf_active_mask(ba, dm, 1, 1),
      m_mf_cell_type(ba, dm, 1, 1), m_mf_diff_coeff(ba, dm, 1, 1), m_mf_chi(ba, dm, 1, 1) {
    BL_PROFILE("EffectiveDiffusivityMLMG::Constructor");

    amrex::Array<int, AMREX_SPACEDIM> is_per{AMREX_D_DECL(1, 1, 1)};
    m_geom.define(geom.Domain(), &(geom.ProbDomain()), amrex::CoordSys::cartesian, is_per.data());

    amrex::ParmParse pp_mlmg("mlmg");
    pp_mlmg.query("eps", m_eps);
    pp_mlmg.query("maxiter", m_maxiter);
    pp_mlmg.query("max_coarsening_level", m_max_coarsening_level);
    pp_mlmg.query("use_gmres", m_use_gmres);
    pp_mlmg.query("gmres_restart", m_gmres_restart);
    std::string prec_str;
    if (pp_mlmg.query("precision", prec_str)) {
        m_precision = parseMLMGPrecision(prec_str);
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_gmres_restart > 0, "GMRES restart length must be positive");

    const amrex::Real* dx_tmp = m_geom.CellSize();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        m_dx[d] = dx_tmp[d];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_dx[d] > 0.0, "Cell size must be positive.");
    }

    amrex::Copy(m_mf_phase_original, mf_phase_input, 0, 0, 1, 0);
    m_mf_phase_original.FillBoundary(m_geom.periodicity());

    if (!phase_coeffs.empty()) {
        m_max_diff_coeff = 0.0;
        for (const auto& kv : phase_coeffs) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(kv.second >= 0.0,
                                             "Phase diffusivities must be non-negative");
            m_phase_coeff_map[kv.first] = kv.second;
            m_max_diff_coeff = std::max(m_max_diff_coeff, kv.second);
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
                "tortuosity.active_phases and tortuosity.phase_diffusivities must match in length");
            m_max_diff_coeff = 0.0;
            for (std::size_t idx = 0; idx < active_phases_vec.size(); ++idx) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(phase_diffs_vec[idx] >= 0.0,
                                                 "Phase diffusivities must be non-negative");
                m_phase_coeff_map[active_phases_vec[idx]] = phase_diffs_vec[idx];
                m_max_diff_coeff = std::max(m_max_diff_coeff, phase_diffs_vec[idx]);
            }
            m_is_multi_phase = true;
        } else {
            m_phase_coeff_map[m_phase_id] = 1.0;
            m_max_diff_coeff = 1.0;
            m_is_multi_phase = false;
        }
    }

    m_mf_diff_coeff.setVal(0.0);
    initializeDiffCoeff();
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    m_mf_active_mask.setVal(CELL_INACTIVE);
    generateActiveMask();
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    m_num_active_cells = countActiveCells();
    m_mf_chi.setVal(0.0);

    if (m_num_active_cells == 0) {
        m_mf_cell_type.setVal(CELL_INACTIVE);
        m_converged = true;
        return;
    }

    if (precomputed_cell_type != nullptr) {
        amrex::Copy(m_mf_cell_type, *precomputed_cell_type, 0, 0, 1, 0);
        m_mf_cell_type.FillBoundary(m_geom.periodicity());
    } else {
        buildPeriodicCellTypeMask(m_geom, m_ba, m_dm, m_mf_active_mask, m_mf_cell_type, 0);
    }
}

void EffectiveDiffusivityMLMG::initializeDiffCoeff() {
    int max_pid = 0;
    for (const auto& kv : m_phase_coeff_map) {
        max_pid = std::max(max_pid, kv.first);
    }
    amrex::Gpu::DeviceVector<amrex::Real> d_coeff_lut(max_pid + 1, 0.0);
    amrex::Gpu::HostVector<amrex::Real> h_coeff_lut(max_pid + 1, 0.0);
    for (const auto& kv : m_phase_coeff_map) {
        if (kv.first >= 0) {
            h_coeff_lut[kv.first] = kv.second;
        }
    }
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_coeff_lut.begin(), h_coeff_lut.end(),
                     d_coeff_lut.begin());
    const amrex::Real* lut_ptr = d_coeff_lut.data();
    const int lut_size = max_pid + 1;

    for (amrex::MFIter mfi(m_mf_diff_coeff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        const auto dc_arr = m_mf_diff_coeff.array(mfi);
        const auto phase_arr = m_mf_phase_original.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int pid = phase_arr(i, j, k, 0);
            dc_arr(i, j, k, 0) = (pid >= 0 && pid < lut_size) ? lut_ptr[pid] : 0.0;
        });
    }
}

void EffectiveDiffusivityMLMG::generateActiveMask() {
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto mask_arr = m_mf_active_mask.array(mfi);
        const auto dc_arr = m_mf_diff_coeff.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            mask_arr(i, j, k, 0) = (dc_arr(i, j, k, 0) > 0.0) ? CELL_ACTIVE : CELL_INACTIVE;
        });
    }
}

long EffectiveDiffusivityMLMG::countActiveCells() const {
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<long> reduce_data(reduce_op);
    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& vbox = mfi.validbox();
        const auto arr = m_mf_active_mask.const_array(mfi);
        reduce_op.eval(vbox, reduce_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<long> {
                           return {(arr(i, j, k, 0) == CELL_ACTIVE) ? 1L : 0L};
                       });
    }
    long total = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceLongSum(total);
    return total;
}

bool EffectiveDiffusivityMLMG::solveFloat64() {
    const int idir = static_cast<int>(m_dir_solve);
    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);
    amrex::MLABecLaplacian op({m_geom}, {m_ba}, {m_dm}, lp_info);
    op.setMaxOrder(2);

    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc{AMREX_D_DECL(
        amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)};
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc{AMREX_D_DECL(
        amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)};
    op.setDomainBC(lo_bc, hi_bc);
    op.setLevelBC(0, nullptr);

    const amrex::Real length_k = m_geom.ProbLength(idir);
    const amrex::Real d_scale = (m_max_diff_coeff > 0.0) ? m_max_diff_coeff : 1.0;
    const amrex::Real alpha_inactive = d_scale / (length_k * length_k);
    const amrex::Real pin_diag = d_scale * (1.0 / (m_dx[0] * m_dx[0]) + 1.0 / (m_dx[1] * m_dx[1]) +
                                            1.0 / (m_dx[2] * m_dx[2]));
    const amrex::Real pinned_acoef = pin_diag / alpha_inactive;
    op.setScalars(alpha_inactive, 1.0);

    amrex::MultiFab acoef(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(acoef, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto a = acoef.array(mfi);
        const auto ct = m_mf_cell_type.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int cell_state = ct(i, j, k, 0);
            if (cell_state == CELL_ACTIVE) {
                a(i, j, k, 0) = 0.0;
            } else if (cell_state == CELL_PINNED) {
                a(i, j, k, 0) = pinned_acoef;
            } else {
                a(i, j, k, 0) = 1.0;
            }
        });
    }
    op.setACoeffs(0, acoef);

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoefs;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        bcoefs[d].define(amrex::convert(m_ba, amrex::IntVect::TheDimensionVector(d)), m_dm, 1, 0);
        const auto shift = amrex::IntVect::TheDimensionVector(d);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(bcoefs[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto b = bcoefs[d].array(mfi);
            const auto dc = m_mf_diff_coeff.const_array(mfi);
            const auto mask = m_mf_active_mask.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::IntVect hi(i, j, k);
                const amrex::IntVect lo = hi - shift;
                const amrex::Real left = dc(lo);
                const amrex::Real right = dc(hi);
                b(i, j, k, 0) =
                    (mask(lo) != CELL_INACTIVE && mask(hi) != CELL_INACTIVE && (left + right > 0.0))
                        ? (2.0 * left * right / (left + right))
                        : 0.0;
            });
        }
    }
    op.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs));

    amrex::MultiFab rhs(m_ba, m_dm, 1, 0);
    const amrex::Real inv_hk = 1.0 / m_dx[idir];
    const auto shift_k = amrex::IntVect::TheDimensionVector(idir);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto f_arr = rhs.array(mfi);
        const auto b_k = bcoefs[idir].const_array(mfi);
        const auto mask = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::IntVect iv(i, j, k);
            if (mask(iv) == CELL_INACTIVE) {
                f_arr(iv) = 0.0;
            } else {
                f_arr(iv) = (b_k(iv + shift_k) - b_k(iv)) * inv_hk;
            }
        });
    }

    const amrex::Real rhs_norm_inf = rhs.norm0();
    const amrex::Real zero_cutoff = 1.0e-30 * d_scale * inv_hk;
    if (rhs_norm_inf <= zero_cutoff) {
        m_mf_chi.FillBoundary(m_geom.periodicity());
        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        return true;
    }

    const amrex::Real rhs_norm_2 = rhs.norm2();
    const amrex::Real tol_abs_inf = m_eps * rhs_norm_inf;
    const amrex::Real tol_abs_2 = m_eps * rhs_norm_2;

    amrex::MLMG mlmg(op);
    mlmg.setMaxIter(m_maxiter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(0);
    mlmg.setThrowException(true);

    amrex::Real abs_res = -1.0;
    try {
        if (m_use_gmres) {
            amrex::GMRESMLMG gmres(mlmg);
            gmres.getGMRES().setRestartLength(m_gmres_restart);
            gmres.setMaxIters(m_maxiter);
            if (m_verbose > 1) {
                gmres.setVerbose(m_verbose - 1);
            }
            gmres.getGMRES().setVerbose(m_verbose);
            gmres.solve({&m_mf_chi}, {&rhs}, m_eps, tol_abs_2);
            abs_res = gmres.getResidualNorm();
            m_num_iterations = gmres.getNumIters();
            m_converged = (gmres.getGMRES().getStatus() == 0);
            m_final_res_norm = (rhs_norm_2 > 0.0) ? (abs_res / rhs_norm_2) : abs_res;
        } else {
            abs_res = mlmg.solve({&m_mf_chi}, {&rhs}, m_eps, tol_abs_inf);
            m_num_iterations = mlmg.getNumIters();
            m_converged = true;
            m_final_res_norm = (rhs_norm_inf > 0.0) ? (abs_res / rhs_norm_inf) : abs_res;
        }
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "EffectiveDiffusivityMLMG: solver failed: " << e.what() << std::endl;
        }
        m_converged = false;
        m_mf_chi.setVal(0.0);
    }

    m_mf_chi.FillBoundary(m_geom.periodicity());
    return m_converged;
}

bool EffectiveDiffusivityMLMG::solveFloat32() {
    const int idir = static_cast<int>(m_dir_solve);
    const amrex::Real h_ref = m_dx[0];
    const amrex::Real d_scale = (m_max_diff_coeff > 0.0) ? m_max_diff_coeff : 1.0;
    const amrex::Real inv_d_scale = 1.0 / d_scale;

    amrex::RealBox rb_nd({AMREX_D_DECL(0.0, 0.0, 0.0)},
                         {AMREX_D_DECL(m_geom.ProbLength(0) / h_ref, m_geom.ProbLength(1) / h_ref,
                                       m_geom.ProbLength(2) / h_ref)});
    amrex::Array<int, AMREX_SPACEDIM> is_per{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom_nd(m_geom.Domain(), &rb_nd, amrex::CoordSys::cartesian, is_per.data());
    const amrex::Real* dx_nd = geom_nd.CellSize();

    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);
    amrex::MLABecLaplacianT<amrex::fMultiFab> op_f({geom_nd}, {m_ba}, {m_dm}, lp_info);
    op_f.setMaxOrder(2);

    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc{AMREX_D_DECL(
        amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)};
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc{AMREX_D_DECL(
        amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)};
    op_f.setDomainBC(lo_bc, hi_bc);
    op_f.setLevelBC(0, nullptr);

    const amrex::Real length_nd = geom_nd.ProbLength(idir);
    const amrex::Real alpha_nd = 1.0 / (length_nd * length_nd);
    const amrex::Real pin_diag_nd =
        1.0 / (dx_nd[0] * dx_nd[0]) + 1.0 / (dx_nd[1] * dx_nd[1]) + 1.0 / (dx_nd[2] * dx_nd[2]);
    const float pinned_acoef_f = static_cast<float>(pin_diag_nd / alpha_nd);
    op_f.setScalars(static_cast<float>(alpha_nd), 1.0f);

    amrex::fMultiFab acoef_f(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(acoef_f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto a = acoef_f.array(mfi);
        const auto ct = m_mf_cell_type.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int cell_state = ct(i, j, k, 0);
            if (cell_state == CELL_ACTIVE) {
                a(i, j, k, 0) = 0.0f;
            } else if (cell_state == CELL_PINNED) {
                a(i, j, k, 0) = pinned_acoef_f;
            } else {
                a(i, j, k, 0) = 1.0f;
            }
        });
    }
    op_f.setACoeffs(0, acoef_f);

    amrex::Array<amrex::fMultiFab, AMREX_SPACEDIM> bcoefs_f;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        bcoefs_f[d].define(amrex::convert(m_ba, amrex::IntVect::TheDimensionVector(d)), m_dm, 1, 0);
        const auto shift = amrex::IntVect::TheDimensionVector(d);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(bcoefs_f[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto b = bcoefs_f[d].array(mfi);
            const auto dc = m_mf_diff_coeff.const_array(mfi);
            const auto mask = m_mf_active_mask.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::IntVect hi(i, j, k);
                const amrex::IntVect lo = hi - shift;
                const amrex::Real left = dc(lo) * inv_d_scale;
                const amrex::Real right = dc(hi) * inv_d_scale;
                b(i, j, k, 0) =
                    (mask(lo) != CELL_INACTIVE && mask(hi) != CELL_INACTIVE && (left + right > 0.0))
                        ? static_cast<float>(2.0 * left * right / (left + right))
                        : 0.0f;
            });
        }
    }
    op_f.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs_f));

    amrex::fMultiFab rhs_f(m_ba, m_dm, 1, 0);
    const float inv_hk_nd = static_cast<float>(1.0 / dx_nd[idir]);
    const auto shift_k = amrex::IntVect::TheDimensionVector(idir);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(rhs_f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto f_arr = rhs_f.array(mfi);
        const auto b_k = bcoefs_f[idir].const_array(mfi);
        const auto mask = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::IntVect iv(i, j, k);
            if (mask(iv) == CELL_INACTIVE) {
                f_arr(iv) = 0.0f;
            } else {
                f_arr(iv) = (b_k(iv + shift_k) - b_k(iv)) * inv_hk_nd;
            }
        });
    }

    const float rhs_norm_inf = rhs_f.norminf(0, 1, amrex::IntVect(0), false);
    if (rhs_norm_inf <= 1.0e-30f * inv_hk_nd) {
        m_mf_chi.FillBoundary(m_geom.periodicity());
        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        return true;
    }

    const float eps_f = std::max(static_cast<float>(m_eps), 5.0e-6f);
    const float rhs_norm_2 =
        std::sqrt(static_cast<float>(amrex::Dot(rhs_f, 0, 1, amrex::IntVect(0), false)));
    const float tol_abs_inf = eps_f * rhs_norm_inf;
    const float tol_abs_2 = eps_f * rhs_norm_2;

    amrex::fMultiFab chi_nd_f(m_ba, m_dm, 1, 1);
    chi_nd_f.setVal(0.0f);

    amrex::MLMGT<amrex::fMultiFab> mlmg_f(op_f);
    mlmg_f.setMaxIter(m_maxiter);
    mlmg_f.setVerbose(m_verbose);
    mlmg_f.setBottomVerbose(0);
    mlmg_f.setThrowException(true);

    float abs_res = -1.0f;
    try {
        if (m_use_gmres) {
            SingleLevelGMRESMLMG gmres_f(mlmg_f, m_ba, m_dm);
            gmres_f.getGMRES().setRestartLength(m_gmres_restart);
            gmres_f.setMaxIters(m_maxiter);
            if (m_verbose > 1) {
                gmres_f.setVerbose(m_verbose - 1);
            }
            gmres_f.getGMRES().setVerbose(m_verbose);
            gmres_f.solve(chi_nd_f, rhs_f, eps_f, tol_abs_2);
            abs_res = gmres_f.getResidualNorm();
            m_num_iterations = gmres_f.getNumIters();
            m_converged = (gmres_f.getGMRES().getStatus() == 0);
            m_final_res_norm =
                static_cast<amrex::Real>((rhs_norm_2 > 0.0f) ? (abs_res / rhs_norm_2) : abs_res);
        } else {
            abs_res = mlmg_f.solve({&chi_nd_f}, {&rhs_f}, eps_f, tol_abs_inf);
            m_num_iterations = mlmg_f.getNumIters();
            m_converged = true;
            m_final_res_norm = static_cast<amrex::Real>(
                (rhs_norm_inf > 0.0f) ? (abs_res / rhs_norm_inf) : abs_res);
        }
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "EffectiveDiffusivityMLMG(float32): solver failed: " << e.what()
                           << std::endl;
        }
        m_converged = false;
        m_mf_chi.setVal(0.0);
        return false;
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_chi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto cd = m_mf_chi.array(mfi);
        auto cf = chi_nd_f.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            cd(i, j, k) = h_ref * static_cast<amrex::Real>(cf(i, j, k));
        });
    }
    m_mf_chi.FillBoundary(m_geom.periodicity());
    return m_converged;
}

bool EffectiveDiffusivityMLMG::solveMixed() {
    const int idir = static_cast<int>(m_dir_solve);
    const amrex::Real h_ref = m_dx[0];
    const amrex::Real d_scale = (m_max_diff_coeff > 0.0) ? m_max_diff_coeff : 1.0;
    const amrex::Real inv_d_scale = 1.0 / d_scale;

    amrex::RealBox rb_nd({AMREX_D_DECL(0.0, 0.0, 0.0)},
                         {AMREX_D_DECL(m_geom.ProbLength(0) / h_ref, m_geom.ProbLength(1) / h_ref,
                                       m_geom.ProbLength(2) / h_ref)});
    amrex::Array<int, AMREX_SPACEDIM> is_per{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom_nd(m_geom.Domain(), &rb_nd, amrex::CoordSys::cartesian, is_per.data());
    const amrex::Real* dx_nd = geom_nd.CellSize();

    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc{AMREX_D_DECL(
        amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)};
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc{AMREX_D_DECL(
        amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic)};

    amrex::LPInfo lp_fine;
    lp_fine.setMaxCoarseningLevel(0);
    amrex::MLABecLaplacian op_fine_d({geom_nd}, {m_ba}, {m_dm}, lp_fine);
    op_fine_d.setMaxOrder(2);
    op_fine_d.setDomainBC(lo_bc, hi_bc);
    op_fine_d.setLevelBC(0, nullptr);

    amrex::LPInfo lp_mg;
    lp_mg.setMaxCoarseningLevel(m_max_coarsening_level);
    amrex::MLABecLaplacianT<amrex::fMultiFab> op_mg_f({geom_nd}, {m_ba}, {m_dm}, lp_mg);
    op_mg_f.setMaxOrder(2);
    op_mg_f.setDomainBC(lo_bc, hi_bc);
    op_mg_f.setLevelBC(0, nullptr);

    const amrex::Real length_nd = geom_nd.ProbLength(idir);
    const amrex::Real alpha_nd = 1.0 / (length_nd * length_nd);
    const amrex::Real pin_diag_nd =
        1.0 / (dx_nd[0] * dx_nd[0]) + 1.0 / (dx_nd[1] * dx_nd[1]) + 1.0 / (dx_nd[2] * dx_nd[2]);
    const amrex::Real pinned_acoef_d = pin_diag_nd / alpha_nd;
    const float pinned_acoef_f = static_cast<float>(pinned_acoef_d);
    op_fine_d.setScalars(alpha_nd, 1.0);
    op_mg_f.setScalars(static_cast<float>(alpha_nd), 1.0f);

    amrex::MultiFab rhs_d(m_ba, m_dm, 1, 0);
    {
        amrex::MultiFab acoef_d(m_ba, m_dm, 1, 0);
        amrex::fMultiFab acoef_f(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(acoef_d, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto ad = acoef_d.array(mfi);
            const auto af = acoef_f.array(mfi);
            const auto ct = m_mf_cell_type.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int cell_state = ct(i, j, k, 0);
                if (cell_state == CELL_ACTIVE) {
                    ad(i, j, k, 0) = 0.0;
                    af(i, j, k, 0) = 0.0f;
                } else if (cell_state == CELL_PINNED) {
                    ad(i, j, k, 0) = pinned_acoef_d;
                    af(i, j, k, 0) = pinned_acoef_f;
                } else {
                    ad(i, j, k, 0) = 1.0;
                    af(i, j, k, 0) = 1.0f;
                }
            });
        }
        op_fine_d.setACoeffs(0, acoef_d);
        op_mg_f.setACoeffs(0, acoef_f);

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoefs_d;
        amrex::Array<amrex::fMultiFab, AMREX_SPACEDIM> bcoefs_f;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            const auto face_ba = amrex::convert(m_ba, amrex::IntVect::TheDimensionVector(d));
            bcoefs_d[d].define(face_ba, m_dm, 1, 0);
            bcoefs_f[d].define(face_ba, m_dm, 1, 0);
            const auto shift = amrex::IntVect::TheDimensionVector(d);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(bcoefs_d[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const auto bd = bcoefs_d[d].array(mfi);
                const auto bf = bcoefs_f[d].array(mfi);
                const auto dc = m_mf_diff_coeff.const_array(mfi);
                const auto mask = m_mf_active_mask.const_array(mfi);
                amrex::ParallelFor(
                    mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::IntVect hi(i, j, k);
                        const amrex::IntVect lo = hi - shift;
                        const amrex::Real left = dc(lo) * inv_d_scale;
                        const amrex::Real right = dc(hi) * inv_d_scale;
                        const amrex::Real val = (mask(lo) != CELL_INACTIVE &&
                                                 mask(hi) != CELL_INACTIVE && (left + right > 0.0))
                                                    ? (2.0 * left * right / (left + right))
                                                    : 0.0;
                        bd(i, j, k, 0) = val;
                        bf(i, j, k, 0) = static_cast<float>(val);
                    });
            }
        }
        op_fine_d.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs_d));
        op_mg_f.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs_f));

        const amrex::Real inv_hk_nd = 1.0 / dx_nd[idir];
        const auto shift_k = amrex::IntVect::TheDimensionVector(idir);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rhs_d, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto f_arr = rhs_d.array(mfi);
            const auto b_k = bcoefs_d[idir].const_array(mfi);
            const auto mask = m_mf_active_mask.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::IntVect iv(i, j, k);
                if (mask(iv) == CELL_INACTIVE) {
                    f_arr(iv) = 0.0;
                } else {
                    f_arr(iv) = (b_k(iv + shift_k) - b_k(iv)) * inv_hk_nd;
                }
            });
        }
    }

    const amrex::Real rhs_norm_inf = rhs_d.norm0();
    if (rhs_norm_inf <= 1.0e-30 / dx_nd[idir]) {
        m_mf_chi.FillBoundary(m_geom.periodicity());
        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        return true;
    }

    const amrex::Real rhs_norm_2 = rhs_d.norm2();
    const amrex::Real tol_abs_inf = m_eps * rhs_norm_inf;
    const amrex::Real tol_abs_2 = m_eps * rhs_norm_2;

    m_mf_chi.setVal(0.0);
    m_mf_chi.FillBoundary(geom_nd.periodicity());

    amrex::MultiFab res_d(m_ba, m_dm, 1, 0);
    amrex::fMultiFab rhs_f(m_ba, m_dm, 1, 0);
    amrex::fMultiFab cor_f(m_ba, m_dm, 1, 1);

    amrex::MLMG mlmg_fine_d(op_fine_d);
    mlmg_fine_d.setVerbose(0);

    amrex::MLMGT<amrex::fMultiFab> mlmg_f(op_mg_f);
    mlmg_f.setMaxIter(m_maxiter);
    mlmg_f.setVerbose(m_verbose > 1 ? m_verbose - 1 : 0);
    mlmg_f.setBottomVerbose(0);
    mlmg_f.setThrowException(true);

    std::unique_ptr<SingleLevelGMRESMLMG> gmres_f;
    if (m_use_gmres) {
        gmres_f = std::make_unique<SingleLevelGMRESMLMG>(mlmg_f, m_ba, m_dm);
        gmres_f->getGMRES().setRestartLength(m_gmres_restart);
        gmres_f->setMaxIters(std::min(m_maxiter, 60));
        gmres_f->setVerbose(m_verbose > 1 ? m_verbose - 1 : 0);
        gmres_f->getGMRES().setVerbose(m_verbose > 1 ? m_verbose - 1 : 0);
    }

    int total_iters = 0;
    amrex::Real cur_res = 0.0;
    m_converged = false;

    constexpr int max_outer = 10;
    constexpr float inner_rel_tol = 1.0e-4f;

    try {
        for (int outer = 0; outer <= max_outer; ++outer) {
            m_mf_chi.FillBoundary(geom_nd.periodicity());
            mlmg_fine_d.compResidual({&res_d}, {&m_mf_chi}, {&rhs_d});
            cur_res = m_use_gmres ? res_d.norm2() : res_d.norm0();
            const amrex::Real abs_target = m_use_gmres ? tol_abs_2 : tol_abs_inf;
            if (cur_res <= abs_target) {
                m_converged = true;
                break;
            }
            if (outer == max_outer || total_iters >= m_maxiter) {
                break;
            }

            const amrex::Real scale_r = res_d.norm0();
            if (scale_r <= 0.0) {
                m_converged = true;
                break;
            }
            const amrex::Real inv_scale_r = 1.0 / scale_r;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(rhs_f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto rf = rhs_f.array(mfi);
                auto rd = res_d.const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    rf(i, j, k) = static_cast<float>(rd(i, j, k) * inv_scale_r);
                });
            }
            cor_f.setVal(0.0f);

            if (m_use_gmres) {
                const int rem = std::max(1, m_maxiter - total_iters);
                gmres_f->setMaxIters(std::min(rem, 60));
                gmres_f->solve(cor_f, rhs_f, inner_rel_tol, 0.0f);
                total_iters += gmres_f->getNumIters();
            } else {
                const int rem = std::max(1, m_maxiter - total_iters);
                mlmg_f.setMaxIter(std::min(rem, 60));
                mlmg_f.solve({&cor_f}, {&rhs_f}, inner_rel_tol, 0.0f);
                total_iters += mlmg_f.getNumIters();
            }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(m_mf_chi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto cd = m_mf_chi.array(mfi);
                auto cf = cor_f.const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    cd(i, j, k) += scale_r * static_cast<amrex::Real>(cf(i, j, k));
                });
            }
        }
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "EffectiveDiffusivityMLMG(mixed): solver failed: " << e.what()
                           << std::endl;
        }
        m_converged = false;
        m_mf_chi.setVal(0.0);
        return false;
    }

    m_num_iterations = total_iters;
    const amrex::Real denom_norm = m_use_gmres ? rhs_norm_2 : rhs_norm_inf;
    m_final_res_norm = (denom_norm > 0.0) ? (cur_res / denom_norm) : cur_res;
    m_mf_chi.mult(h_ref, 0, 1, 0);
    m_mf_chi.FillBoundary(m_geom.periodicity());
    return m_converged;
}

bool EffectiveDiffusivityMLMG::solve() {
    BL_PROFILE("EffectiveDiffusivityMLMG::solve");

    m_mf_chi.setVal(0.0);
    if (m_num_active_cells == 0) {
        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        if (m_write_plotfile) {
            writeChiPlotfile();
        }
        return true;
    }

    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    m_mf_cell_type.FillBoundary(m_geom.periodicity());
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    bool ok = false;
    switch (m_precision) {
    case MLMGPrecision::Float32:
        ok = solveFloat32();
        break;
    case MLMGPrecision::Mixed:
        ok = solveMixed();
        break;
    case MLMGPrecision::Float64:
    default:
        ok = solveFloat64();
        break;
    }

    if (m_write_plotfile && ok) {
        writeChiPlotfile();
    }
    return ok;
}

void EffectiveDiffusivityMLMG::getChiSolution(amrex::MultiFab& chi_field) const {
    AMREX_ALWAYS_ASSERT(chi_field.nComp() >= 1);
    AMREX_ALWAYS_ASSERT(chi_field.boxArray() == m_ba);
    AMREX_ALWAYS_ASSERT(chi_field.DistributionMap() == m_dm);

    amrex::Copy(chi_field, m_mf_chi, 0, 0, 1, 0);
    if (chi_field.nGrow() > 0) {
        chi_field.FillBoundary(m_geom.periodicity());
    }
}

void EffectiveDiffusivityMLMG::writeChiPlotfile() const {
    amrex::MultiFab mf_plot(m_ba, m_dm, 2, 0);
    amrex::Copy(mf_plot, m_mf_chi, 0, 0, 1, 0);

    for (amrex::MFIter mfi(mf_plot, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto plot_arr = mf_plot.array(mfi);
        const auto mask_arr = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            plot_arr(i, j, k, 1) = static_cast<amrex::Real>(mask_arr(i, j, k, 0));
        });
    }

    const std::string plot_dir =
        m_resultspath + "/effdiff_chi_dir" + std::to_string(static_cast<int>(m_dir_solve));
    const amrex::Vector<std::string> varnames = {"chi_k", "active_mask_from_solver"};
    amrex::WriteSingleLevelPlotfile(plot_dir, mf_plot, varnames, m_geom, 0.0, 0);
}

} // namespace OpenImpala
