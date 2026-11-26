/** @file TortuosityMLMG.cpp
 *  @brief Matrix-free AMReX MLMG tortuosity solver with float64, float32, and mixed precision.
 */

#include "TortuosityMLMG.H"
#include "SolverConfig.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_CoordSys.H>
#include <AMReX_GMRES_MLMG.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>

namespace OpenImpala {

TortuosityMLMG::TortuosityMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                               const amrex::DistributionMapping& dm,
                               const amrex::iMultiFab& mf_phase_input, const amrex::Real vf,
                               const int phase, const OpenImpala::Direction dir,
                               const std::string& resultspath, const amrex::Real vlo,
                               const amrex::Real vhi, int verbose, bool write_plotfile,
                               amrex::Real eps, int maxiter, int max_coarsening_level,
                               const std::map<int, amrex::Real>& phase_coeffs,
                               MLMGPrecision precision)
    : TortuositySolverBase(geom, ba, dm, mf_phase_input, vf, phase, dir, resultspath, vlo, vhi,
                           verbose, write_plotfile, phase_coeffs),
      m_eps(eps), m_maxiter(maxiter), m_max_coarsening_level(max_coarsening_level),
      m_precision(precision) {
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

    if (m_precision == MLMGPrecision::Float32) {
        // Pure single-precision solves saturate around ~1e-5 relative residual,
        // so widen the boundary-conservation check accordingly.
        m_flux_tol = std::max(m_flux_tol, static_cast<amrex::Real>(5.0e-3));
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Max iterations must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_gmres_restart > 0, "GMRES restart length must be positive");

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityMLMG: Initialized with eps=" << m_eps
                       << ", maxiter=" << m_maxiter << ", max_coarsening=" << m_max_coarsening_level
                       << ", use_gmres=" << m_use_gmres << ", gmres_restart=" << m_gmres_restart
                       << ", precision=" << mlmgPrecisionString(m_precision) << std::endl;
    }
}

void TortuosityMLMG::initializeLinearRampSolution() {
    const int idir = static_cast<int>(m_dir);
    const amrex::Box& domain = m_geom.Domain();
    const int dom_lo_dir = domain.smallEnd(idir);
    const int dom_hi_dir = domain.bigEnd(idir);
    const amrex::Real inv_len_dir = 1.0 / static_cast<amrex::Real>(domain.length(idir));
    const amrex::Real vlo = m_vlo;
    const amrex::Real vhi = m_vhi;
    const amrex::Box dom_box = domain;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<amrex::Real> const phi = m_mf_solution.array(mfi);
        amrex::Array4<const int> const mask = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::IntVect iv(i, j, k);
            if (iv[idir] < dom_lo_dir) {
                phi(iv) = vlo;
            } else if (iv[idir] > dom_hi_dir) {
                phi(iv) = vhi;
            } else if (dom_box.contains(iv) && mask(iv) != 0) {
                const amrex::Real s =
                    (static_cast<amrex::Real>(iv[idir] - dom_lo_dir) + 0.5) * inv_len_dir;
                phi(iv) = vlo + (vhi - vlo) * s;
            } else {
                phi(iv) = 0.0;
            }
        });
    }
    m_mf_solution.FillBoundary(m_geom.periodicity());
}

bool TortuosityMLMG::solveFloat64() {
    const int idir = static_cast<int>(m_dir);
    const amrex::Box& domain = m_geom.Domain();
    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);
    amrex::MLABecLaplacian op({m_geom}, {m_ba}, {m_dm}, lp_info);
    op.setMaxOrder(2);

    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc{};
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (d == idir) {
            lo_bc[d] = amrex::LinOpBCType::Dirichlet;
            hi_bc[d] = amrex::LinOpBCType::Dirichlet;
        } else {
            lo_bc[d] = amrex::LinOpBCType::Neumann;
            hi_bc[d] = amrex::LinOpBCType::Neumann;
        }
    }
    op.setDomainBC(lo_bc, hi_bc);

    initializeLinearRampSolution();
    op.setLevelBC(0, &m_mf_solution);

    const amrex::Real length = m_geom.ProbLength(idir);
    const amrex::Real d_scale = (m_max_diff_coeff > 0.0) ? m_max_diff_coeff : 1.0;
    op.setScalars(d_scale / (length * length), 1.0);

    amrex::MultiFab acoef(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(acoef, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto a = acoef.array(mfi);
        auto mask = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            a(i, j, k) = mask(i, j, k) ? 0.0 : 1.0;
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
            auto b = bcoefs[d].array(mfi);
            auto dc = m_mf_diff_coeff.const_array(mfi);
            auto mask = m_mf_active_mask.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::IntVect hi(i, j, k), lo = hi - shift;
                if (!domain.contains(lo))
                    lo = hi;
                if (!domain.contains(hi))
                    hi = lo;
                const auto left = dc(lo);
                const auto right = dc(hi);
                b(i, j, k) = mask(lo) && mask(hi) && left + right > 0.0
                                 ? 2.0 * left * right / (left + right)
                                 : 0.0;
            });
        }
    }
    op.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs));
    amrex::MultiFab rhs(m_ba, m_dm, 1, 0);
    rhs.setVal(0.0);

    const amrex::Real dx_dir = m_geom.CellSize(idir);
    const amrex::Real dv = std::abs(m_vhi - m_vlo);
    const amrex::Real n_cross =
        static_cast<amrex::Real>(domain.numPts()) / static_cast<amrex::Real>(domain.length(idir));
    const amrex::Real est_bndry_cells = std::max<amrex::Real>(1.0, m_active_vf * n_cross);
    const amrex::Real bndry_scale = (dv > 0.0) ? (2.0 * d_scale * dv / (dx_dir * dx_dir)) : d_scale;
    const amrex::Real tol_abs_inf = m_eps * bndry_scale;
    const amrex::Real tol_abs_2 = m_eps * bndry_scale * std::sqrt(est_bndry_cells);

    amrex::MLMG mlmg(op);
    mlmg.setMaxIter(m_maxiter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(0);
    mlmg.setThrowException(true);

    amrex::Real res_norm = -1.0;
    try {
        if (m_use_gmres) {
            amrex::GMRESMLMG gmres(mlmg);
            gmres.getGMRES().setRestartLength(m_gmres_restart);
            gmres.setMaxIters(m_maxiter);
            if (m_verbose > 1) {
                gmres.setVerbose(m_verbose - 1);
            }
            gmres.getGMRES().setVerbose(m_verbose);
            gmres.solve({&m_mf_solution}, {&rhs}, m_eps, tol_abs_2);
            res_norm = gmres.getResidualNorm();
            m_num_iterations = gmres.getNumIters();
            m_converged = (gmres.getGMRES().getStatus() == 0);
        } else {
            res_norm = mlmg.solve({&m_mf_solution}, {&rhs}, m_eps, tol_abs_inf);
            m_num_iterations = mlmg.getNumIters();
            m_converged = true;
        }
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityMLMG: MLMG solver failed: " << e.what() << std::endl;
        }
        m_converged = false;
    }

    m_final_res_norm = res_norm;
    m_mf_solution.FillBoundary(m_geom.periodicity());
    return m_converged;
}

bool TortuosityMLMG::solveFloat32() {
    const int idir = static_cast<int>(m_dir);
    const amrex::Box& domain = m_geom.Domain();

    // Non-dimensionalise cell sizes by h_ref and diffusivities by d_scale so all
    // single-precision stencil entries and GMRES 2-norms stay O(1) even when
    // voxel_size is in nanometres/micrometres.
    const amrex::Real h_ref = m_geom.CellSize(0);
    const amrex::Real d_scale = (m_max_diff_coeff > 0.0) ? m_max_diff_coeff : 1.0;
    const amrex::Real inv_d_scale = 1.0 / d_scale;
    amrex::RealBox rb_nd({AMREX_D_DECL(0.0, 0.0, 0.0)},
                         {AMREX_D_DECL(m_geom.ProbLength(0) / h_ref, m_geom.ProbLength(1) / h_ref,
                                       m_geom.ProbLength(2) / h_ref)});
    amrex::Array<int, AMREX_SPACEDIM> is_per{
        AMREX_D_DECL(m_geom.isPeriodic(0), m_geom.isPeriodic(1), m_geom.isPeriodic(2))};
    amrex::Geometry geom_nd(domain, &rb_nd, amrex::CoordSys::cartesian, is_per.data());

    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);
    amrex::MLABecLaplacianT<amrex::fMultiFab> op_f({geom_nd}, {m_ba}, {m_dm}, lp_info);
    op_f.setMaxOrder(2);

    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc{};
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (d == idir) {
            lo_bc[d] = amrex::LinOpBCType::Dirichlet;
            hi_bc[d] = amrex::LinOpBCType::Dirichlet;
        } else {
            lo_bc[d] = amrex::LinOpBCType::Neumann;
            hi_bc[d] = amrex::LinOpBCType::Neumann;
        }
    }
    op_f.setDomainBC(lo_bc, hi_bc);

    initializeLinearRampSolution();
    amrex::fMultiFab sol_f(m_ba, m_dm, 1, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(sol_f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        auto sf = sol_f.array(mfi);
        auto sd = m_mf_solution.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            sf(i, j, k) = static_cast<float>(sd(i, j, k));
        });
    }
    sol_f.FillBoundary(geom_nd.periodicity());
    op_f.setLevelBC(0, &sol_f);

    const float length_nd = static_cast<float>(geom_nd.ProbLength(idir));
    op_f.setScalars(1.0f / (length_nd * length_nd), 1.0f);

    amrex::fMultiFab acoef_f(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(acoef_f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto a = acoef_f.array(mfi);
        auto mask = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            a(i, j, k) = mask(i, j, k) ? 0.0f : 1.0f;
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
            auto b = bcoefs_f[d].array(mfi);
            auto dc = m_mf_diff_coeff.const_array(mfi);
            auto mask = m_mf_active_mask.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::IntVect hi(i, j, k), lo = hi - shift;
                if (!domain.contains(lo))
                    lo = hi;
                if (!domain.contains(hi))
                    hi = lo;
                const amrex::Real left = dc(lo) * inv_d_scale;
                const amrex::Real right = dc(hi) * inv_d_scale;
                b(i, j, k) = mask(lo) && mask(hi) && (left + right > 0.0)
                                 ? static_cast<float>(2.0 * left * right / (left + right))
                                 : 0.0f;
            });
        }
    }
    op_f.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs_f));

    amrex::fMultiFab rhs_f(m_ba, m_dm, 1, 0);
    rhs_f.setVal(0.0f);

    const float eps_f = std::max(static_cast<float>(m_eps), 5.0e-6f);
    const float dx_nd_dir = static_cast<float>(geom_nd.CellSize(idir));
    const float dv_f = static_cast<float>(std::abs(m_vhi - m_vlo));
    const float n_cross =
        static_cast<float>(domain.numPts()) / static_cast<float>(domain.length(idir));
    const float est_bndry_cells = std::max(1.0f, static_cast<float>(m_active_vf) * n_cross);
    const float bndry_scale = (dv_f > 0.0f) ? (2.0f * dv_f / (dx_nd_dir * dx_nd_dir)) : 1.0f;
    const float tol_abs_inf = eps_f * bndry_scale;
    const float tol_abs_2 = eps_f * bndry_scale * std::sqrt(est_bndry_cells);

    amrex::MLMGT<amrex::fMultiFab> mlmg_f(op_f);
    mlmg_f.setMaxIter(m_maxiter);
    mlmg_f.setVerbose(m_verbose);
    mlmg_f.setBottomVerbose(0);
    mlmg_f.setThrowException(true);

    float res_norm_f = -1.0f;
    try {
        if (m_use_gmres) {
            amrex::GMRESMLMGT<amrex::fMultiFab> gmres_f(mlmg_f);
            gmres_f.getGMRES().setRestartLength(m_gmres_restart);
            gmres_f.setMaxIters(m_maxiter);
            if (m_verbose > 1) {
                gmres_f.setVerbose(m_verbose - 1);
            }
            gmres_f.getGMRES().setVerbose(m_verbose);
            gmres_f.solve({&sol_f}, {&rhs_f}, eps_f, tol_abs_2);
            res_norm_f = gmres_f.getResidualNorm();
            m_num_iterations = gmres_f.getNumIters();
            m_converged = (gmres_f.getGMRES().getStatus() == 0);
        } else {
            res_norm_f = mlmg_f.solve({&sol_f}, {&rhs_f}, eps_f, tol_abs_inf);
            m_num_iterations = mlmg_f.getNumIters();
            m_converged = true;
        }
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityMLMG(float32): MLMG solver failed: " << e.what()
                           << std::endl;
        }
        m_converged = false;
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto sd = m_mf_solution.array(mfi);
        auto sf = sol_f.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            sd(i, j, k) = static_cast<amrex::Real>(sf(i, j, k));
        });
    }
    m_mf_solution.FillBoundary(m_geom.periodicity());
    m_final_res_norm = static_cast<amrex::Real>(res_norm_f);
    return m_converged;
}

bool TortuosityMLMG::solveMixed() {
    const int idir = static_cast<int>(m_dir);
    const amrex::Box& domain = m_geom.Domain();

    // Non-dimensionalise geometry and diffusivity so both the fine-level double
    // residual evaluator and the single-precision multigrid hierarchy operate on
    // O(1) numbers regardless of physical SI voxel size.
    const amrex::Real h_ref = m_geom.CellSize(0);
    const amrex::Real d_scale = (m_max_diff_coeff > 0.0) ? m_max_diff_coeff : 1.0;
    const amrex::Real inv_d_scale = 1.0 / d_scale;
    amrex::RealBox rb_nd({AMREX_D_DECL(0.0, 0.0, 0.0)},
                         {AMREX_D_DECL(m_geom.ProbLength(0) / h_ref, m_geom.ProbLength(1) / h_ref,
                                       m_geom.ProbLength(2) / h_ref)});
    amrex::Array<int, AMREX_SPACEDIM> is_per{
        AMREX_D_DECL(m_geom.isPeriodic(0), m_geom.isPeriodic(1), m_geom.isPeriodic(2))};
    amrex::Geometry geom_nd(domain, &rb_nd, amrex::CoordSys::cartesian, is_per.data());

    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc{};
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (d == idir) {
            lo_bc[d] = amrex::LinOpBCType::Dirichlet;
            hi_bc[d] = amrex::LinOpBCType::Dirichlet;
        } else {
            lo_bc[d] = amrex::LinOpBCType::Neumann;
            hi_bc[d] = amrex::LinOpBCType::Neumann;
        }
    }

    initializeLinearRampSolution();

    // 1. Fine-grid-only double operator (MaxCoarseningLevel = 0 -> zero coarse levels)
    //    for exact double-precision residual evaluation r = rhs - L_double(phi).
    amrex::LPInfo lp_fine;
    lp_fine.setMaxCoarseningLevel(0);
    amrex::MLABecLaplacian op_fine_d({geom_nd}, {m_ba}, {m_dm}, lp_fine);
    op_fine_d.setMaxOrder(2);
    op_fine_d.setDomainBC(lo_bc, hi_bc);
    op_fine_d.setLevelBC(0, &m_mf_solution);

    // 2. Full multigrid hierarchy in float32 with homogeneous BCs for defect solves
    //    L_float(cor_f) = r_f.
    amrex::LPInfo lp_mg;
    lp_mg.setMaxCoarseningLevel(m_max_coarsening_level);
    amrex::MLABecLaplacianT<amrex::fMultiFab> op_mg_f({geom_nd}, {m_ba}, {m_dm}, lp_mg);
    op_mg_f.setMaxOrder(2);
    op_mg_f.setDomainBC(lo_bc, hi_bc);
    op_mg_f.setLevelBC(0, nullptr);

    const amrex::Real length_nd = geom_nd.ProbLength(idir);
    const amrex::Real alpha_nd = 1.0 / (length_nd * length_nd);
    op_fine_d.setScalars(alpha_nd, 1.0);
    op_mg_f.setScalars(static_cast<float>(alpha_nd), 1.0f);

    {
        amrex::MultiFab acoef_d(m_ba, m_dm, 1, 0);
        amrex::fMultiFab acoef_f(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(acoef_d, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto ad = acoef_d.array(mfi);
            auto af = acoef_f.array(mfi);
            auto mask = m_mf_active_mask.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const bool act = (mask(i, j, k) != 0);
                ad(i, j, k) = act ? 0.0 : 1.0;
                af(i, j, k) = act ? 0.0f : 1.0f;
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
                auto bd = bcoefs_d[d].array(mfi);
                auto bf = bcoefs_f[d].array(mfi);
                auto dc = m_mf_diff_coeff.const_array(mfi);
                auto mask = m_mf_active_mask.const_array(mfi);
                amrex::ParallelFor(
                    mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        amrex::IntVect hi(i, j, k), lo = hi - shift;
                        if (!domain.contains(lo))
                            lo = hi;
                        if (!domain.contains(hi))
                            hi = lo;
                        const amrex::Real left = dc(lo) * inv_d_scale;
                        const amrex::Real right = dc(hi) * inv_d_scale;
                        const amrex::Real val = (mask(lo) && mask(hi) && (left + right > 0.0))
                                                    ? (2.0 * left * right / (left + right))
                                                    : 0.0;
                        bd(i, j, k) = val;
                        bf(i, j, k) = static_cast<float>(val);
                    });
            }
        }
        op_fine_d.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs_d));
        op_mg_f.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs_f));
    }

    amrex::MultiFab rhs_zero_d(m_ba, m_dm, 1, 0);
    rhs_zero_d.setVal(0.0);
    amrex::MultiFab res_d(m_ba, m_dm, 1, 0);
    amrex::fMultiFab rhs_f(m_ba, m_dm, 1, 0);
    amrex::fMultiFab cor_f(m_ba, m_dm, 1, 1);

    const amrex::Real dx_nd_dir = geom_nd.CellSize(idir);
    const amrex::Real dv = std::abs(m_vhi - m_vlo);
    const amrex::Real n_cross =
        static_cast<amrex::Real>(domain.numPts()) / static_cast<amrex::Real>(domain.length(idir));
    const amrex::Real est_bndry_cells = std::max<amrex::Real>(1.0, m_active_vf * n_cross);
    const amrex::Real bndry_scale = (dv > 0.0) ? (2.0 * dv / (dx_nd_dir * dx_nd_dir)) : 1.0;
    const amrex::Real tol_abs_2 = m_eps * bndry_scale * std::sqrt(est_bndry_cells);
    const amrex::Real tol_abs_inf = m_eps * bndry_scale;

    amrex::MLMG mlmg_fine_d(op_fine_d);
    mlmg_fine_d.setVerbose(0);

    amrex::MLMGT<amrex::fMultiFab> mlmg_f(op_mg_f);
    mlmg_f.setMaxIter(m_maxiter);
    mlmg_f.setVerbose(m_verbose > 1 ? m_verbose - 1 : 0);
    mlmg_f.setBottomVerbose(0);
    mlmg_f.setThrowException(true);

    std::unique_ptr<amrex::GMRESMLMGT<amrex::fMultiFab>> gmres_f;
    if (m_use_gmres) {
        gmres_f = std::make_unique<amrex::GMRESMLMGT<amrex::fMultiFab>>(mlmg_f);
        gmres_f->getGMRES().setRestartLength(m_gmres_restart);
        gmres_f->setMaxIters(std::min(m_maxiter, 60));
        gmres_f->setVerbose(m_verbose > 1 ? m_verbose - 1 : 0);
        gmres_f->getGMRES().setVerbose(m_verbose > 1 ? m_verbose - 1 : 0);
    }

    int total_iters = 0;
    amrex::Real cur_res = 0.0;
    amrex::Real r0_norm = -1.0;
    m_converged = false;

    constexpr int max_outer = 10;
    constexpr float inner_rel_tol = 1.0e-4f;

    try {
        for (int outer = 0; outer <= max_outer; ++outer) {
            mlmg_fine_d.compResidual({&res_d}, {&m_mf_solution}, {&rhs_zero_d});
            cur_res = m_use_gmres ? res_d.norm2() : res_d.norm0();
            const amrex::Real abs_target = m_use_gmres ? tol_abs_2 : tol_abs_inf;
            if (r0_norm < 0.0) {
                r0_norm = cur_res;
            }
            if (cur_res <= std::max(abs_target, m_eps * r0_norm)) {
                m_converged = true;
                break;
            }
            if (outer == max_outer || total_iters >= m_maxiter) {
                break;
            }

            // Normalize defect RHS to O(1) before casting to float32 so tiny residuals
            // (~1e-8..1e-12) never underflow or hit float32 subnormal territory.
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
                gmres_f->solve({&cor_f}, {&rhs_f}, inner_rel_tol, 0.0f);
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
            for (amrex::MFIter mfi(m_mf_solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.tilebox();
                auto sd = m_mf_solution.array(mfi);
                auto cf = cor_f.const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    sd(i, j, k) += scale_r * static_cast<amrex::Real>(cf(i, j, k));
                });
            }
            m_mf_solution.FillBoundary(m_geom.periodicity());
        }
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityMLMG(mixed): MLMG solver failed: " << e.what()
                           << std::endl;
        }
        m_converged = false;
    }

    m_num_iterations = total_iters;
    // Scale normalized residual back to physical operator units for diagnostic parity
    const amrex::Real phys_scale = d_scale / (h_ref * h_ref);
    m_final_res_norm = cur_res * phys_scale;
    m_mf_solution.FillBoundary(m_geom.periodicity());
    return m_converged;
}

bool TortuosityMLMG::solve() {
    BL_PROFILE("TortuosityMLMG::solve");

    m_mf_active_mask.FillBoundary(m_geom.periodicity());
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

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  MLMG solve (" << mlmgPrecisionString(m_precision)
                       << "): residual=" << std::scientific << m_final_res_norm << std::defaultfloat
                       << ", iterations=" << m_num_iterations << ", converged=" << m_converged
                       << std::endl;
    }

    if (m_write_plotfile && ok) {
        writeSolutionPlotfile("tortuosity_mlmg_" + std::to_string(static_cast<int>(m_dir)));
    }

    return ok;
}

} // namespace OpenImpala
