#include "TortuosityMLMG.H"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
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
#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>

namespace OpenImpala {


// --- Constructor ---
TortuosityMLMG::TortuosityMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                               const amrex::DistributionMapping& dm,
                               const amrex::iMultiFab& mf_phase_input, const amrex::Real vf,
                               const int phase, const OpenImpala::Direction dir,
                               const std::string& resultspath, const amrex::Real vlo,
                               const amrex::Real vhi, int verbose, bool write_plotfile,
                               amrex::Real eps, int maxiter, int max_coarsening_level,
                               const std::map<int, amrex::Real>& phase_coeffs)
    : TortuositySolverBase(geom, ba, dm, mf_phase_input, vf, phase, dir, resultspath, vlo, vhi,
                           verbose, write_plotfile, phase_coeffs) {
    m_eps = eps;
    m_maxiter = maxiter;
    m_max_coarsening_level = max_coarsening_level;

    amrex::ParmParse pp_mlmg("mlmg");
    pp_mlmg.query("eps", m_eps);
    pp_mlmg.query("maxiter", m_maxiter);
    pp_mlmg.query("max_coarsening_level", m_max_coarsening_level);
    pp_mlmg.query("use_gmres", m_use_gmres);
    pp_mlmg.query("gmres_restart", m_gmres_restart);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Max iterations must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_gmres_restart > 0, "GMRES restart length must be positive");

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityMLMG: Initialized with eps=" << m_eps
                       << ", maxiter=" << m_maxiter << ", max_coarsening=" << m_max_coarsening_level
                       << ", use_gmres=" << m_use_gmres << ", gmres_restart=" << m_gmres_restart
                       << std::endl;
    }
}

// --- solve ---
bool TortuosityMLMG::solve() {
    BL_PROFILE("TortuosityMLMG::solve");

    const int idir = static_cast<int>(m_dir);
    const amrex::Box& domain = m_geom.Domain();
    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);
    amrex::MLABecLaplacian op({m_geom}, {m_ba}, {m_dm}, lp_info);
    op.setMaxOrder(2);

    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc;
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc;
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

    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    // -----------------------------------------------------------------
    // Set Dirichlet face values in exterior ghost cells and seed active
    // interior voxels with the 1-D linear potential ramp between vlo and
    // vhi.  For uniform domains and straight channels this initial guess
    // is the exact discrete harmonic solution (0 solver iterations); for
    // tortuous media it eliminates the macroscopic low-frequency mode up
    // front while keeping inactive voxels pinned at 0.
    // -----------------------------------------------------------------
    auto& solution = m_mf_solution;
    {
        const int dom_lo_dir = domain.smallEnd(idir);
        const int dom_hi_dir = domain.bigEnd(idir);
        const amrex::Real inv_len_dir = 1.0 / static_cast<amrex::Real>(domain.length(idir));
        const amrex::Real vlo = m_vlo;
        const amrex::Real vhi = m_vhi;
        const amrex::Box dom_box = domain;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const phi = solution.array(mfi);
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
    }
    solution.FillBoundary(m_geom.periodicity());
    op.setLevelBC(0, &solution);

    // -----------------------------------------------------------------
    // Step 5: coefficients.
    // -----------------------------------------------------------------
    // An inactive voxel is an isolated row. Pin its potential and give its
    // faces zero conductance so it cannot carry any current.
    // The pin only fixes an isolated unknown. Scale it to the macroscopic
    // diffusion operator so its coarse-grid average does not overwhelm the
    // diffusion mode on long connected paths.
    const amrex::Real length = m_geom.ProbLength(idir);
    const amrex::Real d_scale = (m_max_diff_coeff > 0.0) ? m_max_diff_coeff : 1.0;
    op.setScalars(d_scale / (length * length), 1.);
    amrex::MultiFab acoef(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(acoef, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto a = acoef.array(mfi);
        auto mask = m_mf_active_mask.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            a(i, j, k) = mask(i, j, k) ? 0. : 1.;
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
                // Use the adjacent cell diffusivity at the physical boundary.
                if (!domain.contains(lo))
                    lo = hi;
                if (!domain.contains(hi))
                    hi = lo;
                const auto left = dc(lo);
                const auto right = dc(hi);
                b(i, j, k) = mask(lo) && mask(hi) && left + right > 0.
                                 ? 2. * left * right / (left + right)
                                 : 0.;
            });
        }
    }
    op.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs));
    amrex::MultiFab rhs(m_ba, m_dm, 1, 0);
    rhs.setVal(0.0);

    // -----------------------------------------------------------------
    // Step 6: run GMRES-accelerated MLMG (or standalone MLMG if disabled).
    // -----------------------------------------------------------------
    // Because rhs == 0 and forcing enters solely through inhomogeneous
    // Dirichlet BCs, an initial guess near the exact solution yields a
    // machine-precision initial residual r0.  Supplying an absolute
    // tolerance anchored to the physical zero-guess boundary residual
    // ||L(0)|| prevents demanding m_eps * 1e-13 when r0 is already tiny.
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
            gmres.solve({&solution}, {&rhs}, m_eps, tol_abs_2);
            res_norm = gmres.getResidualNorm();
            m_num_iterations = gmres.getNumIters();
            m_converged = (gmres.getGMRES().getStatus() == 0);
        } else {
            res_norm = mlmg.solve({&solution}, {&rhs}, m_eps, tol_abs_inf);
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
    solution.FillBoundary(m_geom.periodicity());

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  MLMG solve: residual=" << std::scientific << res_norm
                       << std::defaultfloat << ", iterations=" << m_num_iterations
                       << ", converged=" << m_converged << std::endl;
    }

    if (m_write_plotfile && m_converged) {
        writeSolutionPlotfile("tortuosity_mlmg_" + std::to_string(idir));
    }

    return m_converged;
}

} // namespace OpenImpala
