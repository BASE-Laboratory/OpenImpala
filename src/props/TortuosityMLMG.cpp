// --- TortuosityMLMG.cpp ---
//
// EB-based matrix-free MLMG tortuosity solver. Non-percolating cells are
// encoded as an EB body region via an implicit function over the active
// mask. AMReX builds geometric apertures and coarsens EB metadata across
// MG levels, preserving the fine operator's restriction — recovering the
// O(N) iteration count that the earlier alpha*a pin could not achieve.

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
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_Vector.H>

namespace OpenImpala {

namespace {
constexpr int MaskComp = 0;
constexpr int cell_inactive = 0;
constexpr int cell_active = 1;

// Implicit function for EB: returns < 0 for fluid (active/percolating)
// and > 0 for body (inactive/non-percolating). Out-of-domain queries
// return fluid so that domain-boundary face apertures stay open and
// setDomainBC's Dirichlet/Neumann conditions apply normally.
struct ActiveMaskIF {
    int nx;
    int ny;
    int nz;
    amrex::Real plox;
    amrex::Real ploy;
    amrex::Real ploz;
    amrex::Real dx;
    amrex::Real dy;
    amrex::Real dz;
    const int* mask;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE inline amrex::Real
    operator()(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z)) const noexcept {
        const int i = static_cast<int>(std::floor((x - plox) / dx));
        const int j = static_cast<int>(std::floor((y - ploy) / dy));
        const int k = static_cast<int>(std::floor((z - ploz) / dz));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
            return amrex::Real(-1.0);
        }
        const std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(ny) *
                                    static_cast<std::size_t>(nx) +
                                static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                                static_cast<std::size_t>(i);
        return mask[idx] == cell_active ? amrex::Real(-1.0) : amrex::Real(1.0);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE inline amrex::Real
    operator()(const amrex::RealArray& p) const noexcept {
        return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
    }
};
} // namespace

// --- Constructor ---
TortuosityMLMG::TortuosityMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                               const amrex::DistributionMapping& dm,
                               const amrex::iMultiFab& mf_phase_input, const amrex::Real vf,
                               const int phase, const OpenImpala::Direction dir,
                               const std::string& resultspath, const amrex::Real vlo,
                               const amrex::Real vhi, int verbose, bool write_plotfile,
                               amrex::Real eps, int maxiter, int max_coarsening_level)
    : TortuositySolverBase(geom, ba, dm, mf_phase_input, vf, phase, dir, resultspath, vlo, vhi,
                           verbose, write_plotfile) {
    m_eps = eps;
    m_maxiter = maxiter;
    m_max_coarsening_level = max_coarsening_level;

    // EB cut cells at the active/inactive boundary introduce small
    // geometric flux errors (~0.3% for typical geometries). Loosen
    // the boundary flux conservation tolerance from the HYPRE default
    // of 1e-4 to 1e-2.
    m_flux_tol = 1.0e-2;

    amrex::ParmParse pp_mlmg("mlmg");
    pp_mlmg.query("eps", m_eps);
    pp_mlmg.query("maxiter", m_maxiter);
    pp_mlmg.query("max_coarsening_level", m_max_coarsening_level);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Max iterations must be positive");

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityMLMG: Initialized with eps=" << m_eps
                       << ", maxiter=" << m_maxiter << ", max_coarsening=" << m_max_coarsening_level
                       << std::endl;
    }
}

// --- solve ---
bool TortuosityMLMG::solve() {
    BL_PROFILE("TortuosityMLMG::solve");

    const int idir = static_cast<int>(m_dir);
    const amrex::Box& domain = m_geom.Domain();
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    const std::size_t total_cells =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    // -----------------------------------------------------------------
    // Step 1: gather global mask from local FABs.
    // -----------------------------------------------------------------
    std::vector<int> host_mask(total_cells, cell_inactive);
    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto mask_arr = m_mf_active_mask.const_array(mfi);
        const int nx_l = nx;
        const int ny_l = ny;
        int* hm = host_mask.data();
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
            const std::size_t idx = static_cast<std::size_t>(k) * ny_l * nx_l +
                                    static_cast<std::size_t>(j) * nx_l +
                                    static_cast<std::size_t>(i);
            hm[idx] = mask_arr(i, j, k, 0);
        });
    }

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<int> device_mask(total_cells);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host_mask.data(),
                          host_mask.data() + total_cells, device_mask.data());
    amrex::Gpu::streamSynchronize();
    const int* mask_data_ptr = device_mask.data();
#else
    const int* mask_data_ptr = host_mask.data();
#endif

    // -----------------------------------------------------------------
    // Step 2: build EB from the mask.
    // -----------------------------------------------------------------
    const amrex::Real* dx = m_geom.CellSize();
    ActiveMaskIF if_obj{nx,    ny,    nz,    m_geom.ProbLo(0), m_geom.ProbLo(1), m_geom.ProbLo(2),
                        dx[0], dx[1], dx[2], mask_data_ptr};
    auto gshop = amrex::EB2::makeShop(if_obj);
    amrex::EB2::Build(gshop, m_geom, 0, m_max_coarsening_level);

    const amrex::EB2::IndexSpace& eb_is = amrex::EB2::IndexSpace::top();
    const amrex::EB2::Level& eb_level = eb_is.getLevel(m_geom);

    const amrex::Vector<int> ng{2, 2, 2};
    amrex::EBFArrayBoxFactory factory(eb_level, m_geom, m_ba, m_dm, ng, amrex::EBSupport::full);

    // -----------------------------------------------------------------
    // Step 3: build MLEBABecLap operator.
    // -----------------------------------------------------------------
    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);

    amrex::MLEBABecLap mlebop({m_geom}, {m_ba}, {m_dm}, lp_info,
                              amrex::Vector<amrex::EBFArrayBoxFactory const*>{&factory});

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
    mlebop.setDomainBC(lo_bc, hi_bc);

    // -----------------------------------------------------------------
    // Step 4: initial guess — linear ramp in flow direction.
    // -----------------------------------------------------------------
    amrex::MultiFab sol_eb(m_ba, m_dm, 1, 1, amrex::MFInfo(), factory);
    sol_eb.setVal(0.0);
    {
        const int n_cells = domain.length(idir);
        if (n_cells <= 1) {
            amrex::Abort("TortuosityMLMG: domain must have more than 1 cell in flow direction.");
        }
        const int dom_lo_dir = domain.smallEnd(idir);
        const int dom_hi_dir = domain.bigEnd(idir);
        const amrex::Real vlo = m_vlo;
        const amrex::Real vhi = m_vhi;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(sol_eb, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const phi = sol_eb.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::IntVect iv(i, j, k);
                int idx_in_dir = iv[idir] - dom_lo_dir;
                amrex::Real frac =
                    static_cast<amrex::Real>(idx_in_dir) / static_cast<amrex::Real>(n_cells - 1);
                if (iv[idir] >= dom_lo_dir && iv[idir] <= dom_hi_dir) {
                    phi(i, j, k) = vlo + frac * (vhi - vlo);
                } else if (iv[idir] < dom_lo_dir) {
                    phi(i, j, k) = vlo;
                } else {
                    phi(i, j, k) = vhi;
                }
            });
        }
    }
    sol_eb.FillBoundary(m_geom.periodicity());
    mlebop.setLevelBC(0, &sol_eb);

    // -----------------------------------------------------------------
    // Step 5: coefficients.
    // -----------------------------------------------------------------
    mlebop.setScalars(amrex::Real(0.0), amrex::Real(1.0));

    amrex::MultiFab acoef(m_ba, m_dm, 1, 0, amrex::MFInfo(), factory);
    acoef.setVal(0.0);
    mlebop.setACoeffs(0, acoef);

    // B = 1.0 everywhere. EB apertures handle active/inactive decoupling
    // geometrically — do NOT use the harmonic mean from m_mf_diff_coeff,
    // which would zero B at interface faces and cause 0/0 in the EB
    // stencil at cut cells.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoefs;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        amrex::BoxArray edge_ba = m_ba;
        edge_ba.surroundingNodes(d);
        bcoefs[d].define(edge_ba, m_dm, 1, 0, amrex::MFInfo(), factory);
        bcoefs[d].setVal(1.0);
    }
    mlebop.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs));

    amrex::MultiFab rhs(m_ba, m_dm, 1, 0, amrex::MFInfo(), factory);
    rhs.setVal(0.0);

    // -----------------------------------------------------------------
    // Step 6: run MLMG.
    // -----------------------------------------------------------------
    amrex::MLMG mlmg(mlebop);
    mlmg.setMaxIter(m_maxiter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(0);
    mlmg.setThrowException(true);

    amrex::Real res_norm = -1.0;
    try {
        res_norm = mlmg.solve({&sol_eb}, {&rhs}, m_eps, 0.0);
        m_converged = true;
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityMLMG: MLMG solver failed: " << e.what() << std::endl;
        }
        m_converged = false;
    }

    m_final_res_norm = res_norm;
    m_num_iterations = mlmg.getNumIters();
    if (m_converged && res_norm >= m_eps) {
        m_converged = false;
    }

    // Copy EB solution back into base-class m_mf_solution for globalFluxes.
    sol_eb.FillBoundary(m_geom.periodicity());
    amrex::MultiFab::Copy(m_mf_solution, sol_eb, 0, 0, 1,
                          std::min(sol_eb.nGrow(), m_mf_solution.nGrow()));
    m_mf_solution.FillBoundary(m_geom.periodicity());

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  MLMG solve: residual=" << std::scientific << res_norm
                       << std::defaultfloat << ", iterations=" << m_num_iterations
                       << ", converged=" << m_converged << std::endl;
    }

    if (m_write_plotfile && m_converged) {
        writeSolutionPlotfile("tortuosity_mlmg_" + std::to_string(idir));
    }

    amrex::EB2::IndexSpace::pop();

    return m_converged;
}

} // namespace OpenImpala
