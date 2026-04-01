// --- TortuosityMLMG.cpp ---

#include "TortuosityMLMG.H"

#include <cmath>
#include <iomanip>
#include <string>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
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
#include <AMReX_Vector.H>

namespace OpenImpala {

// --- Constructor ---
TortuosityMLMG::TortuosityMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                               const amrex::DistributionMapping& dm,
                               const amrex::iMultiFab& mf_phase_input, const amrex::Real vf,
                               const int phase, const OpenImpala::Direction dir,
                               const std::string& resultspath, const amrex::Real vlo,
                               const amrex::Real vhi, int verbose, bool write_plotfile)
    : TortuositySolverBase(geom, ba, dm, mf_phase_input, vf, phase, dir, resultspath, vlo, vhi,
                           verbose, write_plotfile) {
    // Parse MLMG-specific solver parameters
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
// Uses AMReX MLABecLaplacian + MLMG to solve div(B grad phi) = 0
// with Dirichlet BCs at inlet/outlet and Neumann on lateral faces.
bool TortuosityMLMG::solve() {
    BL_PROFILE("TortuosityMLMG::solve");

    const int idir = static_cast<int>(m_dir);

    // --- Set up the MLABecLaplacian operator ---
    // Solves: alpha * a * phi - beta * div(B grad phi) = rhs
    // For Laplacian: alpha=0, beta=1, a=0, rhs=0 => -div(B grad phi) = 0
    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);

    amrex::MLABecLaplacian mlabec({m_geom}, {m_ba}, {m_dm}, lp_info);

    // Domain boundary conditions: Dirichlet in flow dir, Neumann on sides
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
    mlabec.setDomainBC(lo_bc, hi_bc);

    // --- Adjust Dirichlet face values for HYPRE-compatible cell-centre BCs ---
    //
    // AMReX MLABecLaplacian applies Dirichlet BCs at domain faces (half a cell
    // outside the boundary cell centre). The shared flux integration code
    // (globalFluxes / value) expects the HYPRE convention where Dirichlet
    // values are at boundary cell centres: cell 0 = vlo, cell N-1 = vhi.
    //
    // To make the MLMG face BC produce the same cell-centre values, we extend
    // the face values outward by half a cell:
    //   face_lo = vlo - 0.5 * (vhi - vlo) / (N - 1)
    //   face_hi = vhi + 0.5 * (vhi - vlo) / (N - 1)
    //
    // This ensures the linear solution through cell centres hits exactly
    // vlo at cell 0 and vhi at cell N-1, matching HYPRE's τ = (N-1)/N.
    const amrex::Box& domain = m_geom.Domain();
    const int n_cells = domain.length(idir);
    if (n_cells <= 1) {
        amrex::Abort("TortuosityMLMG: domain must have more than 1 cell in flow direction.");
    }
    const amrex::Real half_step = 0.5 * (m_vhi - m_vlo) / static_cast<amrex::Real>(n_cells - 1);
    const amrex::Real face_vlo = m_vlo - half_step;
    const amrex::Real face_vhi = m_vhi + half_step;

    // Set initial guess: linear ramp in flow direction for better convergence
    m_mf_solution.setVal(0.0);
    {
        const int dom_lo_dir = domain.smallEnd(idir);
        const int dom_hi_dir = domain.bigEnd(idir);
        const amrex::Real vlo = face_vlo;
        const amrex::Real vhi = face_vhi;
        // Ramp from face_vlo at the low face to face_vhi at the high face.
        // Cell centres at i map to fraction (i - dom_lo + 0.5) / n_cells.
        const amrex::Real inv_n = 1.0 / static_cast<amrex::Real>(n_cells);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(m_mf_solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const phi = m_mf_solution.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::IntVect iv(i, j, k);
                int idx_in_dir = iv[idir] - dom_lo_dir;
                amrex::Real frac = (static_cast<amrex::Real>(idx_in_dir) + 0.5) * inv_n;
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
    m_mf_solution.FillBoundary(m_geom.periodicity());

    // Set level BC (ghost cell values encode the Dirichlet face data)
    mlabec.setLevelBC(0, &m_mf_solution);

    // Set coefficients: alpha*a - beta*div(B*grad)
    mlabec.setScalars(0.0, 1.0); // alpha=0, beta=1

    // A-coefficient (not used since alpha=0, but must be set)
    amrex::MultiFab acoef(m_ba, m_dm, 1, 0);
    acoef.setVal(0.0);
    mlabec.setACoeffs(0, acoef);

    // B-coefficients: face-centred diffusivities via harmonic mean.
    //
    // First, extrapolate m_mf_diff_coeff into physical boundary ghost cells
    // so that boundary-face harmonic means see the correct value (the adjacent
    // interior cell's D) instead of the default 0.  Without this, boundary
    // faces get B=0, effectively imposing zero-flux Neumann everywhere and
    // preventing MLMG from enforcing Dirichlet BCs properly.
    {
        const amrex::Box& domain = m_geom.Domain();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(m_mf_diff_coeff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Array4<amrex::Real> const dc = m_mf_diff_coeff.array(mfi);
            const amrex::Box& vbx = mfi.validbox();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                // Low boundary: copy interior value into ghost cell
                if (vbx.smallEnd(d) == domain.smallEnd(d)) {
                    const amrex::Box lobx = amrex::adjCellLo(vbx, d, 1);
                    const int interior = domain.smallEnd(d);
                    amrex::ParallelFor(lobx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        amrex::IntVect iv(i, j, k);
                        amrex::IntVect iv_int = iv;
                        iv_int[d] = interior;
                        dc(i, j, k) = dc(iv_int);
                    });
                }
                // High boundary: copy interior value into ghost cell
                if (vbx.bigEnd(d) == domain.bigEnd(d)) {
                    const amrex::Box hibx = amrex::adjCellHi(vbx, d, 1);
                    const int interior = domain.bigEnd(d);
                    amrex::ParallelFor(hibx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        amrex::IntVect iv(i, j, k);
                        amrex::IntVect iv_int = iv;
                        iv_int[d] = interior;
                        dc(i, j, k) = dc(iv_int);
                    });
                }
            }
        }
    }

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoefs;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        amrex::BoxArray edge_ba = m_ba;
        edge_ba.surroundingNodes(d);
        bcoefs[d].define(edge_ba, m_dm, 1, 0);
        bcoefs[d].setVal(0.0);
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_diff_coeff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Array4<const amrex::Real> const dc = m_mf_diff_coeff.const_array(mfi);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            const amrex::Box& ebx = amrex::surroundingNodes(mfi.tilebox(), d);
            amrex::Array4<amrex::Real> const bf = bcoefs[d].array(mfi);
            const amrex::IntVect shift = amrex::IntVect::TheDimensionVector(d);
            amrex::ParallelFor(ebx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::IntVect iv(i, j, k);
                amrex::Real D_lo = dc(iv - shift);
                amrex::Real D_hi = dc(iv);
                if (D_lo + D_hi > 0.0) {
                    bf(i, j, k) = 2.0 * D_lo * D_hi / (D_lo + D_hi);
                } else {
                    bf(i, j, k) = 0.0;
                }
            });
        }
    }
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs));

    // RHS = 0 (steady-state Laplacian)
    amrex::MultiFab rhs(m_ba, m_dm, 1, 0);
    rhs.setVal(0.0);

    // --- Run MLMG solver ---
    amrex::MLMG mlmg(mlabec);
    mlmg.setMaxIter(m_maxiter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(0);

    amrex::Real res_norm = -1.0;
    try {
        res_norm = mlmg.solve({&m_mf_solution}, {&rhs}, m_eps, 0.0);
        m_converged = true;
    } catch (const std::exception& e) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityMLMG: MLMG solver failed: " << e.what() << std::endl;
        }
        m_converged = false;
    }

    m_final_res_norm = res_norm;
    m_num_iterations = mlmg.getNumIters();
    // Also verify residual is below tolerance (MLMG may return without exception
    // but with residual above tolerance)
    if (m_converged && res_norm >= m_eps) {
        m_converged = false;
    }

    m_mf_solution.FillBoundary(m_geom.periodicity());

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  MLMG solve: residual=" << std::scientific << res_norm
                       << std::defaultfloat << ", iterations=" << m_num_iterations
                       << ", converged=" << m_converged << std::endl;
    }

    // Write plotfile if requested
    if (m_write_plotfile && m_converged) {
        writeSolutionPlotfile("tortuosity_mlmg_" + std::to_string(idir));
    }

    return m_converged;
}

} // namespace OpenImpala
