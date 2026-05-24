// --- TortuosityMLMG.cpp ---
//
// EB-based reimplementation of the matrix-free MLMG tortuosity solver.
// See issue #289 for the migration rationale: the previous alpha*a row
// pin on inactive cells decoupled them correctly but degraded geometric
// multigrid coarsening (coarse cells averaging mixed active/inactive
// don't preserve fine-grid physics), giving ~5%/V-cycle reduction
// instead of the order-of-magnitude per cycle a healthy MG produces.
//
// Embedded-Boundary fixes that structurally: non-percolating cells are
// encoded as a "body" region via an implicit function over the active
// mask. AMReX builds geometric apertures/centroids from the IF and
// coarsens EB metadata across MG levels in a way that preserves the
// fine operator's restriction. No alpha*a pin, no dc_masked workaround.

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
// Active-mask sentinels — must match TortuositySolverBase.cpp.
constexpr int MaskComp = 0;
constexpr int cell_inactive = 0;
constexpr int cell_active = 1;

// Implicit function defining the EB body region from the precomputed
// active mask. Returns:
//
//   < 0 : fluid (active, percolating phase cell) -> regular EB cell
//   > 0 : body  (inactive, non-percolating cell) -> covered EB cell
//
// AMReX queries this IF at sample points within each cell to determine
// volume fractions and face apertures. Our mask is cell-binary, so the
// IF is a step function aligned to cell boundaries — apertures come out
// either 0 (active/inactive interface) or 1 (active/active), exactly the
// HYPRE row-decoupling analogue but encoded geometrically so MG
// coarsening preserves it.
//
// CRITICAL: out-of-domain queries must return FLUID (-1), not body (+1).
// EB2::Build uses ngrow ghost cells, so it queries the IF beyond the
// domain box. Returning body there makes domain-boundary face apertures
// zero, which overrides the Dirichlet/Neumann domain BCs set via
// setDomainBC — the solve converges to phi=const with zero flux.
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
// Uses AMReX MLEBABecLap + MLMG to solve -div(B grad phi) = 0 on the
// active subdomain, with the inactive (non-percolating) region carved
// out as an EB body. Dirichlet BCs at inlet/outlet, Neumann on sides
// and on the EB interface (the latter is MLEBABecLap's default).
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
    // Step 1: gather a global host-side mask cube.
    //
    // EB2::Build queries the IF at arbitrary points across the entire
    // domain, so the IF must be globally addressable. For single-rank
    // (the notebook workflow), read directly from local FABs. For
    // multi-rank, a ParallelCopy + Bcast would be needed.
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

    // DEBUG: verify mask was read correctly
    {
        int n_active_in_mask =
            static_cast<int>(std::count(host_mask.begin(), host_mask.end(), cell_active));
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  [DEBUG] host_mask: n_active=" << n_active_in_mask << " / "
                           << total_cells << "\n";
        }
    }
    // For multi-rank: would need MPI_Allreduce(MPI_MAX) here to merge
    // partial masks. Single-rank case (notebook workflow) is complete.

    // On GPU builds, copy to device-accessible memory for the IF.
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

    // required_coarsening_level = 0, max_coarsening_level for EB
    // mirrors MLMG's. EB2 pushes the new IndexSpace onto a global
    // stack — we erase it at the end of solve() to avoid leaking
    // Disable small-cell redistribution. Our IF is a cell-binary step
    // function, so cut cells at channel/solid boundaries have vfrac ≈ 0.25
    // at corners (2/8 vertices fluid). AMReX's default small_volfrac
    // threshold marks these as "small" and merges them into neighbors,
    // converting them to COVERED. If any were active channel cells, the
    // subsequent EB_set_covered zeros their solution, and globalFluxes
    // (which uses the active mask, not EB flags) reads zero where it
    // expects a ramp value — producing NaN fluxes. Setting small_volfrac
    // to 0 prevents any merging; all cut cells keep their partial vfrac
    // and the MLMG EB stencil handles them correctly.
    {
        amrex::ParmParse pp_eb2("eb2");
        pp_eb2.add("small_volfrac", 0.0);
    }
    amrex::EB2::Build(gshop, m_geom, 0, m_max_coarsening_level);

    const amrex::EB2::IndexSpace& eb_is = amrex::EB2::IndexSpace::top();
    const amrex::EB2::Level& eb_level = eb_is.getLevel(m_geom);

    // EBFArrayBoxFactory: 2 ghosts for the MLMG stencil; EBSupport::full
    // so we get volume fractions, apertures, and boundary data — all
    // needed by MLEBABecLap.
    const amrex::Vector<int> ng{2, 2, 2};
    amrex::EBFArrayBoxFactory factory(eb_level, m_geom, m_ba, m_dm, ng, amrex::EBSupport::full);

    // -----------------------------------------------------------------
    // Step 3: build MLEBABecLap operator and coefficients.
    // -----------------------------------------------------------------
    amrex::LPInfo lp_info;
    lp_info.setMaxCoarseningLevel(m_max_coarsening_level);

    amrex::MLEBABecLap mlebop({m_geom}, {m_ba}, {m_dm}, lp_info,
                              amrex::Vector<amrex::EBFArrayBoxFactory const*>{&factory});

    // Domain BCs: Dirichlet in flow direction, Neumann on sides.
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

    // EB boundary BC: MLEBABecLap defaults to no-flux Neumann at the
    // EB surface, which is exactly the physics we want at active/
    // inactive interfaces (no transport into solid). We deliberately
    // do NOT call setEBHomogDirichlet or setEBDirichlet.

    // -----------------------------------------------------------------
    // Step 4: initial guess — linear ramp in flow direction.
    // -----------------------------------------------------------------
    // Allocate an EB-aware solution MultiFab (MLEBABecLap requires
    // factory-allocated MultiFabs for its solve vector). Populate from
    // m_mf_solution if needed, then copy back at the end.
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

    // Pure Laplacian: alpha=0, beta=1 -> -div(B grad phi) = rhs.
    // No alpha*a pin needed — EB excludes inactive cells from the
    // operator entirely.
    mlebop.setScalars(amrex::Real(0.0), amrex::Real(1.0));

    // A-coefficient (unused with alpha=0, but the API requires it set).
    amrex::MultiFab acoef(m_ba, m_dm, 1, 0, amrex::MFInfo(), factory);
    acoef.setVal(0.0);
    mlebop.setACoeffs(0, acoef);

    // -----------------------------------------------------------------
    // Step 5: B-coefficients — face-centred harmonic mean of D.
    // -----------------------------------------------------------------
    // No mask-driven zeroing: the EB factory carries the apertures, so
    // active/inactive interface faces get aperture=0 automatically and
    // the B value there is irrelevant.
    //
    // m_mf_diff_coeff's domain-boundary ghost cells are uninitialized
    // (buildDiffusionCoeffField reads phase ghosts that from_numpy's
    // FillBoundary doesn't fill for non-periodic dirs). MLABecLaplacian
    // silently overrides boundary-face B with its internal stencil, but
    // MLEBABecLap uses user-provided B directly — so garbage ghost D
    // → harmonic_mean(garbage,D_inner) → wrong boundary-face B → the
    // solve converges to phi=const instead of the Dirichlet-driven ramp.
    // Fix: extrapolate domain-boundary ghosts from the nearest interior
    // cell before computing face B.
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (m_geom.isPeriodic(d)) {
            continue;
        }
        const amrex::IntVect e = amrex::IntVect::TheDimensionVector(d);
        const int dom_lo = domain.smallEnd(d);
        const int dom_hi = domain.bigEnd(d);
        for (amrex::MFIter mfi(m_mf_diff_coeff); mfi.isValid(); ++mfi) {
            const amrex::Box& fabbox = mfi.fabbox();
            amrex::Array4<amrex::Real> const dc = m_mf_diff_coeff.array(mfi);
            amrex::Box lo_box = fabbox;
            lo_box.setSmall(d, fabbox.smallEnd(d));
            lo_box.setBig(d, dom_lo - 1);
            if (!lo_box.isEmpty()) {
                amrex::ParallelFor(lo_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::IntVect iv(i, j, k);
                    dc(iv) = dc(iv + e);
                });
            }
            amrex::Box hi_box = fabbox;
            hi_box.setSmall(d, dom_hi + 1);
            hi_box.setBig(d, fabbox.bigEnd(d));
            if (!hi_box.isEmpty()) {
                amrex::ParallelFor(hi_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::IntVect iv(i, j, k);
                    dc(iv) = dc(iv - e);
                });
            }
        }
    }
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoefs;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        amrex::BoxArray edge_ba = m_ba;
        edge_ba.surroundingNodes(d);
        bcoefs[d].define(edge_ba, m_dm, 1, 0, amrex::MFInfo(), factory);
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
    mlebop.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoefs));

    // RHS = 0 (steady-state Laplacian, no source).
    amrex::MultiFab rhs(m_ba, m_dm, 1, 0, amrex::MFInfo(), factory);
    rhs.setVal(0.0);

    // -----------------------------------------------------------------
    // Step 6: run MLMG.
    // -----------------------------------------------------------------
    amrex::MLMG mlmg(mlebop);
    mlmg.setMaxIter(m_maxiter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(0);
    // Throw instead of amrex::Abort on non-convergence so the existing
    // catch translates to NaN at the Python boundary.
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

    // DEBUG: check sol_eb min/max before and after Copy
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Real sol_min = sol_eb.min(0);
        amrex::Real sol_max = sol_eb.max(0);
        std::fprintf(stderr, "  [DEBUG] sol_eb min=%g max=%g\n", sol_min, sol_max);
        std::fflush(stderr);
    }

    // Copy EB solution back into the base-class m_mf_solution that
    // globalFluxes() reads.
    sol_eb.FillBoundary(m_geom.periodicity());
    amrex::MultiFab::Copy(m_mf_solution, sol_eb, 0, 0, 1,
                          std::min(sol_eb.nGrow(), m_mf_solution.nGrow()));
    m_mf_solution.FillBoundary(m_geom.periodicity());

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Real dst_min = m_mf_solution.min(0);
        amrex::Real dst_max = m_mf_solution.max(0);
        std::fprintf(stderr, "  [DEBUG] m_mf_solution min=%g max=%g\n", dst_min, dst_max);
        std::fflush(stderr);
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  MLMG solve: residual=" << std::scientific << res_norm
                       << std::defaultfloat << ", iterations=" << m_num_iterations
                       << ", converged=" << m_converged << std::endl;
    }

    if (m_write_plotfile && m_converged) {
        writeSolutionPlotfile("tortuosity_mlmg_" + std::to_string(idir));
    }

    // Pop the EB IndexSpace we pushed in step 2 — otherwise successive
    // TortuosityMLMG solves leak EB metadata on the global stack.
    amrex::EB2::IndexSpace::pop();

    return m_converged;
}

} // namespace OpenImpala
