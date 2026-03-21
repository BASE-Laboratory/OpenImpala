// --- TortuosityMLMG.cpp ---

#include "TortuosityMLMG.H"
#include "TortuosityKernels.H"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Loop.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <mpi.h>

namespace {
constexpr int MaskComp = 0;
constexpr amrex::Real tiny_flux_threshold = 1.e-15;
constexpr int cell_inactive = 0;
constexpr int cell_active = 1;
} // namespace

namespace OpenImpala {

// --- Constructor ---
TortuosityMLMG::TortuosityMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm,
                                const amrex::iMultiFab& mf_phase_input, const amrex::Real vf,
                                const int phase, const OpenImpala::Direction dir,
                                const std::string& resultspath, const amrex::Real vlo,
                                const amrex::Real vhi, int verbose, bool write_plotfile)
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase(ba, dm, mf_phase_input.nComp(), mf_phase_input.nGrow()), m_phase(phase), m_vf(vf),
      m_dir(dir), m_vlo(vlo), m_vhi(vhi), m_resultspath(resultspath),
      m_write_plotfile(write_plotfile), m_verbose(verbose),
      m_mf_solution(ba, dm, 1, 1), m_mf_active_mask(ba, dm, 1, 1),
      m_mf_diff_coeff(ba, dm, 1, 1), m_active_vf(0.0), m_first_call(true),
      m_value(std::numeric_limits<amrex::Real>::quiet_NaN()), m_flux_in(0.0), m_flux_out(0.0) {
    amrex::Copy(m_mf_phase, mf_phase_input, 0, 0, m_mf_phase.nComp(), m_mf_phase.nGrow());

    // Parse MLMG solver parameters
    amrex::ParmParse pp_mlmg("mlmg");
    pp_mlmg.query("eps", m_eps);
    pp_mlmg.query("maxiter", m_maxiter);
    pp_mlmg.query("max_coarsening_level", m_max_coarsening_level);
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("verbose", m_verbose);

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityMLMG: Initializing..." << std::endl;
        amrex::Print() << "  Original Total VF (Phase " << m_phase << "): " << m_vf << std::endl;
        amrex::Print() << "  MLMG Params: eps=" << m_eps << ", maxiter=" << m_maxiter
                       << ", max_coarsening=" << m_max_coarsening_level << std::endl;
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0,
                                     "Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Max iterations must be positive");

    // Precondition phase data (remove isolated cells)
    preconditionPhaseFab();

    // Build diffusion coefficient field: D=1 for target phase, D=0 otherwise
    m_mf_diff_coeff.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_diff_coeff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<amrex::Real> const dc_arr = m_mf_diff_coeff.array(mfi);
        amrex::Array4<const int> const phase_arr = m_mf_phase.const_array(mfi);
        const int target_phase = m_phase;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            dc_arr(i, j, k, 0) = (phase_arr(i, j, k, 0) == target_phase) ? 1.0 : 0.0;
        });
    }
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    // Generate activity mask via flood fill
    generateActivityMask(m_mf_phase, m_phase, m_dir);

    if (m_active_vf <= std::numeric_limits<amrex::Real>::epsilon()) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "WARNING: Active volume fraction is zero. Skipping solve."
                           << std::endl;
        }
        m_first_call = false;
        m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        return;
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityMLMG: Initialization complete." << std::endl;
    }
}


// --- preconditionPhaseFab ---
void TortuosityMLMG::preconditionPhaseFab() {
    BL_PROFILE("TortuosityMLMG::preconditionPhaseFab");
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1,
                              "Phase fab needs ghost cells for preconditionPhaseFab");

    const amrex::Box& domain_box = m_geom.Domain();
    int num_remspot_passes = 0;
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("remspot_passes", num_remspot_passes);

    if (num_remspot_passes <= 0) {
        return;
    }

    for (int pass = 0; pass < num_remspot_passes; ++pass) {
        m_mf_phase.FillBoundary(m_geom.periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tile_box = mfi.tilebox();
            amrex::IArrayBox& fab = m_mf_phase[mfi];
            int ncomp = fab.nComp();
            OpenImpala::removeIsolatedCells(fab.dataPtr(0), fab.loVect(), fab.hiVect(), ncomp,
                                            tile_box.loVect(), tile_box.hiVect(),
                                            domain_box.loVect(), domain_box.hiVect());
        }
    }
    m_mf_phase.FillBoundary(m_geom.periodicity());
}


// --- parallelFloodFill ---
void TortuosityMLMG::parallelFloodFill(amrex::iMultiFab& reachabilityMask,
                                        const amrex::iMultiFab& phaseFab, int phaseID,
                                        const amrex::Vector<amrex::IntVect>& seedPoints) {
    BL_PROFILE("TortuosityMLMG::parallelFloodFill");
    AMREX_ASSERT(reachabilityMask.nGrow() >= 1);
    AMREX_ASSERT(phaseFab.nGrow() >= 1);

    reachabilityMask.setVal(cell_inactive);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = reachabilityMask.array(mfi);
        const auto phase_arr = phaseFab.const_array(mfi, 0);
        for (const auto& seed : seedPoints) {
            if (tileBox.contains(seed)) {
                if (phase_arr(seed) == phaseID) {
                    mask_arr(seed, MaskComp) = cell_active;
                }
            }
        }
    }

    amrex::IntVect domain_size = m_geom.Domain().size();
    const int max_flood_iter = domain_size[0] + domain_size[1] + domain_size[2] + 2;
    bool changed_globally = true;
    const std::vector<amrex::IntVect> offsets = {amrex::IntVect{1, 0, 0}, amrex::IntVect{-1, 0, 0},
                                                 amrex::IntVect{0, 1, 0}, amrex::IntVect{0, -1, 0},
                                                 amrex::IntVect{0, 0, 1}, amrex::IntVect{0, 0, -1}};
    int iter = 0;
    while (changed_globally && iter < max_flood_iter) {
        ++iter;
        changed_globally = false;
        reachabilityMask.FillBoundary(m_geom.periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        {
            bool changed_locally = false;
            for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
                const amrex::Box& tileBox = mfi.tilebox();
                auto mask_arr = reachabilityMask.array(mfi);
                const auto phase_arr = phaseFab.const_array(mfi, 0);
                const amrex::Box& grownTileBox = amrex::grow(tileBox, reachabilityMask.nGrow());
                amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) {
                    amrex::IntVect current_cell(i, j, k);
                    if (mask_arr(current_cell, MaskComp) == cell_active ||
                        phase_arr(current_cell) != phaseID) {
                        return;
                    }
                    for (const auto& offset : offsets) {
                        amrex::IntVect neighbor_cell = current_cell + offset;
                        if (grownTileBox.contains(neighbor_cell)) {
                            if (mask_arr(neighbor_cell, MaskComp) == cell_active) {
                                mask_arr(current_cell, MaskComp) = cell_active;
                                changed_locally = true;
                                break;
                            }
                        }
                    }
                });
            }
#ifdef AMREX_USE_OMP
#pragma omp critical(flood_fill_crit)
#endif
            {
                if (changed_locally) {
                    changed_globally = true;
                }
            }
        }
        amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);
    }

    if (iter >= max_flood_iter && changed_globally) {
        amrex::Warning("TortuosityMLMG::parallelFloodFill reached max iterations.");
    }
    reachabilityMask.FillBoundary(m_geom.periodicity());
}


// --- generateActivityMask ---
void TortuosityMLMG::generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID,
                                           OpenImpala::Direction dir) {
    BL_PROFILE("TortuosityMLMG::generateActivityMask");

    const amrex::Box& domain = m_geom.Domain();
    const int idir = static_cast<int>(dir);

    amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
    amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);
    amrex::Vector<amrex::IntVect> local_inlet_seeds;
    amrex::Vector<amrex::IntVect> local_outlet_seeds;

    amrex::Box domain_lo_face = domain;
    domain_lo_face.setBig(idir, domain.smallEnd(idir));
    amrex::Box domain_hi_face = domain;
    domain_hi_face.setSmall(idir, domain.bigEnd(idir));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phaseFab); mfi.isValid(); ++mfi) {
        const amrex::Box& validBox = mfi.validbox();
        const auto phase_arr = phaseFab.const_array(mfi);
        amrex::Box inlet_intersect = validBox & domain_lo_face;
        if (!inlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(inlet_intersect, [&](int i, int j, int k) {
                if (phase_arr(i, j, k, 0) == phaseID) {
#ifdef AMREX_USE_OMP
#pragma omp critical(inlet_seed_crit)
#endif
                    local_inlet_seeds.push_back(amrex::IntVect(i, j, k));
                }
            });
        }
        amrex::Box outlet_intersect = validBox & domain_hi_face;
        if (!outlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(outlet_intersect, [&](int i, int j, int k) {
                if (phase_arr(i, j, k, 0) == phaseID) {
#ifdef AMREX_USE_OMP
#pragma omp critical(outlet_seed_crit)
#endif
                    local_outlet_seeds.push_back(amrex::IntVect(i, j, k));
                }
            });
        }
    }

    // MPI Allgather seeds
    MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
    int mpi_size = amrex::ParallelDescriptor::NProcs();

    auto gatherSeeds = [&](const amrex::Vector<amrex::IntVect>& local_seeds) {
        const size_t n_local = static_cast<size_t>(local_seeds.size());
        std::vector<int> flat_local(n_local * AMREX_SPACEDIM);
        for (size_t i = 0; i < n_local; ++i) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                flat_local[i * AMREX_SPACEDIM + d] = local_seeds[i][d];
        }
        int local_count = static_cast<int>(flat_local.size());
        std::vector<int> recv_counts(mpi_size);
        MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
        std::vector<int> displacements(mpi_size, 0);
        int total = recv_counts[0];
        for (int i = 1; i < mpi_size; ++i) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
            total += recv_counts[i];
        }
        std::vector<int> flat_gathered(total);
        MPI_Allgatherv(flat_local.data(), local_count, MPI_INT, flat_gathered.data(),
                        recv_counts.data(), displacements.data(), MPI_INT, comm);
        amrex::Vector<amrex::IntVect> seeds;
        seeds.reserve(total / AMREX_SPACEDIM);
        for (size_t i = 0; i < flat_gathered.size(); i += AMREX_SPACEDIM) {
            seeds.emplace_back(flat_gathered[i], flat_gathered[i + 1], flat_gathered[i + 2]);
        }
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
        return seeds;
    };

    auto inlet_seeds = gatherSeeds(local_inlet_seeds);
    auto outlet_seeds = gatherSeeds(local_outlet_seeds);

    if (inlet_seeds.empty() || outlet_seeds.empty()) {
        amrex::Warning("TortuosityMLMG::generateActivityMask: No percolating path found.");
        m_mf_active_mask.setVal(cell_inactive);
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
        m_active_vf = 0.0;
        return;
    }

    parallelFloodFill(mf_reached_inlet, phaseFab, phaseID, inlet_seeds);
    parallelFloodFill(mf_reached_outlet, phaseFab, phaseID, outlet_seeds);

    m_mf_active_mask.setVal(cell_inactive);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = m_mf_active_mask.array(mfi);
        const auto inlet_arr = mf_reached_inlet.const_array(mfi);
        const auto outlet_arr = mf_reached_outlet.const_array(mfi);
        amrex::ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            mask_arr(i, j, k, MaskComp) =
                (inlet_arr(i, j, k, 0) == cell_active && outlet_arr(i, j, k, 0) == cell_active)
                    ? cell_active
                    : cell_inactive;
        });
    }
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    // Count active cells via GPU-compatible reduction
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<long> reduce_data(reduce_op);
    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& mask_arr = m_mf_active_mask.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                -> amrex::GpuTuple<long> {
                return {(mask_arr(i, j, k, MaskComp) == 1) ? 1L : 0L};
            });
    }
    long num_active = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceLongSum(num_active);

    long total_cells = m_geom.Domain().numPts();
    m_active_vf = (total_cells > 0)
                      ? static_cast<amrex::Real>(num_active) / static_cast<amrex::Real>(total_cells)
                      : 0.0;

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Active Volume Fraction (percolating phase " << m_phase
                       << "): " << m_active_vf << std::endl;
    }
}


// --- solve ---
// Uses AMReX MLABecLaplacian + MLMG to solve div(B grad phi) = 0
// with Dirichlet BCs at inlet/outlet and Neumann on lateral faces.
bool TortuosityMLMG::solve() {
    BL_PROFILE("TortuosityMLMG::solve");

    const int idir = static_cast<int>(m_dir);
    const amrex::Real* dx = m_geom.CellSize();

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

    // Set Dirichlet values via ghost cells of the solution MultiFab.
    // For AMReX "external Dirichlet" BCs, the ghost cell value IS the
    // boundary value (not extrapolated).
    m_mf_solution.setVal(0.0);
    // Set initial guess: linear ramp in flow direction for better convergence
    {
        const amrex::Box& domain = m_geom.Domain();
        const int n_cells = domain.length(idir);
        const int dom_lo_dir = domain.smallEnd(idir);
        const int dom_hi_dir = domain.bigEnd(idir);
        const amrex::Real vlo = m_vlo;
        const amrex::Real vhi = m_vhi;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(m_mf_solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.growntilebox();
            amrex::Array4<amrex::Real> const phi = m_mf_solution.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::IntVect iv(i, j, k);
                int idx_in_dir = iv[idir] - dom_lo_dir;
                amrex::Real frac =
                    static_cast<amrex::Real>(idx_in_dir) / static_cast<amrex::Real>(n_cells - 1);
                if (iv[idir] >= dom_lo_dir && iv[idir] <= dom_hi_dir) {
                    // Interior cell: linear ramp
                    phi(i, j, k) = vlo + frac * (vhi - vlo);
                } else if (iv[idir] < dom_lo_dir) {
                    phi(i, j, k) = vlo; // Inlet ghost
                } else {
                    phi(i, j, k) = vhi; // Outlet ghost
                }
            });
        }
    }
    m_mf_solution.FillBoundary(m_geom.periodicity());

    // Set level BC (ghost cell values encode the Dirichlet data)
    mlabec.setLevelBC(0, &m_mf_solution);

    // Set coefficients: alpha*a - beta*div(B*grad)
    mlabec.setScalars(0.0, 1.0); // alpha=0, beta=1

    // A-coefficient (not used since alpha=0, but must be set)
    amrex::MultiFab acoef(m_ba, m_dm, 1, 0);
    acoef.setVal(0.0);
    mlabec.setACoeffs(0, acoef);

    // B-coefficients: face-centred diffusivities
    // For inactive cells, D=0 ensures they are decoupled.
    // MLABecLaplacian uses face-centred B-coefficients. We compute them
    // as the harmonic mean of adjacent cell-centred values, consistent
    // with the HYPRE solver. Inactive cells get D=0 which naturally
    // decouples them from the system.
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
            // For surroundingNodes(d), face (i,j,k) sits between
            // cell (i-e_d,j,k) and cell (i,j,k) in cell-centred indexing.
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

    // MLMG::solve returns the final absolute residual norm.  If it did not
    // throw, the solver reached the requested tolerance (m_eps relative,
    // 0.0 absolute).  Wrap in try/catch to handle non-convergence gracefully.
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

    m_mf_solution.FillBoundary(m_geom.periodicity());

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  MLMG solve: residual=" << std::scientific << res_norm
                       << std::defaultfloat << ", iterations=" << m_num_iterations
                       << ", converged=" << m_converged << std::endl;
    }

    // Write plotfile if requested
    if (m_write_plotfile && m_converged) {
        amrex::MultiFab mf_plot(m_ba, m_dm, 3, 0);
        // Convert integer fabs to Real for plotfile output
        amrex::MultiFab mf_mask_real(m_ba, m_dm, 1, 0);
        amrex::MultiFab mf_phase_real(m_ba, m_dm, 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mf_mask_real, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto const& mask_int = m_mf_active_mask.const_array(mfi);
            auto const& phase_int = m_mf_phase.const_array(mfi);
            auto const& mask_r = mf_mask_real.array(mfi);
            auto const& phase_r = mf_phase_real.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                mask_r(i, j, k) = static_cast<amrex::Real>(mask_int(i, j, k, MaskComp));
                phase_r(i, j, k) = static_cast<amrex::Real>(phase_int(i, j, k, 0));
            });
        }
        amrex::Copy(mf_plot, m_mf_solution, 0, 0, 1, 0);
        amrex::Copy(mf_plot, mf_phase_real, 0, 1, 1, 0);
        amrex::Copy(mf_plot, mf_mask_real, 0, 2, 1, 0);
        std::string plotfilename =
            m_resultspath + "/tortuosity_mlmg_" + std::to_string(idir);
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id", "active_mask"};
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);
    }

    return m_converged;
}


// --- globalFluxes ---
void TortuosityMLMG::globalFluxes() {
    BL_PROFILE("TortuosityMLMG::globalFluxes");
    m_flux_in = 0.0;
    m_flux_out = 0.0;

    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir);
    const amrex::Real dx_dir = dx[idir];
    amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);

    m_mf_solution.FillBoundary(m_geom.periodicity());
    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> flux_reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real> flux_reduce_data(flux_reduce_op);

    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& validBox = mfi.validbox();
        amrex::Array4<const int> const mask = m_mf_active_mask.const_array(mfi);
        amrex::Array4<const amrex::Real> const soln = m_mf_solution.const_array(mfi);
        amrex::Array4<const amrex::Real> const dc = m_mf_diff_coeff.const_array(mfi);
        const amrex::IntVect sh = shift;
        const amrex::Real dxd = dx_dir;

        amrex::Box lobox_face = amrex::bdryLo(domain, idir) & validBox;
        if (!lobox_face.isEmpty()) {
            flux_reduce_op.eval(lobox_face, flux_reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    -> amrex::GpuTuple<amrex::Real, amrex::Real> {
                    amrex::IntVect iv(i, j, k);
                    if (mask(iv) == cell_active) {
                        amrex::IntVect iv_inner = iv + sh;
                        if (mask(iv_inner) == cell_active) {
                            amrex::Real D_bnd = dc(iv);
                            amrex::Real D_inn = dc(iv_inner);
                            amrex::Real D_face =
                                (D_bnd + D_inn > 0.0) ? 2.0 * D_bnd * D_inn / (D_bnd + D_inn) : 0.0;
                            amrex::Real grad = (soln(iv_inner) - soln(iv)) / dxd;
                            return {-D_face * grad, 0.0};
                        }
                    }
                    return {0.0, 0.0};
                });
        }

        amrex::Box domain_hi_face = domain;
        domain_hi_face.setSmall(idir, domain.bigEnd(idir));
        domain_hi_face.setBig(idir, domain.bigEnd(idir));
        amrex::Box hibox_face = domain_hi_face & validBox;
        if (!hibox_face.isEmpty()) {
            flux_reduce_op.eval(hibox_face, flux_reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    -> amrex::GpuTuple<amrex::Real, amrex::Real> {
                    amrex::IntVect iv(i, j, k);
                    if (mask(iv) == cell_active) {
                        amrex::IntVect iv_inner = iv - sh;
                        if (mask(iv_inner) == cell_active) {
                            amrex::Real D_bnd = dc(iv);
                            amrex::Real D_inn = dc(iv_inner);
                            amrex::Real D_face =
                                (D_bnd + D_inn > 0.0) ? 2.0 * D_bnd * D_inn / (D_bnd + D_inn) : 0.0;
                            amrex::Real grad = (soln(iv) - soln(iv_inner)) / dxd;
                            return {0.0, -D_face * grad};
                        }
                    }
                    return {0.0, 0.0};
                });
        }
    }
    auto flux_result = flux_reduce_data.value();
    amrex::Real local_fxin = amrex::get<0>(flux_result);
    amrex::Real local_fxout = amrex::get<1>(flux_result);

    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    amrex::Real face_area_element = 1.0;
    if (AMREX_SPACEDIM == 3) {
        if (idir == 0)
            face_area_element = dx[1] * dx[2];
        else if (idir == 1)
            face_area_element = dx[0] * dx[2];
        else
            face_area_element = dx[0] * dx[1];
    } else if (AMREX_SPACEDIM == 2) {
        face_area_element = (idir == 0) ? dx[1] : dx[0];
    }
    m_flux_in = local_fxin * face_area_element;
    m_flux_out = local_fxout * face_area_element;

    computePlaneFluxes(m_mf_solution);
}


// --- computePlaneFluxes ---
void TortuosityMLMG::computePlaneFluxes(const amrex::MultiFab& mf_soln) {
    BL_PROFILE("TortuosityMLMG::computePlaneFluxes");

    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir);
    const int n_cells = domain.length(idir);
    const int n_faces = n_cells - 1;
    const amrex::Real dx_dir = dx[idir];
    amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);

    amrex::Real face_area_element = 1.0;
    if (AMREX_SPACEDIM == 3) {
        if (idir == 0)
            face_area_element = dx[1] * dx[2];
        else if (idir == 1)
            face_area_element = dx[0] * dx[2];
        else
            face_area_element = dx[0] * dx[1];
    } else if (AMREX_SPACEDIM == 2) {
        face_area_element = (idir == 0) ? dx[1] : dx[0];
    }

    // Plane flux accumulation indexes into a per-plane histogram; kept on CPU.
    // Ensure device data is synchronized before host-side access.
    amrex::Gpu::streamSynchronize();
    std::vector<amrex::Real> local_plane_flux(n_faces, 0.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        std::vector<amrex::Real> priv_plane_flux(n_faces, 0.0);

        for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tileBox = mfi.tilebox();
            amrex::Array4<const int> const mask = m_mf_active_mask.const_array(mfi);
            amrex::Array4<const amrex::Real> const soln = mf_soln.const_array(mfi);
            amrex::Array4<const amrex::Real> const dc = m_mf_diff_coeff.const_array(mfi);

            amrex::Box flux_box = tileBox;
            flux_box.setBig(idir, std::min(tileBox.bigEnd(idir), domain.bigEnd(idir) - 1));

            amrex::LoopOnCpu(flux_box, [&](int i, int j, int k) {
                amrex::IntVect iv(i, j, k);
                amrex::IntVect iv_plus = iv + shift;
                if (mask(iv) == cell_active && mask(iv_plus) == cell_active) {
                    amrex::Real D_lo = dc(iv);
                    amrex::Real D_hi = dc(iv_plus);
                    amrex::Real D_face =
                        (D_lo + D_hi > 0.0) ? 2.0 * D_lo * D_hi / (D_lo + D_hi) : 0.0;
                    amrex::Real grad = (soln(iv_plus) - soln(iv)) / dx_dir;
                    amrex::Real flux = -D_face * grad * face_area_element;
                    int face_idx = iv[idir] - domain.smallEnd(idir);
                    priv_plane_flux[face_idx] += flux;
                }
            });
        }
#ifdef AMREX_USE_OMP
#pragma omp critical
#endif
        {
            for (int f = 0; f < n_faces; ++f) {
                local_plane_flux[f] += priv_plane_flux[f];
            }
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum(local_plane_flux.data(), n_faces);
    m_plane_fluxes = std::move(local_plane_flux);

    amrex::Real sum_flux = 0.0;
    for (int f = 0; f < n_faces; ++f)
        sum_flux += m_plane_fluxes[f];
    amrex::Real mean_flux = sum_flux / n_faces;

    amrex::Real max_abs_dev = 0.0;
    for (int f = 0; f < n_faces; ++f) {
        amrex::Real dev = std::abs(m_plane_fluxes[f] - mean_flux);
        if (dev > max_abs_dev)
            max_abs_dev = dev;
    }

    amrex::Real abs_mean = std::abs(mean_flux);
    m_plane_flux_max_dev = (abs_mean > tiny_flux_threshold) ? max_abs_dev / abs_mean : 0.0;

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Interior Plane Flux Check (" << n_faces << " faces):\n";
        amrex::Print() << "    Mean Plane Flux = " << std::scientific << mean_flux << "\n";
        amrex::Print() << "    Max |F_i - mean| / |mean| = " << std::scientific
                       << m_plane_flux_max_dev << std::defaultfloat << "\n";
    }
}


// --- value ---
amrex::Real TortuosityMLMG::value(const bool refresh) {
    if (m_active_vf <= std::numeric_limits<amrex::Real>::epsilon() && !m_first_call) {
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }

    if (m_first_call || refresh) {
        if (m_active_vf <= std::numeric_limits<amrex::Real>::epsilon()) {
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            m_first_call = false;
            return m_value;
        }

        bool solve_converged = solve();

        if (!solve_converged) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "WARNING: MLMG solver did not converge. Returning NaN."
                               << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            m_first_call = false;
            return m_value;
        }

        globalFluxes();

        // Flux conservation check
        constexpr amrex::Real flux_tol = 1.0e-6;
        bool flux_conserved = true;
        amrex::Real flux_mag_in = std::abs(m_flux_in);
        amrex::Real flux_mag_out = std::abs(m_flux_out);
        amrex::Real flux_mag_avg = 0.5 * (flux_mag_in + flux_mag_out);
        if (flux_mag_avg > tiny_flux_threshold) {
            amrex::Real rel_diff = std::abs(flux_mag_in - flux_mag_out) / flux_mag_avg;
            if (rel_diff > flux_tol)
                flux_conserved = false;
        }

        // Interior plane check
        if (flux_conserved && !m_plane_fluxes.empty()) {
            if (m_plane_flux_max_dev > flux_tol)
                flux_conserved = false;
        }

        if (!flux_conserved) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "WARNING: Flux not conserved. Returning NaN." << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else {
            amrex::Real vf_for_calc = m_active_vf;
            amrex::Real L = m_geom.ProbLength(static_cast<int>(m_dir));
            amrex::Real A = 1.0;
            if (AMREX_SPACEDIM == 3) {
                if (m_dir == Direction::X)
                    A = m_geom.ProbLength(1) * m_geom.ProbLength(2);
                else if (m_dir == Direction::Y)
                    A = m_geom.ProbLength(0) * m_geom.ProbLength(2);
                else
                    A = m_geom.ProbLength(0) * m_geom.ProbLength(1);
            } else if (AMREX_SPACEDIM == 2) {
                A = (m_dir == Direction::X) ? m_geom.ProbLength(1) : m_geom.ProbLength(0);
            }
            amrex::Real gradPhi = (m_vhi - m_vlo) / L;

            // Use mean of interior plane fluxes when available
            amrex::Real avg_flux_mag;
            if (!m_plane_fluxes.empty()) {
                amrex::Real sum_plane = 0.0;
                for (const auto& pf : m_plane_fluxes)
                    sum_plane += std::abs(pf);
                avg_flux_mag = sum_plane / static_cast<amrex::Real>(m_plane_fluxes.size());
            } else {
                avg_flux_mag = 0.5 * (std::abs(m_flux_in) + std::abs(m_flux_out));
            }

            if (avg_flux_mag < tiny_flux_threshold) {
                m_value = (vf_for_calc > std::numeric_limits<amrex::Real>::epsilon())
                              ? std::numeric_limits<amrex::Real>::infinity()
                              : std::numeric_limits<amrex::Real>::quiet_NaN();
            } else if (vf_for_calc <= std::numeric_limits<amrex::Real>::epsilon()) {
                m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            } else if (std::abs(gradPhi) < tiny_flux_threshold) {
                m_value = std::numeric_limits<amrex::Real>::infinity();
            } else {
                amrex::Real Deff = (avg_flux_mag / A) / std::abs(gradPhi);
                if (std::abs(Deff) < tiny_flux_threshold) {
                    m_value = std::numeric_limits<amrex::Real>::infinity();
                } else {
                    m_value = vf_for_calc / Deff;
                }
            }

            if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Tortuosity (MLMG): " << m_value << std::endl;
            }
        }
    }

    m_first_call = false;
    return m_value;
}

} // namespace OpenImpala
