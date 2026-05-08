// --- TortuosityHypre.cpp ---

#include "TortuosityHypre.H"
#include "FloodFill.H"
#include "HypreCheck.H"
#include "TortuosityKernels.H"   // For removeIsolatedCells (replaces tortuosity_remspot)
#include "TortuosityHypreFill.H" // For tortuosityFillMatrix (replaces tortuosity_fillmtx)

// Includes remain the same...
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <vector>
#include <string>
#include <cmath>     // Required for std::isnan, std::isinf, std::abs
#include <limits>    // Required for std::numeric_limits
#include <stdexcept> // For potential error throwing (optional)
#include <iomanip>   // For std::setprecision
#include <iostream>  // For std::cout, std::flush
#include <set>       // For std::set (currently unused, kept for history)
#include <algorithm> // For std::sort, std::unique
#include <numeric>   // For std::accumulate, iota (potentially useful)
#include <sstream>   // For std::stringstream

#include <AMReX_Loop.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H> // Needed for amrex::Copy, amrex::sum
#include <AMReX_PlotFileUtil.H>
#ifdef OPENIMPALA_USE_GPU
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuContainers.H>
#endif
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H> // Needed for AMREX_GPU_DEVICE if used elsewhere
#include <AMReX_Box.H>           // Needed for Box operations
#include <AMReX_IntVect.H>       // Needed for amrex::IntVect
#include <AMReX_IndexType.H>     // For cell/node types if needed later
#include <AMReX_VisMF.H>         // For potential debug writes
#include <AMReX_Loop.H>          // For amrex::LoopOnCpu
#include <AMReX_Reduce.H>        // For GPU-compatible reductions
#include <AMReX_GpuLaunch.H>     // For amrex::ParallelFor

// HYPRE includes remain the same...
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

// MPI include remains the same...
#include <mpi.h>


// Constants namespace remains the same...
namespace {
constexpr int SolnComp = 0;
constexpr int MaskComp = 0;
constexpr int numComponentsPhi = 3;
constexpr amrex::Real tiny_flux_threshold = 1.e-15;
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
} // namespace

// Helper Functions and Class Implementation
namespace OpenImpala {

// loV/hiV are now inline delegates to HypreStructSolver::loV/hiV (defined in header).


// --- Constructor ---
OpenImpala::TortuosityHypre::TortuosityHypre(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                             const amrex::DistributionMapping& dm,
                                             const amrex::iMultiFab& mf_phase_input,
                                             const amrex::Real vf, // Original total VF
                                             const int phase, const OpenImpala::Direction dir,
                                             const SolverType st, const std::string& resultspath,
                                             const amrex::Real vlo, const amrex::Real vhi,
                                             int verbose, bool write_plotfile,
                                             const PrecondType precond_type)
    : HypreStructSolver(geom, ba, dm, st, 1e-9, 200, verbose),
      m_mf_phase(ba, dm, mf_phase_input.nComp(), mf_phase_input.nGrow()), m_phase(phase), m_vf(vf),
      m_dir(dir), m_vlo(vlo), m_vhi(vhi), m_resultspath(resultspath),
      m_write_plotfile(write_plotfile), m_precond_type(precond_type),
      m_mf_phi(ba, dm, numComponentsPhi, 1), m_mf_active_mask(ba, dm, 1, 1),
      m_mf_diff_coeff(ba, dm, 1, 1), m_active_vf(0.0), m_first_call(true),
      m_value(std::numeric_limits<amrex::Real>::quiet_NaN()), m_flux_in(0.0), m_flux_out(0.0) {
    // Ensure HYPRE is initialised exactly once (thread-safe via std::call_once).
    // C++ tests call HYPRE_Init() in main(), but Python bindings have no main().
    static std::once_flag hypre_once;
    std::call_once(hypre_once, []() { HYPRE_Init(); });

    // Copy data from input iMultiFab to member iMultiFab
    amrex::Copy(m_mf_phase, mf_phase_input, 0, 0, m_mf_phase.nComp(), m_mf_phase.nGrow());

    // --- Rest of constructor ---
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initializing..." << std::endl;
        // Optional: Print the original total VF passed in
        amrex::Print() << "  Original Total VF (Phase " << m_phase << "): " << m_vf << std::endl;
    }

    // Set hardcoded default solver parameters.
    m_eps = 1e-9;
    m_maxiter = 200;

    // Allow advanced users to override these defaults from the inputs file
    // by providing a [hypre] block, e.g., hypre.eps = 1e-12
    amrex::ParmParse pp("hypre");
    pp.query("eps", m_eps);
    pp.query("maxiter", m_maxiter);
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("verbose", m_verbose);

    // --- Boundary condition parsing ---
    {
        amrex::ParmParse pp_bc("bc");
        std::string bc_inlet_outlet_str;
        std::string bc_sides_str;
        if (pp_bc.query("inlet_outlet", bc_inlet_outlet_str)) {
            m_bc_inlet_outlet_type = parseBCType(bc_inlet_outlet_str);
        }
        if (pp_bc.query("sides", bc_sides_str)) {
            m_bc_sides_type = parseBCType(bc_sides_str);
        }
        // Allow bc.value_lo / bc.value_hi to override constructor defaults
        pp_bc.query("value_lo", m_vlo);
        pp_bc.query("value_hi", m_vhi);

        m_bc_inlet_outlet = createBC(m_bc_inlet_outlet_type);
        m_bc_sides = createBC(m_bc_sides_type);

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            auto bcTypeStr = [](BCType t) -> std::string {
                switch (t) {
                case BCType::DirichletExternal:
                    return "DirichletExternal";
                case BCType::DirichletPhaseBoundary:
                    return "DirichletPhaseBoundary";
                case BCType::Neumann:
                    return "Neumann";
                case BCType::Periodic:
                    return "Periodic";
                default:
                    return "Unknown";
                }
            };
            amrex::Print() << "  BC Config: inlet_outlet=" << bcTypeStr(m_bc_inlet_outlet_type)
                           << ", sides=" << bcTypeStr(m_bc_sides_type) << std::endl;
            amrex::Print() << "  BC Values: vlo=" << m_vlo << ", vhi=" << m_vhi << std::endl;
        }
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Params: eps=" << m_eps << ", maxiter=" << m_maxiter << std::endl;
        amrex::Print() << "  Class Verbose Level: " << m_verbose << std::endl;
        amrex::Print() << "  Write Plotfile Flag: " << m_write_plotfile << std::endl;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0,
                                     "Original Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");

    // --- Multi-phase transport coefficient parsing ---
    {
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
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(phase_diffs_vec[idx] >= 0.0,
                                                 "Phase diffusivities must be non-negative");
                m_phase_coeff_map[active_phases_vec[idx]] = phase_diffs_vec[idx];
            }
            m_is_multi_phase = true;
            if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Multi-phase mode enabled with " << m_phase_coeff_map.size()
                               << " phases:" << std::endl;
                for (const auto& kv : m_phase_coeff_map) {
                    amrex::Print()
                        << "    Phase " << kv.first << " -> D = " << kv.second << std::endl;
                }
            }
        } else {
            // Default single-phase behaviour: target phase has D=1, everything else D=0
            m_phase_coeff_map[m_phase] = 1.0;
            m_is_multi_phase = false;
            if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Single-phase mode: Phase " << m_phase << " -> D = 1.0"
                               << std::endl;
            }
        }
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "TortuosityHypre: Running preconditionPhaseFab (remspot)..." << std::endl;
    preconditionPhaseFab();

    // --- Build coefficient MultiFab from phase data and coefficient map ---
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "TortuosityHypre: Building diffusion coefficient field..." << std::endl;
    m_mf_diff_coeff.setVal(0.0);
    initializeDiffCoeff();
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    // --- For multi-phase: create binary traversable mask for flood fill ---
    // In single-phase mode, flood through the target phase only (original behavior).
    // In multi-phase mode, flood through ALL phases with D > 0.
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "TortuosityHypre: Generating activity mask via boundary search..."
                       << std::endl;

    if (m_is_multi_phase) {
        amrex::iMultiFab mf_binary_traversable = buildTraversableMask();
        // Flood through all traversable cells (phase ID = 1 in binary fab)
        generateActivityMask(mf_binary_traversable, 1, m_dir);
    } else {
        generateActivityMask(m_mf_phase, m_phase, m_dir);
    }

    // Check if active VF is zero after generation
    if (m_active_vf <= std::numeric_limits<amrex::Real>::epsilon()) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
                << "WARNING: Active volume fraction is zero. Skipping matrix setup and solve."
                << std::endl;
        }
        // No need to setup HYPRE if there's no active phase
        m_first_call = false; // Mark as "calculated" (result is NaN or Inf)
        m_value = std::numeric_limits<amrex::Real>::quiet_NaN(); // Or Inf if you prefer for zero VF
        return;                                                  // Exit constructor early
    }

    // Setup HYPRE structures only if there's an active phase
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "TortuosityHypre: Running setupGrid..." << std::endl;
    bool needs_periodic = (m_bc_sides && m_bc_sides->needsPeriodicGrid());
    setupGrid(needs_periodic);
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    HypreStructSolver::setupStencil();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation();

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
    }
}

// Destructor is defaulted — HypreStructSolver base class handles HYPRE cleanup.

void TortuosityHypre::initializeDiffCoeff() {
    // Flatten phase coefficient map to a device-accessible lookup table
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
        amrex::Array4<const int> const phase_arr = m_mf_phase.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int pid = phase_arr(i, j, k, 0);
            dc_arr(i, j, k, 0) = (pid >= 0 && pid < lut_size) ? lut_ptr[pid] : 0.0;
        });
    }
}

amrex::iMultiFab TortuosityHypre::buildTraversableMask() {
    // Create a temporary binary phase fab: 1 where D > 0, 0 otherwise
    amrex::iMultiFab mf_binary_traversable(m_ba, m_dm, 1, m_mf_phase.nGrow());
    mf_binary_traversable.setVal(0);
    for (amrex::MFIter mfi(mf_binary_traversable, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox();
        amrex::Array4<int> const trav_arr = mf_binary_traversable.array(mfi);
        amrex::Array4<const amrex::Real> const dc_arr = m_mf_diff_coeff.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            trav_arr(i, j, k, 0) = (dc_arr(i, j, k, 0) > 0.0) ? 1 : 0;
        });
    }
    mf_binary_traversable.FillBoundary(m_geom.periodicity());
    return mf_binary_traversable;
}

// setupGrid() and setupStencil() are now provided by HypreStructSolver base class.

// --- preconditionPhaseFab ---
// Remains the same...
void OpenImpala::TortuosityHypre::preconditionPhaseFab() {
    BL_PROFILE("TortuosityHypre::preconditionPhaseFab");
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1,
                              "Phase fab needs ghost cells for preconditionPhaseFab");

    const amrex::Box& domain_box = m_geom.Domain();
    int num_remspot_passes = 0; // Default to 0 based on input file
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("remspot_passes", num_remspot_passes); // Allow override

    if (num_remspot_passes <= 0) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Skipping tortuosity_remspot filter (remspot_passes <= 0)."
                           << std::endl;
        }
        return;
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Applying tortuosity_remspot filter (" << num_remspot_passes
                       << " passes)..." << std::endl;
    }

    for (int pass = 0; pass < num_remspot_passes; ++pass) {
        m_mf_phase.FillBoundary(m_geom.periodicity());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tile_box = mfi.tilebox();
            amrex::IArrayBox& fab = m_mf_phase[mfi];
            const int ncomp = fab.nComp();
            // removeIsolatedCells is a host kernel that dereferences the
            // raw pointer it receives. fab.dataPtr() returns DEVICE memory
            // on CUDA builds; copy the fab to a host buffer, run the
            // kernel there, then push the modified data back.
#ifdef AMREX_USE_GPU
            const size_t total = static_cast<size_t>(fab.box().numPts()) * ncomp;
            std::vector<int> host_buf(total);
            amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, fab.dataPtr(0), fab.dataPtr(0) + total,
                                  host_buf.data());
            amrex::Gpu::streamSynchronize();
            OpenImpala::removeIsolatedCells(host_buf.data(), fab.loVect(), fab.hiVect(), ncomp,
                                            tile_box.loVect(), tile_box.hiVect(),
                                            domain_box.loVect(), domain_box.hiVect());
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host_buf.data(),
                                  host_buf.data() + total, fab.dataPtr(0));
            amrex::Gpu::streamSynchronize();
#else
            OpenImpala::removeIsolatedCells(fab.dataPtr(0), fab.loVect(), fab.hiVect(), ncomp,
                                            tile_box.loVect(), tile_box.hiVect(),
                                            domain_box.loVect(), domain_box.hiVect());
#endif
        }
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "    DEBUG [preconditionPhaseFab]: Finished remspot pass " << pass + 1
                           << std::endl;
        }
    }
    m_mf_phase.FillBoundary(m_geom.periodicity());
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  ...remspot filtering complete." << std::endl;
    }
}


// --- Generate Activity Mask ---
void OpenImpala::TortuosityHypre::generateActivityMask(const amrex::iMultiFab& phaseFab,
                                                       int phaseID, OpenImpala::Direction dir) {
    BL_PROFILE("TortuosityHypre::generateActivityMask");
    AMREX_ASSERT(phaseFab.nGrow() >= 1);
    AMREX_ASSERT(phaseFab.nComp() > 0);

    const int idir = static_cast<int>(dir);

    // Collect boundary seeds (shared utility handles MPI gather + dedup)
    amrex::Vector<amrex::IntVect> inlet_seeds;
    amrex::Vector<amrex::IntVect> outlet_seeds;
    OpenImpala::collectBoundarySeeds(phaseFab, phaseID, idir, m_geom, inlet_seeds, outlet_seeds);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "    generateActivityMask: Found " << inlet_seeds.size()
                       << " unique inlet seeds." << std::endl;
        amrex::Print() << "    generateActivityMask: Found " << outlet_seeds.size()
                       << " unique outlet seeds." << std::endl;
    }

    if (inlet_seeds.empty() || outlet_seeds.empty()) {
        amrex::Warning(
            "TortuosityHypre::generateActivityMask: No percolating path found (zero seeds on inlet "
            "or outlet face for the specified phase). Mask will be empty.");
        m_mf_active_mask.setVal(cell_inactive);
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
        m_active_vf = 0.0;
        return;
    }

    // GPU-compatible flood fill from inlet and outlet (shared utility)
    amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
    amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);
    mf_reached_inlet.setVal(OpenImpala::FLOOD_INACTIVE);
    mf_reached_outlet.setVal(OpenImpala::FLOOD_INACTIVE);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "  Performing flood fill from inlet..." << std::endl;
    OpenImpala::parallelFloodFill(mf_reached_inlet, phaseFab, phaseID, inlet_seeds, m_geom,
                                  m_verbose);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor())
        amrex::Print() << "  Performing flood fill from outlet..." << std::endl;
    OpenImpala::parallelFloodFill(mf_reached_outlet, phaseFab, phaseID, outlet_seeds, m_geom,
                                  m_verbose);

    m_mf_active_mask.setVal(cell_inactive);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = m_mf_active_mask.array(mfi);
        const auto inlet_reach_arr = mf_reached_inlet.const_array(mfi);
        const auto outlet_reach_arr = mf_reached_outlet.const_array(mfi);
        amrex::ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            mask_arr(i, j, k, MaskComp) = (inlet_reach_arr(i, j, k, 0) == FLOOD_ACTIVE &&
                                           outlet_reach_arr(i, j, k, 0) == FLOOD_ACTIVE)
                                              ? cell_active
                                              : cell_inactive;
        });
    }
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    // Debug plotfile writing remains the same...
    bool write_debug_mask = false;
    amrex::ParmParse pp_debug("debug");
    pp_debug.query("write_active_mask", write_debug_mask);
    if (write_debug_mask) { /* ... plotfile code ... */
    }

    // Count active cells via GPU-compatible reduction (avoids ghost cell inclusion)
    amrex::ReduceOps<amrex::ReduceOpSum> count_reduce_op;
    amrex::ReduceData<long> count_reduce_data(count_reduce_op);
    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& mask_arr = m_mf_active_mask.const_array(mfi);
        count_reduce_op.eval(
            bx, count_reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<long> {
                return {(mask_arr(i, j, k, MaskComp) == 1) ? 1L : 0L};
            });
    }
    long num_active = amrex::get<0>(count_reduce_data.value());
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

// --- setupMatrixEquation ---
// Remains the same...
void OpenImpala::TortuosityHypre::setupMatrixEquation() {
    BL_PROFILE("TortuosityHypre::setupMatrixEquation");
    HYPRE_Int ierr = 0;

    // Create HYPRE matrix and vectors via base class
    createMatrixAndVectors();

    const amrex::Box& domain = m_geom.Domain();
    int stencil_indices[stencil_size] = {istn_c,  istn_mx, istn_px, istn_my,
                                         istn_py, istn_mz, istn_pz};
    const int dir_int = static_cast<int>(m_dir);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize();
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        dxinv_sq[i] = (dx[i] > 0.0) ? (1.0 / dx[i]) * (1.0 / dx[i]) : 0.0;
    }

    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    m_mf_phase.FillBoundary(m_geom.periodicity());
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEq: Calling tortuosityFillMatrix C++ kernel (using mask + "
                          "diff_coeff)..."
                       << std::endl;
    }

    std::vector<amrex::Real> matrix_values;
    std::vector<amrex::Real> rhs_values;
    std::vector<amrex::Real> initial_guess;

#ifdef OPENIMPALA_USE_GPU
    // GPU path: keep matrix data device-resident end-to-end. HYPRE 2.31's
    // structured solvers crash on device data when the memory location is
    // HOST but execution policy is DEVICE — see HypreStructSolver::
    // createMatrixAndVectors for the matching HYPRE_SetMemoryLocation
    // call. We:
    //   1. Allocate device buffers for matrix / RHS / initial guess.
    //   2. Run tortuosityFillMatrixGpu kernel to populate the interior.
    //   3. Bounce briefly to host to apply BCs (BC code is host-only) —
    //      copy device→host, snapshot mask/dc to host, run applyBC, copy
    //      results host→device.
    //   4. Pass the *device* pointer to HYPRE_StructMatrixSetBoxValues etc.
    //      With HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE) set, HYPRE
    //      stores and operates on the data entirely on the GPU.
    for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0)
            continue;

        const size_t mtx_size = static_cast<size_t>(npts) * stencil_size;
        amrex::Gpu::DeviceVector<amrex::Real> d_matrix(mtx_size);
        amrex::Gpu::DeviceVector<amrex::Real> d_rhs(npts);
        amrex::Gpu::DeviceVector<amrex::Real> d_xinit(npts);

        const auto mask_arr = m_mf_active_mask.const_array(mfi, MaskComp);
        const auto dc_arr = m_mf_diff_coeff.const_array(mfi, 0);

        // Step 1+2: GPU kernel fills the interior stencil.
        OpenImpala::tortuosityFillMatrixGpu(bx, d_matrix.data(), d_rhs.data(), d_xinit.data(),
                                            mask_arr, dc_arr, domain.loVect(), domain.hiVect(),
                                            dxinv_sq.data(), m_vlo, m_vhi, dir_int);
        amrex::Gpu::streamSynchronize();

        // Step 3: bounce to host for BC application.
        matrix_values.resize(mtx_size);
        rhs_values.resize(npts);
        initial_guess.resize(npts);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_matrix.begin(), d_matrix.end(),
                         matrix_values.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_rhs.begin(), d_rhs.end(), rhs_values.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_xinit.begin(), d_xinit.end(),
                         initial_guess.begin());

        // mask_iab/dc_fab dataPtr() returns device pointers; snapshot to
        // host vectors so applyBC can dereference them safely.
        const amrex::IArrayBox& mask_iab = m_mf_active_mask[mfi];
        const amrex::FArrayBox& dc_fab = m_mf_diff_coeff[mfi];
        const auto& mask_box = mask_iab.box();
        const auto& dc_box = dc_fab.box();
        const size_t mask_comp_size = mask_box.numPts();
        const size_t dc_comp_size = dc_box.numPts();
        std::vector<int> mask_host(mask_comp_size);
        std::vector<amrex::Real> dc_host(dc_comp_size);
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, mask_iab.dataPtr(MaskComp),
                              mask_iab.dataPtr(MaskComp) + mask_comp_size, mask_host.data());
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, dc_fab.dataPtr(0),
                              dc_fab.dataPtr(0) + dc_comp_size, dc_host.data());
        amrex::Gpu::streamSynchronize();
        const int* mask_ptr = mask_host.data();
        const amrex::Real* dc_ptr = dc_host.data();

        if (m_bc_inlet_outlet != nullptr) {
            m_bc_inlet_outlet->applyBC(matrix_values.data(), rhs_values.data(),
                                       initial_guess.data(), npts, mask_ptr, mask_box.loVect(),
                                       mask_box.hiVect(), dc_ptr, dc_box.loVect(), dc_box.hiVect(),
                                       bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                                       dxinv_sq.data(), m_vlo, m_vhi, dir_int);
        }
        if (m_bc_sides != nullptr) {
            m_bc_sides->applyBC(matrix_values.data(), rhs_values.data(), initial_guess.data(), npts,
                                mask_ptr, mask_box.loVect(), mask_box.hiVect(), dc_ptr,
                                dc_box.loVect(), dc_box.hiVect(), bx.loVect(), bx.hiVect(),
                                domain.loVect(), domain.hiVect(), dxinv_sq.data(), m_vlo, m_vhi,
                                dir_int);
        }

        // Step 4a: push BC-applied values back to device buffers.
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, matrix_values.begin(), matrix_values.end(),
                         d_matrix.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, rhs_values.begin(), rhs_values.end(),
                         d_rhs.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, initial_guess.begin(), initial_guess.end(),
                         d_xinit.begin());
        amrex::Gpu::streamSynchronize();

        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // Step 4b: hand DEVICE pointers to HYPRE.
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size,
                                              stencil_indices, d_matrix.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), d_rhs.data());
        HYPRE_CHECK(ierr);
        ierr =
            HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), d_xinit.data());
        HYPRE_CHECK(ierr);
    }
#else
    // CPU path: no OMP parallelism here because HYPRE_StructMatrixSetBoxValues
    // and HYPRE_StructVectorSetBoxValues are not thread-safe for the same object.
    for (amrex::MFIter mfi(m_mf_phase, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0)
            continue;

        matrix_values.resize(static_cast<size_t>(npts) * stencil_size);
        rhs_values.resize(npts);
        initial_guess.resize(npts);

        const amrex::IArrayBox& mask_iab = m_mf_active_mask[mfi];
        const int* mask_ptr = mask_iab.dataPtr(MaskComp);
        const auto& mask_box = mask_iab.box();

        const amrex::FArrayBox& dc_fab = m_mf_diff_coeff[mfi];
        const amrex::Real* dc_ptr = dc_fab.dataPtr(0);
        const auto& dc_box = dc_fab.box();

        OpenImpala::tortuosityFillMatrix(
            matrix_values.data(), rhs_values.data(), initial_guess.data(), npts, mask_ptr,
            mask_box.loVect(), mask_box.hiVect(), dc_ptr, dc_box.loVect(), dc_box.hiVect(),
            bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(), dxinv_sq.data(), m_vlo,
            m_vhi, dir_int, m_verbose, m_bc_inlet_outlet.get(), m_bc_sides.get());

        // NaN/Inf check
        bool data_ok = true;
        for (size_t i = 0; i < matrix_values.size(); ++i) {
            if (std::isnan(matrix_values[i]) || std::isinf(matrix_values[i]))
                data_ok = false;
        }
        for (size_t i = 0; i < rhs_values.size(); ++i) {
            if (std::isnan(rhs_values[i]) || std::isinf(rhs_values[i]))
                data_ok = false;
        }
        if (!data_ok) {
            amrex::Warning("NaN/Inf detected in kernel output before HYPRE SetBoxValues!");
        }

        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size,
                                              stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(),
                                              rhs_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(),
                                              initial_guess.data());
        HYPRE_CHECK(ierr);
    }
#endif
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEq: Finished MFIter loop." << std::endl;
    }

    // Assemble via base class
    assembleSystem();
}


// --- solve ---
bool OpenImpala::TortuosityHypre::solve() {
    BL_PROFILE("TortuosityHypre::solve");

    // Delegate solver dispatch to base class. The preconditioner choice only affects
    // Krylov solvers (PCG/GMRES/FlexGMRES/BiCGSTAB); standalone SMG/PFMG/Jacobi ignore it.
    runSolver(m_precond_type);
    // Plotfile writing remains the same...
    if (m_write_plotfile && m_converged) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Writing solution plotfile..." << std::endl;
        }
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponentsPhi, 0);
        amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 0);
        mf_soln_temp.setVal(0.0);
        // HYPRE_MEMORY_DEVICE on CUDA → GetBoxValues writes to a device pointer.
        // No OMP: HYPRE_StructVectorGetBoxValues is not thread-safe for the same vector.
        std::vector<HYPRE_Real> soln_buffer_host;
        for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            const int npts = static_cast<int>(bx.numPts());
            if (npts == 0)
                continue;
            auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
            auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
#ifdef AMREX_USE_GPU
            amrex::Gpu::DeviceVector<HYPRE_Real> d_soln_buffer(npts);
            HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(
                m_x, hypre_lo.data(), hypre_hi.data(), d_soln_buffer.data());
            const HYPRE_Real* src_ptr = d_soln_buffer.data();
#else
            soln_buffer_host.resize(npts);
            HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(
                m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer_host.data());
            const HYPRE_Real* src_ptr = soln_buffer_host.data();
#endif
            if (get_ierr != 0) {
                amrex::Warning("HYPRE_StructVectorGetBoxValues failed during plotfile writing!");
            }
            amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
            const auto lo = amrex::lbound(bx);
            const int nx_box = bx.length(0);
            const int ny_box = bx.length(1);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const long long lin = static_cast<long long>(k - lo.z) * nx_box * ny_box +
                                      static_cast<long long>(j - lo.y) * nx_box +
                                      static_cast<long long>(i - lo.x);
                soln_arr(i, j, k, 0) = static_cast<amrex::Real>(src_ptr[lin]);
            });
        }
#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
        amrex::MultiFab mf_mask_temp(m_ba, m_dm, 1, 0);
        amrex::Copy(mf_mask_temp, m_mf_active_mask, MaskComp, 0, 1, 0);
        amrex::MultiFab mf_phase_temp(m_ba, m_dm, 1, 0);
        amrex::Copy(mf_phase_temp, m_mf_phase, 0, 0, 1, 0);
        amrex::Copy(mf_plot, mf_soln_temp, 0, 0, 1, 0);
        amrex::Copy(mf_plot, mf_phase_temp, 0, 1, 1, 0);
        amrex::Copy(mf_plot, mf_mask_temp, 0, 2, 1, 0);
        std::string plotfilename =
            m_resultspath + "/tortuosity_solution_" + std::to_string(static_cast<int>(m_dir));
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id", "active_mask"};
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Plotfile written to " << plotfilename << std::endl;
        }
    } else if (m_write_plotfile && !m_converged) {
        // Write a "failed" plotfile for debugging non-convergent solves
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Solver did not converge. Writing failed plotfile for debugging..."
                           << std::endl;
        }
        amrex::MultiFab mf_plot_fail(m_ba, m_dm, numComponentsPhi, 0);
        amrex::MultiFab mf_soln_fail(m_ba, m_dm, 1, 0);
        mf_soln_fail.setVal(0.0);
        // HYPRE_MEMORY_DEVICE on CUDA → GetBoxValues writes to a device pointer.
        // No OMP: HYPRE_StructVectorGetBoxValues is not thread-safe for the same vector.
        std::vector<HYPRE_Real> soln_buf_fail_host;
        for (amrex::MFIter mfi(mf_soln_fail, false); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            const int npts = static_cast<int>(bx.numPts());
            if (npts == 0)
                continue;
            auto hypre_lo_f = OpenImpala::TortuosityHypre::loV(bx);
            auto hypre_hi_f = OpenImpala::TortuosityHypre::hiV(bx);
#ifdef AMREX_USE_GPU
            amrex::Gpu::DeviceVector<HYPRE_Real> d_soln_buf_fail(npts);
            HYPRE_StructVectorGetBoxValues(m_x, hypre_lo_f.data(), hypre_hi_f.data(),
                                           d_soln_buf_fail.data());
            const HYPRE_Real* src_ptr_fail = d_soln_buf_fail.data();
#else
            soln_buf_fail_host.resize(npts);
            HYPRE_StructVectorGetBoxValues(m_x, hypre_lo_f.data(), hypre_hi_f.data(),
                                           soln_buf_fail_host.data());
            const HYPRE_Real* src_ptr_fail = soln_buf_fail_host.data();
#endif
            amrex::Array4<amrex::Real> const soln_arr = mf_soln_fail.array(mfi);
            const auto lo_f = amrex::lbound(bx);
            const int nx_f = bx.length(0);
            const int ny_f = bx.length(1);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const long long lin = static_cast<long long>(k - lo_f.z) * nx_f * ny_f +
                                      static_cast<long long>(j - lo_f.y) * nx_f +
                                      static_cast<long long>(i - lo_f.x);
                soln_arr(i, j, k, 0) = static_cast<amrex::Real>(src_ptr_fail[lin]);
            });
        }
#ifdef AMREX_USE_GPU
        amrex::Gpu::streamSynchronize();
#endif
        amrex::MultiFab mf_mask_fail(m_ba, m_dm, 1, 0);
        amrex::Copy(mf_mask_fail, m_mf_active_mask, MaskComp, 0, 1, 0);
        amrex::MultiFab mf_phase_fail(m_ba, m_dm, 1, 0);
        amrex::Copy(mf_phase_fail, m_mf_phase, 0, 0, 1, 0);
        amrex::Copy(mf_plot_fail, mf_soln_fail, 0, 0, 1, 0);
        amrex::Copy(mf_plot_fail, mf_phase_fail, 0, 1, 1, 0);
        amrex::Copy(mf_plot_fail, mf_mask_fail, 0, 2, 1, 0);
        std::string failedname = m_resultspath + "/failed_tortuosity_solution_" +
                                 std::to_string(static_cast<int>(m_dir));
        amrex::Vector<std::string> varnames_fail = {"solution_potential", "phase_id",
                                                    "active_mask"};
        amrex::WriteSingleLevelPlotfile(failedname, mf_plot_fail, varnames_fail, m_geom, 0.0, 0);
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Failed plotfile written to " << failedname << std::endl;
        }
    }
    return m_converged;
}


// --- Calculate Tortuosity Value (Calls Solve if needed) ---
// <<< MODIFIED to use ACTIVE volume fraction >>>
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh) {
    // If active VF is zero (checked in constructor), return NaN immediately.
    if (m_active_vf <= std::numeric_limits<amrex::Real>::epsilon() && !m_first_call) {
        return std::numeric_limits<amrex::Real>::quiet_NaN(); // Or Inf
    }

    if (m_first_call || refresh) {
        // Check again in case constructor exited early
        if (m_active_vf <= std::numeric_limits<amrex::Real>::epsilon()) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print()
                    << "WARNING: Active volume fraction is zero. Tortuosity is NaN or Inf."
                    << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN(); // Or Inf
            m_first_call = false;
            return m_value;
        }

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Calculating Tortuosity (solve required)..." << std::endl;
        }
        bool solve_converged = solve(); // Call solve, which now sets m_converged

        if (!solve_converged) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "WARNING: Solver did not converge or failed. Tortuosity "
                                  "calculation skipped, returning NaN."
                               << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            m_first_call = false;
            return m_value;
        }

        // Solve converged, now calculate fluxes and check conservation
        global_fluxes(); // Calculates and stores m_flux_in, m_flux_out

        // --- Check Flux Conservation (boundary + interior planes) ---
        constexpr amrex::Real flux_tol = 1.0e-6;
        bool flux_conserved = true;
        amrex::Real rel_diff = 0.0;
        amrex::Real flux_mag_in = std::abs(m_flux_in);
        amrex::Real flux_mag_out = std::abs(m_flux_out);
        amrex::Real flux_mag_avg = 0.5 * (flux_mag_in + flux_mag_out);
        if (flux_mag_avg > tiny_flux_threshold) {
            rel_diff = std::abs(flux_mag_in - flux_mag_out) / flux_mag_avg;
            if (rel_diff > flux_tol) {
                flux_conserved = false;
            }
        } else {
            flux_conserved = true;
            rel_diff = 0.0;
        }
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Flux Conservation Check (|in|-|out|) / avg(|in|,|out|):\n";
            amrex::Print() << "    Flux In  = " << std::fixed << std::setprecision(8) << m_flux_in
                           << "\n";
            amrex::Print() << "    Flux Out = " << std::fixed << std::setprecision(8) << m_flux_out
                           << "\n";
            amrex::Print() << "    Relative Difference = " << std::scientific << rel_diff
                           << std::defaultfloat << " (Tolerance = " << flux_tol << ")\n";
            if (!flux_conserved) {
                amrex::Warning("Boundary flux conservation check failed!");
            } else {
                amrex::Print() << "    Boundary Conservation Check Status: PASS\n";
            }
        }

        // --- Interior Plane Flux Conservation Check ---
        // Verify that flux is conserved at every cross-section, not just boundaries.
        // This catches cases like narrow inlets/outlets where boundary flux is
        // unreliable but interior planes reveal non-conservation.
        if (flux_conserved && !m_plane_fluxes.empty()) {
            if (m_plane_flux_max_dev > flux_tol) {
                flux_conserved = false;
                if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Warning(
                        "Interior plane flux conservation check failed! Max deviation = " +
                        std::to_string(m_plane_flux_max_dev) +
                        " > tolerance = " + std::to_string(flux_tol));
                }
            } else if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "    Interior Plane Conservation Check Status: PASS"
                               << " (max_dev=" << std::scientific << m_plane_flux_max_dev
                               << std::defaultfloat << ")\n";
            }
        }

        // --- Calculate Tortuosity only if flux is conserved ---
        if (!flux_conserved) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print()
                    << "WARNING: Flux not conserved. Tortuosity calculation skipped, returning NaN."
                    << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else {
            // Flux is conserved, proceed with calculation using average flux magnitude

            // <<< CHANGE: Use m_active_vf instead of m_vf >>>
            amrex::Real vf_for_calc = m_active_vf;
            // <<< END CHANGE >>>

            amrex::Real L = m_geom.ProbLength(static_cast<int>(m_dir));
            amrex::Real A = 1.0;
            if (AMREX_SPACEDIM == 3) {
                if (m_dir == OpenImpala::Direction::X)
                    A = m_geom.ProbLength(1) * m_geom.ProbLength(2);
                else if (m_dir == OpenImpala::Direction::Y)
                    A = m_geom.ProbLength(0) * m_geom.ProbLength(2);
                else
                    A = m_geom.ProbLength(0) * m_geom.ProbLength(1);
            } else if (AMREX_SPACEDIM == 2) {
                if (m_dir == OpenImpala::Direction::X)
                    A = m_geom.ProbLength(1);
                else
                    A = m_geom.ProbLength(0);
            }
            amrex::Real gradPhi = (m_vhi - m_vlo) / L; // Assumes L > 0
            amrex::Real Deff = 0.0;

            // Use mean of interior plane fluxes when available (more robust
            // than boundary-only average for geometries with narrow inlets).
            // Falls back to boundary average if plane fluxes are empty.
            amrex::Real avg_flux_mag;
            if (!m_plane_fluxes.empty()) {
                amrex::Real sum_plane = 0.0;
                for (const auto& pf : m_plane_fluxes) {
                    sum_plane += std::abs(pf);
                }
                avg_flux_mag = sum_plane / static_cast<amrex::Real>(m_plane_fluxes.size());
            } else {
                avg_flux_mag = 0.5 * (std::abs(m_flux_in) + std::abs(m_flux_out));
            }

            // Handle edge cases
            if (avg_flux_mag < tiny_flux_threshold) {
                if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print()
                        << "WARNING: Average flux magnitude is near zero (" << avg_flux_mag
                        << "). Tortuosity set to Inf (or NaN if ActiveVF=0)." << std::endl;
                }
                // <<< CHANGE: Check vf_for_calc (active_vf) here >>>
                m_value = (vf_for_calc > std::numeric_limits<amrex::Real>::epsilon())
                              ? std::numeric_limits<amrex::Real>::infinity()
                              : std::numeric_limits<amrex::Real>::quiet_NaN();
            }
            // <<< CHANGE: Check vf_for_calc (active_vf) here >>>
            else if (vf_for_calc <= std::numeric_limits<amrex::Real>::epsilon()) {
                if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print()
                        << "WARNING: Active volume fraction is zero. Tortuosity set to NaN."
                        << std::endl;
                }
                m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            } else if (std::abs(gradPhi) < tiny_flux_threshold) {
                if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print()
                        << "WARNING: Potential gradient is zero (vlo=vhi). Tortuosity set to Inf."
                        << std::endl;
                }
                m_value = std::numeric_limits<amrex::Real>::infinity();
            } else {
                // Calculate Deff using average flux magnitude
                Deff = (avg_flux_mag / A) / std::abs(gradPhi);
                if (std::abs(Deff) < tiny_flux_threshold) {
                    if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "WARNING: Effective Diffusivity (Deff) is near zero ("
                                       << Deff << "). Tortuosity set to Inf." << std::endl;
                    }
                    m_value = std::numeric_limits<amrex::Real>::infinity();
                } else {
                    // <<< CHANGE: This calculation now uses active VF >>>
                    m_value = vf_for_calc / Deff;
                }
            }

            if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                // <<< CHANGE: Update print statement label >>>
                amrex::Print() << "  Calculation Details: ActiveVf=" << vf_for_calc << ", L=" << L
                               << ", A=" << A << ", gradPhi=" << gradPhi
                               << ", AvgFluxMag=" << avg_flux_mag << ", Deff=" << Deff << std::endl;
                amrex::Print() << "  Calculated Tortuosity (using Active Vf): " << m_value
                               << std::endl;
            }
        } // End if flux_conserved
    } // End if m_first_call or refresh

    m_first_call = false; // Mark that solve/flux has been attempted
    return m_value;
}


// --- checkMatrixProperties ---
// Remains the same...
bool OpenImpala::TortuosityHypre::checkMatrixProperties() {
    BL_PROFILE("TortuosityHypre::checkMatrixProperties");
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Checking assembled matrix/vector properties..."
                       << std::endl;
    }
    HYPRE_Int ierr = 0;
    bool checks_passed_local = true;
    const double tol = 1.e-14;
    HYPRE_Int hypre_stencil_size = stencil_size;
    HYPRE_Int stencil_indices_hypre[stencil_size];
    for (int i = 0; i < stencil_size; ++i)
        stencil_indices_hypre[i] = i;
    const int center_stencil_index = istn_c;
    const amrex::Box& domain = m_geom.Domain();
    const int idir = static_cast<int>(m_dir);
    // No OMP: HYPRE GetBoxValues is not thread-safe for the same matrix/vector.
    std::vector<HYPRE_Real> matrix_buffer;
    std::vector<HYPRE_Real> rhs_buffer;
    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    for (amrex::MFIter mfi(m_mf_active_mask, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0)
            continue;
        matrix_buffer.resize(static_cast<size_t>(npts) * stencil_size);
        rhs_buffer.resize(npts);
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
        bool hypre_get_ok = true;
        ierr = HYPRE_StructMatrixGetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size,
                                              stencil_indices_hypre, matrix_buffer.data());
        if (ierr != 0) {
            hypre_get_ok = false;
        }
        ierr = HYPRE_StructVectorGetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(),
                                              rhs_buffer.data());
        if (ierr != 0) {
            hypre_get_ok = false;
        }
        if (!hypre_get_ok) {
            checks_passed_local = false;
            if (m_verbose > 0)
                amrex::Print() << "CHECK FAILED: HYPRE_GetBoxValues error on rank "
                               << amrex::ParallelDescriptor::MyProc() << " for box " << bx
                               << std::endl;
            continue;
        }
        const amrex::IArrayBox& mask_fab = m_mf_active_mask[mfi];
        amrex::Array4<const int> const mask_arr = mask_fab.const_array();
        long long linear_idx = 0;
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
            amrex::IntVect current_cell(i, j, k);
            size_t matrix_start_idx = linear_idx * stencil_size;
            double rhs_val = rhs_buffer[linear_idx];
            double diag_val = matrix_buffer[matrix_start_idx + center_stencil_index];
            bool has_nan_inf = std::isnan(rhs_val) || std::isinf(rhs_val);
            for (int s = 0; s < stencil_size; ++s) {
                has_nan_inf = has_nan_inf || std::isnan(matrix_buffer[matrix_start_idx + s]) ||
                              std::isinf(matrix_buffer[matrix_start_idx + s]);
            }
            if (has_nan_inf) {
                if (m_verbose > 0)
                    amrex::Print()
                        << "CHECK FAILED: NaN/Inf found at cell " << current_cell << std::endl;
                checks_passed_local = false;
            }
            int cell_status = mask_arr(current_cell, MaskComp);
            bool is_dirichlet = false;
            // Only check for Dirichlet at domain faces when using DirichletExternal BC
            if (cell_status == cell_active && m_bc_inlet_outlet_type == BCType::DirichletExternal) {
                if ((idir == 0 && (i == domain.smallEnd(0) || i == domain.bigEnd(0))) ||
                    (idir == 1 && (j == domain.smallEnd(1) || j == domain.bigEnd(1))) ||
                    (idir == 2 && (k == domain.smallEnd(2) || k == domain.bigEnd(2)))) {
                    is_dirichlet = true;
                }
            }
            if (cell_status == cell_inactive) {
                if (std::abs(diag_val - 1.0) > tol || std::abs(rhs_val) > tol) {
                    if (m_verbose > 0)
                        amrex::Print()
                            << "CHECK FAILED: Inactive cell check fail at " << current_cell
                            << " (Aii=" << diag_val << ", b=" << rhs_val << ")" << std::endl;
                    checks_passed_local = false;
                }
                for (int s = 0; s < stencil_size; ++s) {
                    if (s != center_stencil_index &&
                        std::abs(matrix_buffer[matrix_start_idx + s]) > tol) {
                        if (m_verbose > 0)
                            amrex::Print()
                                << "CHECK FAILED: Non-zero off-diag [" << s << "] at inactive cell "
                                << current_cell << " (Aij=" << matrix_buffer[matrix_start_idx + s]
                                << ")" << std::endl;
                        checks_passed_local = false;
                    }
                }
            } else if (is_dirichlet) {
                double expected_rhs = ((idir == 0 && i == domain.smallEnd(0)) ||
                                       (idir == 1 && j == domain.smallEnd(1)) ||
                                       (idir == 2 && k == domain.smallEnd(2)))
                                          ? m_vlo
                                          : m_vhi;
                if (std::abs(diag_val - 1.0) > tol || std::abs(rhs_val - expected_rhs) > tol) {
                    if (m_verbose > 0)
                        amrex::Print() << "CHECK FAILED: Dirichlet cell check fail at "
                                       << current_cell << " (Aii=" << diag_val << ", b=" << rhs_val
                                       << ", exp_b=" << expected_rhs << ")" << std::endl;
                    checks_passed_local = false;
                }
                for (int s = 0; s < stencil_size; ++s) {
                    if (s != center_stencil_index &&
                        std::abs(matrix_buffer[matrix_start_idx + s]) > tol) {
                        if (m_verbose > 0)
                            amrex::Print() << "CHECK FAILED: Non-zero off-diag [" << s
                                           << "] at Dirichlet cell " << current_cell
                                           << " (Aij=" << matrix_buffer[matrix_start_idx + s] << ")"
                                           << std::endl;
                        checks_passed_local = false;
                    }
                }
            } else { // Active Interior
                if (diag_val <= tol) {
                    if (m_verbose > 0)
                        amrex::Print()
                            << "CHECK FAILED: Non-positive diagonal at active interior cell "
                            << current_cell << " (Aii=" << diag_val << ")" << std::endl;
                    checks_passed_local = false;
                }
                if (std::abs(rhs_val) > tol) {
                    if (m_verbose > 0)
                        amrex::Print() << "CHECK FAILED: Non-zero RHS at active interior cell "
                                       << current_cell << " (b=" << rhs_val << ")" << std::endl;
                    checks_passed_local = false;
                }
                double row_sum = 0.0;
                for (int s = 0; s < stencil_size; ++s) {
                    row_sum += matrix_buffer[matrix_start_idx + s];
                }
                if (std::abs(row_sum) > tol) {
                    if (m_verbose > 0)
                        amrex::Print() << "CHECK FAILED: Non-zero row sum at active interior cell "
                                       << current_cell << " (sum=" << row_sum << ")" << std::endl;
                    checks_passed_local = false;
                }
            }
            linear_idx++;
        });
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(checks_passed_local);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        if (checks_passed_local) {
            amrex::Print() << "TortuosityHypre: Matrix/vector property checks passed." << std::endl;
        } else {
            amrex::Print() << "TortuosityHypre: Matrix/vector property checks FAILED." << std::endl;
        }
    }
    return checks_passed_local;
}


// --- getSolution ---
// Remains the same...
void OpenImpala::TortuosityHypre::getSolution(amrex::MultiFab& soln, int ncomp) {
    amrex::Abort("TortuosityHypre::getSolution not fully implemented yet!");
}

// --- getCellTypes ---
// Remains the same...
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
    amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
}


// --- global_fluxes ---
// Remains the same...
void OpenImpala::TortuosityHypre::global_fluxes() {
    BL_PROFILE("TortuosityHypre::global_fluxes");
    m_flux_in = 0.0;
    m_flux_out = 0.0;
    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir);

    // Pull HYPRE's solution back into an AMReX MultiFab for downstream flux
    // integration. Since HYPRE_SetMemoryLocation is set to DEVICE on CUDA
    // builds, GetBoxValues expects a device pointer; on CPU it's a host
    // std::vector.
    amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 1);
    mf_soln_temp.setVal(0.0);
    // No OMP: HYPRE_StructVectorGetBoxValues is not thread-safe for the same vector.
    std::vector<HYPRE_Real> soln_buffer_host;
    for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0)
            continue;
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
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
            amrex::Warning("HYPRE_StructVectorGetBoxValues failed during flux calculation copy!");
        }
        amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
        const auto lo = amrex::lbound(bx);
        const int nx_box = bx.length(0);
        const int ny_box = bx.length(1);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const long long lin = static_cast<long long>(k - lo.z) * nx_box * ny_box +
                                  static_cast<long long>(j - lo.y) * nx_box +
                                  static_cast<long long>(i - lo.x);
            soln_arr(i, j, k, 0) = static_cast<amrex::Real>(src_ptr[lin]);
        });
    }
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();
#endif
    mf_soln_temp.FillBoundary(m_geom.periodicity());
    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    m_mf_diff_coeff.FillBoundary(m_geom.periodicity());

    // Flux calculation loop - uses D_face (harmonic mean) for variable-coefficient diffusion
    const amrex::Real dx_dir = dx[idir];
    if (dx_dir <= 0.0)
        amrex::Abort("Zero cell size in flux calculation direction!");
    amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);
    if (amrex::ParallelDescriptor::IOProcessor() && m_verbose >= 3) {
        amrex::Print() << "\n--- START LIMITED DEBUG_FLUX (global_fluxes, verbose>=3) ---\n";
        amrex::Print() << "DEBUG_FLUX: Printing details for first few active cells per tile...\n";
    }

    // GPU-compatible reduction for inlet/outlet flux and active cell counts
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
        flux_reduce_op;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> flux_reduce_data(
        flux_reduce_op);

    amrex::Box lobox_domain = amrex::bdryLo(domain, idir);
    amrex::Box hibox_domain = domain;
    hibox_domain.setSmall(idir, domain.bigEnd(idir));
    hibox_domain.setBig(idir, domain.bigEnd(idir));

    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& validBox = mfi.validbox();
        amrex::Array4<const int> const mask = m_mf_active_mask.const_array(mfi);
        amrex::Array4<const amrex::Real> const soln = mf_soln_temp.const_array(mfi);
        amrex::Array4<const amrex::Real> const dc = m_mf_diff_coeff.const_array(mfi);

        amrex::Box lobox_face = validBox & lobox_domain;
        if (!lobox_face.isEmpty()) {
            flux_reduce_op.eval(
                lobox_face, flux_reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
                    amrex::Real fxin = 0.0, cells_in = 0.0;
                    if (mask(i, j, k) == cell_active) {
                        cells_in = 1.0;
                        int si = i + shift[0], sj = j + shift[1], sk = k + shift[2];
                        if (mask(si, sj, sk) == cell_active) {
                            amrex::Real D_bnd = dc(i, j, k);
                            amrex::Real D_inn = dc(si, sj, sk);
                            amrex::Real D_face =
                                (D_bnd + D_inn > 0.0) ? 2.0 * D_bnd * D_inn / (D_bnd + D_inn) : 0.0;
                            amrex::Real grad = (soln(si, sj, sk) - soln(i, j, k)) / dx_dir;
                            fxin = -D_face * grad;
                        }
                    }
                    return {fxin, 0.0, cells_in, 0.0};
                });
        }

        amrex::Box hibox_face = validBox & hibox_domain;
        if (!hibox_face.isEmpty()) {
            flux_reduce_op.eval(
                hibox_face, flux_reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
                    amrex::Real fxout = 0.0, cells_out = 0.0;
                    if (mask(i, j, k) == cell_active) {
                        cells_out = 1.0;
                        int si = i - shift[0], sj = j - shift[1], sk = k - shift[2];
                        if (mask(si, sj, sk) == cell_active) {
                            amrex::Real D_bnd = dc(i, j, k);
                            amrex::Real D_inn = dc(si, sj, sk);
                            amrex::Real D_face =
                                (D_bnd + D_inn > 0.0) ? 2.0 * D_bnd * D_inn / (D_bnd + D_inn) : 0.0;
                            amrex::Real grad = (soln(i, j, k) - soln(si, sj, sk)) / dx_dir;
                            fxout = -D_face * grad;
                        }
                    }
                    return {0.0, fxout, 0.0, cells_out};
                });
        }
    } // End MFIter loop

    auto flux_hv = flux_reduce_data.value();
    amrex::Real local_fxin = amrex::get<0>(flux_hv);
    amrex::Real local_fxout = amrex::get<1>(flux_hv);
    amrex::Real local_active_cells_in = amrex::get<2>(flux_hv);
    amrex::Real local_active_cells_out = amrex::get<3>(flux_hv);

    // Reduction and final scaling remain the same...
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);
    amrex::ParallelDescriptor::ReduceRealSum(local_active_cells_in);
    amrex::ParallelDescriptor::ReduceRealSum(local_active_cells_out);
    long global_active_in = static_cast<long>(local_active_cells_in);
    long global_active_out = static_cast<long>(local_active_cells_out);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (m_verbose > 1) {
            amrex::Print() << "  Active boundary cell counts: In=" << global_active_in
                           << ", Out=" << global_active_out << "\n";
        }
        if (m_verbose >= 3) {
            amrex::Print() << "DEBUG_FLUX: After reduction: Summed_fxin=" << local_fxin
                           << " Summed_fxout=" << local_fxout << "\n";
            amrex::Print() << "--- END LIMITED DEBUG_FLUX (global_fluxes, verbose>=3) ---\n\n";
        }
    }
    amrex::Real face_area_element = 1.0;
    if (AMREX_SPACEDIM == 3) {
        if (idir == 0) {
            face_area_element = dx[1] * dx[2];
        } else if (idir == 1) {
            face_area_element = dx[0] * dx[2];
        } else {
            face_area_element = dx[0] * dx[1];
        }
    } else if (AMREX_SPACEDIM == 2) {
        if (idir == 0) {
            face_area_element = dx[1];
        } else {
            face_area_element = dx[0];
        }
    }
    m_flux_in = local_fxin * face_area_element;
    m_flux_out = local_fxout * face_area_element;

    // Compute flux through every interior plane for enhanced convergence checking
    computePlaneFluxes(mf_soln_temp);
}


// --- computePlaneFluxes ---
// Computes total flux through every inter-cell face perpendicular to the flow
// direction.  For a domain with N cells in the flow direction there are (N-1)
// interior faces (between cell i and cell i+1, for i = 0 .. N-2).
// After computation, m_plane_fluxes[i] holds the total (area-weighted) flux
// through the face between cell-plane i and i+1.
// The result allows verification of flux conservation at every cross-section,
// not just at the inlet and outlet boundaries.
void OpenImpala::TortuosityHypre::computePlaneFluxes(const amrex::MultiFab& mf_soln) {
    BL_PROFILE("TortuosityHypre::computePlaneFluxes");

    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir);
    const int n_cells = domain.length(idir); // Number of cells in flow direction
    const int n_faces = n_cells - 1;         // Interior inter-cell faces

    // Compute face area element (product of dx in the two perpendicular directions)
    amrex::Real face_area_element = 1.0;
    if (AMREX_SPACEDIM == 3) {
        if (idir == 0) {
            face_area_element = dx[1] * dx[2];
        } else if (idir == 1) {
            face_area_element = dx[0] * dx[2];
        } else {
            face_area_element = dx[0] * dx[1];
        }
    } else if (AMREX_SPACEDIM == 2) {
        face_area_element = (idir == 0) ? dx[1] : dx[0];
    }

    const amrex::Real dx_dir = dx[idir];
    amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);

    // Each rank accumulates its local contribution per face using device memory
    // for GPU compatibility with atomic scatter-add.
    amrex::Gpu::DeviceVector<amrex::Real> d_plane_flux(n_faces, 0.0);
    amrex::Real* d_plane_flux_ptr = d_plane_flux.data();
    const int domain_lo_idir = domain.smallEnd(idir);

    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& validBox = mfi.validbox();
        amrex::Array4<const int> const mask = m_mf_active_mask.const_array(mfi);
        amrex::Array4<const amrex::Real> const soln = mf_soln.const_array(mfi);
        amrex::Array4<const amrex::Real> const dc = m_mf_diff_coeff.const_array(mfi);

        amrex::Box flux_box = validBox;
        flux_box.setBig(idir, std::min(validBox.bigEnd(idir), domain.bigEnd(idir) - 1));

        amrex::ParallelFor(flux_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int si = i + shift[0], sj = j + shift[1], sk = k + shift[2];
            if (mask(i, j, k) == cell_active && mask(si, sj, sk) == cell_active) {
                amrex::Real D_lo = dc(i, j, k);
                amrex::Real D_hi = dc(si, sj, sk);
                amrex::Real D_face = (D_lo + D_hi > 0.0) ? 2.0 * D_lo * D_hi / (D_lo + D_hi) : 0.0;
                amrex::Real grad = (soln(si, sj, sk) - soln(i, j, k)) / dx_dir;
                amrex::Real flux = -D_face * grad * face_area_element;
                amrex::IntVect iv(i, j, k);
                int face_idx = iv[idir] - domain_lo_idir;
                amrex::Gpu::Atomic::Add(&d_plane_flux_ptr[face_idx], flux);
            }
        });
    }

    // Copy device results to host
    std::vector<amrex::Real> local_plane_flux(n_faces);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_plane_flux.begin(), d_plane_flux.end(),
                     local_plane_flux.begin());

    // MPI reduction across ranks
    amrex::ParallelDescriptor::ReduceRealSum(local_plane_flux.data(), n_faces);

    // Store results
    m_plane_fluxes = std::move(local_plane_flux);

    // Compute statistics: mean flux and max deviation
    amrex::Real sum_flux = 0.0;
    for (int f = 0; f < n_faces; ++f) {
        sum_flux += m_plane_fluxes[f];
    }
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
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "    Per-plane fluxes:\n";
        for (int f = 0; f < n_faces; ++f) {
            amrex::Print() << "      face " << f << ": " << std::scientific << m_plane_fluxes[f]
                           << std::defaultfloat << "\n";
        }
    }
}


} // End namespace OpenImpala
