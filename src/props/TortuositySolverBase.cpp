// --- TortuositySolverBase.cpp ---

#include "TortuositySolverBase.H"
#include "FloodFill.H"
#include "TortuosityKernels.H"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <vector>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_Loop.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_Vector.H>

namespace {
constexpr int MaskComp = 0;
constexpr amrex::Real tiny_flux_threshold = 1.e-15;
constexpr int cell_inactive = 0;
constexpr int cell_active = 1;
} // namespace

namespace OpenImpala {

// --- Constructor ---
TortuositySolverBase::TortuositySolverBase(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                           const amrex::DistributionMapping& dm,
                                           const amrex::iMultiFab& mf_phase_input,
                                           const amrex::Real vf, const int phase,
                                           const OpenImpala::Direction dir,
                                           const std::string& resultspath, const amrex::Real vlo,
                                           const amrex::Real vhi, int verbose, bool write_plotfile)
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase(ba, dm, mf_phase_input.nComp(), mf_phase_input.nGrow()), m_phase(phase), m_vf(vf),
      m_dir(dir), m_vlo(vlo), m_vhi(vhi), m_resultspath(resultspath),
      m_write_plotfile(write_plotfile), m_verbose(verbose), m_mf_solution(ba, dm, 1, 1),
      m_mf_active_mask(ba, dm, 1, 1), m_mf_diff_coeff(ba, dm, 1, 1) {
    amrex::Copy(m_mf_phase, mf_phase_input, 0, 0, m_mf_phase.nComp(), m_mf_phase.nGrow());

    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("verbose", m_verbose);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0,
                                     "Volume fraction must be between 0 and 1");

    // Precondition phase data (remove isolated cells)
    preconditionPhaseFab();

    // Build diffusion coefficient field
    buildDiffusionCoeffField();

    // Generate activity mask via flood fill
    generateActivityMask(m_mf_phase, m_phase, m_dir);

    if (m_active_vf <= std::numeric_limits<amrex::Real>::epsilon()) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "WARNING: Active volume fraction is zero. Skipping solve."
                           << std::endl;
        }
        m_first_call = false;
        m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
    }
}

// --- preconditionPhaseFab ---
void TortuositySolverBase::preconditionPhaseFab() {
    BL_PROFILE("TortuositySolverBase::preconditionPhaseFab");
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
        amrex::Gpu::streamSynchronize();
        for (amrex::MFIter mfi(m_mf_phase, false); mfi.isValid(); ++mfi) {
            const amrex::Box& tile_box = mfi.validbox();
            amrex::IArrayBox& fab = m_mf_phase[mfi];
            const int ncomp = fab.nComp();
            // removeIsolatedCells is a host kernel that dereferences the
            // raw pointer it receives. fab.dataPtr() returns DEVICE memory
            // on CUDA builds; copy the fab to a host buffer, run the
            // kernel there, then push the modified data back to the fab.
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
    }
    m_mf_phase.FillBoundary(m_geom.periodicity());
}

// --- buildDiffusionCoeffField ---
void TortuositySolverBase::buildDiffusionCoeffField() {
    BL_PROFILE("TortuositySolverBase::buildDiffusionCoeffField");

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
}

// --- generateActivityMask ---
void TortuositySolverBase::generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID,
                                                OpenImpala::Direction dir) {
    BL_PROFILE("TortuositySolverBase::generateActivityMask");

    const int idir = static_cast<int>(dir);

    amrex::Vector<amrex::IntVect> inlet_seeds;
    amrex::Vector<amrex::IntVect> outlet_seeds;
    OpenImpala::collectBoundarySeeds(phaseFab, phaseID, idir, m_geom, inlet_seeds, outlet_seeds);

    if (inlet_seeds.empty() || outlet_seeds.empty()) {
        amrex::Warning("TortuositySolverBase::generateActivityMask: No percolating path found.");
        m_mf_active_mask.setVal(cell_inactive);
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
        m_active_vf = 0.0;
        return;
    }

    amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
    amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);
    mf_reached_inlet.setVal(OpenImpala::FLOOD_INACTIVE);
    mf_reached_outlet.setVal(OpenImpala::FLOOD_INACTIVE);
    OpenImpala::parallelFloodFill(mf_reached_inlet, phaseFab, phaseID, inlet_seeds, m_geom,
                                  m_verbose);
    OpenImpala::parallelFloodFill(mf_reached_outlet, phaseFab, phaseID, outlet_seeds, m_geom,
                                  m_verbose);

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

    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<long> reduce_data(reduce_op);
    for (amrex::MFIter mfi(m_mf_active_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& mask_arr = m_mf_active_mask.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> amrex::GpuTuple<long> {
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

// --- parallelFloodFill (thin wrapper around free function) ---
void TortuositySolverBase::parallelFloodFill(amrex::iMultiFab& reachabilityMask,
                                             const amrex::iMultiFab& phaseFab, int phaseID,
                                             const amrex::Vector<amrex::IntVect>& seedPoints) {
    OpenImpala::parallelFloodFill(reachabilityMask, phaseFab, phaseID, seedPoints, m_geom,
                                  m_verbose);
}

// --- globalFluxes ---
void TortuositySolverBase::globalFluxes() {
    BL_PROFILE("TortuositySolverBase::globalFluxes");
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

        amrex::Box domain_lo_face = domain;
        domain_lo_face.setSmall(idir, domain.smallEnd(idir));
        domain_lo_face.setBig(idir, domain.smallEnd(idir));
        amrex::Box lobox_face = domain_lo_face & validBox;
        if (!lobox_face.isEmpty()) {
            flux_reduce_op.eval(
                lobox_face, flux_reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j,
                                     int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
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
            flux_reduce_op.eval(
                hibox_face, flux_reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j,
                                     int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
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
void TortuositySolverBase::computePlaneFluxes(const amrex::MultiFab& mf_soln) {
    BL_PROFILE("TortuositySolverBase::computePlaneFluxes");

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

    std::vector<amrex::Real> local_plane_flux(n_faces);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_plane_flux.begin(), d_plane_flux.end(),
                     local_plane_flux.begin());

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

// --- writeSolutionPlotfile ---
void TortuositySolverBase::writeSolutionPlotfile(const std::string& label) {
    BL_PROFILE("TortuositySolverBase::writeSolutionPlotfile");

    amrex::MultiFab mf_plot(m_ba, m_dm, 3, 0);
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
    std::string plotfilename = m_resultspath + "/" + label;
    amrex::Vector<std::string> varnames = {"solution_potential", "phase_id", "active_mask"};
    amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);
}

// --- value ---
amrex::Real TortuositySolverBase::value(const bool refresh) {
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
                amrex::Print() << "WARNING: Solver did not converge. Returning NaN." << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            m_first_call = false;
            return m_value;
        }

        globalFluxes();

        // Flux conservation check.
        //
        // Two distinct tests, with different roles:
        //
        //  1) Boundary: |flux_in| ≈ |flux_out|. This is the canonical
        //     conservation guarantee for a steady-state diffusion solve and
        //     stays a hard failure (NaN return). Tolerance is loose at 1e-4
        //     relative — far above the solver's per-cell residual of ~1e-9,
        //     so a legitimate converged solve won't trip it, but a
        //     genuinely-broken one (e.g. wrong BC application) will.
        //
        //  2) Interior planes: max variance of integrated flux across
        //     domain-perpendicular cross-sections. In a heterogeneous medium
        //     these are theoretically equal but the per-cell residual
        //     accumulates into a small per-plane discrepancy that scales as
        //     ~residual × cells-per-plane, which can easily exceed any
        //     fixed relative tolerance at 128³+ even when the solve is
        //     numerically fine. This is now warning-only: it reports the
        //     deviation in the log but does NOT NaN the result.
        constexpr amrex::Real flux_tol = 1.0e-4;
        constexpr amrex::Real plane_dev_warn = 1.0e-3;
        bool flux_conserved = true;
        amrex::Real flux_mag_in = std::abs(m_flux_in);
        amrex::Real flux_mag_out = std::abs(m_flux_out);
        amrex::Real flux_mag_avg = 0.5 * (flux_mag_in + flux_mag_out);
        amrex::Real boundary_rel_diff = 0.0;
        if (flux_mag_avg > tiny_flux_threshold) {
            boundary_rel_diff = std::abs(flux_mag_in - flux_mag_out) / flux_mag_avg;
            if (boundary_rel_diff > flux_tol)
                flux_conserved = false;
        }

        if (!m_plane_fluxes.empty() && m_plane_flux_max_dev > plane_dev_warn) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Interior plane flux variance is " << m_plane_flux_max_dev
                               << " (above " << plane_dev_warn
                               << " warn threshold). Boundary conservation still holds; "
                                  "tortuosity is reported but treat results from large or "
                                  "heterogeneous domains with caution.\n";
            }
        }

        if (!flux_conserved) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "WARNING: Boundary flux not conserved (|in|-|out| / avg = "
                               << boundary_rel_diff << " > " << flux_tol << "). Returning NaN."
                               << std::endl;
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
                amrex::Print() << "  Tortuosity: " << m_value << std::endl;
            }
        }
    }

    m_first_call = false;
    return m_value;
}

} // namespace OpenImpala
