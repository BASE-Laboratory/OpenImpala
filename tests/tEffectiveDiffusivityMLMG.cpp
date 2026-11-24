/** @file tEffectiveDiffusivityMLMG.cpp
 *  @brief Analytical and HYPRE-cross-check test suite for EffectiveDiffusivityMLMG.
 *
 *  Exercises the matrix-free AMReX MLMG periodic homogenisation solver across:
 *    1. Uniform medium (exact zero-RHS fast path, D_eff = I_3)
 *    2. Single-phase parallel/blocked slabs (D_xx = D_yy = 0.5, D_zz = 0)
 *    3. Multi-phase Voigt/Reuss laminate (D = {1.0, 0.1}, phi_0 = 0.25 ->
 *       D_xx = D_yy = 0.325, D_zz = 1 / 7.75 with non-zero corrector chi_z)
 *    4. 3-D heterogeneous porous microstructure with an isolated pore island and
 *       anisotropic voxel spacing (dx=1.0, dy=2.0, dz=1.5), verifying symmetry
 *       and < 1e-6 agreement against EffectiveDiffusivityHypre on CPU builds.
 */

#include "DeffTensor.H"
#include "EffectiveDiffusivityHypre.H"
#include "EffectiveDiffusivityMLMG.H"
#include "Tortuosity.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_Utility.H>
#include <AMReX_iMultiFab.H>

#include <HYPRE.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>

namespace OpenImpalaTest {

void fillSlabZ(amrex::iMultiFab& mf_phase, int z_cut, int phase_lo, int phase_hi) {
    for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto arr = mf_phase.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            arr(i, j, k, 0) = (k < z_cut) ? phase_lo : phase_hi;
        });
    }
}

void fillHeterogeneousWithIsolatedPocket(amrex::iMultiFab& mf_phase, int n) {
    for (amrex::MFIter mfi(mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto arr = mf_phase.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Solid block in [3, 12]^3 enclosing an isolated active 2x2x2 pocket at [7, 8]^3,
            // plus asymmetric solid pillars to produce non-trivial 3-D tortuous pathways.
            const bool in_shell = (i >= 3 && i <= 12 && j >= 3 && j <= 12 && k >= 3 && k <= 12);
            const bool in_pocket = (i >= 7 && i <= 8 && j >= 7 && j <= 8 && k >= 7 && k <= 8);
            const bool in_pillar = ((i + 2 * j) % n < 3) && (k % 5 == 1);
            if (in_pocket) {
                arr(i, j, k, 0) = 1;
            } else if (in_shell || in_pillar) {
                arr(i, j, k, 0) = 0;
            } else {
                arr(i, j, k, 0) = 1;
            }
        });
    }
}

bool solveTensorMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                     int phase_id, const std::map<int, amrex::Real>& phase_coeffs,
                     const std::string& resultsdir,
                     amrex::Real tensor_out[AMREX_SPACEDIM][AMREX_SPACEDIM]) {
    amrex::MultiFab chi_x(ba, dm, 1, 1);
    amrex::MultiFab chi_y(ba, dm, 1, 1);
    amrex::MultiFab chi_z(ba, dm, 1, 1);
    chi_x.setVal(0.0);
    chi_y.setVal(0.0);
    chi_z.setVal(0.0);

    amrex::iMultiFab active_mask_0g(ba, dm, 1, 0);
    amrex::MultiFab diff_coeff_0g(ba, dm, 1, 0);
    amrex::iMultiFab shared_cell_type(ba, dm, 1, 1);

    const std::array<OpenImpala::Direction, 3> dirs = {
        OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};
    amrex::MultiFab* chi_ptrs[3] = {&chi_x, &chi_y, &chi_z};

    for (int d = 0; d < 3; ++d) {
        const amrex::iMultiFab* ct_ptr = (d == 0) ? nullptr : &shared_cell_type;
        OpenImpala::EffectiveDiffusivityMLMG solver(geom, ba, dm, mf_phase, phase_id, dirs[d],
                                                    resultsdir, 0, false, 1.0e-10, 200, 30,
                                                    phase_coeffs, ct_ptr);
        if (d == 0) {
            amrex::Copy(active_mask_0g, solver.getActiveMask(), 0, 0, 1, 0);
            amrex::Copy(diff_coeff_0g, solver.getDiffCoeff(), 0, 0, 1, 0);
            amrex::Copy(shared_cell_type, solver.getCellTypeMask(), 0, 0, 1, 0);
            shared_cell_type.FillBoundary(geom.periodicity());
        }
        if (!solver.solve() || !solver.getSolverConverged()) {
            return false;
        }
        solver.getChiSolution(*chi_ptrs[d]);
        chi_ptrs[d]->FillBoundary(geom.periodicity());
    }

    OpenImpala::calculateDeffTensor(tensor_out, chi_x, chi_y, chi_z, active_mask_0g, geom, 0,
                                    &diff_coeff_0g);
    return true;
}

#ifndef AMREX_USE_GPU
bool solveTensorHypre(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                      const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                      int phase_id, const std::map<int, amrex::Real>& phase_coeffs,
                      const std::string& resultsdir,
                      amrex::Real tensor_out[AMREX_SPACEDIM][AMREX_SPACEDIM]) {
    amrex::MultiFab chi_x(ba, dm, 1, 1);
    amrex::MultiFab chi_y(ba, dm, 1, 1);
    amrex::MultiFab chi_z(ba, dm, 1, 1);
    chi_x.setVal(0.0);
    chi_y.setVal(0.0);
    chi_z.setVal(0.0);

    amrex::iMultiFab active_mask_0g(ba, dm, 1, 0);
    amrex::MultiFab diff_coeff_0g(ba, dm, 1, 0);

    const std::array<OpenImpala::Direction, 3> dirs = {
        OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};
    amrex::MultiFab* chi_ptrs[3] = {&chi_x, &chi_y, &chi_z};

    for (int d = 0; d < 3; ++d) {
        OpenImpala::EffectiveDiffusivityHypre solver(geom, ba, dm, mf_phase, phase_id, dirs[d],
                                                     OpenImpala::SolverType::FlexGMRES, resultsdir,
                                                     0, false, phase_coeffs);
        if (d == 0) {
            amrex::Copy(active_mask_0g, solver.getActiveMask(), 0, 0, 1, 0);
            amrex::Copy(diff_coeff_0g, solver.getDiffCoeff(), 0, 0, 1, 0);
        }
        if (!solver.solve() || !solver.getSolverConverged()) {
            return false;
        }
        solver.getChiSolution(*chi_ptrs[d]);
        chi_ptrs[d]->FillBoundary(geom.periodicity());
    }

    OpenImpala::calculateDeffTensor(tensor_out, chi_x, chi_y, chi_z, active_mask_0g, geom, 0,
                                    &diff_coeff_0g);
    return true;
}
#endif

} // namespace OpenImpalaTest

int main(int argc, char* argv[]) {
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
        std::fprintf(stderr, "FATAL: HYPRE_Init() failed with code %d\n", hypre_ierr);
        return 1;
    }

    amrex::Initialize(argc, argv);
    {
        const int n = 16;
        const int box_size = 8; // Multi-box decomposition to exercise periodic patch boundaries
        const std::string resultsdir = "./tEffectiveDiffusivityMLMG_results";
        const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(n - 1, n - 1, n - 1));
        const amrex::Array<int, AMREX_SPACEDIM> is_per{AMREX_D_DECL(1, 1, 1)};

        amrex::RealBox rb_iso({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(n), amrex::Real(n), amrex::Real(n))});
        amrex::Geometry geom_iso;
        geom_iso.define(domain, &rb_iso, 0, is_per.data());

        amrex::BoxArray ba(domain);
        ba.maxSize(box_size);
        amrex::DistributionMapping dm(ba);
        amrex::iMultiFab mf_phase(ba, dm, 1, 1);

        auto require_near = [](amrex::Real actual, amrex::Real expected, amrex::Real tol,
                               const std::string& label) {
            if (std::abs(actual - expected) > tol) {
                amrex::Abort("tEffectiveDiffusivityMLMG FAIL [" + label + "]: got " +
                             std::to_string(actual) + ", expected " + std::to_string(expected));
            }
        };

        // 1. Uniform single-phase domain -> D_eff = I_3
        mf_phase.setVal(1);
        mf_phase.FillBoundary(geom_iso.periodicity());
        amrex::Real D_unif[AMREX_SPACEDIM][AMREX_SPACEDIM];
        if (!OpenImpalaTest::solveTensorMLMG(geom_iso, ba, dm, mf_phase, 1, {}, resultsdir,
                                             D_unif)) {
            amrex::Abort("tEffectiveDiffusivityMLMG FAIL: uniform solve did not converge.");
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                require_near(D_unif[r][c], (r == c) ? 1.0 : 0.0, 1.0e-10, "uniform");
            }
        }

        // 2. Single-phase parallel/blocked slabs along Z (k < 8 active, k >= 8 solid)
        OpenImpalaTest::fillSlabZ(mf_phase, n / 2, 1, 0);
        mf_phase.FillBoundary(geom_iso.periodicity());
        amrex::Real D_slab[AMREX_SPACEDIM][AMREX_SPACEDIM];
        if (!OpenImpalaTest::solveTensorMLMG(geom_iso, ba, dm, mf_phase, 1, {}, resultsdir,
                                             D_slab)) {
            amrex::Abort("tEffectiveDiffusivityMLMG FAIL: slab solve did not converge.");
        }
        require_near(D_slab[0][0], 0.5, 1.0e-7, "slab D_xx");
        require_near(D_slab[1][1], 0.5, 1.0e-7, "slab D_yy");
        require_near(D_slab[2][2], 0.0, 1.0e-7, "slab D_zz");

        // 3. Two-phase Voigt / Reuss laminate (phi_0 = 0.25 with D_0 = 1.0, phi_1 = 0.75 with D_1 =
        // 0.1)
        OpenImpalaTest::fillSlabZ(mf_phase, n / 4, 0, 1);
        mf_phase.FillBoundary(geom_iso.periodicity());
        const std::map<int, amrex::Real> lam_coeffs = {{0, 1.0}, {1, 0.1}};
        amrex::Real D_lam[AMREX_SPACEDIM][AMREX_SPACEDIM];
        if (!OpenImpalaTest::solveTensorMLMG(geom_iso, ba, dm, mf_phase, 0, lam_coeffs, resultsdir,
                                             D_lam)) {
            amrex::Abort("tEffectiveDiffusivityMLMG FAIL: laminate solve did not converge.");
        }
        require_near(D_lam[0][0], 0.325, 1.0e-6, "laminate Voigt D_xx");
        require_near(D_lam[1][1], 0.325, 1.0e-6, "laminate Voigt D_yy");
        require_near(D_lam[2][2], 1.0 / 7.75, 1.0e-6, "laminate Reuss D_zz");

        // 4. Heterogeneous 3-D medium with isolated periodic pocket + anisotropic voxels
        amrex::RealBox rb_aniso(
            {AMREX_D_DECL(0.0, 0.0, 0.0)},
            {AMREX_D_DECL(amrex::Real(n) * 1.0, amrex::Real(n) * 2.0, amrex::Real(n) * 1.5)});
        amrex::Geometry geom_aniso;
        geom_aniso.define(domain, &rb_aniso, 0, is_per.data());

        OpenImpalaTest::fillHeterogeneousWithIsolatedPocket(mf_phase, n);
        mf_phase.FillBoundary(geom_aniso.periodicity());

        amrex::Real D_het_mlmg[AMREX_SPACEDIM][AMREX_SPACEDIM];
        if (!OpenImpalaTest::solveTensorMLMG(geom_aniso, ba, dm, mf_phase, 1, {}, resultsdir,
                                             D_het_mlmg)) {
            amrex::Abort("tEffectiveDiffusivityMLMG FAIL: heterogeneous MLMG solve failed.");
        }
        // Symmetry check on D_eff
        require_near(D_het_mlmg[0][1], D_het_mlmg[1][0], 1.0e-6, "heterogeneous symmetry XY");
        require_near(D_het_mlmg[0][2], D_het_mlmg[2][0], 1.0e-6, "heterogeneous symmetry XZ");
        require_near(D_het_mlmg[1][2], D_het_mlmg[2][1], 1.0e-6, "heterogeneous symmetry YZ");

#ifndef AMREX_USE_GPU
        amrex::Real D_het_hypre[AMREX_SPACEDIM][AMREX_SPACEDIM];
        if (!OpenImpalaTest::solveTensorHypre(geom_aniso, ba, dm, mf_phase, 1, {}, resultsdir,
                                              D_het_hypre)) {
            amrex::Abort("tEffectiveDiffusivityMLMG FAIL: heterogeneous HYPRE solve failed.");
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                require_near(D_het_mlmg[r][c], D_het_hypre[r][c], 1.0e-6,
                             "MLMG vs HYPRE [" + std::to_string(r) + "," + std::to_string(c) + "]");
            }
        }
#endif

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- tEffectiveDiffusivityMLMG: ALL CHECKS PASSED ---\n";
        }
    }
    amrex::Finalize();
    HYPRE_Finalize();
    return 0;
}
