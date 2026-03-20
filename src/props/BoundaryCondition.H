/** @file BoundaryCondition.H
 *  @brief Boundary condition abstraction for tortuosity and transport solvers.
 *
 *  Provides an abstract BoundaryCondition base class with box-level applyBC()
 *  that modifies HYPRE stencil/RHS/initial-guess arrays for boundary cells.
 *  Concrete implementations handle different BC strategies (Dirichlet at domain
 *  faces, Dirichlet at phase boundaries, Neumann, Periodic).
 *
 *  Runtime selection via BCType enum and parseBCType() string conversion,
 *  following the SolverConfig.H pattern.
 */
#ifndef OPENIMPALA_BOUNDARY_CONDITION_H_
#define OPENIMPALA_BOUNDARY_CONDITION_H_

#include <AMReX_BLassert.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

namespace OpenImpala {

// Forward-reference stencil constants from TortuosityHypreFill.H
// (redeclared here to keep this header self-contained)
namespace bc_detail {
constexpr int STN_C = 0;
constexpr int STN_MX = 1;
constexpr int STN_PX = 2;
constexpr int STN_MY = 3;
constexpr int STN_PY = 4;
constexpr int STN_MZ = 5;
constexpr int STN_PZ = 6;
constexpr int NSTENCIL = 7;
constexpr int CELL_INACTIVE = 0;
constexpr int CELL_ACTIVE = 1;
constexpr amrex::Real SMALL_REAL = 1.0e-15;
} // namespace bc_detail

/** @brief Boundary condition types for transport solvers. */
enum class BCType {
    DirichletExternal,      /**< Dirichlet at domain faces (standard, v1.0.0 default) */
    DirichletPhaseBoundary, /**< Dirichlet at active cells adjacent to inactive cells near boundary
                             */
    Neumann,                /**< Zero-flux Neumann (implicit — no stencil modification needed) */
    Periodic                /**< Periodic wrapping (requires HYPRE periodic grid) */
};

/** @brief Convert a case-insensitive string to BCType.
 *  @param bc_str  BC name (e.g. "DirichletExternal", "neumann", "periodic").
 *  @return Corresponding BCType enum value.
 *  @note Calls amrex::Abort if bc_str is unrecognised.
 */
inline BCType parseBCType(const std::string& bc_str) {
    std::string s = bc_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "dirichletexternal" || s == "dirichlet_external" || s == "dirichlet")
        return BCType::DirichletExternal;
    if (s == "dirichletphaseboundary" || s == "dirichlet_phase_boundary" || s == "phaseboundary")
        return BCType::DirichletPhaseBoundary;
    if (s == "neumann" || s == "zeroflux" || s == "zero_flux")
        return BCType::Neumann;
    if (s == "periodic")
        return BCType::Periodic;
    amrex::Abort("Invalid boundary condition string: '" + bc_str +
                 "'. Valid options: DirichletExternal, DirichletPhaseBoundary, Neumann, Periodic.");
    return BCType::DirichletExternal;
}

/** @brief Abstract base class for boundary conditions.
 *
 *  Operates at box level: one virtual call per MFIter box, not per cell.
 *  The fill kernel builds the interior stencil first, then calls applyBC()
 *  to modify boundary cells.
 */
class BoundaryCondition {
public:
    virtual ~BoundaryCondition() = default;

    /** @brief Apply BC modifications to stencil/rhs/xinit for one box.
     *
     *  Called after the interior stencil is fully built. Only modifies cells
     *  that require BC treatment (domain faces, phase boundaries, etc.).
     *
     *  @param[in,out] a           Flattened stencil array, size nval*7.
     *  @param[in,out] rhs         Right-hand side, size nval.
     *  @param[in,out] xinit       Initial guess, size nval.
     *  @param[in]     nval        Number of cells in this box.
     *  @param[in]     active_mask 3-D activity mask (1=active, 0=inactive), with ghosts.
     *  @param[in]     mask_lo     Lower bounds of active_mask (3 ints).
     *  @param[in]     mask_hi     Upper bounds of active_mask (3 ints).
     *  @param[in]     diff_coeff  3-D cell-centred diffusion coefficients, with ghosts.
     *  @param[in]     dc_lo       Lower bounds of diff_coeff (3 ints).
     *  @param[in]     dc_hi       Upper bounds of diff_coeff (3 ints).
     *  @param[in]     bxlo        Lower corner of valid box (3 ints).
     *  @param[in]     bxhi        Upper corner of valid box (3 ints).
     *  @param[in]     domlo       Domain lower corner (3 ints).
     *  @param[in]     domhi       Domain upper corner (3 ints).
     *  @param[in]     dxinv       Inverse grid spacing squared: [1/dx^2, 1/dy^2, 1/dz^2].
     *  @param[in]     vlo         BC value on the low face.
     *  @param[in]     vhi         BC value on the high face.
     *  @param[in]     dir         Flow direction (0=X, 1=Y, 2=Z).
     */
    virtual void applyBC(amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit, int nval,
                         const int* active_mask, const int* mask_lo, const int* mask_hi,
                         const amrex::Real* diff_coeff, const int* dc_lo, const int* dc_hi,
                         const int* bxlo, const int* bxhi, const int* domlo, const int* domhi,
                         const amrex::Real* dxinv, amrex::Real vlo, amrex::Real vhi,
                         int dir) const = 0;

    /** @brief Whether this BC requires HYPRE periodic grid setup.
     *  @return true if setupGrid() should be called with periodic=true.
     */
    virtual bool needsPeriodicGrid() const {
        return false;
    }
};

// ---------------------------------------------------------------------------
// DirichletExternalBC — Dirichlet at domain faces (current v1.0.0 default)
// ---------------------------------------------------------------------------

/** @brief Applies Dirichlet BC at domain boundary faces perpendicular to the flow direction.
 *
 *  Active cells on the low/high domain face in the flow direction are pinned:
 *  A_ii = 1, A_ij = 0, rhs = vlo or vhi. Also sets a linear initial guess.
 */
class DirichletExternalBC : public BoundaryCondition {
public:
    void applyBC(amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit, int nval,
                 const int* active_mask, const int* mask_lo, const int* mask_hi,
                 const amrex::Real* /*diff_coeff*/, const int* /*dc_lo*/, const int* /*dc_hi*/,
                 const int* bxlo, const int* bxhi, const int* domlo, const int* domhi,
                 const amrex::Real* /*dxinv*/, amrex::Real vlo, amrex::Real vhi,
                 int dir) const override {
        using namespace bc_detail;

        const int mask_nx = mask_hi[0] - mask_lo[0] + 1;
        const int mask_ny = mask_hi[1] - mask_lo[1] + 1;
        auto mask = [&](int i, int j, int k) -> int {
            return active_mask[(i - mask_lo[0]) + (j - mask_lo[1]) * mask_nx +
                               (k - mask_lo[2]) * mask_nx * mask_ny];
        };

        int m_idx = 0;
        for (int k = bxlo[2]; k <= bxhi[2]; ++k) {
            for (int j = bxlo[1]; j <= bxhi[1]; ++j) {
                for (int i = bxlo[0]; i <= bxhi[0]; ++i) {
                    if (mask(i, j, k) == CELL_INACTIVE) {
                        ++m_idx;
                        continue;
                    }

                    const int stn_start = NSTENCIL * m_idx;

                    // Check if on Dirichlet boundary face
                    bool on_dirichlet = false;
                    amrex::Real bc_val = 0.0;
                    if (dir == 0) {
                        if (i == domlo[0]) {
                            on_dirichlet = true;
                            bc_val = vlo;
                        } else if (i == domhi[0]) {
                            on_dirichlet = true;
                            bc_val = vhi;
                        }
                    } else if (dir == 1) {
                        if (j == domlo[1]) {
                            on_dirichlet = true;
                            bc_val = vlo;
                        } else if (j == domhi[1]) {
                            on_dirichlet = true;
                            bc_val = vhi;
                        }
                    } else if (dir == 2) {
                        if (k == domlo[2]) {
                            on_dirichlet = true;
                            bc_val = vlo;
                        } else if (k == domhi[2]) {
                            on_dirichlet = true;
                            bc_val = vhi;
                        }
                    }

                    if (on_dirichlet) {
                        for (int s = 0; s < NSTENCIL; ++s)
                            a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = bc_val;
                    }

                    // Linear initial guess along flow direction
                    amrex::Real domain_extent = static_cast<amrex::Real>(domhi[dir] - domlo[dir]);
                    amrex::Real coord = 0.0;
                    if (dir == 0)
                        coord = static_cast<amrex::Real>(i - domlo[0]);
                    else if (dir == 1)
                        coord = static_cast<amrex::Real>(j - domlo[1]);
                    else
                        coord = static_cast<amrex::Real>(k - domlo[2]);

                    amrex::Real factor =
                        (std::abs(domain_extent) < SMALL_REAL) ? 0.0 : 1.0 / domain_extent;
                    xinit[m_idx] = vlo + (vhi - vlo) * coord * factor;

                    ++m_idx;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// DirichletPhaseBoundaryBC — Dirichlet at pore-solid interface near boundary
// ---------------------------------------------------------------------------

/** @brief Applies Dirichlet BC at active cells adjacent to inactive cells near the domain boundary.
 *
 *  Instead of pinning cells on the domain face itself, this pins active cells that
 *  have an inactive (solid) neighbor in the flow direction at the inlet/outlet boundary
 *  region. This is the "soil branch" BC for materials where the pore space does not
 *  extend to the domain face.
 */
class DirichletPhaseBoundaryBC : public BoundaryCondition {
public:
    void applyBC(amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit, int nval,
                 const int* active_mask, const int* mask_lo, const int* mask_hi,
                 const amrex::Real* /*diff_coeff*/, const int* /*dc_lo*/, const int* /*dc_hi*/,
                 const int* bxlo, const int* bxhi, const int* domlo, const int* domhi,
                 const amrex::Real* /*dxinv*/, amrex::Real vlo, amrex::Real vhi,
                 int dir) const override {
        using namespace bc_detail;

        const int mask_nx = mask_hi[0] - mask_lo[0] + 1;
        const int mask_ny = mask_hi[1] - mask_lo[1] + 1;
        auto mask = [&](int i, int j, int k) -> int {
            return active_mask[(i - mask_lo[0]) + (j - mask_lo[1]) * mask_nx +
                               (k - mask_lo[2]) * mask_nx * mask_ny];
        };

        // Determine which neighbor offset to check for inactive cells
        // For low face: check the -dir neighbor; for high face: check the +dir neighbor
        int di_lo[3] = {0, 0, 0};
        int di_hi[3] = {0, 0, 0};
        di_lo[dir] = -1; // neighbor toward inlet
        di_hi[dir] = 1;  // neighbor toward outlet

        int m_idx = 0;
        for (int k = bxlo[2]; k <= bxhi[2]; ++k) {
            for (int j = bxlo[1]; j <= bxhi[1]; ++j) {
                for (int i = bxlo[0]; i <= bxhi[0]; ++i) {
                    if (mask(i, j, k) == CELL_INACTIVE) {
                        ++m_idx;
                        continue;
                    }

                    const int stn_start = NSTENCIL * m_idx;
                    bool apply_lo = false;
                    bool apply_hi = false;

                    // Near low face: active cell with inactive neighbor toward inlet
                    int coords[3] = {i, j, k};
                    if (coords[dir] == domlo[dir]) {
                        // On the domain face itself — always apply Dirichlet
                        apply_lo = true;
                    } else {
                        // Check if neighbor toward low face is inactive
                        int ni = i + di_lo[0];
                        int nj = j + di_lo[1];
                        int nk = k + di_lo[2];
                        if (mask(ni, nj, nk) == CELL_INACTIVE) {
                            // Check we're in the "near boundary" region (within a few cells)
                            // For the phase boundary BC, any active cell with an inactive
                            // neighbor in the flow direction near the low boundary counts
                            if (coords[dir] <= domlo[dir] + (domhi[dir] - domlo[dir]) / 4) {
                                apply_lo = true;
                            }
                        }
                    }

                    // Near high face: active cell with inactive neighbor toward outlet
                    if (coords[dir] == domhi[dir]) {
                        apply_hi = true;
                    } else {
                        int ni = i + di_hi[0];
                        int nj = j + di_hi[1];
                        int nk = k + di_hi[2];
                        if (mask(ni, nj, nk) == CELL_INACTIVE) {
                            if (coords[dir] >= domhi[dir] - (domhi[dir] - domlo[dir]) / 4) {
                                apply_hi = true;
                            }
                        }
                    }

                    if (apply_lo || apply_hi) {
                        amrex::Real bc_val = apply_lo ? vlo : vhi;
                        for (int s = 0; s < NSTENCIL; ++s)
                            a[stn_start + s] = 0.0;
                        a[stn_start + STN_C] = 1.0;
                        rhs[m_idx] = bc_val;
                    }

                    // Linear initial guess along flow direction
                    amrex::Real domain_extent = static_cast<amrex::Real>(domhi[dir] - domlo[dir]);
                    amrex::Real coord = 0.0;
                    if (dir == 0)
                        coord = static_cast<amrex::Real>(i - domlo[0]);
                    else if (dir == 1)
                        coord = static_cast<amrex::Real>(j - domlo[1]);
                    else
                        coord = static_cast<amrex::Real>(k - domlo[2]);

                    amrex::Real factor =
                        (std::abs(domain_extent) < SMALL_REAL) ? 0.0 : 1.0 / domain_extent;
                    xinit[m_idx] = vlo + (vhi - vlo) * coord * factor;

                    ++m_idx;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// NeumannBC — Zero-flux (no-op, already implicit in the stencil)
// ---------------------------------------------------------------------------

/** @brief Zero-flux Neumann BC — no stencil modification needed.
 *
 *  The default stencil already produces zero-flux at boundaries where neighbors
 *  are inactive (off-diagonal = 0). This class only sets the initial guess.
 */
class NeumannBC : public BoundaryCondition {
public:
    void applyBC(amrex::Real* /*a*/, amrex::Real* /*rhs*/, amrex::Real* /*xinit*/, int /*nval*/,
                 const int* /*active_mask*/, const int* /*mask_lo*/, const int* /*mask_hi*/,
                 const amrex::Real* /*diff_coeff*/, const int* /*dc_lo*/, const int* /*dc_hi*/,
                 const int* /*bxlo*/, const int* /*bxhi*/, const int* /*domlo*/,
                 const int* /*domhi*/, const amrex::Real* /*dxinv*/, amrex::Real /*vlo*/,
                 amrex::Real /*vhi*/, int /*dir*/) const override {
        // No-op: zero-flux is implicit in the stencil (inactive neighbors have zero coupling)
    }
};

// ---------------------------------------------------------------------------
// PeriodicBC — Periodic wrapping (HYPRE handles the stencil connectivity)
// ---------------------------------------------------------------------------

/** @brief Periodic BC — requires HYPRE periodic grid setup.
 *
 *  No stencil modification is needed because HYPRE's periodic grid handles the
 *  wrap-around connectivity. This class only signals that periodic grid setup
 *  is required via needsPeriodicGrid().
 */
class PeriodicBC : public BoundaryCondition {
public:
    void applyBC(amrex::Real* /*a*/, amrex::Real* /*rhs*/, amrex::Real* /*xinit*/, int /*nval*/,
                 const int* /*active_mask*/, const int* /*mask_lo*/, const int* /*mask_hi*/,
                 const amrex::Real* /*diff_coeff*/, const int* /*dc_lo*/, const int* /*dc_hi*/,
                 const int* /*bxlo*/, const int* /*bxhi*/, const int* /*domlo*/,
                 const int* /*domhi*/, const amrex::Real* /*dxinv*/, amrex::Real /*vlo*/,
                 amrex::Real /*vhi*/, int /*dir*/) const override {
        // No-op: HYPRE handles periodic connectivity at the grid level
    }

    bool needsPeriodicGrid() const override {
        return true;
    }
};

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/** @brief Create a BoundaryCondition object from a BCType enum.
 *  @param type The BC type to create.
 *  @return Unique pointer to the created BoundaryCondition.
 */
inline std::unique_ptr<BoundaryCondition> createBC(BCType type) {
    switch (type) {
    case BCType::DirichletExternal:
        return std::make_unique<DirichletExternalBC>();
    case BCType::DirichletPhaseBoundary:
        return std::make_unique<DirichletPhaseBoundaryBC>();
    case BCType::Neumann:
        return std::make_unique<NeumannBC>();
    case BCType::Periodic:
        return std::make_unique<PeriodicBC>();
    default:
        amrex::Abort("Unknown BCType in createBC");
        return nullptr;
    }
}

} // namespace OpenImpala

#endif // OPENIMPALA_BOUNDARY_CONDITION_H_
