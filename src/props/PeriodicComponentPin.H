/** @file PeriodicComponentPin.H
 *  @brief Periodic 6-connected active-component labelling with single-cell nullspace pinning.
 *
 *  Assigns CELL_INACTIVE (0) to non-conducting voxels, CELL_ACTIVE (1) to active voxels,
 *  and CELL_PINNED (2) to the unique minimum-index voxel of each periodic 6-connected
 *  active component so periodic cell-problem solvers (HYPRE and matrix-free MLMG)
 *  eliminate per-component constant nullspaces cleanly across CPU, MPI, and CUDA.
 */
#ifndef OPENIMPALA_PERIODIC_COMPONENT_PIN_H_
#define OPENIMPALA_PERIODIC_COMPONENT_PIN_H_

#include "EffDiffFillMtx.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>

namespace OpenImpala {

/** @brief Classify cells into CELL_INACTIVE (0), CELL_ACTIVE (1), and CELL_PINNED (2).
 *
 *  Uses a fast single-rank O(N) periodic BFS on CPU when running on 1 MPI rank without
 *  GPU offload, and falls back to parallel MPI + CUDA min-label propagation otherwise.
 *
 *  @param geom            Periodic AMReX Geometry defining domain bounds and periodicity.
 *  @param ba              BoxArray of the domain.
 *  @param dm              DistributionMapping of the domain.
 *  @param mf_active_mask  Binary active mask (1 = active, 0 = inactive).
 *  @param mf_cell_type    Output cell-type iMultiFab (populated with 0, 1, or 2 + periodic ghosts).
 *  @param mask_comp       Component index to read from *mf_active_mask* (default 0).
 *  @return Total number of periodic 6-connected active components discovered.
 */
int buildPeriodicCellTypeMask(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm,
                              const amrex::iMultiFab& mf_active_mask,
                              amrex::iMultiFab& mf_cell_type, int mask_comp = 0);

} // namespace OpenImpala

#endif // OPENIMPALA_PERIODIC_COMPONENT_PIN_H_
