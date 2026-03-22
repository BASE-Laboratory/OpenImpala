/** @file FloodFill.H
 *  @brief GPU-compatible parallel flood fill for 3D phase data on AMReX grids.
 *
 *  Provides a shared flood fill implementation used by PercolationCheck,
 *  TortuosityHypre, TortuosityMLMG, and ConnectedComponents.  The algorithm
 *  is a synchronous wavefront BFS: each iteration propagates the reachable
 *  set by one cell in each direction, with FillBoundary for inter-patch
 *  ghost exchange.
 *
 *  When AMReX is built with GPU support, the inner loop uses ParallelFor
 *  with AMREX_GPU_DEVICE lambdas and a device-side atomic flag for early
 *  termination.  On CPU builds, the same code falls back to threaded
 *  execution via AMReX's ParallelFor + OpenMP backend.
 */
#ifndef OPENIMPALA_FLOOD_FILL_H
#define OPENIMPALA_FLOOD_FILL_H

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

namespace OpenImpala {

/// Marker values used in reachability masks.
constexpr int FLOOD_INACTIVE = 0;
constexpr int FLOOD_ACTIVE = 1;

/** @brief GPU-compatible parallel flood fill on an iMultiFab.
 *
 *  Marks all cells of @p phaseID that are 6-connected to any seed point
 *  as FLOOD_ACTIVE in @p reachabilityMask.  Uses iterative wavefront
 *  expansion with FillBoundary for MPI ghost exchange.
 *
 *  The inner expansion loop uses amrex::ParallelFor, which maps to GPU
 *  kernels when AMReX is built with CUDA/HIP, and to OpenMP threads on
 *  CPU builds.  Early termination uses a device-compatible atomic flag.
 *
 *  @param[out] reachabilityMask  iMultiFab to fill (1 comp, >=1 ghost).
 *                                Set to @p label for reachable cells.
 *  @param[in]  phaseFab          Phase data (1 comp, >=1 ghost).
 *  @param[in]  phaseID           Phase ID to flood through.
 *  @param[in]  seedPoints        Initial seed cell coordinates.
 *  @param[in]  geom              AMReX Geometry for FillBoundary periodicity.
 *  @param[in]  verbose           Verbosity (0=silent, >1 prints iteration count).
 *  @param[in]  label             Integer label to write into reachable cells.
 *                                Default FLOOD_ACTIVE (1).  ConnectedComponents
 *                                passes a unique label per component.
 */
void parallelFloodFill(amrex::iMultiFab& reachabilityMask, const amrex::iMultiFab& phaseFab,
                       int phaseID, const amrex::Vector<amrex::IntVect>& seedPoints,
                       const amrex::Geometry& geom, int verbose = 0, int label = FLOOD_ACTIVE);

/** @brief Collect boundary seeds on inlet/outlet faces of a phase.
 *
 *  Scans the lo and hi faces of the domain in direction @p dir for cells
 *  matching @p phaseID, gathers them across MPI ranks via Allgatherv,
 *  and returns deduplicated seed vectors.
 *
 *  @param[in]  phaseFab      Phase data.
 *  @param[in]  phaseID       Phase ID to seed.
 *  @param[in]  dir           Flow direction (determines which faces to scan).
 *  @param[in]  geom          AMReX Geometry.
 *  @param[out] inletSeeds    Seeds on the low face (populated on return).
 *  @param[out] outletSeeds   Seeds on the high face (populated on return).
 */
void collectBoundarySeeds(const amrex::iMultiFab& phaseFab, int phaseID, int dir,
                          const amrex::Geometry& geom, amrex::Vector<amrex::IntVect>& inletSeeds,
                          amrex::Vector<amrex::IntVect>& outletSeeds);

} // namespace OpenImpala

#endif // OPENIMPALA_FLOOD_FILL_H
