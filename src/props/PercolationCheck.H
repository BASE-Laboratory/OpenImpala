#ifndef OPENIMPALA_PERCOLATION_CHECK_H
#define OPENIMPALA_PERCOLATION_CHECK_H

#include "Tortuosity.H" // For OpenImpala::Direction

#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>
#include <AMReX_REAL.H>

#include <string>

namespace OpenImpala {

/**
 * @brief Checks whether a phase percolates across the domain in a given direction.
 *
 * Uses a parallel flood fill algorithm (6-connectivity) to determine if a connected
 * path exists from the inlet face (low boundary) to the outlet face (high boundary)
 * in the specified direction. This is a lightweight check that does not require
 * any solver setup (no HYPRE dependency).
 */
class PercolationCheck {
public:
    /**
     * @brief Construct a percolation checker.
     *
     * @param geom AMReX Geometry (should be non-periodic in the check direction).
     * @param ba BoxArray for the domain decomposition.
     * @param dm DistributionMapping for processor assignments.
     * @param mf_phase iMultiFab containing phase IDs (>=1 ghost cell required).
     * @param phase_id The phase ID to check percolation for.
     * @param dir Direction to check (X, Y, or Z).
     * @param verbose Verbosity level (0=silent, 1=basic, 2=detailed). Default 0.
     */
    PercolationCheck(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                     int phase_id, OpenImpala::Direction dir, int verbose = 0);

    /** @brief Returns true if the phase percolates in the specified direction. */
    bool percolates() const { return m_percolates; }

    /** @brief Returns the volume fraction of the percolating (active) subset of the phase. */
    amrex::Real activeVolumeFraction() const { return m_active_vf; }

    /** @brief Returns a human-readable direction string ("X", "Y", or "Z"). */
    static std::string directionString(OpenImpala::Direction dir);

private:
    void run(const amrex::iMultiFab& mf_phase, int phase_id, OpenImpala::Direction dir);

    void parallelFloodFill(amrex::iMultiFab& reachabilityMask, const amrex::iMultiFab& phaseFab,
                           int phaseID, const amrex::Vector<amrex::IntVect>& seedPoints);

    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    int m_verbose;

    bool m_percolates = false;
    amrex::Real m_active_vf = 0.0;
};

} // namespace OpenImpala

#endif // OPENIMPALA_PERCOLATION_CHECK_H
