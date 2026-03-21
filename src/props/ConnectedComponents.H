/** @file ConnectedComponents.H
 *  @brief Parallel connected component labeling for phase data via iterative flood-fill.
 */
#ifndef OPENIMPALA_CONNECTED_COMPONENTS_H
#define OPENIMPALA_CONNECTED_COMPONENTS_H

#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <vector>

namespace OpenImpala {

/**
 * @brief Labels connected components of a phase using parallel flood-fill.
 *
 * Each contiguous region of the target phase (6-connectivity) is assigned
 * a unique positive integer label. Cells not belonging to the target phase
 * are labeled 0.
 *
 * The algorithm uses iterative global flood-fill (similar to PercolationCheck)
 * with sequential seed discovery: each unlabeled cell found globally is used
 * as a seed to flood-fill one component at a time.
 *
 * @note Requires >= 1 ghost cell in the input iMultiFab.
 */
class ConnectedComponents {
public:
    /**
     * @brief Construct and run connected component labeling.
     *
     * @param geom AMReX Geometry.
     * @param ba BoxArray for domain decomposition.
     * @param dm DistributionMapping.
     * @param mf_phase iMultiFab containing phase IDs (>= 1 ghost cell).
     * @param phase_id Phase to label components for.
     * @param verbose Verbosity level (default 0).
     */
    ConnectedComponents(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                        const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                        int phase_id, int verbose = 0);

    /** @brief Number of connected components found. */
    int numComponents() const {
        return m_num_components;
    }

    /** @brief Volume (cell count) of each component, indexed by label (1-based). */
    const std::vector<long long>& componentVolumes() const {
        return m_volumes;
    }

    /** @brief The label field (each cell labeled 0 or 1..N). */
    const amrex::iMultiFab& labelField() const {
        return m_labels;
    }

private:
    void run(const amrex::iMultiFab& mf_phase, int phase_id);

    amrex::IntVect findNextUnlabeled(const amrex::iMultiFab& labelMF,
                                     const amrex::iMultiFab& phaseFab, int phaseID) const;

    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;
    int m_verbose;

    amrex::iMultiFab m_labels;
    int m_num_components = 0;
    std::vector<long long> m_volumes;
};

} // namespace OpenImpala

#endif // OPENIMPALA_CONNECTED_COMPONENTS_H
