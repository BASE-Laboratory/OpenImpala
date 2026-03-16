/** @file ParticleSizeDistribution.H
 *  @brief Computes particle size distribution from connected component labeling.
 */
#ifndef OPENIMPALA_PARTICLE_SIZE_DISTRIBUTION_H
#define OPENIMPALA_PARTICLE_SIZE_DISTRIBUTION_H

#include "ConnectedComponents.H"

#include <AMReX_REAL.H>

#include <vector>

namespace OpenImpala {

/**
 * @brief Computes equivalent sphere radii from connected component volumes.
 *
 * For each connected component with volume V (in voxels), the equivalent
 * sphere radius is R = (3V / (4*pi))^(1/3). This is the standard
 * "equivalent sphere diameter" approach used in materials science.
 */
class ParticleSizeDistribution {
public:
    /**
     * @brief Construct PSD from connected components.
     *
     * @param ccl ConnectedComponents result.
     */
    explicit ParticleSizeDistribution(const ConnectedComponents& ccl);

    /** @brief Mean equivalent sphere radius (in voxel units). */
    amrex::Real meanRadius() const {
        return m_mean_radius;
    }

    /** @brief Number of particles (connected components). */
    int numParticles() const {
        return static_cast<int>(m_radii.size());
    }

    /** @brief Equivalent sphere radius for each component (in voxel units). */
    const std::vector<amrex::Real>& radii() const {
        return m_radii;
    }

    /** @brief Volume (cell count) of each component. */
    const std::vector<long long>& volumes() const {
        return m_volumes;
    }

private:
    std::vector<amrex::Real> m_radii;
    std::vector<long long> m_volumes;
    amrex::Real m_mean_radius = 0.0;
};

} // namespace OpenImpala

#endif // OPENIMPALA_PARTICLE_SIZE_DISTRIBUTION_H
