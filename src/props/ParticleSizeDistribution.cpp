#include "ParticleSizeDistribution.H"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OpenImpala {

ParticleSizeDistribution::ParticleSizeDistribution(const ConnectedComponents& ccl) {
    m_volumes = ccl.componentVolumes();
    int n = static_cast<int>(m_volumes.size());
    m_radii.resize(n);

    amrex::Real sum_radius = 0.0;
    for (int i = 0; i < n; ++i) {
        amrex::Real vol = static_cast<amrex::Real>(m_volumes[i]);
        // Equivalent sphere radius: R = (3V / (4*pi))^(1/3)
        m_radii[i] = std::cbrt(3.0 * vol / (4.0 * M_PI));
        sum_radius += m_radii[i];
    }

    m_mean_radius = (n > 0) ? sum_radius / static_cast<amrex::Real>(n) : 0.0;
}

} // namespace OpenImpala
