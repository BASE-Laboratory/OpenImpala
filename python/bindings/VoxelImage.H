/** @file VoxelImage.H
 *  @brief Opaque container for AMReX grid objects, exposed to Python via pybind11.
 *
 *  Encapsulates the Geometry, BoxArray, DistributionMapping, and iMultiFab
 *  so that Python users never need to interact with AMReX types directly.
 *  This eliminates the pyamrex dependency entirely.
 */
#ifndef OPENIMPALA_VOXEL_IMAGE_H
#define OPENIMPALA_VOXEL_IMAGE_H

#include <memory>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>

namespace OpenImpala {

/** @brief Opaque handle that bundles AMReX grid infrastructure for a 3-D voxel image.
 *
 *  Created from a NumPy array (via pybind11) or from a native C++ reader.
 *  Passed by shared_ptr so pybind11 can safely prevent premature destruction.
 */
struct VoxelImage {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    std::shared_ptr<amrex::iMultiFab> mf;
};

} // namespace OpenImpala

#endif // OPENIMPALA_VOXEL_IMAGE_H
