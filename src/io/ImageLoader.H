/** @file ImageLoader.H
 *  @brief Loads 3D voxel images from multiple file formats and thresholds into phase data.
 *
 *  Consolidates file format detection, reader instantiation, and phase thresholding
 *  into a single utility. Supports TIFF, HDF5, and DAT formats (RAW files require
 *  RawReader directly due to needing explicit dimensions and data type). The output is always
 *  an amrex::iMultiFab containing integer phase IDs, ready for solver consumption.
 */
#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <string>
#include <filesystem>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

namespace OpenImpala {

/** @brief Result of loading and thresholding a 3D image file.
 *
 *  Contains all the AMReX data structures needed to proceed with
 *  physics calculations: geometry, grid decomposition, and phase field.
 */
struct ImageData {
    amrex::Box domain_box;         ///< Domain bounding box
    amrex::Geometry geom;          ///< Geometry (periodic by default)
    amrex::BoxArray ba;            ///< Grid decomposition
    amrex::DistributionMapping dm; ///< MPI distribution
    amrex::iMultiFab mf_phase;     ///< Phase field (1 component, 1 ghost cell)
};

/** @brief Loads a 3D image from file and thresholds it into a binary phase field.
 *
 *  Detects the file format from the extension (.tif/.tiff, .h5/.hdf5, .dat),
 *  reads the image using the appropriate reader class, and applies a threshold
 *  to produce a binary phase field (0/1).
 *
 *  The resulting geometry uses periodic boundary conditions, consistent with
 *  the homogenization cell-problem formulation.
 *
 *  @param filepath       Full path to the input image file.
 *  @param hdf5_dataset   HDF5 dataset name (only used for .h5/.hdf5 files).
 *  @param threshold_val  Threshold value: voxels > threshold → phase 1, else → phase 0.
 *  @param box_size       Maximum grid box size for AMReX BoxArray decomposition.
 *  @param verbose        Verbosity level (0=silent, 1+=progress output).
 *  @return ImageData struct containing geometry, grid, and phase field.
 *
 *  @throws std::runtime_error if the file cannot be read or has unsupported format.
 */
ImageData loadImage(const std::filesystem::path& filepath, const std::string& hdf5_dataset,
                    amrex::Real threshold_val, int box_size, int verbose = 1);

} // namespace OpenImpala

#endif // IMAGE_LOADER_H
