/** @file ImageLoader.cpp
 *  @brief Implementation of the unified image loading and thresholding utility.
 */

#include "ImageLoader.H"
#include "TiffReader.H"
#include "DatReader.H"
#include "HDF5Reader.H"

#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

#include <algorithm>
#include <stdexcept>

namespace OpenImpala {

ImageData loadImage(const std::filesystem::path& filepath, const std::string& hdf5_dataset,
                    amrex::Real threshold_val, int box_size, int verbose) {
    ImageData result;

    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("Input file does not exist: " + filepath.string());
    }

    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Reading image data from: " << filepath.string() << std::endl;
    }

    // Detect file format from extension
    std::string ext;
    if (filepath.has_extension()) {
        ext = filepath.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    } else {
        throw std::runtime_error("File has no extension: " + filepath.string());
    }

    // Phase mapping: voxels > threshold → 1, else → 0
    const int phase_active = 1;
    const int phase_inactive = 0;

    if (ext == ".tif" || ext == ".tiff") {
        TiffReader reader(filepath.string());
        if (!reader.isRead()) {
            throw std::runtime_error("TiffReader failed to read metadata.");
        }
        result.domain_box = reader.box();
        result.ba.define(result.domain_box);
        result.ba.maxSize(box_size);
        result.dm.define(result.ba);
        result.mf_phase.define(result.ba, result.dm, 1, 1);
        amrex::iMultiFab mf_temp(result.ba, result.dm, 1, 0);
        reader.threshold(threshold_val, phase_active, phase_inactive, mf_temp);
        amrex::Copy(result.mf_phase, mf_temp, 0, 0, 1, 0);

    } else if (ext == ".dat") {
        DatReader reader(filepath.string());
        if (!reader.isRead()) {
            throw std::runtime_error("DatReader failed to read metadata.");
        }
        result.domain_box = reader.box();
        result.ba.define(result.domain_box);
        result.ba.maxSize(box_size);
        result.dm.define(result.ba);
        result.mf_phase.define(result.ba, result.dm, 1, 1);
        amrex::iMultiFab mf_temp(result.ba, result.dm, 1, 0);
        reader.threshold(static_cast<DatReader::DataType>(threshold_val), phase_active,
                         phase_inactive, mf_temp);
        amrex::Copy(result.mf_phase, mf_temp, 0, 0, 1, 0);

    } else if (ext == ".h5" || ext == ".hdf5") {
        HDF5Reader reader(filepath.string(), hdf5_dataset);
        if (!reader.isRead()) {
            throw std::runtime_error("HDF5Reader failed to read metadata.");
        }
        result.domain_box = reader.box();
        result.ba.define(result.domain_box);
        result.ba.maxSize(box_size);
        result.dm.define(result.ba);
        result.mf_phase.define(result.ba, result.dm, 1, 1);
        amrex::iMultiFab mf_temp(result.ba, result.dm, 1, 0);
        reader.threshold(threshold_val, phase_active, phase_inactive, mf_temp);
        amrex::Copy(result.mf_phase, mf_temp, 0, 0, 1, 0);

    } else {
        throw std::runtime_error("Unsupported file extension: " + ext +
                                 ". Supported: .tif, .tiff, .dat, .h5, .hdf5");
    }

    // Set up periodic geometry
    amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                      {AMREX_D_DECL(amrex::Real(result.domain_box.length(0)),
                                    amrex::Real(result.domain_box.length(1)),
                                    amrex::Real(result.domain_box.length(2)))});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic = {AMREX_D_DECL(1, 1, 1)};
    result.geom.define(result.domain_box, &rb, 0, is_periodic.data());
    result.mf_phase.FillBoundary(result.geom.periodicity());

    return result;
}

} // namespace OpenImpala
