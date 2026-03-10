/** @file io.cpp
 *  @brief pybind11 bindings for OpenImpala I/O readers.
 *
 *  Readers produce VoxelImage handles natively — no pyamrex dependency.
 */

#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_RealBox.H>
#include <AMReX_iMultiFab.H>

#include "TiffReader.H"
#include "HDF5Reader.H"
#include "RawReader.H"
#include "DatReader.H"
#include "VoxelImage.H"

namespace py = pybind11;
using namespace OpenImpala;

/** @brief Build a VoxelImage from a reader's Box, then run threshold to fill it. */
template <typename ReaderT, typename ThreshT>
static std::shared_ptr<VoxelImage>
reader_to_voxelimage(const ReaderT& reader, ThreshT threshold_value, int max_grid_size) {
    auto img = std::make_shared<VoxelImage>();

    const amrex::Box& box = reader.box();
    img->ba.define(box);
    img->ba.maxSize(max_grid_size);
    img->dm.define(img->ba);
    img->mf = std::make_shared<amrex::iMultiFab>(img->ba, img->dm, 1, 1);

    int nx = box.length(0);
    int ny = box.length(1);
    int nz = box.length(2);
    amrex::RealBox rb({0.0, 0.0, 0.0},
                      {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{0, 0, 0};
    img->geom.define(box, &rb, amrex::CoordSys::cartesian, is_periodic.data());

    reader.threshold(threshold_value, *(img->mf));
    return img;
}

/** @brief Build a VoxelImage with custom threshold output values. */
template <typename ReaderT, typename ThreshT>
static std::shared_ptr<VoxelImage>
reader_to_voxelimage_custom(const ReaderT& reader, ThreshT threshold_value, int value_if_true,
                            int value_if_false, int max_grid_size) {
    auto img = std::make_shared<VoxelImage>();

    const amrex::Box& box = reader.box();
    img->ba.define(box);
    img->ba.maxSize(max_grid_size);
    img->dm.define(img->ba);
    img->mf = std::make_shared<amrex::iMultiFab>(img->ba, img->dm, 1, 1);

    int nx = box.length(0);
    int ny = box.length(1);
    int nz = box.length(2);
    amrex::RealBox rb({0.0, 0.0, 0.0},
                      {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)});
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{0, 0, 0};
    img->geom.define(box, &rb, amrex::CoordSys::cartesian, is_periodic.data());

    reader.threshold(threshold_value, value_if_true, value_if_false, *(img->mf));
    return img;
}

void init_io(py::module_& m) {
    // =========================================================================
    // TiffReader
    // =========================================================================
    py::class_<TiffReader>(
        m, "TiffReader",
        "Reads 3-D image data from single/multi-directory TIFFs or file sequences.")

        .def(py::init<>(),
             "Create an empty reader.  Call read_file() or read_file_sequence() next.")

        .def(py::init<const std::string&>(), py::arg("filename"),
             "Open a single TIFF file and read its metadata (no pixel data yet).")

        .def(py::init<const std::string&, int, int, int, const std::string&>(),
             py::arg("base_pattern"), py::arg("num_files"), py::arg("start_index") = 0,
             py::arg("digits") = 1, py::arg("suffix") = ".tif",
             "Open a numbered TIFF sequence and read metadata from the first file.")

        .def("read_file", &TiffReader::readFile, py::arg("filename"),
             "Read metadata from a single TIFF (clears previous state).")

        .def("read_file_sequence", &TiffReader::readFileSequence, py::arg("base_pattern"),
             py::arg("num_files"), py::arg("start_index") = 0, py::arg("digits") = 1,
             py::arg("suffix") = ".tif",
             "Read metadata from a TIFF sequence (clears previous state).")

        // Two-arg threshold → VoxelImage
        .def(
            "threshold",
            [](const TiffReader& self, double raw_threshold, int max_grid_size) {
                return reader_to_voxelimage(self, raw_threshold, max_grid_size);
            },
            py::arg("raw_threshold"), py::arg("max_grid_size") = 32,
            "Threshold the image and return a VoxelImage (1 where pixel > threshold, else 0).")

        // Four-arg threshold → VoxelImage
        .def(
            "threshold",
            [](const TiffReader& self, double raw_threshold, int value_if_true, int value_if_false,
               int max_grid_size) {
                return reader_to_voxelimage_custom(self, raw_threshold, value_if_true,
                                                   value_if_false, max_grid_size);
            },
            py::arg("raw_threshold"), py::arg("value_if_true"), py::arg("value_if_false"),
            py::arg("max_grid_size") = 32,
            "Threshold the image with custom output values and return a VoxelImage.")

        // Metadata properties
        .def_property_readonly("box", &TiffReader::box)
        .def_property_readonly("width", &TiffReader::width)
        .def_property_readonly("height", &TiffReader::height)
        .def_property_readonly("depth", &TiffReader::depth)
        .def_property_readonly("is_read", &TiffReader::isRead)
        .def_property_readonly("bits_per_sample", &TiffReader::bitsPerSample)
        .def_property_readonly("sample_format", &TiffReader::sampleFormat)
        .def_property_readonly("samples_per_pixel", &TiffReader::samplesPerPixel)
        .def_property_readonly("fill_order", &TiffReader::getFillOrder)

        .def("__repr__", [](const TiffReader& r) {
            if (!r.isRead())
                return std::string("<TiffReader (no file loaded)>");
            return "<TiffReader " + std::to_string(r.width()) + "x" + std::to_string(r.height()) +
                   "x" + std::to_string(r.depth()) + ">";
        });

    // =========================================================================
    // HDF5Reader
    // =========================================================================
    py::class_<HDF5Reader>(m, "HDF5Reader", "Reads 3-D datasets from HDF5 files.")

        .def(py::init<>())

        .def(py::init<const std::string&, const std::string&>(), py::arg("filename"),
             py::arg("hdf5dataset"),
             "Open an HDF5 file and read metadata for the specified dataset.")

        .def("read_file", &HDF5Reader::readFile, py::arg("filename"), py::arg("hdf5dataset"),
             "Read metadata from an HDF5 file + dataset (clears previous state).")

        .def(
            "threshold",
            [](const HDF5Reader& self, double raw_threshold, int max_grid_size) {
                return reader_to_voxelimage(self, raw_threshold, max_grid_size);
            },
            py::arg("raw_threshold"), py::arg("max_grid_size") = 32)

        .def(
            "threshold",
            [](const HDF5Reader& self, double raw_threshold, int value_if_true, int value_if_false,
               int max_grid_size) {
                return reader_to_voxelimage_custom(self, raw_threshold, value_if_true,
                                                   value_if_false, max_grid_size);
            },
            py::arg("raw_threshold"), py::arg("value_if_true"), py::arg("value_if_false"),
            py::arg("max_grid_size") = 32)

        .def_property_readonly("box", &HDF5Reader::box)
        .def_property_readonly("width", &HDF5Reader::width)
        .def_property_readonly("height", &HDF5Reader::height)
        .def_property_readonly("depth", &HDF5Reader::depth)
        .def_property_readonly("is_read", &HDF5Reader::isRead)

        .def("get_attribute", &HDF5Reader::getAttribute, py::arg("attr_name"),
             "Read a string attribute from the HDF5 dataset.")

        .def("get_all_attributes", &HDF5Reader::getAllAttributes,
             "Read all attributes from the HDF5 dataset as a dict.")

        .def("__repr__", [](const HDF5Reader& r) {
            if (!r.isRead())
                return std::string("<HDF5Reader (no file loaded)>");
            return "<HDF5Reader " + std::to_string(r.width()) + "x" + std::to_string(r.height()) +
                   "x" + std::to_string(r.depth()) + ">";
        });

    // =========================================================================
    // RawReader
    // =========================================================================
    py::class_<RawReader>(m, "RawReader", "Reads 3-D voxel data from flat binary (RAW) files.")

        .def(py::init<>())

        .def(py::init<const std::string&, int, int, int, RawDataType>(), py::arg("filename"),
             py::arg("width"), py::arg("height"), py::arg("depth"), py::arg("data_type"),
             "Open a raw binary file with externally supplied dimensions and data type.")

        .def("read_file", &RawReader::readFile, py::arg("filename"), py::arg("width"),
             py::arg("height"), py::arg("depth"), py::arg("data_type"))

        .def(
            "threshold",
            [](const RawReader& self, double threshold_value, int max_grid_size) {
                return reader_to_voxelimage(self, threshold_value, max_grid_size);
            },
            py::arg("threshold_value"), py::arg("max_grid_size") = 32)

        .def(
            "threshold",
            [](const RawReader& self, double threshold_value, int value_if_true, int value_if_false,
               int max_grid_size) {
                return reader_to_voxelimage_custom(self, threshold_value, value_if_true,
                                                   value_if_false, max_grid_size);
            },
            py::arg("threshold_value"), py::arg("value_if_true"), py::arg("value_if_false"),
            py::arg("max_grid_size") = 32)

        .def_property_readonly("box", &RawReader::box)
        .def_property_readonly("width", &RawReader::width)
        .def_property_readonly("height", &RawReader::height)
        .def_property_readonly("depth", &RawReader::depth)
        .def_property_readonly("is_read", &RawReader::isRead)
        .def_property_readonly("data_type", &RawReader::getDataType)

        .def("get_value", &RawReader::getValue, py::arg("i"), py::arg("j"), py::arg("k"),
             "Retrieve a single voxel value (converted to float).")

        .def("__repr__", [](const RawReader& r) {
            if (!r.isRead())
                return std::string("<RawReader (no file loaded)>");
            return "<RawReader " + std::to_string(r.width()) + "x" + std::to_string(r.height()) +
                   "x" + std::to_string(r.depth()) + ">";
        });

    // =========================================================================
    // DatReader
    // =========================================================================
    py::class_<DatReader>(m, "DatReader", "Reads 3-D image data from legacy DAT binary files.")

        .def(py::init<>())

        .def(py::init<const std::string&>(), py::arg("filename"))

        .def("read_file", &DatReader::readFile, py::arg("filename"))

        .def(
            "threshold",
            [](const DatReader& self, DatReader::DataType raw_threshold, int max_grid_size) {
                return reader_to_voxelimage(self, raw_threshold, max_grid_size);
            },
            py::arg("raw_threshold"), py::arg("max_grid_size") = 32)

        .def(
            "threshold",
            [](const DatReader& self, DatReader::DataType raw_threshold, int value_if_true,
               int value_if_false, int max_grid_size) {
                return reader_to_voxelimage_custom(self, raw_threshold, value_if_true,
                                                   value_if_false, max_grid_size);
            },
            py::arg("raw_threshold"), py::arg("value_if_true"), py::arg("value_if_false"),
            py::arg("max_grid_size") = 32)

        .def_property_readonly("box", &DatReader::box)
        .def_property_readonly("width", &DatReader::width)
        .def_property_readonly("height", &DatReader::height)
        .def_property_readonly("depth", &DatReader::depth)
        .def_property_readonly("is_read", &DatReader::isRead)

        .def("get_raw_value", &DatReader::getRawValue, py::arg("i"), py::arg("j"), py::arg("k"))

        .def("__repr__", [](const DatReader& r) {
            if (!r.isRead())
                return std::string("<DatReader (no file loaded)>");
            return "<DatReader " + std::to_string(r.width()) + "x" + std::to_string(r.height()) +
                   "x" + std::to_string(r.depth()) + ">";
        });
}
