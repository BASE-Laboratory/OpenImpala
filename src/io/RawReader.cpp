/** @file RawReader.cpp
 *  @brief Collective MPI-IO subarray and slab-streaming implementation of RawReader.
 */

#include "RawReader.H"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <AMReX_Box.H>
#include <AMReX_Extension.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Print.H>
#include <AMReX_ccse-mpi.H>
#include <AMReX_iMultiFab.H>

namespace OpenImpala {

namespace {

bool determineHostEndianness() {
    std::uint16_t test_value = 1;
    unsigned char first_byte = 0;
    std::memcpy(&first_byte, &test_value, 1);
    return (first_byte == 1);
}

const bool host_is_little_endian = determineHostEndianness();

template <typename T>
AMREX_FORCE_INLINE T reconstructValue(const RawReader::ByteType* src_ptr,
                                      const size_t bytes_to_read, const bool needs_swap) {
    static_assert(sizeof(T) <= 8, "reconstructValue: Type size mismatch or too large");
    RawReader::ByteType temp_buffer[8];

    if (needs_swap) {
        for (size_t b = 0; b < bytes_to_read; ++b) {
            temp_buffer[b] = src_ptr[bytes_to_read - 1 - b];
        }
    } else {
        std::memcpy(temp_buffer, src_ptr, bytes_to_read);
    }

    T val;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bytes_to_read == sizeof(T),
                                     "reconstructValue: bytes_to_read does not match sizeof(T)");
    std::memcpy(&val, temp_buffer, sizeof(T));
    return val;
}

AMREX_FORCE_INLINE double decodeVoxelAsDouble(const RawReader::ByteType* src_ptr, RawDataType dtype,
                                              bool needs_swap) {
    switch (dtype) {
    case RawDataType::UINT8:
        return static_cast<double>(reconstructValue<uint8_t>(src_ptr, 1, needs_swap));
    case RawDataType::INT8:
        return static_cast<double>(reconstructValue<int8_t>(src_ptr, 1, needs_swap));
    case RawDataType::UINT16_LE:
    case RawDataType::UINT16_BE:
        return static_cast<double>(reconstructValue<uint16_t>(src_ptr, 2, needs_swap));
    case RawDataType::INT16_LE:
    case RawDataType::INT16_BE:
        return static_cast<double>(reconstructValue<int16_t>(src_ptr, 2, needs_swap));
    case RawDataType::UINT32_LE:
    case RawDataType::UINT32_BE:
        return static_cast<double>(reconstructValue<uint32_t>(src_ptr, 4, needs_swap));
    case RawDataType::INT32_LE:
    case RawDataType::INT32_BE:
        return static_cast<double>(reconstructValue<int32_t>(src_ptr, 4, needs_swap));
    case RawDataType::FLOAT32_LE:
    case RawDataType::FLOAT32_BE:
        static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
                      "Require IEEE 754 32-bit float");
        return static_cast<double>(reconstructValue<float>(src_ptr, 4, needs_swap));
    case RawDataType::FLOAT64_LE:
    case RawDataType::FLOAT64_BE:
        static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
                      "Require IEEE 754 64-bit double");
        return reconstructValue<double>(src_ptr, 8, needs_swap);
    case RawDataType::UNKNOWN:
    default:
        amrex::Abort("[RawReader] Unknown or unsupported data type encountered.");
        return 0.0;
    }
}

struct PatchSlab {
    amrex::IArrayBox* fab = nullptr;
    amrex::Box box;
};

} // namespace

bool RawReader::isHostLittleEndian() const {
    return host_is_little_endian;
}

bool RawReader::needsByteSwap() const {
    const size_t bytes_per_voxel = getBytesPerVoxel();
    if (bytes_per_voxel <= 1) {
        return false;
    }
    bool data_is_le = host_is_little_endian;
    switch (m_data_type) {
    case RawDataType::INT16_LE:
    case RawDataType::UINT16_LE:
    case RawDataType::INT32_LE:
    case RawDataType::UINT32_LE:
    case RawDataType::FLOAT32_LE:
    case RawDataType::FLOAT64_LE:
        data_is_le = true;
        break;
    case RawDataType::INT16_BE:
    case RawDataType::UINT16_BE:
    case RawDataType::INT32_BE:
    case RawDataType::UINT32_BE:
    case RawDataType::FLOAT32_BE:
    case RawDataType::FLOAT64_BE:
        data_is_le = false;
        break;
    default:
        break;
    }
    return (data_is_le != host_is_little_endian);
}

RawDataType RawReader::parseDataType(const std::string& type_str) {
    std::string s = type_str;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (s == "UINT8" || s == "U1")
        return RawDataType::UINT8;
    if (s == "INT8" || s == "I1")
        return RawDataType::INT8;
    if (s == "UINT16" || s == "UINT16_LE" || s == "<U2")
        return RawDataType::UINT16_LE;
    if (s == "UINT16_BE" || s == ">U2")
        return RawDataType::UINT16_BE;
    if (s == "INT16" || s == "INT16_LE" || s == "<I2")
        return RawDataType::INT16_LE;
    if (s == "INT16_BE" || s == ">I2")
        return RawDataType::INT16_BE;
    if (s == "UINT32" || s == "UINT32_LE" || s == "<U4")
        return RawDataType::UINT32_LE;
    if (s == "UINT32_BE" || s == ">U4")
        return RawDataType::UINT32_BE;
    if (s == "INT32" || s == "INT32_LE" || s == "<I4")
        return RawDataType::INT32_LE;
    if (s == "INT32_BE" || s == ">I4")
        return RawDataType::INT32_BE;
    if (s == "FLOAT32" || s == "FLOAT32_LE" || s == "<F4")
        return RawDataType::FLOAT32_LE;
    if (s == "FLOAT32_BE" || s == ">F4")
        return RawDataType::FLOAT32_BE;
    if (s == "FLOAT64" || s == "FLOAT64_LE" || s == "<F8")
        return RawDataType::FLOAT64_LE;
    if (s == "FLOAT64_BE" || s == ">F8")
        return RawDataType::FLOAT64_BE;
    return RawDataType::UNKNOWN;
}

RawReader::RawReader() = default;

RawReader::RawReader(const std::string& filename, int width, int height, int depth,
                     RawDataType data_type)
    : RawReader() {
    if (!readFile(filename, width, height, depth, data_type)) {
        throw std::runtime_error("RawReader: Failed to read raw file '" + filename +
                                 "' during construction.");
    }
}

bool RawReader::readFile(const std::string& filename, int width, int height, int depth,
                         RawDataType data_type) {
    m_raw_bytes.clear();
    m_raw_bytes.shrink_to_fit();
    m_buffer_loaded = false;
    m_is_read = false;
    m_expected_bytes = 0;
    m_filename = filename;
    m_width = width;
    m_height = height;
    m_depth = depth;
    m_data_type = data_type;

    if (m_filename.empty()) {
        amrex::Print() << "Error: [RawReader] No filename provided.\n";
        return false;
    }
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0) {
        amrex::Print() << "Error: [RawReader] Invalid dimensions provided (W=" << m_width
                       << ", H=" << m_height << ", D=" << m_depth << "). Must be positive.\n";
        return false;
    }
    if (m_data_type == RawDataType::UNKNOWN) {
        amrex::Print() << "Error: [RawReader] Data type cannot be UNKNOWN.\n";
        return false;
    }

    const bool success = validateRawFileMetadata();
    m_is_read = success;

    if (success) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Successfully validated Raw File: " << m_filename
                           << " (Dimensions: " << m_width << "x" << m_height << "x" << m_depth
                           << ", Type: " << static_cast<int>(m_data_type)
                           << ", HostLE: " << isHostLittleEndian() << ")\n";
        }
    } else {
        m_width = m_height = m_depth = 0;
        m_expected_bytes = 0;
        m_data_type = RawDataType::UNKNOWN;
    }

    return success;
}

bool RawReader::validateRawFileMetadata() {
    const size_t bytes_per_voxel = getBytesPerVoxel();
    if (bytes_per_voxel == 0) {
        amrex::Print()
            << "Internal Error: [RawReader] Invalid data type resulted in zero bytes per voxel.\n";
        return false;
    }

    const long long total_voxels = static_cast<long long>(m_width) * m_height * m_depth;
    const long long expected_bytes_ll = total_voxels * static_cast<long long>(bytes_per_voxel);

    if (expected_bytes_ll <= 0) {
        amrex::Print() << "Error: [RawReader] Calculated data size (" << expected_bytes_ll
                       << ") is zero or negative.\n";
        return false;
    }

    if (static_cast<unsigned long long>(expected_bytes_ll) > std::numeric_limits<size_t>::max()) {
        amrex::Print() << "Error: [RawReader] Calculated data size (" << expected_bytes_ll
                       << ") exceeds maximum value representable by size_t ("
                       << std::numeric_limits<size_t>::max() << ").\n";
        return false;
    }
    const size_t expected_bytes = static_cast<size_t>(expected_bytes_ll);

    std::ifstream file(m_filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        amrex::Print() << "Error: [RawReader] Could not open file: " << m_filename << "\n";
        return false;
    }

    const std::streamsize file_size = file.tellg();
    file.close();

    if (file_size < 0) {
        amrex::Print()
            << "Error: [RawReader] Could not determine file size or file size is negative for: "
            << m_filename << "\n";
        return false;
    }
    if (static_cast<unsigned long long>(file_size) < expected_bytes) {
        amrex::Print() << "Error: [RawReader] File size (" << file_size
                       << ") is smaller than expected data size (" << expected_bytes
                       << ") for: " << m_filename << "\n";
        return false;
    }
    if (static_cast<size_t>(file_size) != expected_bytes) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Warning: [RawReader] File size (" << file_size
                           << ") does not exactly match expected data size (" << expected_bytes
                           << ") for: " << m_filename << ". Reading only expected bytes.\n";
        }
    }

    m_expected_bytes = expected_bytes;
    return true;
}

bool RawReader::ensureHostBufferLoaded() const {
    if (m_buffer_loaded) {
        return true;
    }
    if (!m_is_read || m_expected_bytes == 0) {
        return false;
    }
    std::ifstream file(m_filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    m_raw_bytes.resize(m_expected_bytes);
    if (!file.read(reinterpret_cast<char*>(m_raw_bytes.data()),
                   static_cast<std::streamsize>(m_expected_bytes))) {
        m_raw_bytes.clear();
        return false;
    }
    m_buffer_loaded = true;
    return true;
}

size_t RawReader::getBytesPerVoxel() const {
    switch (m_data_type) {
    case RawDataType::UINT8:
    case RawDataType::INT8:
        return 1;
    case RawDataType::INT16_LE:
    case RawDataType::INT16_BE:
    case RawDataType::UINT16_LE:
    case RawDataType::UINT16_BE:
        return 2;
    case RawDataType::INT32_LE:
    case RawDataType::INT32_BE:
    case RawDataType::UINT32_LE:
    case RawDataType::UINT32_BE:
    case RawDataType::FLOAT32_LE:
    case RawDataType::FLOAT32_BE:
        return 4;
    case RawDataType::FLOAT64_LE:
    case RawDataType::FLOAT64_BE:
        return 8;
    case RawDataType::UNKNOWN:
    default:
        return 0;
    }
}

bool RawReader::isRead() const {
    return m_is_read;
}
int RawReader::width() const {
    return m_width;
}
int RawReader::height() const {
    return m_height;
}
int RawReader::depth() const {
    return m_depth;
}
RawDataType RawReader::getDataType() const {
    return m_data_type;
}

amrex::Box RawReader::box() const {
    if (!m_is_read)
        return amrex::Box();
    return amrex::Box(amrex::IntVect(0, 0, 0),
                      amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

double RawReader::getValue(int i, int j, int k) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_is_read, "[RawReader::getValue] Data not read yet.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        i >= 0 && i < m_width && j >= 0 && j < m_height && k >= 0 && k < m_depth,
        "[RawReader::getValue] Index (" + std::to_string(i) + "," + std::to_string(j) + "," +
            std::to_string(k) + ") out of bounds (W:" + std::to_string(m_width) +
            ", H:" + std::to_string(m_height) + ", D:" + std::to_string(m_depth) + ").");

    if (!ensureHostBufferLoaded()) {
        throw std::runtime_error("[RawReader::getValue] Failed to load raw byte buffer from '" +
                                 m_filename + "'.");
    }

    const size_t bytes_per_voxel = getBytesPerVoxel();
    const size_t plane_size = static_cast<size_t>(m_width) * static_cast<size_t>(m_height);
    const size_t row_offset = static_cast<size_t>(j) * static_cast<size_t>(m_width);
    const size_t idx_1d = static_cast<size_t>(k) * plane_size + row_offset + static_cast<size_t>(i);
    const size_t offset = idx_1d * bytes_per_voxel;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE((offset + bytes_per_voxel) <= m_raw_bytes.size(),
                                     "[RawReader::getValue] Calculated byte offset out of bounds.");

    return decodeVoxelAsDouble(m_raw_bytes.data() + offset, m_data_type, needsByteSwap());
}

void RawReader::threshold(double threshold_value, int value_if_true, int value_if_false,
                          amrex::iMultiFab& mf) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_is_read, "[RawReader::threshold] Cannot threshold, data not read successfully.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        mf.nComp() == 1, "[RawReader::threshold] Output iMultiFab must have 1 component.");

    const size_t bytes_per_voxel = getBytesPerVoxel();
    AMREX_ALWAYS_ASSERT(bytes_per_voxel > 0);
    const bool swap_bytes = needsByteSwap();
    const RawDataType current_dtype = m_data_type;
    const amrex::Box file_domain(amrex::IntVect(0, 0, 0),
                                 amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));

#ifdef AMREX_USE_GPU
    amrex::iMultiFab mf_host(mf.boxArray(), mf.DistributionMap(), mf.nComp(), mf.nGrow(),
                             amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::iMultiFab& target = mf_host;
#else
    amrex::iMultiFab& target = mf;
#endif

    target.setVal(value_if_false);

    // Split any giant patch into bounded Z-slabs (<= 64M voxels/slab) so both the
    // per-rank I/O scratch buffer and MPI int element counts stay strictly bounded.
    constexpr long long max_slab_voxels = 64LL * 1024LL * 1024LL;
    std::vector<PatchSlab> slabs;
    for (amrex::MFIter mfi(target, false); mfi.isValid(); ++mfi) {
        const amrex::Box overlap = mfi.validbox() & file_domain;
        if (!overlap.ok()) {
            continue;
        }
        const long long xy_pts =
            static_cast<long long>(overlap.length(0)) * static_cast<long long>(overlap.length(1));
        const int z_step = static_cast<int>(std::max(
            1LL, std::min<long long>(overlap.length(2), max_slab_voxels / std::max(1LL, xy_pts))));
        for (int z0 = overlap.smallEnd(2); z0 <= overlap.bigEnd(2); z0 += z_step) {
            const int z1 = std::min(overlap.bigEnd(2), z0 + z_step - 1);
            amrex::Box sb = overlap;
            sb.setSmall(2, z0);
            sb.setBig(2, z1);
            slabs.push_back(PatchSlab{&target[mfi], sb});
        }
    }

    auto fill_slab_from_buffer = [&](amrex::IArrayBox& fab, const amrex::Box& sb,
                                     const std::vector<ByteType>& buf) {
        const int lo_x = sb.smallEnd(0);
        const int lo_y = sb.smallEnd(1);
        const int lo_z = sb.smallEnd(2);
        const size_t nx_s = static_cast<size_t>(sb.length(0));
        const size_t ny_s = static_cast<size_t>(sb.length(1));
        const ByteType* buf_ptr = buf.data();

        amrex::LoopOnCpu(sb, [&](int i, int j, int k) {
            const size_t local_idx =
                (static_cast<size_t>(k - lo_z) * ny_s + static_cast<size_t>(j - lo_y)) * nx_s +
                static_cast<size_t>(i - lo_x);
            const double val = decodeVoxelAsDouble(buf_ptr + local_idx * bytes_per_voxel,
                                                   current_dtype, swap_bytes);
            fab(amrex::IntVect(i, j, k), 0) =
                (val > threshold_value) ? value_if_true : value_if_false;
        });
    };

#ifdef BL_USE_MPI
    MPI_Comm comm = amrex::ParallelContext::CommunicatorSub();
    MPI_File fh = MPI_FILE_NULL; // NOLINT(bugprone-casting-through-void)
    const int open_rc = MPI_File_open(comm, m_filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL,
                                      &fh); // NOLINT(bugprone-casting-through-void)
    if (open_rc != MPI_SUCCESS) {
        amrex::Abort("[RawReader::threshold] MPI_File_open failed on '" + m_filename + "'.");
    }

    MPI_Datatype voxel_mpi_type = MPI_DATATYPE_NULL; // NOLINT(bugprone-casting-through-void)
    MPI_Type_contiguous(static_cast<int>(bytes_per_voxel), MPI_BYTE,
                        &voxel_mpi_type); // NOLINT(bugprone-casting-through-void)
    MPI_Type_commit(&voxel_mpi_type);

    const int local_slabs = static_cast<int>(slabs.size());
    int max_slabs = local_slabs;
    amrex::ParallelAllReduce::Max(max_slabs, comm);

    const int full_sizes_zyx[3] = {m_depth, m_height, m_width};
    std::vector<ByteType> slab_bytes;
    ByteType dummy_byte = 0;

    for (int s = 0; s < max_slabs; ++s) {
        MPI_Datatype view_type = MPI_DATATYPE_NULL; // NOLINT(bugprone-casting-through-void)
        if (s < local_slabs) {
            const amrex::Box& sb = slabs[static_cast<size_t>(s)].box;
            const int subsizes_zyx[3] = {sb.length(2), sb.length(1), sb.length(0)};
            const int starts_zyx[3] = {sb.smallEnd(2), sb.smallEnd(1), sb.smallEnd(0)};
            const int n_vox = static_cast<int>(sb.numPts());

            MPI_Type_create_subarray(3, full_sizes_zyx, subsizes_zyx, starts_zyx, MPI_ORDER_C,
                                     voxel_mpi_type, &view_type);
            MPI_Type_commit(&view_type);
            MPI_File_set_view(fh, 0, voxel_mpi_type, view_type, "native",
                              MPI_INFO_NULL); // NOLINT(bugprone-casting-through-void)

            slab_bytes.resize(static_cast<size_t>(n_vox) * bytes_per_voxel);
            const int read_rc =
                MPI_File_read_all(fh, slab_bytes.data(), n_vox, voxel_mpi_type, MPI_STATUS_IGNORE);
            MPI_Type_free(&view_type);
            if (read_rc != MPI_SUCCESS) {
                MPI_Type_free(&voxel_mpi_type);
                MPI_File_close(&fh);
                amrex::Abort("[RawReader::threshold] MPI_File_read_all failed on '" + m_filename +
                             "'.");
            }

            fill_slab_from_buffer(*slabs[static_cast<size_t>(s)].fab, sb, slab_bytes);
        } else {
            // Ranks with fewer local boxes participate collectively with a zero-count read.
            const int dummy_subsizes[3] = {1, 1, 1};
            const int dummy_starts[3] = {0, 0, 0};
            MPI_Type_create_subarray(3, full_sizes_zyx, dummy_subsizes, dummy_starts, MPI_ORDER_C,
                                     voxel_mpi_type, &view_type);
            MPI_Type_commit(&view_type);
            MPI_File_set_view(fh, 0, voxel_mpi_type, view_type, "native",
                              MPI_INFO_NULL); // NOLINT(bugprone-casting-through-void)
            const int read_rc =
                MPI_File_read_all(fh, &dummy_byte, 0, voxel_mpi_type, MPI_STATUS_IGNORE);
            MPI_Type_free(&view_type);
            if (read_rc != MPI_SUCCESS) {
                MPI_Type_free(&voxel_mpi_type);
                MPI_File_close(&fh);
                amrex::Abort("[RawReader::threshold] MPI_File_read_all failed on idle rank.");
            }
        }
    }

    MPI_Type_free(&voxel_mpi_type);
    MPI_File_close(&fh);
#else
    if (!slabs.empty()) {
        std::ifstream ifs(m_filename, std::ios::binary);
        if (!ifs.is_open()) {
            amrex::Abort("[RawReader::threshold] Failed to open '" + m_filename + "'.");
        }
        const size_t plane_voxels = static_cast<size_t>(m_width) * static_cast<size_t>(m_height);
        std::vector<ByteType> slab_bytes;

        for (const auto& slab : slabs) {
            const amrex::Box& sb = slab.box;
            const size_t nx_s = static_cast<size_t>(sb.length(0));
            const size_t ny_s = static_cast<size_t>(sb.length(1));
            const size_t row_bytes = nx_s * bytes_per_voxel;
            slab_bytes.resize(static_cast<size_t>(sb.numPts()) * bytes_per_voxel);

            for (int k = sb.smallEnd(2); k <= sb.bigEnd(2); ++k) {
                for (int j = sb.smallEnd(1); j <= sb.bigEnd(1); ++j) {
                    const size_t file_vox_off =
                        static_cast<size_t>(k) * plane_voxels +
                        static_cast<size_t>(j) * static_cast<size_t>(m_width) +
                        static_cast<size_t>(sb.smallEnd(0));
                    const size_t dst_row_off = ((static_cast<size_t>(k - sb.smallEnd(2)) * ny_s +
                                                 static_cast<size_t>(j - sb.smallEnd(1))) *
                                                row_bytes);
                    ifs.seekg(static_cast<std::streamoff>(file_vox_off * bytes_per_voxel),
                              std::ios::beg);
                    if (!ifs.read(reinterpret_cast<char*>(slab_bytes.data() + dst_row_off),
                                  static_cast<std::streamsize>(row_bytes))) {
                        amrex::Abort("[RawReader::threshold] Failed reading slab from '" +
                                     m_filename + "'.");
                    }
                }
            }
            fill_slab_from_buffer(*slab.fab, sb, slab_bytes);
        }
    }
#endif

#ifdef AMREX_USE_GPU
    amrex::Copy(mf, mf_host, 0, 0, mf.nComp(), mf.nGrow());
    amrex::Gpu::streamSynchronize();
#endif
}

void RawReader::threshold(double threshold_value, amrex::iMultiFab& mf) const {
    threshold(threshold_value, 1, 0, mf);
}

} // namespace OpenImpala
