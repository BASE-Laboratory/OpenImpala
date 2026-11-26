/** @file RawReader.H
 *  @brief Reads 3D image data from flat binary (RAW) files with collective MPI-IO streaming.
 */
#ifndef OPENIMPALA_RAW_READER_H
#define OPENIMPALA_RAW_READER_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>

namespace OpenImpala {

/**
 * @brief Specifies the expected primitive data type and endianness of voxels
 * within a raw binary data file.
 */
enum class RawDataType {
    UNKNOWN,    ///< Default, indicating an uninitialized or invalid type.
    UINT8,      ///< 8-bit unsigned integer (char, byte).
    INT8,       ///< 8-bit signed integer (signed char).
    INT16_LE,   ///< 16-bit signed integer, Little Endian.
    INT16_BE,   ///< 16-bit signed integer, Big Endian.
    UINT16_LE,  ///< 16-bit unsigned integer, Little Endian.
    UINT16_BE,  ///< 16-bit unsigned integer, Big Endian.
    INT32_LE,   ///< 32-bit signed integer, Little Endian.
    INT32_BE,   ///< 32-bit signed integer, Big Endian.
    UINT32_LE,  ///< 32-bit unsigned integer, Little Endian.
    UINT32_BE,  ///< 32-bit unsigned integer, Big Endian.
    FLOAT32_LE, ///< 32-bit single-precision float (IEEE 754), Little Endian.
    FLOAT32_BE, ///< 32-bit single-precision float (IEEE 754), Big Endian.
    FLOAT64_LE, ///< 64-bit double-precision float (IEEE 754), Little Endian.
    FLOAT64_BE  ///< 64-bit double-precision float (IEEE 754), Big Endian.
};

/**
 * @brief Reads and thresholds 3-D voxel data from headerless binary (`.raw`) files.
 *
 * `readFile()` validates file existence and expected byte size without allocating a
 * full-volume host buffer on every MPI rank. `threshold()` streams each MPI rank's
 * owned 3-D sub-boxes directly via collective `MPI_File_read_all` + `MPI_Type_create_subarray`
 * (or per-slab `std::ifstream` reads when built without MPI). Random-access `getValue(i,j,k)`
 * lazily materialises the host byte buffer on first call for single-voxel inspection.
 */
class RawReader {
public:
    using ByteType = unsigned char;

    RawReader();

    explicit RawReader(const std::string& filename, int width, int height, int depth,
                       RawDataType data_type);

    virtual ~RawReader() = default;

    RawReader(const RawReader&) = delete;
    RawReader& operator=(const RawReader&) = delete;
    RawReader(RawReader&&) = default;
    RawReader& operator=(RawReader&&) = default;

    bool readFile(const std::string& filename, int width, int height, int depth,
                  RawDataType data_type);

    bool isRead() const;
    int width() const;
    int height() const;
    int depth() const;
    amrex::Box box() const;
    RawDataType getDataType() const;

    /** @brief Parse a case-insensitive datatype string (e.g. "UINT8", "FLOAT32_LE") to RawDataType.
     */
    static RawDataType parseDataType(const std::string& type_str);

    /** @brief Return true if the full-volume host buffer has been materialised (via getValue). */
    bool hasFullBufferAllocated() const {
        return m_buffer_loaded;
    }

    double getValue(int i, int j, int k) const;

    void threshold(double threshold_value, int value_if_true, int value_if_false,
                   amrex::iMultiFab& mf) const;

    void threshold(double threshold_value, amrex::iMultiFab& mf) const;

private:
    bool validateRawFileMetadata();
    bool ensureHostBufferLoaded() const;
    size_t getBytesPerVoxel() const;
    bool isHostLittleEndian() const;
    bool needsByteSwap() const;

    std::string m_filename;
    int m_width = 0;
    int m_height = 0;
    int m_depth = 0;
    size_t m_expected_bytes = 0;
    RawDataType m_data_type = RawDataType::UNKNOWN;

    mutable std::vector<ByteType> m_raw_bytes;
    mutable bool m_buffer_loaded = false;

    bool m_is_read = false;
};

} // namespace OpenImpala

#endif // OPENIMPALA_RAW_READER_H
