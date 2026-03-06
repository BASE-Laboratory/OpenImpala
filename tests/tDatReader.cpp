// tests/tDatReader.cpp
//
// Integration test for OpenImpala::DatReader.
//
// Creates temporary binary DAT files with known content and validates:
//   - File reading (header + voxel data)
//   - Dimension getters (width, height, depth, box)
//   - Raw value access (getRawValue, getRawData)
//   - Thresholding into amrex::iMultiFab
//   - Error handling (missing file, truncated file, invalid dimensions)

#include "DatReader.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

// Write a DAT file: 3 x int32 header (W, H, D) + W*H*D x uint16 data (LE)
bool writeDatFile(const std::string& filename, int w, int h, int d,
                  const std::vector<std::uint16_t>& data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open())
        return false;

    std::int32_t dims[3] = {static_cast<std::int32_t>(w), static_cast<std::int32_t>(h),
                            static_cast<std::int32_t>(d)};
    ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    ofs.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(std::uint16_t)));
    return ofs.good();
}

struct TestStatus {
    bool passed = true;
    std::string fail_reason;

    void recordFail(const std::string& reason) {
        passed = false;
        fail_reason = reason;
    }
};

} // anonymous namespace


int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        TestStatus status;
        int verbose = 1;
        {
            amrex::ParmParse pp;
            pp.query("verbose", verbose);
        }

        // ================================================================
        // Test 1: Create and read a small DAT file with known data
        // ================================================================
        const int W = 4, H = 3, D = 2;
        const int num_voxels = W * H * D;
        std::vector<std::uint16_t> known_data(num_voxels);
        // Fill with values: index-based pattern
        for (int k = 0; k < D; ++k) {
            for (int j = 0; j < H; ++j) {
                for (int i = 0; i < W; ++i) {
                    int idx = k * W * H + j * W + i;
                    known_data[idx] = static_cast<std::uint16_t>(idx * 100);
                }
            }
        }

        std::string test_file = "tDatReader_test.dat";
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (!writeDatFile(test_file, W, H, D, known_data)) {
                status.recordFail("Failed to write test DAT file");
            }
        }
        amrex::ParallelDescriptor::Barrier();

        if (status.passed) {
            OpenImpala::DatReader reader;
            bool read_ok = reader.readFile(test_file);

            if (!read_ok) {
                status.recordFail("DatReader::readFile() returned false for valid file");
            } else {
                // Check dimensions
                if (reader.width() != W || reader.height() != H || reader.depth() != D) {
                    status.recordFail("Dimension mismatch: got (" + std::to_string(reader.width()) +
                                      "," + std::to_string(reader.height()) + "," +
                                      std::to_string(reader.depth()) + "), expected (" +
                                      std::to_string(W) + "," + std::to_string(H) + "," +
                                      std::to_string(D) + ")");
                }

                // Check isRead
                if (!reader.isRead()) {
                    status.recordFail("isRead() returned false after successful readFile()");
                }

                // Check box
                amrex::Box expected_box(amrex::IntVect(0, 0, 0),
                                        amrex::IntVect(W - 1, H - 1, D - 1));
                if (reader.box() != expected_box) {
                    status.recordFail("box() mismatch");
                }

                // Check raw data vector
                const auto& raw = reader.getRawData();
                if (static_cast<int>(raw.size()) != num_voxels) {
                    status.recordFail("getRawData() size mismatch: got " +
                                      std::to_string(raw.size()));
                }

                // Check individual voxel values
                for (int k = 0; k < D && status.passed; ++k) {
                    for (int j = 0; j < H && status.passed; ++j) {
                        for (int i = 0; i < W && status.passed; ++i) {
                            int idx = k * W * H + j * W + i;
                            std::uint16_t expected = static_cast<std::uint16_t>(idx * 100);
                            std::uint16_t actual = reader.getRawValue(i, j, k);
                            if (actual != expected) {
                                status.recordFail("getRawValue(" + std::to_string(i) + "," +
                                                  std::to_string(j) + "," + std::to_string(k) +
                                                  ") = " + std::to_string(actual) + ", expected " +
                                                  std::to_string(expected));
                            }
                        }
                    }
                }

                if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << " Test 1 (read + getters):  PASS\n";
                }
            }
        }

        // ================================================================
        // Test 2: getRawValue bounds checking
        // ================================================================
        if (status.passed) {
            OpenImpala::DatReader reader(test_file);
            bool caught = false;
            try {
                reader.getRawValue(-1, 0, 0);
            } catch (const std::out_of_range&) {
                caught = true;
            }
            if (!caught) {
                status.recordFail("getRawValue(-1,0,0) did not throw std::out_of_range");
            }

            caught = false;
            try {
                reader.getRawValue(W, 0, 0);
            } catch (const std::out_of_range&) {
                caught = true;
            }
            if (!caught) {
                status.recordFail("getRawValue(W,0,0) did not throw std::out_of_range");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 2 (bounds checking): PASS\n";
            }
        }

        // ================================================================
        // Test 3: Threshold into iMultiFab
        // ================================================================
        if (status.passed) {
            OpenImpala::DatReader reader(test_file);
            amrex::Box domain_box = reader.box();
            amrex::BoxArray ba(domain_box);
            ba.maxSize(8);
            amrex::DistributionMapping dm(ba);
            amrex::iMultiFab mf(ba, dm, 1, 0);
            mf.setVal(-1);

            // Threshold at 500: values > 500 become 1, else 0
            // known_data has values 0, 100, 200, ..., 2300
            // Values > 500: indices 6,7,8,...,23 (i.e., value >= 600)
            reader.threshold(static_cast<std::uint16_t>(500), mf);

            // Count phase 0 and phase 1 cells
            long long count0 = 0, count1 = 0;
            for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                const auto& arr = mf.const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    int val = arr(i, j, k, 0);
                    if (val == 0)
                        count0++;
                    else if (val == 1)
                        count1++;
                });
            }
            amrex::ParallelAllReduce::Sum(count0, amrex::ParallelContext::CommunicatorSub());
            amrex::ParallelAllReduce::Sum(count1, amrex::ParallelContext::CommunicatorSub());

            // Values 0,100,200,300,400,500 → 6 cells with val <= 500 → phase 0
            // Values 600,...,2300 → 18 cells with val > 500 → phase 1
            if (count0 != 6 || count1 != 18) {
                status.recordFail("Threshold count mismatch: phase0=" + std::to_string(count0) +
                                  " (exp 6), phase1=" + std::to_string(count1) + " (exp 18)");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 3 (threshold):       PASS\n";
            }
        }

        // ================================================================
        // Test 4: Threshold with custom output values
        // ================================================================
        if (status.passed) {
            OpenImpala::DatReader reader(test_file);
            amrex::Box domain_box = reader.box();
            amrex::BoxArray ba(domain_box);
            ba.maxSize(8);
            amrex::DistributionMapping dm(ba);
            amrex::iMultiFab mf(ba, dm, 1, 0);

            reader.threshold(static_cast<std::uint16_t>(500), 42, 99, mf);

            int min_val = mf.min(0);
            int max_val = mf.max(0);
            // Should contain only 42 and 99
            if (!((min_val == 42 && max_val == 99) || (min_val == 99 && max_val == 42) ||
                  (min_val == 42 && max_val == 42) || (min_val == 99 && max_val == 99))) {
                // More precisely: min should be 42, max should be 99
                // (since we have both above and below threshold)
                if (min_val != 42 || max_val != 99) {
                    status.recordFail("Custom threshold values wrong: min=" +
                                      std::to_string(min_val) + " max=" + std::to_string(max_val));
                }
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 4 (custom threshold): PASS\n";
            }
        }

        // ================================================================
        // Test 5: Constructor throws on missing file
        // ================================================================
        if (status.passed) {
            bool caught = false;
            try {
                OpenImpala::DatReader reader("nonexistent_file_xyzzy.dat");
            } catch (const std::runtime_error&) {
                caught = true;
            }
            if (!caught) {
                status.recordFail("Constructor did not throw for missing file");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 5 (missing file):    PASS\n";
            }
        }

        // ================================================================
        // Test 6: readFile returns false for truncated file (too small)
        // ================================================================
        if (status.passed) {
            std::string trunc_file = "tDatReader_trunc.dat";
            if (amrex::ParallelDescriptor::IOProcessor()) {
                // Write only header, no data
                std::ofstream ofs(trunc_file, std::ios::binary);
                std::int32_t dims[3] = {10, 10, 10};
                ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));
            }
            amrex::ParallelDescriptor::Barrier();

            OpenImpala::DatReader reader;
            bool read_ok = reader.readFile(trunc_file);
            if (read_ok) {
                status.recordFail("readFile() should return false for truncated file");
            }
            if (reader.isRead()) {
                status.recordFail("isRead() should be false for truncated file");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 6 (truncated file):  PASS\n";
            }
        }

        // ================================================================
        // Test 7: Default constructor state
        // ================================================================
        if (status.passed) {
            OpenImpala::DatReader reader;
            if (reader.isRead()) {
                status.recordFail("Default-constructed reader should not be read");
            }
            if (reader.width() != 0 || reader.height() != 0 || reader.depth() != 0) {
                status.recordFail("Default-constructed reader should have zero dimensions");
            }
            amrex::Box empty_box = reader.box();
            if (!empty_box.isEmpty()) {
                status.recordFail("Default-constructed reader should return empty box");
            }

            // getRawValue should throw on unread reader
            bool caught = false;
            try {
                reader.getRawValue(0, 0, 0);
            } catch (const std::out_of_range&) {
                caught = true;
            }
            if (!caught) {
                status.recordFail("getRawValue on unread reader did not throw");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 7 (default state):   PASS\n";
            }
        }

        // ================================================================
        // Test 8: File with invalid (zero/negative) dimensions in header
        // ================================================================
        if (status.passed) {
            std::string bad_file = "tDatReader_baddims.dat";
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream ofs(bad_file, std::ios::binary);
                std::int32_t dims[3] = {0, 10, 10};
                ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));
            }
            amrex::ParallelDescriptor::Barrier();

            OpenImpala::DatReader reader;
            bool read_ok = reader.readFile(bad_file);
            if (read_ok) {
                status.recordFail("readFile() should fail for zero-dimension header");
            }

            if (status.passed && verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Test 8 (invalid dims):    PASS\n";
            }
        }

        // ================================================================
        // Cleanup temp files
        // ================================================================
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::remove("tDatReader_test.dat");
            std::remove("tDatReader_trunc.dat");
            std::remove("tDatReader_baddims.dat");
        }

        // ================================================================
        // Final summary
        // ================================================================
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (status.passed) {
                amrex::Print() << "\n--- TEST RESULT: PASS ---\n";
            } else {
                amrex::Print() << "\n--- TEST RESULT: FAIL ---\n";
                amrex::Print() << "  Reason: " << status.fail_reason << "\n";
            }
        }

        if (!status.passed) {
            amrex::Abort("tDatReader Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
