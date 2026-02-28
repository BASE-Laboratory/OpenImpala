// Test driver for the OpenImpala::PercolationCheck class.
// Reads phase data from a TIFF file, thresholds it, then runs percolation
// checks in all three directions. Reports PASS/FAIL based on optional
// expected values.

#include "PercolationCheck.H"
#include "TiffReader.H"
#include "VolumeFraction.H"
#include "Tortuosity.H"

#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

// Default test parameters
const std::string default_tiff_filename = "data/SampleData_2Phase_stack_3d_1bit.tif";
constexpr double default_threshold_value = 0.5;
constexpr int default_phase_id = 1;
constexpr int default_box_size = 32;

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::Real strt_time = amrex::second();
        bool all_passed = true;

        // --- Configuration via ParmParse ---
        std::string tifffile = default_tiff_filename;
        double threshold_val = default_threshold_value;
        int phase_id = default_phase_id;
        int box_size = default_box_size;
        int verbose = 1;

        // Expected percolation results (-1 = don't check, 0 = expect no, 1 = expect yes)
        int expected_percolates_x = -1;
        int expected_percolates_y = -1;
        int expected_percolates_z = -1;

        {
            amrex::ParmParse pp;
            pp.query("tifffile", tifffile);
            pp.query("threshold", threshold_val);
            pp.query("phase_id", phase_id);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("expected_percolates_x", expected_percolates_x);
            pp.query("expected_percolates_y", expected_percolates_y);
            pp.query("expected_percolates_z", expected_percolates_z);
        }

        // --- Validate ---
        if (box_size <= 0) {
            amrex::Abort("Error: 'box_size' must be positive (got " + std::to_string(box_size) +
                         ").");
        }
        {
            std::ifstream test_ifs(tifffile);
            if (!test_ifs) {
                amrex::Abort("Error: Cannot open input tifffile: " + tifffile +
                             "\n       Specify path using 'tifffile=/path/to/file.tif'");
            }
        }

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- PercolationCheck Test Configuration ---\n";
            amrex::Print() << "  TIF File:      " << tifffile << "\n";
            amrex::Print() << "  Threshold:     " << threshold_val << "\n";
            amrex::Print() << "  Phase ID:      " << phase_id << "\n";
            amrex::Print() << "  Box Size:      " << box_size << "\n";
            amrex::Print() << "  Verbose:       " << verbose << "\n";
            amrex::Print() << "----------------------------------------\n\n";
        }

        // --- Read TIFF and Setup Grids ---
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::iMultiFab mf_phase;
        amrex::Box domain_box;

        try {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << " Reading file " << tifffile << "...\n";

            auto reader = std::make_unique<OpenImpala::TiffReader>(tifffile);
            if (!reader->isRead()) {
                throw std::runtime_error("TiffReader::isRead() returned false.");
            }

            domain_box = reader->box();
            if (domain_box.isEmpty()) {
                amrex::Abort("FAIL: TiffReader returned an empty box.");
            }

            // Non-periodic geometry (required for percolation checking)
            amrex::RealBox rb(
                {AMREX_D_DECL(0.0, 0.0, 0.0)},
                {AMREX_D_DECL(amrex::Real(domain_box.length(0)), amrex::Real(domain_box.length(1)),
                              amrex::Real(domain_box.length(2)))});
            amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
            geom.define(domain_box, &rb, 0, is_periodic.data());

            ba.define(domain_box);
            ba.maxSize(box_size);
            dm.define(ba);

            // Need 1 ghost cell for flood fill
            mf_phase.define(ba, dm, 1, 1);
            mf_phase.setVal(-1);

            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << " Thresholding data...\n";

            // Threshold into a temp no-ghost mfab, then copy with ghost cells
            amrex::iMultiFab mf_phase_noghost(ba, dm, 1, 0);
            reader->threshold(threshold_val, 1, 0, mf_phase_noghost);
            amrex::Copy(mf_phase, mf_phase_noghost, 0, 0, 1, 0);
            mf_phase.FillBoundary(geom.periodicity());

            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << " Domain: " << domain_box.length(0) << " x "
                               << domain_box.length(1) << " x " << domain_box.length(2) << "\n";
            }
        } catch (const std::exception& e) {
            amrex::Abort("Error during setup: " + std::string(e.what()));
        }

        // --- Volume fraction check ---
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
            amrex::Print() << "\n--- Volume Fraction ---\n";

        OpenImpala::VolumeFraction vf_calc(mf_phase, phase_id);
        amrex::Real vf = vf_calc.value_vf(false);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Phase " << phase_id << " VF: " << std::fixed
                           << std::setprecision(6) << vf << "\n";
        }

        if (vf <= 0.0) {
            amrex::Print() << "FAIL: Phase " << phase_id << " has zero volume fraction.\n";
            all_passed = false;
        }

        // --- Percolation checks in all 3 directions ---
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
            amrex::Print() << "\n--- Percolation Checks (Phase " << phase_id << ") ---\n";

        struct DirCheck {
            OpenImpala::Direction dir;
            int expected; // -1=skip, 0=expect false, 1=expect true
        };
        std::vector<DirCheck> checks = {{OpenImpala::Direction::X, expected_percolates_x},
                                        {OpenImpala::Direction::Y, expected_percolates_y},
                                        {OpenImpala::Direction::Z, expected_percolates_z}};

        for (const auto& check : checks) {
            OpenImpala::PercolationCheck pc(geom, ba, dm, mf_phase, phase_id, check.dir, verbose);

            std::string dir_str = OpenImpala::PercolationCheck::directionString(check.dir);
            bool result = pc.percolates();
            amrex::Real active_vf = pc.activeVolumeFraction();

            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  " << dir_str
                               << "-direction: " << (result ? "PERCOLATES" : "DOES NOT PERCOLATE")
                               << " (active VF = " << std::fixed << std::setprecision(6)
                               << active_vf << ")\n";
            }

            // Verify active VF is sensible
            if (result && active_vf <= 0.0) {
                if (amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << "  FAIL: Percolates but active VF is zero!\n";
                all_passed = false;
            }
            if (!result && active_vf > 0.0) {
                if (amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << "  FAIL: Does not percolate but active VF > 0!\n";
                all_passed = false;
            }
            if (active_vf > vf + 1e-9) {
                if (amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << "  FAIL: Active VF (" << active_vf
                                   << ") exceeds total phase VF (" << vf << ")!\n";
                all_passed = false;
            }

            // Check against expected value if provided
            if (check.expected >= 0) {
                bool expected_result = (check.expected == 1);
                if (result != expected_result) {
                    if (amrex::ParallelDescriptor::IOProcessor())
                        amrex::Print() << "  FAIL: Expected "
                                       << (expected_result ? "percolation" : "no percolation")
                                       << " in " << dir_str << " direction.\n";
                    all_passed = false;
                } else {
                    if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                        amrex::Print() << "  " << dir_str << "-direction check: PASS\n";
                }
            }
        }

        // --- Check for unused input parameters ---
        if (amrex::ParmParse::QueryUnusedInputs() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Warning("There are unused parameters in the inputs file (see list above). "
                           "These may be typos.");
        }

        // --- Final Result ---
        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time,
                                                 amrex::ParallelDescriptor::IOProcessorNumber());

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n Run time = " << stop_time << " sec\n";
            if (all_passed) {
                amrex::Print() << "\n tPercolationCheck Test Completed Successfully.\n";
            }
        }

        if (!all_passed) {
            amrex::Abort("tPercolationCheck Test FAILED.");
        }
    }
    amrex::Finalize();
    return 0;
}
