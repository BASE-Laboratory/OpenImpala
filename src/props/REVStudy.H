/** @file REVStudy.H
 *  @brief Representative Elementary Volume (REV) study module.
 *
 *  Performs an REV convergence study by extracting random sub-volumes of increasing
 *  size from a full domain, solving the cell problem on each, and writing the
 *  effective diffusivity tensor to a CSV file.
 */
#ifndef REV_STUDY_H
#define REV_STUDY_H

#include "EffectiveDiffusivityHypre.H"

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <filesystem>
#include <string>
#include <vector>

namespace OpenImpala {

/** @brief Configuration for an REV study. */
struct REVConfig {
    int num_samples = 3;                ///< Number of random sub-volumes per size
    std::vector<int> sizes;             ///< Target sub-volume edge lengths
    int phase_id = 1;                   ///< Phase ID to analyze
    int box_size = 32;                  ///< Max AMReX box size
    int verbose = 1;                    ///< Verbosity level
    bool write_plotfiles = false;       ///< Write plotfiles for each chi_k solve
    EffectiveDiffusivityHypre::SolverType solver_type =
        EffectiveDiffusivityHypre::SolverType::FlexGMRES;
    std::filesystem::path results_path; ///< Directory for output files
    std::string csv_filename = "rev_study_Deff.csv"; ///< CSV output filename
};

/** @brief Runs an REV convergence study on the given full-domain phase field.
 *
 *  For each target size and sample, extracts a random sub-volume from the full
 *  domain, solves the cell problem in 3 directions using EffectiveDiffusivityHypre,
 *  assembles the D_eff tensor, and appends the result to a CSV file.
 *
 *  @param geom_full       Geometry of the full domain.
 *  @param ba_full         BoxArray of the full domain.
 *  @param dm_full         DistributionMapping of the full domain.
 *  @param mf_phase_full   Phase field of the full domain.
 *  @param config          REV study configuration.
 */
void runREVStudy(const amrex::Geometry& geom_full, const amrex::BoxArray& ba_full,
                 const amrex::DistributionMapping& dm_full,
                 const amrex::iMultiFab& mf_phase_full, const REVConfig& config);

} // namespace OpenImpala

#endif // REV_STUDY_H
