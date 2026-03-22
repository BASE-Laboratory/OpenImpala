#ifndef EFFECTIVEDIFFUSIVITYHYPRE_H
#define EFFECTIVEDIFFUSIVITYHYPRE_H

#include <string>
#include <vector>
#include <limits>
#include <map>

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H> // For m_dx

#include <HYPRE.h>
#include <HYPRE_struct_ls.h>

#include "HypreStructSolver.H"
#include "Tortuosity.H" // For OpenImpala::Direction, SolverType

/** @file EffectiveDiffusivityHypre.H
 * @brief Defines a class to solve the cell problem for effective diffusivity calculation using
 * HYPRE.
 */

namespace OpenImpala {

/** @brief Solves the cell problem ∇_ξ ⋅ (D ∇_ξ χ_k) = -∇_ξ ⋅ (D ê_k) for a corrector function χ_k.
 *
 * This class uses HYPRE's structured-grid solvers to find the corrector function χ_k
 * for a given direction k (X, Y, or Z). The material phase where D=D_material is
 * identified from the input phase data. The class sets up and solves the PDE
 * with periodic boundary conditions for χ_k and specific Neumann conditions
 * on internal pore-solid interfaces.
 *
 * The full effective diffusivity tensor D_eff calculation, which requires
 * all three χ_k fields, is intended to be performed outside this class.
 */
class EffectiveDiffusivityHypre : public HypreStructSolver {

public:
    /** @brief Alias for the shared HYPRE solver type enum (defined in Tortuosity.H). */
    using SolverType = OpenImpala::SolverType;

    /** @brief Construct new solver for the effective diffusivity cell problem.
     *
     * @param geom AMReX Geometry object defining the domain (REV).
     * @param ba AMReX BoxArray defining the distribution of grid patches.
     * @param dm AMReX DistributionMapping defining processor assignments.
     * @param mf_phase_input Input iMultiFab containing phase IDs (expects 1 component, >=1 ghost
     * cell).
     * @param phase_id The ID of the phase considered to be the conductive material (where
     * D=D_material, e.g., pores).
     * @param dir_of_chi_k The direction 'k' (X, Y, or Z) for which the corrector function χ_k is
     * being solved. This determines the ê_k vector in the RHS of the cell problem.
     * @param solver_type Solver type to use (from SolverType enum).
     * @param resultspath Path to directory for writing output files (e.g., plotfiles for χ_k).
     * @param verbose_level Verbosity level (0=silent, 1=basic, 2+=detailed HYPRE output).
     * Default 1.
     * @param write_plotfile_flag If true, write a plotfile of the solved χ_k field. Default false.
     */
    EffectiveDiffusivityHypre(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm,
                              const amrex::iMultiFab& mf_phase_input, const int phase_id,
                              const OpenImpala::Direction dir_of_chi_k,
                              const SolverType solver_type, const std::string& resultspath,
                              int verbose_level = 1, bool write_plotfile_flag = false);

    /** @brief Destructor. Base class handles HYPRE resource cleanup. */
    ~EffectiveDiffusivityHypre() override = default;

    // Deleted Copy Operations
    EffectiveDiffusivityHypre(const EffectiveDiffusivityHypre&) = delete;
    EffectiveDiffusivityHypre& operator=(const EffectiveDiffusivityHypre&) = delete;

    /** @brief Solves the cell problem for the configured χ_k.
     *
     *  Sets up and calls the HYPRE solver. The solved χ_k field can be retrieved
     *  using `getChiSolution()`.
     *
     * @return True if the solver converged to the specified tolerance, false otherwise.
     */
    bool solve();

    /** @brief Retrieves the solved corrector function χ_k.
     *
     *  This method should be called after `solve()` has returned true.
     *  It copies the solution from the internal HYPRE vector to the provided
     *  AMReX MultiFab. The `chi_field` should be pre-allocated with the same
     *  BoxArray, DistributionMapping, 1 component, and desired number of ghost cells
     *  (typically 1 for subsequent gradient calculations).
     *
     * @param chi_field Output MultiFab to store the solved χ_k.
     */
    void getChiSolution(amrex::MultiFab& chi_field);


    // --- Public Getters for Status and Solver Information ---
    // Solver statistics are inherited from HypreStructSolver:
    //   getSolverConverged(), getFinalRelativeResidualNorm(), getSolverIterations()

    const amrex::iMultiFab& getActiveMask() const {
        return m_mf_active_mask;
    }
    const amrex::MultiFab& getDiffCoeff() const {
        return m_mf_diff_coeff;
    }

    // Static Helper Functions (delegate to base class)
    static amrex::Array<HYPRE_Int, AMREX_SPACEDIM> loV(const amrex::Box& b) {
        return HypreStructSolver::loV(b);
    }
    static amrex::Array<HYPRE_Int, AMREX_SPACEDIM> hiV(const amrex::Box& b) {
        return HypreStructSolver::hiV(b);
    }


private:
    // --- Private Methods ---
    void setupMatrixEquation();
    void generateActiveMask(); // Simpler version for D=0/1 based on phase_id

    // --- Member Variables ---
    // Configuration (solver config m_solvertype/m_eps/m_maxiter/m_verbose are in base class)
    std::string m_resultspath;
    const int m_phase_id;                    // Phase ID where D=D_material
    const OpenImpala::Direction m_dir_solve; // Direction k for current chi_k
    bool m_write_plotfile;
    amrex::RealVect m_dx; // Cell sizes [dx, dy, dz]

    // AMReX Data (m_geom/m_ba/m_dm are in base class)
    amrex::iMultiFab m_mf_phase_original; // Stores the input phase data
    amrex::iMultiFab m_mf_active_mask;    // Binary mask (1 for active phase, 0 otherwise)
    amrex::MultiFab m_mf_diff_coeff;      // Spatially varying transport coefficient D(x)
    amrex::MultiFab m_mf_chi;             // Stores the solved chi_k for internal use/plotting

    // Multi-phase configuration
    std::map<int, amrex::Real> m_phase_coeff_map; // Maps phase ID -> transport coefficient
    bool m_is_multi_phase = false;

    // Solver statistics and HYPRE Data Structures are in base class
};



} // namespace OpenImpala

#endif // EFFECTIVEDIFFUSIVITYHYPRE_H
