/** @file EffectiveDiffusivityMLMG.H
 *  @brief Matrix-free AMReX MLMG solver for 3-D periodic homogenisation cell problems.
 *
 *  Solves -div(D (e_k + grad chi_k)) = 0 on a periodic 3-torus using AMReX's
 *  MLABecLaplacian + GMRESMLMG without assembling explicit HYPRE matrices,
 *  cutting peak RAM by ~3.3x and enabling native CUDA GPU execution.
 */
#ifndef EFFECTIVEDIFFUSIVITYMLMG_H
#define EFFECTIVEDIFFUSIVITYMLMG_H

#include <map>
#include <string>

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealVect.H>
#include <AMReX_iMultiFab.H>

#include "Tortuosity.H"

namespace OpenImpala {

/** @brief Matrix-free periodic homogenisation cell-problem solver using AMReX MLMG. */
class EffectiveDiffusivityMLMG {
public:
    /** @brief Construct a matrix-free periodic cell-problem solver for direction *dir_of_chi_k*.
     *
     *  @param geom                   AMReX Geometry defining domain bounds and cell sizes.
     *  @param ba                     AMReX BoxArray defining patch decomposition.
     *  @param dm                     AMReX DistributionMapping for MPI rank assignment.
     *  @param mf_phase_input         Input iMultiFab of discrete phase IDs.
     *  @param phase_id               Conducting phase ID (used when *phase_coeffs* is empty).
     *  @param dir_of_chi_k           Unit-forcing direction k (X, Y, or Z).
     *  @param resultspath            Directory for optional AMReX plotfiles.
     *  @param verbose_level          Verbosity level (0 = silent).
     *  @param write_plotfile_flag    Write AMReX plotfile for chi_k when true.
     *  @param eps                    Relative residual tolerance (default 1e-8).
     *  @param maxiter                Maximum outer iterations / V-cycles (default 200).
     *  @param max_coarsening_level   Maximum multigrid coarsening depth (default 30).
     *  @param phase_coeffs           Optional multi-phase diffusivity map {phase_id -> D_i}.
     *  @param precomputed_cell_type  Optional precomputed periodic cell-type mask (0/1/2)
     *                                shared across X, Y, Z solves to avoid repeating BFS.
     */
    EffectiveDiffusivityMLMG(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                             const amrex::DistributionMapping& dm,
                             const amrex::iMultiFab& mf_phase_input, int phase_id,
                             OpenImpala::Direction dir_of_chi_k, const std::string& resultspath,
                             int verbose_level = 0, bool write_plotfile_flag = false,
                             amrex::Real eps = 1.0e-8, int maxiter = 200,
                             int max_coarsening_level = 30,
                             const std::map<int, amrex::Real>& phase_coeffs = {},
                             const amrex::iMultiFab* precomputed_cell_type = nullptr);

    ~EffectiveDiffusivityMLMG() = default;

    EffectiveDiffusivityMLMG(const EffectiveDiffusivityMLMG&) = delete;
    EffectiveDiffusivityMLMG& operator=(const EffectiveDiffusivityMLMG&) = delete;

    /** @brief Solve the periodic cell problem for chi_k. */
    bool solve();

    /** @brief Copy the solved corrector field chi_k into *chi_field* (including periodic ghosts).
     */
    void getChiSolution(amrex::MultiFab& chi_field) const;

    const amrex::iMultiFab& getActiveMask() const {
        return m_mf_active_mask;
    }
    const amrex::MultiFab& getDiffCoeff() const {
        return m_mf_diff_coeff;
    }
    const amrex::iMultiFab& getCellTypeMask() const {
        return m_mf_cell_type;
    }

    bool getSolverConverged() const {
        return m_converged;
    }
    int getSolverIterations() const {
        return m_num_iterations;
    }
    amrex::Real getFinalRelativeResidualNorm() const {
        return m_final_res_norm;
    }

    // Public for CUDA __device__ lambda compatibility (CUDA Programming Guide §I.4.1)
    void initializeDiffCoeff();
    void generateActiveMask();
    long countActiveCells() const;
    void writeChiPlotfile() const;

private:
    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    std::string m_resultspath;
    int m_phase_id = 1;
    OpenImpala::Direction m_dir_solve = OpenImpala::Direction::X;
    int m_verbose = 0;
    bool m_write_plotfile = false;
    amrex::RealVect m_dx;

    amrex::Real m_eps = 1.0e-8;
    int m_maxiter = 200;
    int m_max_coarsening_level = 30;
    bool m_use_gmres = true;
    int m_gmres_restart = 24;
    amrex::Real m_max_diff_coeff = 1.0;
    long m_num_active_cells = 0;

    bool m_converged = false;
    int m_num_iterations = 0;
    amrex::Real m_final_res_norm = 0.0;

    amrex::iMultiFab m_mf_phase_original;
    amrex::iMultiFab m_mf_active_mask;
    amrex::iMultiFab m_mf_cell_type;
    amrex::MultiFab m_mf_diff_coeff;
    amrex::MultiFab m_mf_chi;

    std::map<int, amrex::Real> m_phase_coeff_map;
    bool m_is_multi_phase = false;
};

} // namespace OpenImpala

#endif // EFFECTIVEDIFFUSIVITYMLMG_H
