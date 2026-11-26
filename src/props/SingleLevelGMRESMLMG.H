/**
 * @file SingleLevelGMRESMLMG.H
 * @brief EB-safe single-level float32 GMRES wrapper preconditioned by AMReX MLMGT<fMultiFab>.
 *
 * Upstream AMReX 25.03's `MLMGT<MF>::apply` contains an unguarded `EB_average_down`
 * call inside `if constexpr (IsMultiFabLike_v<MF>)` when compiled with `AMREX_USE_EB`,
 * which fails template instantiation for `MF = amrex::fMultiFab` even on single-level
 * (`NAMRLevels() == 1`) regular grids where the AMR reflux/average-down branch is dead.
 *
 * `SingleLevelGMRESMLMG` drives `amrex::GMRES<amrex::fMultiFab, SingleLevelGMRESMLMG>`
 * directly against `MLLinOpT<amrex::fMultiFab>::solutionResidual` and
 * `MLMGT<amrex::fMultiFab>::precond` on `amrlev = 0`, remaining algebraically identical
 * to `amrex::GMRESMLMGT<amrex::fMultiFab>` while compiling cleanly under both
 * `AMReX_EB=OFF` and `AMReX_EB=ON` (CPU, MPI, and CUDA).
 */

#ifndef OPENIMPALA_SINGLE_LEVEL_GMRES_MLMG_H
#define OPENIMPALA_SINGLE_LEVEL_GMRES_MLMG_H

#include <AMReX_GMRES.H>
#include <AMReX_MLMG.H>
#include <utility>

namespace OpenImpala {

class SingleLevelGMRESMLMG {
public:
    using MF = amrex::fMultiFab;
    using MG = amrex::MLMGT<MF>;
    using RT = typename MG::RT;
    using GM = amrex::GMRES<MF, SingleLevelGMRESMLMG>;

    SingleLevelGMRESMLMG(MG& mlmg, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
        : m_mlmg(&mlmg), m_linop(&mlmg.getLinOp()), m_ba(ba), m_dm(dm) {
        AMREX_ALWAYS_ASSERT(m_linop->NAMRLevels() == 1);
        m_mlmg->preparePrecond();
        m_gmres.define(*this);
    }

    ~SingleLevelGMRESMLMG() = default;
    SingleLevelGMRESMLMG(const SingleLevelGMRESMLMG&) = delete;
    SingleLevelGMRESMLMG(SingleLevelGMRESMLMG&&) = delete;
    SingleLevelGMRESMLMG& operator=(const SingleLevelGMRESMLMG&) = delete;
    SingleLevelGMRESMLMG& operator=(SingleLevelGMRESMLMG&&) = delete;

    void setVerbose(int v) {
        m_verbose = v;
        m_gmres.setVerbose(v);
    }

    void setMaxIters(int niters) {
        m_gmres.setMaxIters(niters);
    }

    [[nodiscard]] int getNumIters() const {
        return m_gmres.getNumIters();
    }

    [[nodiscard]] RT getResidualNorm() const {
        return m_gmres.getResidualNorm();
    }

    GM& getGMRES() {
        return m_gmres;
    }

    [[nodiscard]] MF makeVecRHS() const {
        MF mf(m_ba, m_dm, m_linop->getNComp(), 0);
        amrex::setVal(mf, RT(0));
        return mf;
    }

    [[nodiscard]] MF makeVecLHS() const {
        MF mf(m_ba, m_dm, m_linop->getNComp(), 1);
        amrex::setVal(mf, RT(0));
        return mf;
    }

    [[nodiscard]] RT norm2(MF const& mf) const {
        return m_linop->norm2Precond({&mf});
    }

    static void scale(MF& mf, RT scale_factor) {
        amrex::Scale(mf, scale_factor, 0, amrex::nComp(mf), 0);
    }

    [[nodiscard]] RT dotProduct(MF const& mf1, MF const& mf2) const {
        return m_linop->dotProductPrecond({&mf1}, {&mf2});
    }

    static void setToZero(MF& lhs) {
        amrex::setVal(lhs, RT(0));
    }

    static void assign(MF& lhs, MF const& rhs) {
        amrex::LocalCopy(lhs, rhs, 0, 0, amrex::nComp(lhs), amrex::IntVect(0));
    }

    static void increment(MF& lhs, MF const& rhs, RT a) {
        amrex::Saxpy(lhs, a, rhs, 0, 0, amrex::nComp(lhs), amrex::IntVect(0));
    }

    static void linComb(MF& lhs, RT a, MF const& rhs_a, RT b, MF const& rhs_b) {
        amrex::LinComb(lhs, a, rhs_a, 0, b, rhs_b, 0, 0, amrex::nComp(lhs), amrex::IntVect(0));
    }

    void apply(MF& lhs, MF const& rhs) const {
        m_linop->beginPrecondBC();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        applySingleLevel(lhs, const_cast<MF&>(rhs));
        m_linop->endPrecondBC();
    }

    void precond(MF& lhs, MF const& rhs) const {
        if (m_use_precond) {
            m_mlmg->setPrecondIter(m_precond_niters);
            setToZero(lhs);
            m_mlmg->precond({&lhs}, {&rhs}, RT(0), RT(0));
        } else {
            assign(lhs, rhs);
        }
    }

    void solve(MF& a_sol, MF const& a_rhs, RT a_tol_rel, RT a_tol_abs) {
        const int mlmg_verbose = m_mlmg->getVerbose();
        const int mlmg_bottom_verbose = m_mlmg->getBottomVerbose();
        m_mlmg->setVerbose(m_verbose);
        const auto mlmg_bottom_solver = m_mlmg->getBottomSolver();

        if (mlmg_bottom_solver != amrex::BottomSolver::smoother &&
            mlmg_bottom_solver != amrex::BottomSolver::hypre &&
            mlmg_bottom_solver != amrex::BottomSolver::petsc) {
            m_mlmg->setBottomSolver(amrex::BottomSolver::smoother);
        }

        MF res = makeVecLHS();
        MF cor = makeVecLHS();

        applySingleLevel(res, a_sol); // res = L(a_sol)
        const bool need_to_scale_rhs = m_linop->scaleRHS(0, nullptr);
        if (need_to_scale_rhs) {
            amrex::LocalCopy(cor, a_rhs, 0, 0, amrex::nComp(cor), amrex::IntVect(0));
            const bool scaled = m_linop->scaleRHS(0, &cor);
            amrex::ignore_unused(scaled);
            amrex::Saxpy(res, RT(-1), cor, 0, 0, amrex::nComp(res), amrex::IntVect(0));
        } else {
            amrex::Saxpy(res, RT(-1), a_rhs, 0, 0, amrex::nComp(res), amrex::IntVect(0));
        }
        m_linop->setDirichletNodesToZero(0, 0, res);

        m_gmres.solve(cor, res, a_tol_rel, a_tol_abs); // L_homog(cor) = res
        amrex::Saxpy(a_sol, RT(-1), cor, 0, 0, amrex::nComp(a_sol), amrex::IntVect(0));

        m_mlmg->setBottomSolver(mlmg_bottom_solver);
        m_mlmg->setVerbose(mlmg_verbose);
        m_mlmg->setBottomVerbose(mlmg_bottom_verbose);
    }

private:
    void applySingleLevel(MF& out, MF& a_in) const {
        const int ncomp = m_linop->getNComp();
        const amrex::IntVect ng_sol(1);
        MF in_raii;
        MF* in_ptr = &a_in;
        if (amrex::nGrowVect(a_in) != ng_sol) {
            in_raii.define(m_ba, m_dm, ncomp, ng_sol);
            amrex::setVal(in_raii, RT(0));
            amrex::LocalCopy(in_raii, a_in, 0, 0, ncomp, amrex::IntVect(0));
            in_ptr = &in_raii;
        }
        MF rh(m_ba, m_dm, ncomp, 0);
        amrex::setVal(rh, RT(0));
        m_linop->applyInhomogNeumannTerm(0, rh);
        m_linop->solutionResidual(0, out, *in_ptr, rh, nullptr);
        amrex::Scale(out, RT(-1), 0, amrex::nComp(out), 0);
    }

    GM m_gmres;
    MG* m_mlmg;
    amrex::MLLinOpT<MF>* m_linop;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;
    int m_verbose = 0;
    bool m_use_precond = true;
    int m_precond_niters = 1;
};

} // namespace OpenImpala

#endif // OPENIMPALA_SINGLE_LEVEL_GMRES_MLMG_H
