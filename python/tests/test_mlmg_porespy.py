"""MLMG-on-porous-media regression tests.

Validates the EB-based MLMG solver (``MLEBABecLap``) on heterogeneous
porous geometry where non-percolating phase-target islands are carved
out as an EB body region. The embedded-boundary framework preserves MG
coarsening quality across levels, recovering the O(N) iteration count
that the earlier alpha*a pin could not achieve (see issue #289).

Skipped if porespy is not installed (it is not a hard dependency of the
openimpala wheel itself, only of the wheel-test job).
"""

import numpy as np
import pytest

import openimpala as oi

porespy = pytest.importorskip("porespy", reason="porespy not installed")


@pytest.fixture(scope="module")
def porous_blobs():
    """Deterministic porespy blob structure at 32^3, ~50% porosity.

    Small enough that the test runs in ~5 s on CPU CI yet heterogeneous
    enough to exercise the EB masking path (porespy blobs at 50% porosity
    always produce isolated pore islands that the flood fill marks
    inactive).
    """
    np.random.seed(42)
    im = porespy.generators.blobs(shape=[32, 32, 32], porosity=0.5, blobiness=1.5)
    return im.astype(np.int32)


class TestMLMGOnPorousMedia:
    def test_mlmg_returns_finite_tau(self, porous_blobs):
        """MLMG on porous data must return a finite, physical tortuosity.

        Pre-EB-fix, the boundary flux guard rejected MLMG's result on
        this geometry (non-percolating islands were indeterminate).
        Post-fix, ``tau`` is a finite, positive number greater than 1.
        """
        res = oi.tortuosity(porous_blobs, phase=0, direction="z", solver="mlmg")
        assert np.isfinite(res.tortuosity)
        assert res.tortuosity > 1.0
        assert res.solver_converged

    def test_mlmg_matches_hypre_within_one_percent(self, porous_blobs):
        """MLMG and HYPRE+SMG should agree on the same discrete problem.

        Both solve the identical 7-point discretisation with harmonic-mean
        face coefficients; they only differ in how the linear system is
        solved. Agreement to ~1% relative confirms the EB-based MLMG is
        converging to the correct answer, not just *an* answer.
        """
        mlmg = oi.tortuosity(porous_blobs, phase=0, direction="z", solver="mlmg")
        hypre = oi.tortuosity(
            porous_blobs, phase=0, direction="z", solver="pcg", preconditioner="smg"
        )
        assert mlmg.solver_converged
        assert hypre.solver_converged
        rel_diff = abs(mlmg.tortuosity - hypre.tortuosity) / hypre.tortuosity
        assert rel_diff < 0.01, (
            f"MLMG and HYPRE+SMG disagree on porous geometry: "
            f"MLMG={mlmg.tortuosity:.6f}, HYPRE+SMG={hypre.tortuosity:.6f}, "
            f"rel_diff={rel_diff:.2%}"
        )

    def test_mlmg_converges_in_bounded_iterations(self, porous_blobs):
        """EB-aware MG coarsening must keep the iteration count low.

        With the alpha*a pin (pre-EB), MLMG needed ~250 iterations on
        this geometry (~5% reduction per V-cycle). With EB coarsening,
        the iteration count should be bounded by a small constant
        regardless of problem size — the acceptance criterion from #289
        is <=20 V-cycles at 32^3.
        """
        res = oi.tortuosity(porous_blobs, phase=0, direction="z", solver="mlmg")
        assert res.solver_converged
        assert res.iterations <= 20, (
            f"MLMG took {res.iterations} iterations on 32^3 porespy "
            f"(target <=20). EB coarsening may not be working correctly."
        )

    def test_mlmg_directional_symmetry(self, porous_blobs):
        """An isotropic blob field should give similar tau in all 3 directions.

        Not an equality test — porespy blobs at 32^3 have meaningful sample
        variance — but cross-directional ratios should be within ~30 %.
        """
        taus = [
            oi.tortuosity(porous_blobs, phase=0, direction=d, solver="mlmg").tortuosity
            for d in ("x", "y", "z")
        ]
        assert all(np.isfinite(t) for t in taus)
        tau_min, tau_max = min(taus), max(taus)
        assert tau_max / tau_min < 1.3, (
            f"Directional tortuosities differ by more than 30%: {taus}"
        )
