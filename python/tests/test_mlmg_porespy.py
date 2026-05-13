"""MLMG-on-porous-media regression tests.

The MLMG solver references its relative residual against ``||r_initial||``
(``rhs = 0`` for the steady-state Laplacian) rather than against ``||b||``
as HYPRE does. On heterogeneous porous geometry that leaves a per-cell
absolute residual that, integrated over a plane, used to trip the boundary
flux conservation guard (``1e-4`` relative) at the default ``eps = 1e-9``.

The default was tightened to ``eps = 1e-11`` in ``TortuosityMLMG.H`` and
mirrored in the pybind11 binding. These tests lock that in by running the
same user-facing call (``oi.tortuosity(...)``) on a real porespy blob
structure — the exact workload the bake-off in
``notebooks/profiling_and_tuning.ipynb`` exercises.

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
    enough to trip the pre-fix flux guard.
    """
    np.random.seed(42)
    im = porespy.generators.blobs(shape=[32, 32, 32], porosity=0.5, blobiness=1.5)
    return im.astype(np.int32)


class TestMLMGOnPorousMedia:
    def test_mlmg_returns_finite_tau(self, porous_blobs):
        """The headline regression: MLMG on porous data must not NaN.

        Pre-fix, the boundary flux guard rejected MLMG's result on this
        geometry. Post-fix, ``tau`` is a finite, positive number greater
        than 1 (tortuosity is always >= 1 for a porous medium).
        """
        res = oi.tortuosity(porous_blobs, phase=0, direction="z", solver="mlmg")
        assert np.isfinite(res.tortuosity)
        assert res.tortuosity > 1.0
        assert res.solver_converged

    def test_mlmg_matches_hypre_within_one_percent(self, porous_blobs):
        """MLMG and HYPRE+SMG should agree on the same discrete problem.

        Both solve the identical 7-point discretisation with harmonic-mean
        face coefficients; they only differ in how the linear system is
        solved. Agreement to ~1% relative confirms MLMG is not just
        converging to *something* but to the correct answer.
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
