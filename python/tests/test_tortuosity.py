"""Tests for TortuosityHypre bindings and facade.

Includes the analytical uniform-block benchmark: tau = (N-1)/N.
"""

import numpy as np
import pytest

from openimpala import _core
from openimpala.exceptions import ConvergenceError, PercolationError


def _make_mf(data: np.ndarray, max_grid_size: int = 32):
    from openimpala.facade import _numpy_to_imultifab

    return _numpy_to_imultifab(data, max_grid_size)


class TestTortuosityHypreCore:
    """Low-level TortuosityHypre tests."""

    def test_uniform_block_analytical(self):
        """tau = (N-1)/N for a uniform N-cell domain."""
        N = 16
        data = np.zeros((N, N, N), dtype=np.int32)
        geom, ba, dm, mf = _make_mf(data, max_grid_size=N)

        vf = 1.0  # all phase 0
        solver = _core.TortuosityHypre(
            geom, ba, dm, mf,
            vf, 0, _core.Direction.X, _core.SolverType.FlexGMRES, ".",
            0.0, 1.0, 0, False,
        )
        tau = solver.value()
        expected = (N - 1.0) / N
        assert tau == pytest.approx(expected, rel=1e-6)

    def test_solver_diagnostics(self):
        N = 8
        data = np.zeros((N, N, N), dtype=np.int32)
        geom, ba, dm, mf = _make_mf(data, max_grid_size=N)

        solver = _core.TortuosityHypre(
            geom, ba, dm, mf,
            1.0, 0, _core.Direction.X, _core.SolverType.FlexGMRES, ".",
        )
        solver.value()
        assert solver.solver_converged is True
        assert solver.iterations >= 0
        assert solver.residual_norm >= 0.0
        assert abs(solver.flux_in) > 0.0


class TestTortuosityFacade:
    """High-level tortuosity() function tests."""

    def test_uniform_block(self):
        N = 16
        data = np.zeros((N, N, N), dtype=np.int32)
        from openimpala.facade import tortuosity

        result = tortuosity(data, phase=0, direction="x", max_grid_size=N)
        expected = (N - 1.0) / N
        assert result.tortuosity == pytest.approx(expected, rel=1e-6)
        assert result.solver_converged is True

    def test_non_percolating_raises(self, disconnected_phase):
        from openimpala.facade import tortuosity

        with pytest.raises(PercolationError):
            tortuosity(disconnected_phase, phase=0, direction="x")

    def test_html_repr(self):
        N = 8
        data = np.zeros((N, N, N), dtype=np.int32)
        from openimpala.facade import tortuosity

        result = tortuosity(data, phase=0, direction="x", max_grid_size=N)
        html = result._repr_html_()
        assert "<table>" in html
        assert "Tortuosity" in html
