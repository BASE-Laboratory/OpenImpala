"""Tests for TortuosityHypre bindings and facade.

Includes the analytical uniform-block benchmark: tau = 1.
"""

from pathlib import Path

import numpy as np
import pytest

try:
    from openimpala import _core
except ImportError:
    _core = None
from openimpala.exceptions import ConvergenceError, PercolationError


def _make_img(data: np.ndarray, max_grid_size: int = 32):
    return _core.VoxelImage.from_numpy(np.ascontiguousarray(data, dtype=np.int32), max_grid_size)


@pytest.mark.skipif(_core is None, reason="Requires compiled _core extension")
class TestTortuosityHypreCore:
    """Low-level TortuosityHypre tests."""

    def test_uniform_block_analytical(self):
        """tau = 1 for a uniform N-cell domain."""
        N = 16
        data = np.zeros((N, N, N), dtype=np.int32)
        img = _make_img(data, max_grid_size=N)

        vf = 1.0  # all phase 0
        solver = _core.TortuosityHypre(
            img, vf, 0, _core.Direction.X, _core.SolverType.FlexGMRES, ".",
            0.0, 1.0, 0, False,
        )
        tau = solver.value()
        expected = 1.0
        assert tau == pytest.approx(expected, rel=1e-6)

        # Explicitly clean up C++ objects before pytest caches the frame
        del solver, img

    def test_solver_diagnostics(self):
        N = 8
        data = np.zeros((N, N, N), dtype=np.int32)
        img = _make_img(data, max_grid_size=N)

        solver = _core.TortuosityHypre(
            img, 1.0, 0, _core.Direction.X, _core.SolverType.FlexGMRES, ".",
        )
        solver.value()
        assert solver.solver_converged is True
        assert solver.iterations >= 0
        assert solver.residual_norm >= 0.0
        assert abs(solver.flux_in) > 0.0

        # Explicitly clean up C++ objects before pytest caches the frame
        del solver, img


class TestTortuosityFacade:
    """High-level tortuosity() function tests."""

    @pytest.mark.parametrize("direction", ["x", "y", "z"])
    @pytest.mark.parametrize("length", [8, 16, 24, 32])
    @pytest.mark.parametrize("solver", ["pcg", "auto"])
    def test_straight_channel(self, direction, length, solver):
        from openimpala.facade import tortuosity

        axis = "zyx".index(direction)
        data = np.ones((length, 8, 8), dtype=np.int32)
        data[:, 2:6, 2:6] = 0
        data = np.ascontiguousarray(np.moveaxis(data, 0, axis))
        result = tortuosity(data, phase=0, direction=direction, solver=solver)
        assert result.solver_converged
        assert result.tortuosity == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("solver", ["pcg", "auto"])
    def test_intersecting_solid_blocks(self, solver):
        from openimpala.facade import tortuosity

        image = np.ones((12, 12, 12), dtype=np.int32)
        image[4:6, 4:6, 4:6] = 0
        image[4:6, 5:7, 5:7] = 0
        result = tortuosity(image, phase=1, direction="z", solver=solver)
        # Independent direct solve of the six-neighbour resistor network.
        assert result.solver_converged
        assert result.tortuosity == pytest.approx(1.0094612463163157, rel=1e-6)
        assert result.flux_in == pytest.approx(result.flux_out, rel=1e-6)

    def test_porous_volume_against_taufactor(self):
        from openimpala.facade import tortuosity

        # Seed 42, 24^3 Gaussian noise smoothed at sigma=1.2, thresholded at
        # the 45th percentile. Keep six-connected pores spanning axis 0.
        with np.load(Path(__file__).parent / "data" / "porous24.npz") as fixture:
            image = fixture["pore"].astype(np.int32)
        result = tortuosity(image, phase=1, direction="z")
        # TauFactor 1.2.1, CPU, same mask and axis, conv_crit=1e-5.
        assert result.solver_converged
        assert result.tortuosity == pytest.approx(2.10421252, rel=1e-6)

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
