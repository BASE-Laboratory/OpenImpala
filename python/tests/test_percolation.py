"""Tests for PercolationCheck bindings and facade."""

import numpy as np
import pytest

from openimpala import _core


def _make_mf(data: np.ndarray, max_grid_size: int = 32):
    from openimpala.facade import _numpy_to_imultifab

    return _numpy_to_imultifab(data, max_grid_size)


class TestPercolationCheckCore:

    def test_connected_channel_percolates(self, connected_channel):
        geom, ba, dm, mf = _make_mf(connected_channel)
        pc = _core.PercolationCheck(geom, ba, dm, mf, 0, _core.Direction.X, 0)
        assert pc.percolates is True
        assert pc.active_volume_fraction > 0.0

        del pc, mf, dm, ba, geom

    def test_disconnected_does_not_percolate(self, disconnected_phase):
        geom, ba, dm, mf = _make_mf(disconnected_phase)
        pc = _core.PercolationCheck(geom, ba, dm, mf, 0, _core.Direction.X, 0)
        assert pc.percolates is False

        del pc, mf, dm, ba, geom

    def test_direction_string(self):
        assert _core.PercolationCheck.direction_string(_core.Direction.X) == "X"
        assert _core.PercolationCheck.direction_string(_core.Direction.Y) == "Y"
        assert _core.PercolationCheck.direction_string(_core.Direction.Z) == "Z"


class TestPercolationFacade:

    def test_connected(self, connected_channel):
        from openimpala.facade import percolation_check

        result = percolation_check(connected_channel, phase=0, direction="x")
        assert result.percolates is True
        assert result.direction == "X"

    def test_disconnected(self, disconnected_phase):
        from openimpala.facade import percolation_check

        result = percolation_check(disconnected_phase, phase=0, direction="x")
        assert result.percolates is False
