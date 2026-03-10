"""Tests for PercolationCheck bindings and facade."""

import numpy as np
import pytest

from openimpala import _core


def _make_img(data: np.ndarray, max_grid_size: int = 32):
    return _core.VoxelImage.from_numpy(np.ascontiguousarray(data, dtype=np.int32), max_grid_size)


class TestPercolationCheckCore:

    def test_connected_channel_percolates(self, connected_channel):
        img = _make_img(connected_channel)
        pc = _core.PercolationCheck(img, 0, _core.Direction.X, 0)
        assert pc.percolates is True
        assert pc.active_volume_fraction > 0.0

        del pc, img

    def test_disconnected_does_not_percolate(self, disconnected_phase):
        img = _make_img(disconnected_phase)
        pc = _core.PercolationCheck(img, 0, _core.Direction.X, 0)
        assert pc.percolates is False

        del pc, img

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
