"""Tests for VolumeFraction bindings and facade."""

import numpy as np
import pytest

from openimpala import _core


def _make_img(data: np.ndarray, max_grid_size: int = 32):
    """Helper: convert ndarray to VoxelImage."""
    return _core.VoxelImage.from_numpy(np.ascontiguousarray(data, dtype=np.int32), max_grid_size)


class TestVolumeFractionCore:
    """Tests using the low-level _core.VolumeFraction class."""

    def test_uniform_phase_zero(self, uniform_block):
        img = _make_img(uniform_block)
        vf = _core.VolumeFraction(img, 0, 0)
        pc, tc = vf.value()
        assert tc == 16 ** 3
        assert pc == 16 ** 3

    def test_uniform_vf(self, uniform_block):
        img = _make_img(uniform_block)
        vf = _core.VolumeFraction(img, 0, 0)
        assert vf.value_vf() == pytest.approx(1.0)

    def test_two_phase_half(self, two_phase_block):
        img = _make_img(two_phase_block)
        vf0 = _core.VolumeFraction(img, 0, 0)
        vf1 = _core.VolumeFraction(img, 1, 0)
        assert vf0.value_vf() == pytest.approx(0.5)
        assert vf1.value_vf() == pytest.approx(0.5)

    def test_absent_phase(self, uniform_block):
        img = _make_img(uniform_block)
        vf = _core.VolumeFraction(img, 99, 0)
        pc, tc = vf.value()
        assert pc == 0
        assert tc == 16 ** 3


class TestVolumeFractionFacade:
    """Tests using the high-level volume_fraction() function."""

    def test_simple(self, uniform_block):
        from openimpala.facade import volume_fraction

        result = volume_fraction(uniform_block, phase=0)
        assert result.fraction == pytest.approx(1.0)
        assert result.total_count == 16 ** 3

    def test_html_repr(self, uniform_block):
        from openimpala.facade import volume_fraction

        result = volume_fraction(uniform_block, phase=0)
        html = result._repr_html_()
        assert "<table>" in html
        assert "100.00%" in html
