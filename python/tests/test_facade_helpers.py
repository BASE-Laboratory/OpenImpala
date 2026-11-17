"""Tests for facade helper functions that don't need AMReX."""

import pytest

from openimpala import _core
from openimpala.facade import _auto_grid_size, _parse_direction, _parse_solver


class TestParseDirection:

    def test_string_lowercase(self):
        assert _parse_direction("x") == _core.Direction.X
        assert _parse_direction("y") == _core.Direction.Y
        assert _parse_direction("z") == _core.Direction.Z

    def test_string_uppercase(self):
        assert _parse_direction("X") == _core.Direction.X

    def test_enum_passthrough(self):
        assert _parse_direction(_core.Direction.Y) == _core.Direction.Y

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown direction"):
            _parse_direction("w")


class TestParseSolver:

    def test_string_aliases(self):
        assert _parse_solver("hypre") == _core.SolverType.FlexGMRES
        assert _parse_solver("pcg") == _core.SolverType.PCG
        assert _parse_solver("PFMG") == _core.SolverType.PFMG

    def test_auto_selects_mlmg(self):
        """``'auto'`` resolves to MLMG (the matrix-free default)."""
        assert _parse_solver("auto") == "mlmg"
        assert _parse_solver("AUTO") == "mlmg"
        assert _parse_solver("mlmg") == "mlmg"

    def test_enum_passthrough(self):
        assert _parse_solver(_core.SolverType.SMG) == _core.SolverType.SMG

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            _parse_solver("magic")


class TestAutoGridSize:

    def test_single_rank_avoids_box_splitting(self, monkeypatch):
        monkeypatch.setattr(_core, "mpi_ranks", lambda: 1)
        assert _auto_grid_size((8, 8, 8)) == 16
        assert _auto_grid_size((64, 64, 64)) == 64
        assert _auto_grid_size((128, 128, 128)) == 128
        assert _auto_grid_size((512, 512, 512)) == 256

    def test_multi_rank_splits_for_load_balance(self, monkeypatch):
        monkeypatch.setattr(_core, "mpi_ranks", lambda: 4)
        assert _auto_grid_size((128, 128, 128)) == 64
        assert _auto_grid_size((64, 64, 64)) == 32
