"""Tests for facade helper functions that don't need AMReX."""

import pytest

from openimpala import _core
from openimpala.facade import _parse_direction, _parse_solver


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

    def test_enum_passthrough(self):
        assert _parse_solver(_core.SolverType.SMG) == _core.SolverType.SMG

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            _parse_solver("magic")
