"""Tests for enum bindings."""

from openimpala import _core


def test_direction_values():
    assert _core.Direction.X.value == 0
    assert _core.Direction.Y.value == 1
    assert _core.Direction.Z.value == 2


def test_cell_type_values():
    assert _core.CellType.BLOCKED.value == 0
    assert _core.CellType.FREE.value == 1
    assert _core.CellType.BOUNDARY_Z_HI.value == 7


def test_raw_data_type_roundtrip():
    for member in _core.RawDataType.__members__.values():
        assert _core.RawDataType(member.value) == member


def test_solver_type_names():
    names = [s.name for s in _core.SolverType.__members__.values()]
    assert "FlexGMRES" in names
    assert "PFMG" in names


def test_physics_type_has_diffusion():
    assert hasattr(_core.PhysicsType, "Diffusion")
    assert hasattr(_core.PhysicsType, "ElectricalConductivity")
