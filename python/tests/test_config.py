"""Tests for PhysicsConfig, ResultsJSON, and CathodeParams bindings."""

import pytest

from openimpala import _core


class TestPhysicsConfig:

    def test_from_type_string_diffusion(self):
        cfg = _core.PhysicsConfig.from_type_string("diffusion")
        assert cfg.name == "Diffusion"
        assert cfg.type == _core.PhysicsType.Diffusion
        assert cfg.coeff_label == "D"

    def test_from_type_string_electrical(self):
        cfg = _core.PhysicsConfig.from_type_string("electrical_conductivity")
        assert cfg.name == "Electrical Conductivity"
        assert cfg.type == _core.PhysicsType.ElectricalConductivity

    def test_from_type_string_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown physics type"):
            _core.PhysicsConfig.from_type_string("antimatter")

    def test_effective_property(self):
        cfg = _core.PhysicsConfig.from_type_string("diffusion")
        cfg.bulk_property = 2.0
        assert cfg.effective_property(0.5) == pytest.approx(1.0)

    def test_tortuosity_factor(self):
        cfg = _core.PhysicsConfig.from_type_string("diffusion")
        # tau = vf / D_eff_ratio
        assert cfg.tortuosity_factor(0.5, 0.5) == pytest.approx(1.0)
        assert cfg.tortuosity_factor(0.25, 0.5) == pytest.approx(2.0)

    def test_formation_factor(self):
        cfg = _core.PhysicsConfig.from_type_string("electrical_conductivity")
        assert cfg.formation_factor(0.5) == pytest.approx(2.0)

    def test_repr(self):
        cfg = _core.PhysicsConfig.from_type_string("diffusion")
        assert "Diffusion" in repr(cfg)


class TestResultsJSON:

    def test_build_json_string(self):
        import json

        rj = _core.ResultsJSON()
        cfg = _core.PhysicsConfig.from_type_string("diffusion")
        rj.set_physics_config(cfg)
        rj.set_input_file("test.tif")
        rj.set_phase_id(0)
        rj.set_grid_info(16, 16, 16, 8)
        rj.set_solver_info("FlexGMRES", True)
        rj.set_volume_fraction(0.5)
        rj.add_direction_result("X", 0.25)

        s = rj.build_json_string()
        d = json.loads(s)
        assert d["openimpala"]["physics_type"] == "Diffusion"
        assert d["openimpala"]["volume_fraction"] == 0.5
        assert "X" in d["openimpala"]["results"]


class TestCathodeParams:

    def test_default_values(self):
        p = _core.CathodeParams()
        assert p.volume_fraction_solid == pytest.approx(0.5)
        assert p.particle_radius == pytest.approx(5e-6)

    def test_readwrite(self):
        p = _core.CathodeParams()
        p.volume_fraction_solid = 0.6
        assert p.volume_fraction_solid == pytest.approx(0.6)
