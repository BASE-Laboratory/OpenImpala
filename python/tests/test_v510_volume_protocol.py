"""Tests for v5.1 BASE-Laboratory ECOSYSTEM.md Layer 3 solver boundary:
zero-dependency VolumeProtocol duck-typing, SI-metres physical unit enforcement,
UnitCalibrationError, and OME-Zarr v0.4 labels/phase + conformal envelope support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

import openimpala as oi

REPO_ROOT = Path(__file__).resolve().parents[2]


class _DuckCalibratedVolume:
    """Standalone VolumeProtocol double (verifies zero basecoat runtime coupling)."""

    def __init__(self, arr: np.ndarray, spacing_um: Tuple[float, float, float] | None):
        self._data = np.asarray(arr)
        self._spacing_um = spacing_um

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(s) for s in self._data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return int(self._data.ndim)

    def spacing_in(self, unit: str) -> Tuple[float, ...]:
        if self._spacing_um is None:
            raise ValueError("Volume is uncalibrated (unit='voxel').")
        factors = {"m": 1e-6, "um": 1.0, "mm": 1e-3}
        if unit not in factors:
            raise ValueError(f"Unsupported unit {unit!r}")
        scale = factors[unit]
        return tuple(float(s * scale) for s in self._spacing_um)


class TestVolumeProtocolZeroDependency:
    """Verify structural VolumeProtocol works without importing basecoat."""

    def test_duck_typed_volume_protocol_si_metres_and_calibration_guard(self):
        # Straight pore slab (x < 6 is phase 1, x >= 6 is phase 0) on a (12, 12, 12) grid
        arr = np.zeros((12, 12, 12), dtype=np.int32)
        arr[:, :, :6] = 1

        cal = _DuckCalibratedVolume(arr, spacing_um=(1.6, 0.8, 0.4))
        assert isinstance(cal, oi.VolumeProtocol)

        with oi.Session():
            ssa_proto = oi.specific_surface_area(cal, phase_a=1, phase_b=0)
            ssa_ref_m = oi.specific_surface_area(
                arr,
                phase_a=1,
                phase_b=0,
                voxel_size=(1.6e-6, 0.8e-6, 0.4e-6),
            )
            assert ssa_proto.ssa == pytest.approx(ssa_ref_m.ssa, rel=1e-12)
            # Single planar x-interface: 12*12 faces across 12*12*12 cells of dx=0.4 um -> 1 / (12 * 0.4e-6) m^-1
            assert ssa_proto.ssa == pytest.approx(1.0 / (12.0 * 0.4e-6), rel=1e-12)

            ttp_proto = oi.through_thickness_profile(cal, phase=1, direction="z")
            assert ttp_proto.positions[0] == pytest.approx(0.5 * 1.6e-6, rel=1e-12)

            psd_proto = oi.particle_size_distribution(cal, phase=1)
            psd_ref_m = oi.particle_size_distribution(
                arr, phase=1, voxel_size=(1.6e-6, 0.8e-6, 0.4e-6)
            )
            assert psd_proto.mean_radius == pytest.approx(psd_ref_m.mean_radius, rel=1e-12)

            tau_proto = oi.tortuosity(cal, phase=1, direction="z")
            assert tau_proto.tortuosity == pytest.approx(1.0, rel=1e-6)

            # Explicit voxel_size= overrides spacing_in("m")
            ssa_override = oi.specific_surface_area(
                cal, phase_a=1, phase_b=0, voxel_size=(1.6, 0.8, 0.4)
            )
            assert ssa_override.ssa == pytest.approx(1.0 / (12.0 * 0.4), rel=1e-12)

            # Uncalibrated VolumeProtocol raises oi.UnitCalibrationError unless voxel_size= is passed
            uncal = _DuckCalibratedVolume(arr, spacing_um=None)
            assert isinstance(uncal, oi.VolumeProtocol)
            with pytest.raises(oi.UnitCalibrationError, match="uncalibrated"):
                oi.specific_surface_area(uncal, phase_a=1, phase_b=0)
            with pytest.raises(oi.UnitCalibrationError, match="uncalibrated"):
                oi.tortuosity(uncal, phase=1, direction="z")

            tau_uncal_explicit = oi.tortuosity(
                uncal, phase=1, direction="z", voxel_size=1.0
            )
            assert tau_uncal_explicit.tortuosity == pytest.approx(1.0, rel=1e-6)


class TestBasecoatAndOmeZarrContract:
    """Verify live interoperability with basecoat v0.2+ Volume, VolumeSet, and .ome.zarr."""

    def test_layer3_ast_boundary_bans_basecoat_and_upstream_tools(self):
        bc = pytest.importorskip("basecoat")
        violations = bc.check_layer_imports(REPO_ROOT, raise_on_violation=True)
        assert violations == []

    def test_basecoat_volume_volumeset_and_conformal_ome_zarr_roundtrip(
        self, tmp_path: Path
    ):
        bc = pytest.importorskip("basecoat")
        pytest.importorskip("zarr")

        # Analytical straight slab along Z:
        # lower envelope: x < 4 (VF = 0.25), point phase: x < 8 (VF = 0.50), upper: x < 12 (VF = 0.75)
        nz, ny, nx = 12, 12, 16
        gray = np.linspace(0.1, 0.9, nz * ny * nx, dtype=np.float32).reshape((nz, ny, nx))
        phase = np.zeros((nz, ny, nx), dtype=np.uint8)
        phase[:, :, :8] = 1
        phase_lo = np.zeros((nz, ny, nx), dtype=np.uint8)
        phase_lo[:, :, :4] = 1
        phase_hi = np.zeros((nz, ny, nx), dtype=np.uint8)
        phase_hi[:, :, :12] = 1

        img_vol = bc.Volume.from_array(gray, voxel_size_um=(1.0, 0.5, 0.5))
        lbl_phase = bc.Volume(
            phase, axes=img_vol.axes, label_names={0: "solid", 1: "pore"}
        )
        lbl_lo = bc.Volume(
            phase_lo, axes=img_vol.axes, label_names={0: "solid", 1: "pore"}
        )
        lbl_hi = bc.Volume(
            phase_hi, axes=img_vol.axes, label_names={0: "solid", 1: "pore"}
        )
        vset = bc.VolumeSet(
            img_vol,
            labels={
                "phase": lbl_phase,
                "phase_lower": lbl_lo,
                "phase_upper": lbl_hi,
            },
        )

        store = bc.write_ome_zarr(vset, tmp_path / "catalyst.ome.zarr", multiscale_levels=2)

        with oi.Session():
            # 1. Direct basecoat.Volume (calibrated -> SI metres; uncalibrated -> UnitCalibrationError)
            assert isinstance(lbl_phase, oi.VolumeProtocol)
            ssa_bc = oi.specific_surface_area(lbl_phase, phase_a=1, phase_b=0)
            assert ssa_bc.ssa == pytest.approx(1.0 / (16.0 * 0.5e-6), rel=1e-12)

            uncal_bc = bc.Volume.uncalibrated(phase)
            with pytest.raises(oi.UnitCalibrationError):
                oi.tortuosity(uncal_bc, phase=1, direction="z")

            # 2. Direct basecoat.VolumeSet unwraps labels["phase"] automatically
            vf_vset = oi.volume_fraction(vset, phase=1)
            assert vf_vset.fraction == pytest.approx(0.50)
            ssa_vset = oi.specific_surface_area(vset, phase_a=1, phase_b=0)
            assert ssa_vset.ssa == pytest.approx(ssa_bc.ssa, rel=1e-12)

            # 3. Direct .ome.zarr store path (root '0' is float32 gray -> auto-selects labels/phase!)
            tau_store = oi.tortuosity(store, phase=1, direction="z")
            assert tau_store.tortuosity == pytest.approx(1.0, rel=1e-6)
            assert tau_store.active_volume_fraction == pytest.approx(0.50)

            # 4. Conformal lower/upper envelope labels + unit='m' conversion via read_zarr_volume & from_zarr
            lo_arr, lo_vz_m = oi.read_zarr_volume(store, label_key="phase_lower", unit="m")
            hi_arr, hi_vz_m = oi.read_zarr_volume(store, label_key="phase_upper", unit="m")
            assert lo_vz_m == pytest.approx((1.0e-6, 0.5e-6, 0.5e-6))
            assert hi_vz_m == pytest.approx((1.0e-6, 0.5e-6, 0.5e-6))

            vf_lo = oi.volume_fraction(lo_arr, phase=1).fraction
            vf_hi = oi.volume_fraction(hi_arr, phase=1).fraction
            assert vf_lo == pytest.approx(0.25)
            assert vf_hi == pytest.approx(0.75)
            assert vf_lo < vf_vset.fraction < vf_hi

            vimg_m = oi.VoxelImage.from_zarr(store, label_key="phase", unit="m")
            assert vimg_m.voxel_size == pytest.approx((1.0e-6, 0.5e-6, 0.5e-6))

            # Coarse pyramid level=1 doubles physical voxel spacing
            _, l1_vz_m = oi.read_zarr_volume(
                store, label_key="phase", level=1, unit="m"
            )
            assert l1_vz_m == pytest.approx((2.0e-6, 1.0e-6, 1.0e-6))

            # Uncalibrated OME-Zarr store raises UnitCalibrationError when unit='m' is requested
            uncal_store = bc.write_ome_zarr(
                bc.Volume.uncalibrated(phase),
                tmp_path / "uncalibrated.ome.zarr",
            )
            with pytest.raises(oi.UnitCalibrationError):
                oi.read_zarr_volume(uncal_store, unit="m")
