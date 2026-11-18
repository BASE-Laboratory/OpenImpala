"""Comprehensive tests for OpenImpala v4.6.0 Python API features:
  1. 3-D solution field output (return_field=True / False)
  2. Multi-phase variable-diffusivity transport (phase_diffusivities)
  3. Non-cubic / anisotropic and physical SI voxel_size support
  4. Full 3x3 homogenised effective diffusivity tensor (effective_diffusivity)
  5. Microstructural metrics (specific_surface_area, connected_components,
     particle_size_distribution, through_thickness_profile, rev_study)
  6. C++ (_core) vs pure-Python (_solver) parity when _core is compiled
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import openimpala as oi
from openimpala import _solver
from openimpala.session import Session, _has_core


SOLVERS_TO_TEST = ["auto", "pcg"] if _has_core() else ["auto"]


class TestSolutionFieldOutput:
    """Tests for return_field=False (default) and return_field=True."""

    def test_default_does_not_materialize_field(self, uniform_block):
        res = oi.tortuosity(uniform_block, phase=0, direction="x")
        assert res.solution_field is None
        assert res.effective_diffusivity == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("direction", ["x", "y", "z"])
    @pytest.mark.parametrize("solver", SOLVERS_TO_TEST)
    def test_linear_potential_profile_on_uniform_block(self, direction, solver):
        n = 12
        data = np.zeros((n, n, n), dtype=np.int32)
        res = oi.tortuosity(
            data, phase=0, direction=direction, solver=solver, return_field=True
        )
        assert res.solution_field is not None
        assert res.solution_field.shape == (n, n, n)
        assert res.solution_field.dtype == np.float64

        axis = {"z": 0, "y": 1, "x": 2}[direction]
        expected_1d = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
        slice_means = np.mean(
            res.solution_field, axis=tuple(a for a in range(3) if a != axis)
        )
        np.testing.assert_allclose(slice_means, expected_1d, atol=1e-5)

    def test_inactive_voxels_zero_in_solution_field(self, connected_channel):
        res = oi.tortuosity(
            connected_channel, phase=0, direction="x", return_field=True
        )
        assert res.solution_field is not None
        solid_vals = res.solution_field[connected_channel != 0]
        np.testing.assert_allclose(solid_vals, 0.0, atol=1e-12)
        channel_vals = res.solution_field[connected_channel == 0]
        assert np.all((channel_vals > 0.0) & (channel_vals < 1.0))


class TestMultiPhaseTransport:
    """Analytical Reuss (series) and Voigt (parallel) multi-phase transport tests."""

    @pytest.mark.parametrize("solver", SOLVERS_TO_TEST)
    def test_series_laminate_harmonic_mean(self, solver):
        # Layers perpendicular to X: x < 8 has D=1.0 (phase 1), x >= 8 has D=0.25 (phase 2)
        # Exact harmonic mean: D_eff = 2 * 1.0 * 0.25 / (1.0 + 0.25) = 0.4, tau = 1.0 / 0.4 = 2.5
        data = np.ones((8, 8, 16), dtype=np.int32)
        data[:, :, 8:] = 2
        res = oi.tortuosity(
            data,
            direction="x",
            solver=solver,
            phase_diffusivities={1: 1.0, 2: 0.25},
            return_field=True,
        )
        assert res.solver_converged
        assert res.active_volume_fraction == pytest.approx(1.0, rel=1e-6)
        assert res.effective_diffusivity == pytest.approx(0.4, rel=1e-5)
        assert res.tortuosity == pytest.approx(2.5, rel=1e-5)
        assert res.solution_field is not None

    @pytest.mark.parametrize("solver", SOLVERS_TO_TEST)
    def test_parallel_laminate_arithmetic_mean(self, solver):
        # Layers parallel to X (split along Y): y < 4 has D=1.0, y >= 4 has D=0.5
        # Exact arithmetic mean: D_eff = 0.5 * (1.0 + 0.5) = 0.75, tau = 1.0 / 0.75 = 4/3
        data = np.ones((8, 8, 16), dtype=np.int32)
        data[:, 4:, :] = 2
        res = oi.tortuosity(
            data,
            direction="x",
            solver=solver,
            phase_diffusivities={1: 1.0, 2: 0.5},
        )
        assert res.solver_converged
        assert res.effective_diffusivity == pytest.approx(0.75, rel=1e-5)
        assert res.tortuosity == pytest.approx(4.0 / 3.0, rel=1e-5)

    @pytest.mark.parametrize("solver", SOLVERS_TO_TEST)
    def test_physical_si_diffusivity_and_insulating_solid(self, solver):
        # Three-phase battery electrode style: pore (1: 2e-10 m^2/s), CBD (2: 5e-11 m^2/s), AM (0: 0)
        # Parallel channels along Z: 25% pore, 25% CBD, 50% solid AM
        # Exact D_eff = 0.25 * 2e-10 + 0.25 * 5e-11 = 6.25e-11 m^2/s, active_vf = 0.5
        data = np.zeros((16, 8, 8), dtype=np.int32)
        data[:, 0:2, :] = 1
        data[:, 2:4, :] = 2
        res = oi.tortuosity(
            data,
            direction="z",
            solver=solver,
            voxel_size=50e-9,
            phase_diffusivities={0: 0.0, 1: 2.0e-10, 2: 0.5e-10},
        )
        assert res.solver_converged
        assert res.active_volume_fraction == pytest.approx(0.5, rel=1e-6)
        assert res.effective_diffusivity == pytest.approx(6.25e-11, rel=1e-4)


class TestAnisotropicVoxelSize:
    """Tests for scalar SI voxel_size and anisotropic (dz, dy, dx) voxel_size."""

    @pytest.mark.parametrize("solver", SOLVERS_TO_TEST)
    def test_isotropic_scale_invariance_on_porous_media(self, solver):
        with np.load(Path(__file__).parent / "data" / "porous24.npz") as fixture:
            image = fixture["pore"].astype(np.int32)

        res_unit = oi.tortuosity(image, phase=1, direction="z", solver=solver)
        res_nm = oi.tortuosity(
            image, phase=1, direction="z", solver=solver, voxel_size=50e-9
        )
        assert res_nm.tortuosity == pytest.approx(res_unit.tortuosity, rel=1e-5)
        assert res_nm.effective_diffusivity == pytest.approx(
            res_unit.effective_diffusivity, rel=1e-5
        )

    @pytest.mark.parametrize("solver", SOLVERS_TO_TEST)
    def test_anisotropic_tortuosity_sensitivity(self, solver):
        # Zig-zag / obstructed path where lateral Y detour distance depends on dy vs dx
        data = np.ones((8, 12, 12), dtype=np.int32)
        data[:, :7, 4] = 0  # baffle 1 forcing flow to high Y
        data[:, 5:, 8] = 0  # baffle 2 forcing flow back to low Y

        tau_fine_y = oi.tortuosity(
            data, phase=1, direction="x", solver=solver, voxel_size=(1.0, 0.5, 1.0)
        ).tortuosity
        tau_coarse_y = oi.tortuosity(
            data, phase=1, direction="x", solver=solver, voxel_size=(1.0, 2.0, 1.0)
        ).tortuosity
        # Stretching dy increases lateral detour length relative to axial length -> higher tortuosity
        assert tau_coarse_y > tau_fine_y * 1.3

    def test_invalid_voxel_size_raises(self, uniform_block):
        with pytest.raises(ValueError):
            oi.tortuosity(uniform_block, phase=0, voxel_size=-1.0)
        with pytest.raises(ValueError):
            oi.tortuosity(uniform_block, phase=0, voxel_size=(1.0, 0.0, 1.0))
        with pytest.raises(ValueError):
            oi.tortuosity(uniform_block, phase=0, voxel_size=(1.0, 2.0))


class TestEffectiveDiffusivityTensor:
    """Tests for 3x3 periodic homogenisation D_eff tensor."""

    def test_uniform_medium_is_identity(self):
        data = np.ones((10, 10, 10), dtype=np.int32)
        res = oi.effective_diffusivity(data, phase=1, return_fields=True)
        assert res.solver_converged
        assert res.active_volume_fraction == pytest.approx(1.0)
        np.testing.assert_allclose(res.tensor, np.eye(3), atol=1e-6)
        assert res.corrector_fields is not None
        assert set(res.corrector_fields.keys()) == {"x", "y", "z"}
        assert res.corrector_fields["x"].shape == (10, 10, 10)

    def test_empty_phase_returns_zero_tensor(self):
        data = np.zeros((8, 8, 8), dtype=np.int32)
        res = oi.effective_diffusivity(data, phase=1)
        assert res.solver_converged
        assert res.active_volume_fraction == 0.0
        np.testing.assert_allclose(res.tensor, np.zeros((3, 3)), atol=1e-12)

    def test_blocked_slab_zero_normal_and_voigt_parallel(self):
        # Insulating wall normal to X: half active (phase=1), half solid (phase=0)
        # Exact periodic D_eff: D_xx = 0, D_yy = D_zz = 0.5
        slab = np.zeros((8, 8, 16), dtype=np.int32)
        slab[:, :, :8] = 1
        res = oi.effective_diffusivity(slab, phase=1)
        assert res.solver_converged
        assert res.active_volume_fraction == pytest.approx(0.5)
        expected = np.diag([0.0, 0.5, 0.5])
        np.testing.assert_allclose(res.tensor, expected, atol=1e-6)

    def test_multiphase_reuss_voigt_laminate(self):
        # Two-phase periodic laminate normal to X: D_1 = 1.0, D_0 = 0.25 (equal thickness)
        # Exact Reuss normal to X: 2 * 1.0 * 0.25 / (1.0 + 0.25) = 0.4
        # Exact Voigt parallel (Y, Z): 0.5 * (1.0 + 0.25) = 0.625
        lam = np.zeros((8, 8, 16), dtype=np.int32)
        lam[:, :, :8] = 1
        res = oi.effective_diffusivity(lam, phase_diffusivities={0: 0.25, 1: 1.0})
        assert res.solver_converged
        assert res.active_volume_fraction == pytest.approx(1.0)
        expected = np.diag([0.4, 0.625, 0.625])
        np.testing.assert_allclose(res.tensor, expected, atol=1e-5)

    def test_diagonal_channels_off_diagonal_symmetry(self):
        # Periodic 45-degree diagonal sheets in the (x, y) plane: (ix - iy) % 8 < 4
        # Note: staircase diagonal sheets require width >= 2 voxels so active cells
        # connect across faces; width 4 gives percolating 45-degree sheets with D_xy = D_yx > 0.
        nz, ny, nx = 8, 16, 16
        iz, iy, ix = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
        )
        data = (((ix - iy) % 8) < 4).astype(np.int32)

        res = oi.effective_diffusivity(data, phase=1)
        assert res.solver_converged
        D = res.tensor
        np.testing.assert_allclose(D, D.T, atol=1e-6)
        assert D[0, 0] == pytest.approx(D[1, 1], rel=1e-5)
        assert D[2, 2] == pytest.approx(0.5, rel=1e-5)
        assert D[0, 1] > 0.1
        assert abs(D[0, 2]) < 1e-6
        assert abs(D[1, 2]) < 1e-6


class TestMicrostructuralMetrics:
    """Tests for SSA, CCL, PSD, through-thickness profile, and REV study."""

    def test_specific_surface_area_planar_and_anisotropic(self):
        # Single planar interface normal to X (between ix=3 and ix=4 in an 8x8x8 cube):
        # 8*8 = 64 x-normal faces, 0 y-faces, 0 z-faces; total_count = 512.
        data = np.zeros((8, 8, 8), dtype=np.int32)
        data[:, :, 4:] = 1

        ssa_unit = oi.specific_surface_area(data, phase_a=0, phase_b=1)
        assert ssa_unit.face_count == 64
        assert ssa_unit.face_counts_xyz == (64, 0, 0)
        assert ssa_unit.total_count == 512
        assert ssa_unit.ssa == pytest.approx(64.0 / 512.0, rel=1e-12)
        assert ssa_unit.ssa_corrected == pytest.approx((2.0 / 3.0) * 64.0 / 512.0, rel=1e-12)

        # With anisotropic voxel_size = (dz=5.0, dy=3.0, dx=0.25):
        # Physical area = 64 * (dy * dz), physical volume = 512 * (dx * dy * dz)
        # => SSA_phys = 64 / (512 * dx) = 0.125 / 0.25 = 0.5
        ssa_aniso = oi.specific_surface_area(
            data, phase_a=0, phase_b=1, voxel_size=(5.0, 3.0, 0.25)
        )
        assert ssa_aniso.ssa_voxel == pytest.approx(0.125, rel=1e-12)
        assert ssa_aniso.ssa == pytest.approx(0.5, rel=1e-12)

    def test_connected_components_and_psd(self):
        data = np.zeros((12, 12, 12), dtype=np.int32)
        data[1:3, 1:3, 1:3] = 1  # 2x2x2 = 8 voxels
        data[6:9, 6:9, 6:9] = 1  # 3x3x3 = 27 voxels

        ccl = oi.connected_components(data, phase=1, return_labels=True)
        assert ccl.num_components == 2
        assert sorted(ccl.volumes.tolist()) == [8, 27]
        assert ccl.labels is not None
        assert ccl.labels.shape == (12, 12, 12)
        assert set(np.unique(ccl.labels)).issubset({0, 1, 2})
        assert np.all(ccl.labels[data == 0] == 0)
        assert len(set(ccl.labels[1:3, 1:3, 1:3].ravel())) == 1
        assert len(set(ccl.labels[6:9, 6:9, 6:9].ravel())) == 1
        assert ccl.labels[1, 1, 1] != ccl.labels[6, 6, 6]

        psd = oi.particle_size_distribution(data, phase=1, voxel_size=2.0)
        assert psd.num_particles == 2
        expected_r_vox = sorted(
            [((3.0 * v) / (4.0 * np.pi)) ** (1.0 / 3.0) for v in (8.0, 27.0)]
        )
        np.testing.assert_allclose(sorted(psd.radii_voxels), expected_r_vox, rtol=1e-6)
        # Physical radius scales linearly with isotropic voxel_size = 2.0
        np.testing.assert_allclose(
            sorted(psd.radii), [2.0 * r for r in expected_r_vox], rtol=1e-6
        )

    def test_through_thickness_profile(self):
        # Graded porosity along Z: slice k has k rows of phase 1 out of 8 -> vf(k) = k / 8
        data = np.zeros((8, 8, 8), dtype=np.int32)
        for k in range(8):
            data[k, :k, :] = 1

        prof = oi.through_thickness_profile(
            data, phase=1, direction="z", voxel_size=(2.0, 1.0, 1.0)
        )
        assert prof.direction == "z"
        assert prof.num_slices == 8
        np.testing.assert_allclose(prof.profile, np.arange(8) / 8.0, atol=1e-12)
        np.testing.assert_allclose(prof.positions, (np.arange(8) + 0.5) * 2.0, atol=1e-12)

    def test_rev_study_uniform_domain(self, tmp_path):
        data = np.ones((16, 16, 16), dtype=np.int32)
        csv_file = tmp_path / "rev_out.csv"
        rev = oi.rev_study(
            data, sizes=[8, 12], num_samples=2, phase=1, csv_path=str(csv_file)
        )
        assert rev.sizes == [8, 12]
        assert len(rev.samples) == 4
        assert csv_file.exists()
        for sz in (8, 12):
            np.testing.assert_allclose(rev.mean_diagonal[sz], [1.0, 1.0, 1.0], atol=1e-5)


@pytest.mark.skipif(not _has_core(), reason="Requires compiled _core extension for parity check")
class TestCppPurePythonParity:
    """Verify compiled C++ backend and pure-Python fallback agree closely."""

    def test_parity_across_all_v460_apis(self, monkeypatch):
        rng = np.random.default_rng(123)
        data = (rng.random((12, 12, 12)) < 0.7).astype(np.int32)
        # Guarantee straight percolation path along X, Y, Z plus multi-phase values
        data[4:8, 4:8, :] = 1
        data[4:8, :, 4:8] = 1
        data[:, 4:8, 4:8] = 1
        vsize = (1.5, 1.0, 0.75)

        tau_cpp = oi.tortuosity(
            data, phase=1, direction="x", voxel_size=vsize, return_field=True
        )
        deff_cpp = oi.effective_diffusivity(data, phase=1, voxel_size=vsize)
        ssa_cpp = oi.specific_surface_area(data, phase_a=0, phase_b=1, voxel_size=vsize)
        ccl_cpp = oi.connected_components(data, phase=0, return_labels=True)
        ttp_cpp = oi.through_thickness_profile(
            data, phase=1, direction="x", voxel_size=vsize
        )

        monkeypatch.setattr(Session, "_pure_python", True)
        tau_py = oi.tortuosity(
            data, phase=1, direction="x", voxel_size=vsize, return_field=True
        )
        deff_py = oi.effective_diffusivity(data, phase=1, voxel_size=vsize)
        ssa_py = oi.specific_surface_area(data, phase_a=0, phase_b=1, voxel_size=vsize)
        ccl_py = oi.connected_components(data, phase=0, return_labels=True)
        ttp_py = oi.through_thickness_profile(
            data, phase=1, direction="x", voxel_size=vsize
        )

        assert tau_cpp.tortuosity == pytest.approx(tau_py.tortuosity, rel=1e-4)
        assert tau_cpp.effective_diffusivity == pytest.approx(
            tau_py.effective_diffusivity, rel=1e-4
        )
        np.testing.assert_allclose(
            tau_cpp.solution_field, tau_py.solution_field, atol=1e-4
        )
        np.testing.assert_allclose(deff_cpp.tensor, deff_py.tensor, atol=1e-4)
        assert ssa_cpp.ssa == pytest.approx(ssa_py.ssa, rel=1e-12)
        assert ccl_cpp.num_components == ccl_py.num_components
        assert sorted(ccl_cpp.volumes.tolist()) == sorted(ccl_py.volumes.tolist())
        np.testing.assert_allclose(ttp_cpp.profile, ttp_py.profile, atol=1e-12)
