"""Tests for OpenImpala v5.2.0 features across C++ and pure-Python backends:

1. 3-D cell-centred diffusive flux vector ``(Nz, Ny, Nx, 3)`` ``(Jz, Jy, Jx)`` and
   magnitude ``(Nz, Ny, Nx)`` via ``return_flux=True`` on ``tortuosity`` and
   ``tortuosity_all``, plus 7-variable AMReX plotfile export.
2. Electrochemically active / participating interfacial specific surface area via
   ``specific_surface_area(..., active_only=True, direction=..., active_phase=...)``.
3. Single- and mixed-precision MLMG solves (``mlmg_precision="float64" | "float32" | "mixed"``)
   across ``tortuosity``, ``tortuosity_all``, ``effective_diffusivity``, ``rev_study``,
   ``estimate_memory(..., precision=...)``, and ``openimpala analyze --precision --flux-stats``.
4. Out-of-core ``RawReader`` subarray streaming (zero full-volume host buffer allocation)
   and deterministic ``rev_study(..., resume=True)`` checkpoint/restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import openimpala as oi
from openimpala import _solver
from openimpala.cli import main as cli_main


def _make_channel_with_isolated_pocket(n: int = 16) -> np.ndarray:
    """Return (Nz, Ny, Nx) binary volume with X-spanning channel + isolated 2^3 pocket."""
    vol = np.zeros((n, n, n), dtype=np.int32)
    # 4x4 X-spanning channel at z in [2, 6), y in [2, 6) -> 256 lateral faces on N=16
    vol[2:6, 2:6, :] = 1
    # Isolated 2x2x2 pocket at [10:12, 10:12, 10:12] -> 24 faces, non-percolating
    vol[10:12, 10:12, 10:12] = 1
    return vol


class TestFluxFieldAndPlotfile:
    """Verify 3-D cell-centred diffusive flux fields and C++ / pure-Python parity."""

    def test_flux_field_conservation_inactive_mask_and_backend_parity(self, tmp_path: Path):
        vol = oi.synthetic.gaussian_blobs(shape=(16, 16, 16), porosity=0.62, blobiness=1.1, seed=7)
        # Embed a small solid shell enclosing an isolated pore pocket
        vol[9:15, 9:15, 9:15] = 1
        vol[11:13, 11:13, 11:13] = 0
        dx = 50.0e-9

        with oi.Session():
            res = oi.tortuosity(
                vol,
                phase=0,
                direction="x",
                voxel_size=dx,
                solver="mlmg",
                return_field=True,
                return_flux=True,
                write_plotfile=True,
                results_path=str(tmp_path / "plt_x"),
            )

        assert res.flux_field is not None
        assert res.flux_magnitude is not None
        assert res.flux_field.shape == (16, 16, 16, 3)
        assert res.flux_magnitude.shape == (16, 16, 16)

        # Isolated pocket and solid phase must have identically zero flux
        assert np.all(res.flux_magnitude[11:13, 11:13, 11:13] == 0.0)
        assert np.all(res.flux_magnitude[vol == 1] == 0.0)

        # Pointwise magnitude identity |J| == sqrt(Jz^2 + Jy^2 + Jx^2)
        np.testing.assert_allclose(
            res.flux_magnitude,
            np.linalg.norm(res.flux_field, axis=-1),
            rtol=1e-12,
            atol=1e-15,
        )

        # Exact volume-average conservation: mean(-Jx) * Lx == D_eff
        mean_neg_jx = float(np.mean(-res.flux_field[..., 2]))
        assert mean_neg_jx * (16 * dx) == pytest.approx(res.effective_diffusivity, rel=1e-5)

        # 7-component AMReX plotfile written when write_plotfile=True and return_flux=True
        assert res.plotfile_path is not None
        hdr_text = (Path(res.plotfile_path) / "Header").read_text(encoding="utf-8")
        for comp_name in ("flux_x", "flux_y", "flux_z", "flux_magnitude"):
            assert comp_name in hdr_text

        # Cross-check pure-Python solver flux output directly against facade result
        # (NumPy axis 2 == physical X)
        pure_res = _solver.solve_tortuosity(
            vol,
            phase=0,
            direction=2,
            voxel_size=(dx, dx, dx),
            return_field=True,
            return_flux=True,
        )
        assert pure_res["flux_field"].shape == (16, 16, 16, 3)
        np.testing.assert_allclose(
            pure_res["flux_magnitude"],
            res.flux_magnitude,
            rtol=2e-3,
            atol=1e-4 * float(np.max(res.flux_magnitude)),
        )

    def test_tortuosity_all_return_flux_axis_ordering(self):
        # Uniform anisotropic domain: J along flow axis equals -1 / L_dir everywhere
        vol = np.ones((8, 10, 12), dtype=np.int32)
        spacing = (2.0e-6, 1.5e-6, 1.0e-6)  # (dz, dy, dx)
        with oi.Session():
            all_res = oi.tortuosity_all(
                vol,
                phase=1,
                voxel_size=spacing,
                solver="mlmg",
                return_flux=True,
            )

        # flux_field last axis is ordered (Jz, Jy, Jx) matching NumPy (Z, Y, X)
        lx = 12 * spacing[2]
        ly = 10 * spacing[1]
        lz = 8 * spacing[0]
        np.testing.assert_allclose(all_res["x"].flux_field[..., 2], -1.0 / lx, rtol=1e-5)
        np.testing.assert_allclose(all_res["y"].flux_field[..., 1], -1.0 / ly, rtol=1e-5)
        np.testing.assert_allclose(all_res["z"].flux_field[..., 0], -1.0 / lz, rtol=1e-5)


class TestActiveSpecificSurfaceArea:
    """Verify percolation-filtered electrochemically active SSA."""

    def test_active_ssa_excludes_isolated_pockets_and_matches_pure_python(self):
        vol = _make_channel_with_isolated_pocket(16)
        dx = 0.5e-6

        with oi.Session():
            ssa_tot = oi.specific_surface_area(vol, phase_a=1, phase_b=0, voxel_size=dx)
            ssa_act_x = oi.specific_surface_area(
                vol,
                phase_a=1,
                phase_b=0,
                voxel_size=dx,
                active_only=True,
                direction="x",
                active_phase=1,
            )
            ssa_act_z = oi.specific_surface_area(
                vol,
                phase_a=1,
                phase_b=0,
                voxel_size=dx,
                active_only=True,
                direction="z",
                active_phase=1,
            )

        assert sum(ssa_tot.face_counts_xyz) == 280
        assert sum(ssa_act_x.face_counts_xyz) == 256
        assert ssa_act_x.ssa < ssa_tot.ssa
        # Channel does not percolate along Z -> zero active surface area along Z
        assert sum(ssa_act_z.face_counts_xyz) == 0
        assert ssa_act_z.ssa == 0.0

        # Pure-Python parity check (NumPy axis 2 == physical X)
        perc_ok, _, act_mask = _solver.percolation_check(vol, phase=1, direction=2)
        assert perc_ok is True
        pure_act_x = _solver.specific_surface_area(
            vol,
            phase_a=1,
            phase_b=0,
            voxel_size=(dx, dx, dx),
            active_mask=act_mask,
            active_phase=1,
        )
        assert pure_act_x["face_count"] == 256
        assert pure_act_x["ssa"] == pytest.approx(ssa_act_x.ssa, rel=1e-12)


class TestMixedAndSinglePrecisionMLMG:
    """Verify float64, float32, and mixed-precision MLMG + memory estimation + CLI."""

    def test_mlmg_precision_modes_tortuosity_and_effective_diffusivity(self):
        vol = oi.synthetic.gaussian_blobs(shape=(24, 24, 24), porosity=0.58, blobiness=1.1, seed=21)
        dx = 50.0e-9  # nanometre SI scale exercises coordinate non-dimensionalisation

        with oi.Session():
            t64 = oi.tortuosity(
                vol, phase=0, direction="x", voxel_size=dx, solver="mlmg", mlmg_precision="float64"
            )
            t32 = oi.tortuosity(
                vol, phase=0, direction="x", voxel_size=dx, solver="mlmg", mlmg_precision="float32"
            )
            tmix = oi.tortuosity(
                vol, phase=0, direction="x", voxel_size=dx, solver="mlmg", mlmg_precision="mixed"
            )

            d64 = oi.effective_diffusivity(
                vol, phase=0, voxel_size=dx, solver="mlmg", mlmg_precision="float64"
            )
            d32 = oi.effective_diffusivity(
                vol, phase=0, voxel_size=dx, solver="mlmg", mlmg_precision="float32"
            )
            dmix = oi.effective_diffusivity(
                vol, phase=0, voxel_size=dx, solver="mlmg", mlmg_precision="mixed"
            )

        assert tmix.tortuosity == pytest.approx(t64.tortuosity, rel=1e-6)
        assert t32.tortuosity == pytest.approx(t64.tortuosity, rel=2e-3)
        np.testing.assert_allclose(dmix.tensor, d64.tensor, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(d32.tensor, d64.tensor, rtol=2e-3, atol=1e-5)

    def test_estimate_memory_precision_and_cli_analyze_flux_stats(self, tmp_path: Path, capsys):
        mem64 = oi.estimate_memory((256, 256, 256), solver="mlmg", precision="float64")
        mem_mix = oi.estimate_memory((256, 256, 256), solver="mlmg", precision="mixed")
        mem32 = oi.estimate_memory((256, 256, 256), solver="mlmg", precision="float32")
        assert mem32["per_rank_gb"] < mem_mix["per_rank_gb"] < mem64["per_rank_gb"]

        vol = _make_channel_with_isolated_pocket(12)
        npy_path = tmp_path / "chan12.npy"
        np.save(npy_path, vol)

        rc = cli_main(
            [
                "analyze",
                str(npy_path),
                "--phase",
                "1",
                "--direction",
                "x",
                "--voxel-size",
                "1e-6",
                "--solver",
                "mlmg",
                "--precision",
                "mixed",
                "--flux-stats",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["precision"] == "mixed"
        assert "flux_stats" in payload
        assert payload["flux_stats"]["mean_magnitude"] > 0.0
        assert payload["flux_stats"]["max_magnitude"] >= payload["flux_stats"]["mean_magnitude"]


class TestHPCStreamingAndCheckpointResume:
    """Verify zero-full-buffer RawReader streaming and deterministic REV resume."""

    def test_raw_reader_streams_without_full_host_buffer(self, tmp_path: Path):
        arr = (np.arange(10 * 12 * 14, dtype=np.uint8).reshape(10, 12, 14) % 17)
        raw_path = tmp_path / "sample_14x12x10.raw"
        arr.tofile(raw_path)

        with oi.Session():
            reader, img = oi.read_image(
                str(raw_path),
                raw_width=14,
                raw_height=12,
                raw_depth=10,
                raw_data_type="UINT8",
                threshold=8.0,
                dark_is_pore=True,
                voxel_size=1.0e-6,
                max_grid_size=6,
            )
            if hasattr(reader, "has_full_buffer_allocated"):
                assert reader.has_full_buffer_allocated is False
            vf = oi.volume_fraction(img, phase=1)
            assert vf.phase_count == int(np.count_nonzero(arr > 8))
            assert vf.total_count == arr.size

    def test_rev_study_deterministic_resume(self, tmp_path: Path):
        vol = oi.synthetic.gaussian_blobs(shape=(20, 20, 20), porosity=0.6, blobiness=1.0, seed=9)
        csv_file = tmp_path / "rev_checkpoint.csv"

        with oi.Session():
            r_init = oi.rev_study(
                vol,
                sizes=[8, 12],
                phase=0,
                num_samples=2,
                voxel_size=1.0e-6,
                solver="mlmg",
                mlmg_precision="mixed",
                csv_path=str(csv_file),
            )
            assert csv_file.exists()
            lines_before = [ln for ln in csv_file.read_text(encoding="utf-8").splitlines() if ln]
            assert len(lines_before) == 5  # header + 4 rows

            # Simulate interrupted job after 2 rows
            csv_file.write_text("\n".join(lines_before[:3]) + "\n", encoding="utf-8")

            r_resumed = oi.rev_study(
                vol,
                sizes=[8, 12],
                phase=0,
                num_samples=2,
                voxel_size=1.0e-6,
                solver="mlmg",
                mlmg_precision="mixed",
                csv_path=str(csv_file),
                resume=True,
            )

        lines_after = [ln for ln in csv_file.read_text(encoding="utf-8").splitlines() if ln]
        assert len(lines_after) == 5
        assert lines_after[1] == lines_before[1]
        assert lines_after[2] == lines_before[2]
        np.testing.assert_allclose(r_resumed.mean_diagonal[8], r_init.mean_diagonal[8], rtol=1e-6)
        np.testing.assert_allclose(r_resumed.mean_diagonal[12], r_init.mean_diagonal[12], rtol=1e-5)
