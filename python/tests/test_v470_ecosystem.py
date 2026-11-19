"""Tests for v4.7.0 ecosystem features: Otsu segmentation, verified dataset registry,
AMReX plotfile export + yt integration, cloud-native OME-Zarr I/O, and synthetic
porous-media / ML surrogate dataset generation.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import openimpala as oi
from openimpala.cli import main as cli_main


def _write_uncompressed_u16_tiff(path: Path, vol: np.ndarray) -> None:
    arr = np.ascontiguousarray(vol, dtype="<u2")
    nz, ny, nx = arr.shape
    chunks = [b"II" + struct.pack("<HI", 42, 8)]
    pos = 8
    for z in range(nz):
        raw = arr[z].tobytes()
        data_off = pos + 2 + 10 * 12 + 4
        next_ifd = (data_off + len(raw)) if z < nz - 1 else 0
        entries = [
            (256, 4, struct.pack("<I", nx)),
            (257, 4, struct.pack("<I", ny)),
            (258, 3, struct.pack("<HH", 16, 0)),
            (259, 3, struct.pack("<HH", 1, 0)),
            (262, 3, struct.pack("<HH", 1, 0)),
            (273, 4, struct.pack("<I", data_off)),
            (277, 3, struct.pack("<HH", 1, 0)),
            (278, 4, struct.pack("<I", ny)),
            (279, 4, struct.pack("<I", len(raw))),
            (339, 3, struct.pack("<HH", 1, 0)),
        ]
        ifd = struct.pack("<H", len(entries))
        for tag, ftype, val4 in entries:
            ifd += struct.pack("<HHI4s", tag, ftype, 1, val4)
        ifd += struct.pack("<I", next_ifd)
        chunks.extend([ifd, raw])
        pos = next_ifd
    path.write_bytes(b"".join(chunks))


class TestOtsuSegmentation:
    def test_otsu_bimodal_dtypes_and_edge_cases(self):
        # Exact symmetric bimodal uint8 valley midpoint: (20 + 200) / 2 = 110.0
        u8 = np.zeros((10, 10, 10), dtype=np.uint8)
        u8[:5] = 20
        u8[5:] = 200
        assert oi.calculate_otsu_threshold(u8) == pytest.approx(110.0, abs=1.0)
        assert oi.otsu_threshold(u8) == pytest.approx(110.0, abs=1.0)

        # uint16 bimodal
        rng = np.random.default_rng(12)
        gt = np.zeros((16, 16, 16), dtype=np.int32)
        gt[:, :, 8:] = 1
        u16 = np.where(
            gt == 0,
            rng.normal(6000.0, 300.0, size=gt.shape),
            rng.normal(45000.0, 600.0, size=gt.shape),
        ).astype(np.uint16)
        seg, t_val = oi.binarize_otsu(u16, pore_value=0, solid_value=1, dark_is_pore=True)
        assert 10000.0 < t_val < 40000.0
        np.testing.assert_array_equal(seg, gt)

        # Inverted contrast (bright pores)
        seg_inv, _ = oi.binarize_otsu(u16, pore_value=0, solid_value=1, dark_is_pore=False)
        np.testing.assert_array_equal(seg_inv, 1 - gt)

        # Boolean & constant arrays
        assert oi.otsu_threshold(np.array([False, True])) == pytest.approx(0.5)
        assert oi.otsu_threshold(np.full((4, 4, 4), 42.0)) == pytest.approx(42.0)

        with pytest.raises(ValueError, match="empty"):
            oi.otsu_threshold(np.array([]))
        with pytest.raises(ValueError, match="nbins"):
            oi.otsu_threshold(u8, nbins=1)
        with pytest.raises(ValueError, match="NaN"):
            oi.otsu_threshold(np.array([0.0, np.nan, 1.0]))

    def test_voxelimage_read_image_and_cli_otsu(self, tmp_path, capsys):
        # Slab along Z: pore phase 0 in left half (x < 6), solid phase 1 in right half
        gt = np.zeros((12, 12, 12), dtype=np.int32)
        gt[:, :, 6:] = 1
        gray = np.where(gt == 0, 5000, 50000).astype(np.uint16)

        img = oi.VoxelImage.from_grayscale_otsu(gray, pore_value=0, solid_value=1)
        tau = oi.tortuosity(img, phase=0, direction="z")
        assert tau.tortuosity == pytest.approx(1.0, rel=1e-6)

        tif_path = tmp_path / "gray_ct.tif"
        _write_uncompressed_u16_tiff(tif_path, gray)
        _, img_read = oi.read_image(tif_path, threshold="otsu")
        np.testing.assert_array_equal(img_read.to_numpy(), gt)

        with pytest.raises(ValueError, match="otsu"):
            oi.read_image(tif_path, threshold="not_a_method")

        rc = cli_main(["analyze", str(tif_path), "--phase", "0", "--threshold", "otsu", "--direction", "z"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["volume_fraction"] == pytest.approx(0.5)
        assert payload["tortuosity"] == pytest.approx(1.0, rel=1e-6)


class TestDatasetsRegistry:
    def test_list_datasets_and_fetch_example(self):
        reg = oi.datasets.list_datasets()
        assert {"two_phase_stack", "two_phase_cubic", "sphere_pack", "two_phase_hdf5", "two_phase_raw"} <= set(reg)
        for meta in reg.values():
            assert len(meta["sha256"]) == 64

        arr = oi.datasets.fetch_example("porespy_blobs_64")
        assert arr.shape == (64, 64, 64) and arr.dtype == np.int32

        arr_raw = oi.datasets.fetch_example("two_phase_raw")
        assert arr_raw.shape == (100, 100, 100)

        vimg = oi.datasets.fetch_example(
            "two_phase_cubic", as_voxel_image=True, voxel_size=(0.5, 1.0, 2.0)
        )
        assert vimg.shape == (64, 64, 64)
        assert vimg.voxel_size == pytest.approx((0.5, 1.0, 2.0))

        with pytest.raises(KeyError, match="Unknown dataset"):
            oi.datasets.fetch_example("nonexistent_benchmark")

    def test_sha256_corruption_recovery_and_mismatch_error(self, tmp_path, monkeypatch):
        real_spheres = oi.datasets._repo_data_dir() / "spheres.tif"
        real_bytes = real_spheres.read_bytes()

        missing_repo = tmp_path / "empty_repo"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(oi.datasets, "_repo_data_dir", lambda: missing_repo)
        monkeypatch.setenv("OPENIMPALA_DATA_DIR", str(cache_dir))

        # 1. Corrupted cached file is evicted and replaced by verified download
        corrupt_target = cache_dir / "spheres.tif"
        corrupt_target.write_bytes(b"CORRUPTED_PARTIAL_CACHE")

        def _good_download(url: str, dest: str):
            Path(dest).write_bytes(real_bytes)
            return (dest, None)

        monkeypatch.setattr(urllib.request, "urlretrieve", _good_download)
        resolved = oi.datasets.fetch_path("spheres.tif")
        assert resolved == corrupt_target
        assert resolved.read_bytes() == real_bytes

        # 2. Tampered download raises ValueError and leaves no partial/corrupt file
        corrupt_target.unlink()

        def _tampered_download(url: str, dest: str):
            Path(dest).write_bytes(b"TAMPERED_DOWNLOAD")
            return (dest, None)

        monkeypatch.setattr(urllib.request, "urlretrieve", _tampered_download)
        with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
            oi.datasets.fetch_path("spheres.tif")
        assert not corrupt_target.exists()


class TestAmrexPlotfilesAndYt:
    def test_write_plotfile_from_facade_and_low_level(self, tmp_path):
        vol = np.zeros((10, 12, 14), dtype=np.int32)
        vol[:, :, :7] = 1

        tau_res = oi.tortuosity(
            vol,
            phase=1,
            direction="z",
            voxel_size=(2.0, 1.5, 0.5),
            write_plotfile=True,
            results_path=str(tmp_path),
        )
        assert tau_res.solution_field is None
        assert tau_res.plotfile_path is not None
        plt_dir = Path(tau_res.plotfile_path)
        assert (plt_dir / "Header").is_file()
        assert (plt_dir / "Level_0" / "Cell_H").is_file()
        hdr_lines = (plt_dir / "Header").read_text(encoding="ascii").splitlines()
        assert hdr_lines[0] == "HyperCLaw-V1.1"
        assert "solution_potential" in hdr_lines

        deff_res = oi.effective_diffusivity(
            vol,
            phase=1,
            voxel_size=(2.0, 1.5, 0.5),
            write_plotfile=True,
            results_path=str(tmp_path),
        )
        assert deff_res.corrector_fields is None
        assert deff_res.plotfile_paths is not None
        assert set(deff_res.plotfile_paths.keys()) == {"x", "y", "z"}
        for p_str in deff_res.plotfile_paths.values():
            assert (Path(p_str) / "Header").is_file()

        with pytest.raises(ValueError):
            oi.write_amrex_plotfile(tmp_path / "empty_plt", {})
        with pytest.raises(ValueError):
            oi.write_amrex_plotfile(tmp_path / "bad_2d", {"phi": np.ones((4, 4))})
        with pytest.raises(ValueError):
            oi.write_amrex_plotfile(
                tmp_path / "mismatch",
                {"a": np.ones((4, 4, 4)), "b": np.ones((4, 4, 5))},
            )

    def test_yt_roundtrip_exact_values(self, tmp_path):
        yt = pytest.importorskip("yt")
        yt.set_log_level(40)

        nz, ny, nx = 8, 10, 12
        zz, yy, xx = np.meshgrid(
            np.arange(nz, dtype=np.float64),
            np.arange(ny, dtype=np.float64),
            np.arange(nx, dtype=np.float64),
            indexing="ij",
        )
        phi_expected = 0.1 * xx + 1.0 * yy + 10.0 * zz
        mask_expected = ((xx + zz) % 2 == 0).astype(np.float64)

        plt_dir = oi.write_amrex_plotfile(
            tmp_path / "plt_exact",
            {"solution_potential": phi_expected, "active_mask": mask_expected},
            voxel_size=(0.4, 0.3, 0.2),  # (dz, dy, dx)
        )
        ds = yt.load(str(plt_dir))
        assert tuple(int(d) for d in ds.domain_dimensions) == (nx, ny, nz)
        np.testing.assert_allclose(
            np.asarray(ds.domain_right_edge.d),
            [nx * 0.2, ny * 0.3, nz * 0.4],
            rtol=1e-12,
        )
        cg = ds.covering_grid(level=0, left_edge=[0.0, 0.0, 0.0], dims=ds.domain_dimensions)
        phi_yt_zyx = np.transpose(np.asarray(cg[("boxlib", "solution_potential")].d), (2, 1, 0))
        np.testing.assert_allclose(phi_yt_zyx, phi_expected, atol=1e-14)


class TestOmeZarrIO:
    def test_ome_zarr_roundtrip_voxel_scale_and_solvers(self, tmp_path):
        zarr = pytest.importorskip("zarr")

        vol = np.zeros((12, 14, 16), dtype=np.int32)
        vol[:, :, :8] = 1
        zpath = oi.write_zarr_volume(
            tmp_path / "sample.zarr",
            vol,
            voxel_size=(0.5, 0.25, 0.25),
            chunks=(8, 8, 8),
        )

        with pytest.raises(FileExistsError):
            oi.write_zarr_volume(zpath, vol, overwrite=False)

        loaded, vz = oi.read_zarr_volume(zpath)
        np.testing.assert_array_equal(loaded, vol)
        assert vz == pytest.approx((0.5, 0.25, 0.25))

        vimg = oi.VoxelImage.from_zarr(zpath)
        assert vimg.voxel_size == pytest.approx((0.5, 0.25, 0.25))
        exported = vimg.to_zarr(tmp_path / "reexported.zarr")
        assert exported.is_dir()

        # Direct solver input inherits OME-NGFF scale automatically
        ssa_zarr = oi.specific_surface_area(zpath, phase_a=1, phase_b=0)
        ssa_ref = oi.specific_surface_area(vol, phase_a=1, phase_b=0, voxel_size=(0.5, 0.25, 0.25))
        assert ssa_zarr.ssa == pytest.approx(ssa_ref.ssa, rel=1e-12)

        tau_zarr = oi.tortuosity(zpath, phase=1, direction="z")
        assert tau_zarr.tortuosity == pytest.approx(1.0, rel=1e-6)

        # Grayscale Zarr with Otsu thresholding via read_image & read_zarr_volume
        gray_zarr = oi.write_zarr_volume(
            tmp_path / "gray.zarr",
            np.where(vol == 1, 220, 25).astype(np.uint8),
            voxel_size=(1.0, 1.0, 1.0),
        )
        _, img_otsu = oi.read_image(gray_zarr, threshold="otsu")
        np.testing.assert_array_equal(img_otsu.to_numpy(), vol)

        # Root Zarr array (non-group) compatibility
        root_arr_path = tmp_path / "plain_array.zarr"
        z_arr = zarr.open_array(str(root_arr_path), mode="w", shape=vol.shape, dtype="i4")
        z_arr[:] = vol
        plain_loaded, _ = oi.read_zarr_volume(root_arr_path)
        np.testing.assert_array_equal(plain_loaded, vol)

        with pytest.raises(FileNotFoundError):
            oi.read_zarr_volume(tmp_path / "missing.zarr")


class TestSyntheticAndMlGenerator:
    def test_procedural_generators_and_multiphase_electrode(self):
        blobs = oi.synthetic.gaussian_blobs(
            (24, 24, 24), porosity=0.45, blobiness=1.2, anisotropy=(1.5, 1.0, 1.0), seed=11
        )
        assert blobs.shape == (24, 24, 24) and blobs.dtype == np.int32
        assert np.mean(blobs == 0) == pytest.approx(0.45, abs=0.02)

        assert np.all(oi.synthetic.gaussian_blobs(8, porosity=0.0) == 1)
        assert np.all(oi.synthetic.gaussian_blobs(8, porosity=1.0) == 0)

        spheres = oi.synthetic.overlapping_spheres(
            (24, 24, 24), porosity=0.55, radius=3.5, periodic=True, seed=19
        )
        assert spheres.shape == (24, 24, 24)
        assert 0.35 <= float(np.mean(spheres == 0)) <= 0.65

        elec = oi.synthetic.multiphase_electrode(
            (20, 20, 20),
            pore_fraction=0.35,
            active_fraction=0.50,
            binder_fraction=0.15,
            seed=5,
        )
        assert set(np.unique(elec)) == {0, 1, 2}
        assert np.mean(elec == 1) == pytest.approx(0.50, abs=0.02)
        assert np.mean(elec == 2) == pytest.approx(0.15, abs=0.02)

        res_mp = oi.effective_diffusivity(elec, phase_diffusivities={0: 1.0, 2: 0.15})
        assert res_mp.solver_converged
        assert np.all(np.diag(res_mp.tensor) > 0.0)

        with pytest.raises(ValueError):
            oi.synthetic.multiphase_electrode(8, 0.5, 0.5, 0.5)

    def test_ml_dataset_generation_cli_and_sample_script(self, tmp_path, capsys):
        npz_out = tmp_path / "train.npz"
        rc = cli_main(
            [
                "generate-dataset",
                str(npz_out),
                "--samples",
                "3",
                "--shape",
                "12",
                "--generator",
                "blobs",
                "--porosity-range",
                "0.45",
                "0.65",
                "--seed",
                "99",
            ]
        )
        assert rc == 0
        cli_summary = json.loads(capsys.readouterr().out)
        assert cli_summary["n_samples"] == 3
        assert npz_out.is_file()

        with np.load(npz_out) as archive:
            assert archive["volumes"].shape == (3, 12, 12, 12)
            assert archive["deff_tensor"].shape == (3, 3, 3)
            assert archive["tortuosity_xyz"].shape == (3, 3)
            # Symmetric homogenised tensors
            np.testing.assert_allclose(
                archive["deff_tensor"],
                np.swapaxes(archive["deff_tensor"], 1, 2),
                atol=1e-6,
            )

        # Also verify HDF5 and Zarr ML archive writers + spheres generator
        h5_out = tmp_path / "train_spheres.h5"
        ds_h5 = oi.synthetic.generate_ml_dataset(
            h5_out,
            n_samples=2,
            shape=(12, 12, 12),
            generator="spheres",
            porosity_range=(0.45, 0.60),
            seed=77,
        )
        assert h5_out.is_file()
        assert ds_h5["volumes"].shape == (2, 12, 12, 12)

        try:
            import zarr as _zarr_mod  # noqa: F401
        except ImportError:
            pass
        else:
            zarr_out = tmp_path / "train_blobs.zarr"
            ds_zarr = oi.synthetic.generate_ml_dataset(
                zarr_out,
                n_samples=2,
                shape=(10, 10, 10),
                generator="blobs",
                seed=88,
            )
            assert zarr_out.is_dir()
            assert ds_zarr["deff_tensor"].shape == (2, 3, 3)

        # Verify modernised data/create_sample_structure.py runs without PoreSpy
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "create_sample_structure",
            Path(__file__).resolve().parents[2] / "data" / "create_sample_structure.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sample_out = tmp_path / "generated_samples"
        assert mod.main(["--output-dir", str(sample_out), "--shape", "12", "--porosity", "0.4"]) == 0
        assert (sample_out / "SampleData_2Phase_stack_3d_uint8.raw").is_file()
