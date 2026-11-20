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

        # Verify spheres generator on .npz (always available) + HDF5/Zarr when installed
        spheres_npz = tmp_path / "train_spheres.npz"
        ds_sph = oi.synthetic.generate_ml_dataset(
            spheres_npz,
            n_samples=2,
            shape=(12, 12, 12),
            generator="spheres",
            porosity_range=(0.45, 0.60),
            seed=77,
        )
        assert spheres_npz.is_file()
        assert ds_sph["volumes"].shape == (2, 12, 12, 12)

        try:
            import h5py as _h5py_mod  # noqa: F401
        except ImportError:
            pass
        else:
            h5_out = tmp_path / "train_spheres.h5"
            ds_h5 = oi.synthetic.generate_ml_dataset(
                h5_out,
                n_samples=2,
                shape=(10, 10, 10),
                generator="spheres",
                porosity_range=(0.45, 0.60),
                seed=77,
            )
            assert h5_out.is_file()
            assert ds_h5["volumes"].shape == (2, 10, 10, 10)

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


class TestRedTeamHardening:
    """Adversarial regression tests for v4.7.0 Red Team hardening."""

    def test_safe_store_rmtree_guards_cwd_symlinks_and_user_dirs(self, tmp_path):
        fields = {"phi": np.ones((4, 4, 4), dtype=np.float64)}

        # Refuses current working directory '.' and ancestor '..'
        with pytest.raises(ValueError, match="protected directory"):
            oi.write_amrex_plotfile(".", fields)

        # Refuses non-empty directory missing plotfile markers
        user_dir = tmp_path / "important_project"
        user_dir.mkdir()
        precious = user_dir / "notes.txt"
        precious.write_text("do not delete", encoding="utf-8")
        with pytest.raises(ValueError, match="non-AMReX plotfile"):
            oi.write_amrex_plotfile(user_dir, fields)
        assert precious.read_text(encoding="utf-8") == "do not delete"

        # Refuses symlinks
        real_dir = tmp_path / "real_target"
        real_dir.mkdir()
        link_dir = tmp_path / "symlink_plt"
        link_dir.symlink_to(real_dir, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            oi.write_amrex_plotfile(link_dir, fields)

        # Allows overwriting an existing valid plotfile directory
        valid_plt = tmp_path / "plt00000"
        oi.write_amrex_plotfile(valid_plt, fields)
        oi.write_amrex_plotfile(valid_plt, {"phi": np.zeros((4, 4, 4), dtype=np.float64)})
        assert (valid_plt / "Header").is_file()

    def test_exact_uint16_otsu_zero_quantisation_with_outliers(self):
        # Symmetric bimodal uint16 peaks at 1000 and 41000 spanning [0, 65535]:
        # a coarse 256-bin histogram shifts the threshold by ~8 ADU (bin width 256),
        # whereas exact 65536-bin integer Otsu recovers the exact midpoint 21000.0.
        vol = np.full((16, 16, 16), 1000, dtype=np.uint16)
        vol[8:, :, :] = 41000
        vol[0, 0, 0] = 0
        vol[-1, -1, -1] = 65535
        thresh = oi.calculate_otsu_threshold(vol)
        assert thresh == pytest.approx(21000.0, abs=1e-9)

    def test_read_image_bright_pore_polarity_and_cli_flags(self, tmp_path, capsys):
        # Bright-pore image: pore voxels have intensity 220 (37.5%), solid has 30 (62.5%)
        raw_vol = np.full((8, 8, 8), 30, dtype=np.uint8)
        raw_vol[:3, :, :] = 220
        npy_path = tmp_path / "bright_pore.npy"
        np.save(npy_path, raw_vol)

        with oi.Session():
            _, img_default = oi.read_image(npy_path, threshold="otsu", dark_is_pore=True)
            _, img_inverted = oi.read_image(npy_path, threshold="otsu", dark_is_pore=False)
            assert oi.volume_fraction(img_default, phase=0).fraction == pytest.approx(5.0 / 8.0)
            assert oi.volume_fraction(img_inverted, phase=0).fraction == pytest.approx(3.0 / 8.0)

        rc = cli_main(["vf", str(npy_path), "--threshold", "otsu", "--bright-is-pore"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["volume_fraction"] == pytest.approx(3.0 / 8.0)

    def test_unsegmented_float_volume_rejected_with_clear_error(self, tmp_path):
        gray_f32 = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape((4, 4, 4))
        npy_f32 = tmp_path / "unsegmented.npy"
        np.save(npy_f32, gray_f32)

        with oi.Session():
            with pytest.raises(ValueError, match="floating-point grayscale"):
                oi.volume_fraction(gray_f32, phase=0)
            with pytest.raises(ValueError, match="floating-point grayscale"):
                oi.volume_fraction(npy_f32, phase=0)
            # Integer-valued float arrays (e.g. np.zeros / np.ones) remain accepted
            vf_ok = oi.volume_fraction(np.zeros((4, 4, 4), dtype=np.float64), phase=0)
            assert vf_ok.fraction == pytest.approx(1.0)

    def test_ome_zarr_v2_v3_and_5d_axis_handling(self, tmp_path):
        zarr = pytest.importorskip("zarr")

        vol = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
        p_v2 = tmp_path / "ome_v2.zarr"
        oi.write_zarr_volume(p_v2, vol, voxel_size=(1.5, 2.5, 3.5), zarr_format=2)
        assert (p_v2 / ".zgroup").is_file()
        r_v2, vz_v2 = oi.read_zarr_volume(p_v2)
        np.testing.assert_array_equal(r_v2, vol)
        assert vz_v2 == pytest.approx((1.5, 2.5, 3.5))

        if int(str(zarr.__version__).split(".")[0]) >= 3:
            p_v3 = tmp_path / "ome_v3.zarr"
            oi.write_zarr_volume(p_v3, vol, voxel_size=(0.5, 0.6, 0.7), zarr_format=3)
            assert (p_v3 / "zarr.json").is_file()
            r_v3, vz_v3 = oi.read_zarr_volume(p_v3)
            np.testing.assert_array_equal(r_v3, vol)
            assert vz_v3 == pytest.approx((0.5, 0.6, 0.7))

        # 5-D (t, c, z, y, x) OME-NGFF store with 5-axis scale
        p_5d = tmp_path / "synchrotron_5d.zarr"
        try:
            grp5 = zarr.open_group(str(p_5d), mode="w", zarr_format=2)
        except TypeError:
            grp5 = zarr.open_group(str(p_5d), mode="w")
        data_5d = vol[np.newaxis, np.newaxis, ...]
        if hasattr(grp5, "create_array"):
            grp5.create_array("0", data=data_5d, overwrite=True)
        else:
            grp5.create_dataset("0", data=data_5d, overwrite=True)
        grp5.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 4.0, 5.0, 6.0]}
                        ],
                    }
                ],
            }
        ]
        r_5d, vz_5d = oi.read_zarr_volume(p_5d)
        np.testing.assert_array_equal(r_5d, vol)
        assert vz_5d == pytest.approx((4.0, 5.0, 6.0))

    def test_mpi_env_rank_guard_and_streaming_dataset(self, tmp_path, monkeypatch):
        # Simulate non-zero MPI rank via launcher env when AMReX / mpi4py are inactive
        if oi.build_info()["backend"] == "pure-python":
            monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
            assert oi.synthetic._is_io_rank() is False
            monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
            assert oi.synthetic._is_io_rank() is True
            monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)

        try:
            import zarr  # noqa: F401
        except ImportError:
            pass
        else:
            stream_zarr = tmp_path / "streamed.zarr"
            ds = oi.synthetic.generate_ml_dataset(
                stream_zarr,
                n_samples=2,
                shape=10,
                compute_deff=False,
                compute_tortuosity=False,
                return_volumes=False,
            )
            assert "volumes" not in ds
            grp = zarr.open(str(stream_zarr), mode="r")
            assert grp["volumes"].shape == (2, 10, 10, 10)

    def test_datasets_urlopen_timeout_and_tag_fallback(self, tmp_path, monkeypatch):
        import hashlib
        import io
        import urllib.error
        import urllib.request

        payload = b"verified-payload-bytes"
        digest = hashlib.sha256(payload).hexdigest()
        monkeypatch.setenv("OPENIMPALA_DATA_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(oi.datasets, "_repo_data_dir", lambda: tmp_path / "no_repo")

        seen_calls: list[tuple[str, float]] = []

        class _FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def _fake_urlopen(url, timeout=None):
            seen_calls.append((url, float(timeout)))
            if "/master/" not in url:
                raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
            return _FakeResp(payload)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        out = oi.datasets.fetch_path("tag_fallback.bin", expected_sha256=digest, timeout=12.5)
        assert out.read_bytes() == payload
        assert len(seen_calls) == 2
        assert seen_calls[0][1] == pytest.approx(12.5)
        assert "/master/data/tag_fallback.bin" in seen_calls[1][0]

        with pytest.raises(ValueError, match="timeout"):
            oi.datasets.fetch_path("tag_fallback.bin", timeout=0.0)

    def test_empty_phase_deff_plotfile_overwrites_stale_dir(self, tmp_path):
        with oi.Session():
            vol_full = np.zeros((8, 8, 8), dtype=np.int32)
            r1 = oi.effective_diffusivity(
                vol_full, phase=0, write_plotfile=True, results_path=str(tmp_path)
            )
            assert r1.plotfile_paths is not None

            # Second run into same results_path with an empty phase (phase=9) must
            # overwrite stale plotfiles with zero fields rather than leaving r1's data.
            r2 = oi.effective_diffusivity(
                vol_full, phase=9, write_plotfile=True, results_path=str(tmp_path)
            )
            assert np.allclose(r2.tensor, 0.0)
            assert r2.plotfile_paths is not None
            assert Path(r2.plotfile_paths["x"], "Header").is_file()


class TestPhase2HardeningAndBridges:
    """Tests for downsample_volume (#7), CIL/xarray duck-typing (#210), and runtime hardening."""

    def test_downsample_volume_majority_mean_nearest_and_voxelimage(self):
        # 3-phase volume {0, 10, 255} with non-multiple trailing slice (9, 8, 8)
        vol = np.zeros((9, 8, 8), dtype=np.int32)
        vol[:4, :, :] = 10
        vol[4:8, :, :4] = 255
        # Tie block at [4:6, 0:2, 4:6]: four 0s and four 255s -> tie broken to smallest ID (0)
        vol[4:6, 0:2, 4] = 255

        with pytest.warns(UserWarning, match="truncating trailing slices"):
            ds, vz = oi.downsample_volume(
                vol, factor=2, mode="majority", voxel_size=(0.5, 1.0, 1.5)
            )
        assert ds.shape == (4, 4, 4)
        assert ds.dtype == np.int32
        assert set(np.unique(ds)).issubset({0, 10, 255})
        assert ds[0, 0, 0] == 10
        assert ds[2, 0, 0] == 255
        assert ds[2, 0, 2] == 0
        assert vz == pytest.approx((1.0, 2.0, 3.0))

        # >32 unique labels exercises scipy.stats.mode path
        many = np.repeat(np.repeat(np.repeat(np.arange(64, dtype=np.int32).reshape(4, 4, 4), 2, axis=0), 2, axis=1), 2, axis=2)
        ds_many = oi.downsample_volume(many, factor=2, mode="majority")
        np.testing.assert_array_equal(ds_many, np.arange(64, dtype=np.int32).reshape(4, 4, 4))

        # Grayscale mean & float32 preservation + anisotropic factors
        gray = np.arange(64, dtype=np.float32).reshape((4, 4, 4))
        ds_mean, vz_aniso = oi.downsample_volume(gray, factor=(1, 2, 4), mode="mean", voxel_size=2.0)
        assert ds_mean.shape == (4, 2, 1)
        assert ds_mean.dtype == np.float32
        assert vz_aniso == pytest.approx((2.0, 4.0, 8.0))
        assert ds_mean[0, 0, 0] == pytest.approx(np.mean(gray[0, :2, :4]))

        # Nearest / factor=1 / validation errors
        ds_near = oi.downsample_volume(vol[:8, :8, :8], factor=2, mode="nearest")
        np.testing.assert_array_equal(ds_near, vol[:8:2, :8:2, :8:2])
        ds_one = oi.downsample_volume(vol[:4, :4, :4], factor=1, mode="majority")
        np.testing.assert_array_equal(ds_one, vol[:4, :4, :4])

        with pytest.raises(ValueError, match="mode='mean'"):
            oi.downsample_volume(np.linspace(0.1, 0.9, 8).reshape(2, 2, 2), factor=2, mode="majority")
        with pytest.raises(ValueError, match="exceeds volume shape"):
            oi.downsample_volume(np.zeros((2, 2, 2), dtype=np.int32), factor=4)

        with oi.Session():
            vimg = oi.VoxelImage.from_numpy(
                np.ones((8, 8, 8), dtype=np.int32), max_grid_size=8, dx=0.25, dy=0.5, dz=1.0
            )
            vimg_ds = vimg.downsample(factor=2)
            assert vimg_ds.shape == (4, 4, 4)
            assert vimg_ds.voxel_size == pytest.approx((2.0, 1.0, 0.5))

    def test_cil_imagedata_xarray_and_sliceable_duck_typing(self):
        # 1. Mock STFC CIL ImageData + ImageGeometry
        class _MockCILGeometry:
            voxel_size_z = 2.0
            voxel_size_y = 3.0
            voxel_size_x = 4.0

        class _MockCILImageData:
            def __init__(self, arr):
                self._arr = arr
                self.geometry = _MockCILGeometry()

            def as_array(self):
                return self._arr

        slab = np.zeros((8, 8, 8), dtype=np.int32)
        slab[:4, :, :] = 1  # planar z-interface at z = 4 * dz -> area = (8*3)*(8*4), V = 8^3 * 2*3*4
        cil_obj = _MockCILImageData(slab)

        # 2. Mock xarray.DataArray with physical coordinate vectors
        class _MockDataArray:
            def __init__(self, arr):
                self.values = arr
                self.dims = ("z", "y", "x")
                self.coords = {
                    "z": np.arange(arr.shape[0], dtype=float) * 0.5,
                    "y": np.arange(arr.shape[1], dtype=float) * 0.5,
                    "x": np.arange(arr.shape[2], dtype=float) * 0.5,
                }

        xr_obj = _MockDataArray(np.ones((8, 8, 8), dtype=np.int32))

        # 3. Mock sliceable dataset (like h5py.Dataset without __array__)
        class _MockSliceableDataset:
            def __init__(self, arr):
                self._arr = arr
                self.shape = arr.shape
                self.dtype = arr.dtype

            def __getitem__(self, item):
                return self._arr[item]

        ds_obj = _MockSliceableDataset(np.ones((8, 8, 8), dtype=np.int32))

        with oi.Session():
            ssa_cil = oi.specific_surface_area(cil_obj, phase_a=1, phase_b=0)
            # Raw 1-plane interface across z has area = Ny*Nx*dy*dx; total volume = Nz*Ny*Nx*dz*dy*dx -> 1/(Nz*dz) = 1/16
            assert ssa_cil.ssa == pytest.approx(1.0 / (8.0 * 2.0), rel=1e-5)

            tau_xr = oi.tortuosity(xr_obj, phase=1, direction="z")
            assert tau_xr.tortuosity == pytest.approx(1.0, abs=1e-4)

            vf_ds = oi.volume_fraction(ds_obj, phase=1)
            assert vf_ds.fraction == pytest.approx(1.0)

    def test_backend_env_override_py_typed_version_guard_and_estimate_memory(
        self, tmp_path, monkeypatch
    ):
        from openimpala import session as sess_mod

        # PEP 561 py.typed marker present
        pkg_dir = Path(oi.__file__).resolve().parent
        assert (pkg_dir / "py.typed").is_file()

        # OPENIMPALA_BACKEND validation
        monkeypatch.setenv("OPENIMPALA_BACKEND", "invalid_backend")
        with pytest.raises(ValueError, match="OPENIMPALA_BACKEND"):
            sess_mod._requested_backend()

        monkeypatch.setenv("OPENIMPALA_BACKEND", "pure")
        info_pure = oi.build_info()
        assert info_pure["backend"] == "pure-python"
        assert info_pure["core_available"] is False
        assert "OPENIMPALA_BACKEND=pure" in str(info_pure["core_load_error"])
        monkeypatch.delenv("OPENIMPALA_BACKEND", raising=False)

        # Simulate pip install --target inside an unrelated host project's pyproject.toml
        fake_root = tmp_path / "host_project"
        fake_pkg = fake_root / "vendor" / "openimpala"
        fake_pkg.mkdir(parents=True)
        (fake_root / "pyproject.toml").write_text(
            '[project]\nname = "unrelated-host-app"\nversion = "0.0.1"\n',
            encoding="utf-8",
        )
        fake_init = fake_pkg / "__init__.py"
        fake_init.write_text("# stub\n", encoding="utf-8")
        monkeypatch.setattr(oi, "__file__", str(fake_init))
        # Must NOT return "0.0.1" from the unrelated host project!
        assert oi._detect_version() != "0.0.1"

        # Getting-started estimate_memory signature compatibility
        est_mlmg = oi.estimate_memory((100, 100, 100), solver="mlmg", mpi_ranks=4)
        est_hypre = oi.estimate_memory((100, 100, 100), num_ranks=4)
        assert est_mlmg["mpi_ranks"] == 4
        assert est_mlmg["per_rank_gb"] == est_mlmg["gb_per_rank"]
        assert est_mlmg["bytes_per_rank"] < est_hypre["bytes_per_rank"]


