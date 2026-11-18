"""Tests for openimpala.datasets, pure-Python read_image(), tortuosity_all(), and HTML reprs."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import openimpala as oi
from openimpala._io import read_tiff_volume
from openimpala.exceptions import PercolationError


class TestDatasets:
    def test_loaders_shape_dtype_and_values(self):
        stack = oi.datasets.two_phase_stack()
        assert stack.shape == (100, 100, 100)
        assert stack.dtype == np.int32
        assert stack.flags["C_CONTIGUOUS"]
        assert set(np.unique(stack)) == {0, 1}
        assert np.mean(stack == 1) == pytest.approx(0.398309, rel=1e-6)

        cubic = oi.datasets.two_phase_cubic()
        assert cubic.shape == (64, 64, 64)
        assert cubic.dtype == np.int32
        assert np.mean(cubic == 1) == pytest.approx(0.3993263244628906, rel=1e-6)

        spheres = oi.datasets.sphere_pack()
        assert spheres.shape == (100, 100, 100)
        assert spheres.dtype == np.int32
        assert np.mean(spheres == 0) == pytest.approx(0.111976, rel=1e-6)

    def test_stdlib_tiff_matches_tifffile(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        for fn in (
            "SampleData_2Phase_stack_3d_1bit.tif",
            "SampleData_2Phase_squared.tif",
            "spheres.tif",
        ):
            p = oi.datasets.fetch_path(fn)
            expected = tifffile.imread(str(p)).astype(np.int32)
            actual = read_tiff_volume(p).astype(np.int32)
            np.testing.assert_array_equal(actual, expected)

        # Also verify 16-bit big-endian and tiled uncompressed 3-D TIFF decoding
        rng = np.random.default_rng(7)
        vol_u16 = rng.integers(0, 1000, size=(6, 19, 23), dtype=np.uint16)
        p_be = tmp_path / "u16_be.tif"
        tifffile.imwrite(str(p_be), vol_u16.astype(">u2"))
        np.testing.assert_array_equal(read_tiff_volume(p_be), vol_u16)

        p_tiled = tmp_path / "u16_tiled.tif"
        tifffile.imwrite(str(p_tiled), vol_u16, tile=(16, 16))
        np.testing.assert_array_equal(read_tiff_volume(p_tiled), vol_u16)

    def test_fetch_path_rejects_traversal(self):
        with pytest.raises(ValueError):
            oi.datasets.fetch_path("../README.md")


class TestReadImageAndPathInputs:
    def test_read_image_tiff_and_voxel_size(self):
        path = oi.datasets.fetch_path("SampleData_2Phase_squared.tif")
        reader, img = oi.read_image(path, threshold=0.5, voxel_size=(2.0, 1.5, 1.0))
        assert reader.is_read
        assert (reader.width, reader.height, reader.depth) == (64, 64, 64)
        assert img.shape == (64, 64, 64)
        assert img.voxel_size == pytest.approx((2.0, 1.5, 1.0))
        assert img.cell_size_xyz == pytest.approx((1.0, 1.5, 2.0))

        vf = oi.volume_fraction(img, phase=1)
        assert vf.fraction == pytest.approx(0.3993263244628906, rel=1e-6)

        # Direct path input to facade functions
        vf_direct = oi.volume_fraction(path, phase=1)
        assert vf_direct.phase_count == vf.phase_count

    def test_read_image_dat_raw_npy_and_hdf5(self, tmp_path):
        rng = np.random.default_rng(11)
        vol = rng.integers(0, 2, size=(6, 8, 10), dtype=np.int32)

        # .dat (3x int32 LE header W,H,D + uint16 LE voxels)
        dat_path = tmp_path / "sample.dat"
        header = struct.pack("<iii", 10, 8, 6)
        dat_path.write_bytes(header + (vol.astype("<u2") * 100).tobytes())
        _, img_dat = oi.read_image(dat_path, threshold=50)
        np.testing.assert_array_equal(img_dat.to_numpy(), vol)

        # .raw (float32 LE)
        raw_path = tmp_path / "sample.raw"
        raw_path.write_bytes(vol.astype("<f4").tobytes())
        _, img_raw = oi.read_image(
            raw_path,
            threshold=0.5,
            raw_width=10,
            raw_height=8,
            raw_depth=6,
            raw_data_type="float32_le",
        )
        np.testing.assert_array_equal(img_raw.to_numpy(), vol)

        # .npy (preserves multi-phase labels when passed directly to facade)
        multi = rng.integers(0, 3, size=(8, 8, 8), dtype=np.int32)
        npy_path = tmp_path / "multi.npy"
        np.save(npy_path, multi)
        vf2 = oi.volume_fraction(npy_path, phase=2)
        assert vf2.phase_count == int(np.sum(multi == 2))

        # .hdf5
        h5py = pytest.importorskip("h5py")
        h5_path = oi.datasets.fetch_path("SampleData_2Phase_3d.hdf5")
        _, img_h5 = oi.read_image(h5_path, threshold=0.5, hdf5_dataset="image")
        assert img_h5.shape == (100, 100, 100)


class TestTortuosityAllAndHtmlReprs:
    def test_tortuosity_all_handles_partial_percolation(self):
        # Slab structure: phase 0 spans X and Y, blocked along Z
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[4:6, :, :] = 1

        res = oi.tortuosity_all(vol, phase=0)
        assert set(res.keys()) == {"x", "y", "z"}
        assert res["x"] is not None and res["x"].tortuosity == pytest.approx(1.0, rel=1e-6)
        assert res["y"] is not None and res["y"].tortuosity == pytest.approx(1.0, rel=1e-6)
        assert res["z"] is None

        with pytest.raises(PercolationError):
            oi.tortuosity_all(vol, phase=0, strict=True)

    def test_all_result_dataclasses_have_html_repr(self):
        vol = np.zeros((12, 12, 12), dtype=np.int32)
        vol[:, :, 6:] = 1

        results = [
            oi.volume_fraction(vol, phase=1),
            oi.percolation_check(vol, phase=1, direction="z"),
            oi.tortuosity(vol, phase=1, direction="z"),
            oi.effective_diffusivity(vol, phase=1),
            oi.specific_surface_area(vol, phase_a=0, phase_b=1),
            oi.connected_components(vol, phase=1),
            oi.particle_size_distribution(vol, phase=1),
            oi.through_thickness_profile(vol, phase=1, direction="x"),
            oi.rev_study(vol, sizes=[6, 10], num_samples=2, phase=1),
        ]
        for r in results:
            html = r._repr_html_()
            assert html.startswith("<table>") and html.endswith("</table>")
