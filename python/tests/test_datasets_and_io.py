"""Tests for openimpala.datasets, multi-dtype I/O round-trips, tortuosity_all(), and HTML reprs."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import openimpala as oi
from openimpala._io import (
    PureReader,
    PureVoxelImage,
    read_dat_volume,
    read_hdf5_volume,
    read_raw_volume,
    read_tiff_volume,
)
from openimpala.exceptions import PercolationError
from openimpala.session import Session, _has_core


def _write_uncompressed_tiff(path: Path, vol: np.ndarray) -> None:
    """Write a dependency-free uncompressed little-endian 2-D or 3-D Classic TIFF."""
    is_2d = vol.ndim == 2
    arr = np.ascontiguousarray(vol[np.newaxis, :, :] if is_2d else vol)
    dt = arr.dtype.newbyteorder("<")
    arr = arr.astype(dt, copy=False)
    nz, ny, nx = arr.shape
    bps = dt.itemsize * 8
    if dt.kind == "u":
        sfmt = 1
    elif dt.kind == "i":
        sfmt = 2
    elif dt.kind == "f":
        sfmt = 3
    else:
        raise ValueError(f"Unsupported dtype {dt}")

    chunks = [b"II" + struct.pack("<HI", 42, 8)]
    pos = 8
    for z in range(nz):
        raw = arr[z].tobytes()
        pad = b"\x00" if (len(raw) % 2) else b""
        data_off = pos + 2 + 10 * 12 + 4
        next_ifd = (data_off + len(raw) + len(pad)) if z < nz - 1 else 0
        entries = [
            (256, 4, struct.pack("<I", nx)),
            (257, 4, struct.pack("<I", ny)),
            (258, 3, struct.pack("<HH", bps, 0)),
            (259, 3, struct.pack("<HH", 1, 0)),
            (262, 3, struct.pack("<HH", 1, 0)),
            (273, 4, struct.pack("<I", data_off)),
            (277, 3, struct.pack("<HH", 1, 0)),
            (278, 4, struct.pack("<I", ny)),
            (279, 4, struct.pack("<I", len(raw))),
            (339, 3, struct.pack("<HH", sfmt, 0)),
        ]
        ifd = struct.pack("<H", len(entries))
        for tag, ftype, val4 in entries:
            ifd += struct.pack("<HHI4s", tag, ftype, 1, val4)
        ifd += struct.pack("<I", next_ifd)
        chunks.extend([ifd, raw, pad])
        pos = next_ifd
    path.write_bytes(b"".join(chunks))


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

    def test_stdlib_tiff_matches_tifffile_and_bigtiff(self, tmp_path):
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

        rng = np.random.default_rng(7)
        vol_u16 = rng.integers(0, 1000, size=(6, 19, 23), dtype=np.uint16)
        p_be = tmp_path / "u16_be.tif"
        tifffile.imwrite(str(p_be), vol_u16.astype(">u2"))
        np.testing.assert_array_equal(read_tiff_volume(p_be), vol_u16)

        p_tiled = tmp_path / "u16_tiled.tif"
        tifffile.imwrite(str(p_tiled), vol_u16, tile=(16, 16))
        np.testing.assert_array_equal(read_tiff_volume(p_tiled), vol_u16)

        # BigTIFF and compressed TIFF fallback via tifffile
        p_big = tmp_path / "u16_big.tif"
        tifffile.imwrite(str(p_big), vol_u16, bigtiff=True)
        np.testing.assert_array_equal(read_tiff_volume(p_big), vol_u16)

        p_zlib = tmp_path / "u16_zlib.tif"
        tifffile.imwrite(str(p_zlib), vol_u16, compression="zlib")
        np.testing.assert_array_equal(read_tiff_volume(p_zlib), vol_u16)

    def test_fetch_path_rejects_traversal(self):
        for bad in ("../README.md", "sub/file.tif", "sub\\file.tif", ".hidden.tif"):
            with pytest.raises(ValueError):
                oi.datasets.fetch_path(bad)

    def test_fetch_path_cache_hit_and_download_fallback(self, tmp_path, monkeypatch):
        import urllib.request

        missing_repo = tmp_path / "no_repo_data"
        cache_dir = tmp_path / "user_cache"
        monkeypatch.setattr(oi.datasets, "_repo_data_dir", lambda: missing_repo)
        monkeypatch.setenv("OPENIMPALA_DATA_DIR", str(cache_dir))

        # 1. Pre-populated cache hit does not touch network
        cached_file = cache_dir / "cached_sample.tif"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_bytes(b"VALID_BYTES")

        def _forbid_network(*args, **kwargs):
            raise AssertionError("Network should not be called on cache hit")

        monkeypatch.setattr(urllib.request, "urlretrieve", _forbid_network)
        assert oi.datasets.fetch_path("cached_sample.tif") == cached_file

        # 2. Cache miss downloads atomically into $OPENIMPALA_DATA_DIR
        def _fake_download(url: str, dest: str):
            assert url.endswith("/downloaded.tif")
            Path(dest).write_bytes(b"DOWNLOADED_PAYLOAD")
            return (dest, None)

        monkeypatch.setattr(urllib.request, "urlretrieve", _fake_download)
        dl_path = oi.datasets.fetch_path("downloaded.tif")
        assert dl_path == cache_dir / "downloaded.tif"
        assert dl_path.read_bytes() == b"DOWNLOADED_PAYLOAD"

        # 3. Failed download cleans up temporary .part file
        def _failing_download(url: str, dest: str):
            Path(dest).write_bytes(b"PARTIAL")
            raise OSError("simulated connection drop")

        monkeypatch.setattr(urllib.request, "urlretrieve", _failing_download)
        with pytest.raises(OSError, match="simulated connection drop"):
            oi.datasets.fetch_path("broken_download.tif")
        assert list(cache_dir.glob("*.part")) == []
        assert not (cache_dir / "broken_download.tif").exists()


class TestReadImageAndMultiDtypeIO:
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

    @pytest.mark.parametrize(
        "dtype",
        [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64],
    )
    def test_read_image_tiff_all_numeric_dtypes(self, tmp_path, dtype):
        rng = np.random.default_rng(19)
        mask = rng.integers(0, 2, size=(5, 7, 9), dtype=np.int32)
        scaled = (mask * 20 + 5).astype(dtype)
        tif_path = tmp_path / f"vol_{np.dtype(dtype).name}.tif"
        _write_uncompressed_tiff(tif_path, scaled)

        _, img = oi.read_image(tif_path, threshold=12.0)
        np.testing.assert_array_equal(img.to_numpy(), mask)

    @pytest.mark.parametrize(
        ("raw_name", "np_dt"),
        [
            ("uint8", "<u1"),
            ("int8", "<i1"),
            ("int16_le", "<i2"),
            ("int16_be", ">i2"),
            ("uint16_le", "<u2"),
            ("uint16_be", ">u2"),
            ("int32_le", "<i4"),
            ("int32_be", ">i4"),
            ("uint32_le", "<u4"),
            ("uint32_be", ">u4"),
            ("float32_le", "<f4"),
            ("float32_be", ">f4"),
            ("float64_le", "<f8"),
            ("float64_be", ">f8"),
        ],
    )
    def test_read_image_raw_all_dtypes_and_endianness(self, tmp_path, raw_name, np_dt):
        rng = np.random.default_rng(23)
        mask = rng.integers(0, 2, size=(4, 6, 8), dtype=np.int32)
        payload = (mask * 30 + 2).astype(np.dtype(np_dt))
        raw_path = tmp_path / f"sample_{raw_name}.raw"
        raw_path.write_bytes(payload.tobytes())

        _, img = oi.read_image(
            raw_path,
            threshold=15.0,
            raw_width=8,
            raw_height=6,
            raw_depth=4,
            raw_data_type=raw_name,
        )
        np.testing.assert_array_equal(img.to_numpy(), mask)

    def test_read_image_dat_npy_and_numpy_dtype_coercion(self, tmp_path):
        rng = np.random.default_rng(11)
        vol = rng.integers(0, 2, size=(6, 8, 10), dtype=np.int32)

        # .dat (3x int32 LE header W,H,D + uint16 LE voxels)
        dat_path = tmp_path / "sample.dat"
        header = struct.pack("<iii", 10, 8, 6)
        dat_path.write_bytes(header + (vol.astype("<u2") * 100).tobytes())
        _, img_dat = oi.read_image(dat_path, threshold=50)
        np.testing.assert_array_equal(img_dat.to_numpy(), vol)

        # .raw with np.dtype object and explicit file_format override
        raw_path = tmp_path / "sample.bin"
        raw_path.write_bytes(vol.astype("<f4").tobytes())
        _, img_raw = oi.read_image(
            raw_path,
            threshold=0.5,
            file_format="raw",
            raw_width=10,
            raw_height=8,
            raw_depth=6,
            raw_data_type=np.dtype("<f4"),
        )
        np.testing.assert_array_equal(img_raw.to_numpy(), vol)

        # .npy preserves multi-phase labels when passed directly to facade, and thresholds via read_image
        multi = rng.integers(0, 3, size=(8, 8, 8), dtype=np.int32)
        npy_path = tmp_path / "multi.npy"
        np.save(npy_path, multi)
        vf2 = oi.volume_fraction(npy_path, phase=2)
        assert vf2.phase_count == int(np.sum(multi == 2))
        _, img_npy = oi.read_image(npy_path, threshold=1.5)
        np.testing.assert_array_equal(img_npy.to_numpy(), (multi > 1).astype(np.int32))

    @pytest.mark.parametrize(
        "dtype",
        [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64],
    )
    def test_read_image_hdf5_all_numeric_dtypes(self, tmp_path, dtype):
        h5py = pytest.importorskip("h5py")
        rng = np.random.default_rng(31)
        mask = rng.integers(0, 2, size=(5, 7, 9), dtype=np.int32)
        scaled = (mask * 40 + 4).astype(dtype)

        h5_path = tmp_path / f"vol_{np.dtype(dtype).name}.h5"
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset("micro", data=scaled)
            ds.attrs["material"] = "NMC811"
            ds.attrs["porosity_target"] = float(0.4)
            ds.attrs["scan_id"] = np.int32(42)

        _, img = oi.read_image(h5_path, threshold=20.0, hdf5_dataset="micro")
        np.testing.assert_array_equal(img.to_numpy(), mask)

        # Canonical dataset check
        canon_h5 = oi.datasets.fetch_path("SampleData_2Phase_3d.hdf5")
        _, img_canon = oi.read_image(canon_h5, threshold=0.5, hdf5_dataset="image")
        assert img_canon.shape == (100, 100, 100)

    def test_core_cpp_readers_sequence_attributes_and_custom_thresholds(self, tmp_path):
        """Exercise C++ pybind11 reader methods (sequence, attributes, custom threshold) when _core is built."""
        if not _has_core():
            pytest.skip("Compiled _core extension not present in pure-Python run")

        Session.ensure_active()
        import openimpala._core as _core

        rng = np.random.default_rng(41)
        mask = rng.integers(0, 2, size=(4, 6, 8), dtype=np.int32)

        # 1. TiffReader: empty repr, 2-D slice sequence, metadata, and 4-arg custom threshold
        empty_tiff = _core.TiffReader()
        assert "no file loaded" in repr(empty_tiff)
        for z in range(4):
            _write_uncompressed_tiff(tmp_path / f"slice_{z:02d}.tif", (mask[z] * 100).astype(np.uint16))
        seq_reader = _core.TiffReader(str(tmp_path / "slice_"), 4, 0, 2, ".tif")
        assert seq_reader.is_read
        assert (seq_reader.width, seq_reader.height, seq_reader.depth) == (8, 6, 4)
        assert seq_reader.bits_per_sample == 16
        assert seq_reader.samples_per_pixel == 1
        assert "8x6x4" in repr(seq_reader)
        custom_img = seq_reader.threshold(50.0, 7, 3, 16)
        np.testing.assert_array_equal(custom_img.to_numpy(), np.where(mask == 1, 7, 3))

        # 2. RawReader: empty repr, get_value(i, j, k), data_type, and 4-arg custom threshold
        empty_raw = _core.RawReader()
        assert "no file loaded" in repr(empty_raw)
        raw_vals = np.arange(4 * 6 * 8, dtype="<f4").reshape(4, 6, 8)
        raw_path = tmp_path / "grad.raw"
        raw_path.write_bytes(raw_vals.tobytes())
        raw_reader = _core.RawReader(str(raw_path), 8, 6, 4, _core.RawDataType.FLOAT32_LE)
        assert raw_reader.data_type == _core.RawDataType.FLOAT32_LE
        assert raw_reader.get_value(5, 3, 2) == pytest.approx(float(raw_vals[2, 3, 5]))
        assert "8x6x4" in repr(raw_reader)
        raw_custom = raw_reader.threshold(50.0, 9, 1, 16)
        np.testing.assert_array_equal(raw_custom.to_numpy(), np.where(raw_vals > 50.0, 9, 1))

        # 3. DatReader: empty repr, get_raw_value(i, j, k), and 4-arg custom threshold
        empty_dat = _core.DatReader()
        assert "no file loaded" in repr(empty_dat)
        dat_vals = np.arange(4 * 6 * 8, dtype="<u2").reshape(4, 6, 8)
        dat_path = tmp_path / "grad.dat"
        dat_path.write_bytes(struct.pack("<iii", 8, 6, 4) + dat_vals.tobytes())
        dat_reader = _core.DatReader(str(dat_path))
        assert dat_reader.get_raw_value(5, 3, 2) == int(dat_vals[2, 3, 5])
        assert "8x6x4" in repr(dat_reader)
        dat_custom = dat_reader.threshold(50, 5, 2, 16)
        np.testing.assert_array_equal(dat_custom.to_numpy(), np.where(dat_vals > 50, 5, 2))

        # 4. HDF5Reader: attributes and 4-arg custom threshold (if h5py available to write fixture)
        try:
            import h5py
        except ImportError:
            return
        h5_path = tmp_path / "attrs.h5"
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset("vol", data=(mask * 20).astype(np.int16))
            ds.attrs["sample"] = "electrode_A"
            ds.attrs["voxel_um"] = np.float64(0.25)
            ds.attrs["phase_id"] = np.int32(1)
        empty_h5 = _core.HDF5Reader()
        assert "no file loaded" in repr(empty_h5)
        h5_reader = _core.HDF5Reader(str(h5_path), "vol")
        assert "8x6x4" in repr(h5_reader)
        assert h5_reader.get_attribute("sample") == "electrode_A"
        assert "not found" in h5_reader.get_attribute("nonexistent").lower()
        all_attrs = h5_reader.get_all_attributes()
        assert all_attrs["sample"] == "electrode_A"
        assert "voxel_um" in all_attrs and "phase_id" in all_attrs
        h5_custom = h5_reader.threshold(10.0, 4, 0, 16)
        np.testing.assert_array_equal(h5_custom.to_numpy(), np.where(mask == 1, 4, 0))

    def test_pure_io_validation_and_helpers(self, tmp_path):
        # PureVoxelImage & PureReader validation + protocol helpers
        arr = np.ones((4, 5, 6), dtype=np.int32)
        pimg = PureVoxelImage.from_numpy(arr, dx=0.5, dy=1.0, dz=2.0)
        assert "6x5x4" in repr(pimg)
        assert np.asarray(pimg, dtype=np.float32).dtype == np.float32
        with pytest.raises(ValueError):
            PureVoxelImage(np.ones((4, 5), dtype=np.int32))
        with pytest.raises(ValueError):
            PureVoxelImage(arr, dx=-1.0)

        pread_2d = PureReader(np.array([[0, 10], [20, 30]], dtype=np.uint8), kind="SliceReader")
        assert (pread_2d.depth, pread_2d.height, pread_2d.width) == (1, 2, 2)
        assert "SliceReader 2x2x1" in repr(pread_2d)
        np.testing.assert_array_equal(
            pread_2d.threshold(15, 9, 2).to_numpy(),
            np.array([[[2, 2], [9, 9]]], dtype=np.int32),
        )
        with pytest.raises(ValueError):
            PureReader(np.ones((2, 2, 2, 2)))

        # Malformed file errors in pure-Python readers
        tiny = tmp_path / "tiny.tif"
        tiny.write_bytes(b"short")
        with pytest.raises(ValueError):
            read_tiff_volume(tiny)

        bad_hdr = tmp_path / "bad.tif"
        bad_hdr.write_bytes(b"XX00112233")
        with pytest.raises(ValueError):
            read_tiff_volume(bad_hdr)

        bad_dat = tmp_path / "short.dat"
        bad_dat.write_bytes(b"1234")
        with pytest.raises(ValueError):
            read_dat_volume(bad_dat)

        trunc_dat = tmp_path / "trunc.dat"
        trunc_dat.write_bytes(struct.pack("<iii", 4, 4, 4) + b"\x00\x00")
        with pytest.raises(ValueError):
            read_dat_volume(trunc_dat)

        with pytest.raises(ValueError):
            read_raw_volume(trunc_dat, 0, 4, 4)
        with pytest.raises(ValueError):
            read_raw_volume(trunc_dat, 10, 10, 10, raw_data_type="uint8")
        with pytest.raises(ValueError):
            oi.read_image(tmp_path / "unknown.xyz")
        with pytest.raises(ValueError):
            oi.read_image(trunc_dat, file_format="unsupported_fmt")

        h5py = pytest.importorskip("h5py")
        h5_bad = tmp_path / "2d.h5"
        with h5py.File(h5_bad, "w") as f:
            f.create_dataset("flat", data=np.zeros((4, 4), dtype=np.uint8))
        with pytest.raises(KeyError):
            read_hdf5_volume(h5_bad, dataset="missing")
        with pytest.raises(ValueError):
            read_hdf5_volume(h5_bad, dataset="flat")


class TestTortuosityAllHtmlReprsAndSession:
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

    def test_all_result_dataclasses_have_html_repr_and_estimate_memory(self):
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

        mem = oi.estimate_memory((100, 100, 100), num_ranks=4)
        assert mem["total_voxels"] == 1_000_000
        assert mem["voxels_per_rank"] == 250_000
        assert mem["num_ranks"] == 4

        # Nested Session re-entrancy preserves active session state
        with oi.Session():
            with oi.Session():
                assert oi.volume_fraction(vol, phase=1).fraction == pytest.approx(0.5)

