"""Pure-Python 3-D voxel image readers matching OpenImpala C++ I/O semantics.

Supports uncompressed 1/8/16/32/64-bit stripped and tiled TIFF stacks using the
Python standard library (falling back to ``tifffile`` for compressed TIFFs),
HDF5 datasets (via optional ``h5py``), legacy ``.dat`` files, flat binary ``.raw``
volumes, and ``.npy`` arrays.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# RawDataType integer codes matching src/io/RawReader.H & python/bindings/enums.cpp
_RAW_DTYPE_MAP: dict[str, str] = {
    "uint8": "<u1",
    "int8": "<i1",
    "int16_le": "<i2",
    "int16_be": ">i2",
    "uint16_le": "<u2",
    "uint16_be": ">u2",
    "int32_le": "<i4",
    "int32_be": ">i4",
    "uint32_le": "<u4",
    "uint32_be": ">u4",
    "float32_le": "<f4",
    "float32_be": ">f4",
    "float64_le": "<f8",
    "float64_be": ">f8",
}

_TIFF_TYPE_INFO: dict[int, tuple[int, str]] = {
    1: (1, "B"),   # BYTE
    2: (1, "s"),   # ASCII
    3: (2, "H"),   # SHORT
    4: (4, "I"),   # LONG
    5: (8, "II"),  # RATIONAL
    6: (1, "b"),   # SBYTE
    7: (1, "B"),   # UNDEFINED
    8: (2, "h"),   # SSHORT
    9: (4, "i"),   # SLONG
    11: (4, "f"),  # FLOAT
    12: (8, "d"),  # DOUBLE
    16: (8, "Q"),  # LONG8 (BigTIFF)
    17: (8, "q"),  # SLONG8 (BigTIFF)
    18: (8, "Q"),  # IFD8 (BigTIFF)
}


class PureVoxelImage:
    """Pure-Python counterpart to ``openimpala._core.VoxelImage``."""

    def __init__(
        self,
        arr: np.ndarray,
        *,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        periodic: Sequence[bool] = (False, False, False),
    ) -> None:
        arr_c = np.ascontiguousarray(arr, dtype=np.int32)
        if arr_c.ndim != 3:
            raise ValueError(f"Expected a 3-D array, got shape {arr_c.shape}")
        if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            raise ValueError("Voxel sizes (dx, dy, dz) must all be strictly positive.")
        self._arr = arr_c
        self._dx = float(dx)
        self._dy = float(dy)
        self._dz = float(dz)
        self._periodic = (bool(periodic[0]), bool(periodic[1]), bool(periodic[2]))

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        max_grid_size: int = 32,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        periodic: Sequence[bool] = (False, False, False),
    ) -> "PureVoxelImage":
        del max_grid_size
        return cls(arr, dx=dx, dy=dy, dz=dz, periodic=periodic)

    def with_voxel_size(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        periodic: Sequence[bool] = (False, False, False),
    ) -> "PureVoxelImage":
        return PureVoxelImage(self._arr, dx=dx, dy=dy, dz=dz, periodic=periodic)

    @property
    def shape(self) -> tuple[int, int, int]:
        nz, ny, nx = self._arr.shape
        return (int(nz), int(ny), int(nx))

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        return (self._dz, self._dy, self._dx)

    @property
    def cell_size_xyz(self) -> tuple[float, float, float]:
        return (self._dx, self._dy, self._dz)

    def to_numpy(self) -> np.ndarray:
        return self._arr.copy()

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype is None:
            return self._arr
        return self._arr.astype(dtype, copy=False)

    def __repr__(self) -> str:
        nz, ny, nx = self._arr.shape
        return f"<VoxelImage {nx}x{ny}x{nz}>"


class PureReader:
    """Pure-Python reader handle matching C++ reader metadata & thresholding."""

    def __init__(self, raw_volume: np.ndarray, kind: str = "PureReader") -> None:
        if raw_volume.ndim == 2:
            raw_volume = raw_volume[np.newaxis, :, :]
        if raw_volume.ndim != 3:
            raise ValueError(f"Expected 3-D volume (Nz, Ny, Nx), got shape {raw_volume.shape}")
        self._raw = raw_volume
        self._kind = kind

    @property
    def depth(self) -> int:
        return int(self._raw.shape[0])

    @property
    def height(self) -> int:
        return int(self._raw.shape[1])

    @property
    def width(self) -> int:
        return int(self._raw.shape[2])

    @property
    def is_read(self) -> bool:
        return True

    def threshold(
        self,
        raw_threshold: float,
        value_if_true_or_grid: int = 32,
        value_if_false: Optional[int] = None,
        max_grid_size: int = 32,
    ) -> PureVoxelImage:
        del max_grid_size
        if value_if_false is None:
            v_true, v_false = 1, 0
        else:
            v_true = int(value_if_true_or_grid)
            v_false = int(value_if_false)
        binary = np.where(self._raw > raw_threshold, v_true, v_false).astype(np.int32)
        return PureVoxelImage(binary)

    def __repr__(self) -> str:
        return f"<{self._kind} {self.width}x{self.height}x{self.depth}>"


def _sample_numpy_dtype(endian: str, bps: int, sample_format: int) -> np.dtype:
    bytes_per_sample = bps // 8
    if sample_format == 1:  # unsigned int
        code = "u"
    elif sample_format == 2:  # signed int
        code = "i"
    elif sample_format == 3:  # IEEE float
        code = "f"
    else:
        raise ValueError(f"Unsupported TIFF SampleFormat={sample_format}")
    return np.dtype(f"{endian}{code}{bytes_per_sample}")


def _decode_uncompressed_chunk(
    buf: bytes,
    rows: int,
    cols: int,
    bps: int,
    fill_order: int,
    endian: str,
    sample_format: int,
) -> np.ndarray:
    if bps == 1:
        row_bytes = (cols + 7) // 8
        needed = rows * row_bytes
        raw_bytes = np.frombuffer(buf[:needed], dtype=np.uint8)
        if raw_bytes.size < needed:
            padded = np.zeros(needed, dtype=np.uint8)
            padded[: raw_bytes.size] = raw_bytes
            raw_bytes = padded
        bitorder = "big" if fill_order == 1 else "little"
        bits = np.unpackbits(raw_bytes.reshape(rows, row_bytes), axis=1, bitorder=bitorder)
        return bits[:, :cols]

    if bps in (8, 16, 32, 64):
        dt = _sample_numpy_dtype(endian, bps, sample_format)
        needed_elems = rows * cols
        needed_bytes = needed_elems * dt.itemsize
        chunk = np.frombuffer(buf[:needed_bytes], dtype=dt)
        if chunk.size < needed_elems:
            padded = np.zeros(needed_elems, dtype=dt)
            padded[: chunk.size] = chunk
            chunk = padded
        return chunk.reshape(rows, cols)

    raise ValueError(f"Unsupported uncompressed TIFF BitsPerSample={bps}")


def read_tiff_volume(path: str | os.PathLike) -> np.ndarray:
    """Read a 2-D or multi-page 3-D TIFF into a ``(Nz, Ny, Nx)`` NumPy array.

    Uncompressed 1/8/16/32/64-bit stripped and tiled TIFFs (Classic TIFF and
    BigTIFF) are decoded directly with the Python standard library, matching
    ``OpenImpala::TiffReader`` raw bit/sample semantics. Compressed TIFFs
    transparently fall back to ``tifffile`` when installed.
    """
    fpath = Path(path)
    data = fpath.read_bytes()
    if len(data) < 8:
        raise ValueError(f"File '{fpath}' is too small to be a valid TIFF.")

    byte_order = data[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        raise ValueError(f"File '{fpath}' does not have a valid TIFF header.")

    magic = struct.unpack_from(f"{endian}H", data, 2)[0]
    if magic == 42:
        bigtiff = False
        ifd_offset = struct.unpack_from(f"{endian}I", data, 4)[0]
    elif magic == 43:
        bigtiff = True
        offset_size, zero = struct.unpack_from(f"{endian}HH", data, 4)
        if offset_size != 8 or zero != 0:
            raise ValueError(f"Malformed BigTIFF header in '{fpath}'.")
        ifd_offset = struct.unpack_from(f"{endian}Q", data, 8)[0]
    else:
        raise ValueError(f"Unsupported TIFF magic number {magic} in '{fpath}'.")

    pages: list[np.ndarray] = []
    inline_limit = 8 if bigtiff else 4

    while ifd_offset != 0:
        if bigtiff:
            num_entries = struct.unpack_from(f"{endian}Q", data, ifd_offset)[0]
            pos = ifd_offset + 8
            entry_stride = 20
        else:
            num_entries = struct.unpack_from(f"{endian}H", data, ifd_offset)[0]
            pos = ifd_offset + 2
            entry_stride = 12

        tags: dict[int, tuple] = {}
        for _ in range(num_entries):
            if bigtiff:
                tag, ftype, count = struct.unpack_from(f"{endian}HHQ", data, pos)
                val_bytes = data[pos + 12 : pos + 20]
            else:
                tag, ftype, count = struct.unpack_from(f"{endian}HHI", data, pos)
                val_bytes = data[pos + 8 : pos + 12]
            pos += entry_stride

            if ftype not in _TIFF_TYPE_INFO:
                continue
            unit_size, fmt_char = _TIFF_TYPE_INFO[ftype]
            total_bytes = unit_size * count
            if total_bytes <= inline_limit:
                raw_val = val_bytes[:total_bytes]
            else:
                ext_off = struct.unpack_from(
                    f"{endian}{'Q' if bigtiff else 'I'}", val_bytes, 0
                )[0]
                raw_val = data[ext_off : ext_off + total_bytes]

            if fmt_char == "s":
                tags[tag] = (raw_val.rstrip(b"\x00").decode("latin1", errors="replace"),)
            elif ftype == 5:
                vals = struct.unpack(f"{endian}{2 * count}I", raw_val)
                tags[tag] = tuple(
                    vals[2 * i] / vals[2 * i + 1] if vals[2 * i + 1] else 0.0
                    for i in range(count)
                )
            else:
                tags[tag] = struct.unpack(f"{endian}{count}{fmt_char}", raw_val)

        ifd_offset = struct.unpack_from(
            f"{endian}{'Q' if bigtiff else 'I'}", data, pos
        )[0]

        width = int(tags[256][0])
        height = int(tags[257][0])
        bps = int(tags.get(258, (1,))[0])
        compression = int(tags.get(259, (1,))[0])
        fill_order = int(tags.get(266, (1,))[0])
        spp = int(tags.get(277, (1,))[0])
        sample_format = int(tags.get(339, (1,))[0])

        if compression != 1 or spp != 1:
            try:
                import tifffile
            except ImportError as exc:
                raise RuntimeError(
                    f"TIFF '{fpath}' uses compression={compression}, samples_per_pixel={spp}. "
                    "Install 'tifffile' (`pip install tifffile`) to read compressed or multi-channel TIFFs."
                ) from exc
            arr_tf = tifffile.imread(str(fpath))
            if arr_tf.ndim == 2:
                arr_tf = arr_tf[np.newaxis, :, :]
            return arr_tf

        out_dtype = np.uint8 if bps == 1 else _sample_numpy_dtype(endian, bps, sample_format)
        page = np.zeros((height, width), dtype=out_dtype)

        if 324 in tags:  # Tiled TIFF
            tw = int(tags[322][0])
            th = int(tags[323][0])
            t_offsets = tags[324]
            t_counts = tags[325]
            tiles_across = (width + tw - 1) // tw
            tiles_down = (height + th - 1) // th
            idx = 0
            for ty in range(tiles_down):
                for tx in range(tiles_across):
                    off = int(t_offsets[idx])
                    cnt = int(t_counts[idx])
                    idx += 1
                    tile = _decode_uncompressed_chunk(
                        data[off : off + cnt], th, tw, bps, fill_order, endian, sample_format
                    )
                    y0, x0 = ty * th, tx * tw
                    rh = min(th, height - y0)
                    rw = min(tw, width - x0)
                    page[y0 : y0 + rh, x0 : x0 + rw] = tile[:rh, :rw]
        else:  # Stripped TIFF
            rps = int(tags.get(278, (height,))[0])
            if rps <= 0 or rps > height:
                rps = height
            s_offsets = tags[273]
            s_counts = tags[279]
            for s_idx, (off, cnt) in enumerate(zip(s_offsets, s_counts)):
                y0 = s_idx * rps
                rows = min(rps, height - y0)
                if rows <= 0:
                    break
                strip = _decode_uncompressed_chunk(
                    data[int(off) : int(off) + int(cnt)],
                    rows,
                    width,
                    bps,
                    fill_order,
                    endian,
                    sample_format,
                )
                page[y0 : y0 + rows, :] = strip

        pages.append(page)

    if not pages:
        raise ValueError(f"No image directories found in TIFF '{fpath}'.")
    return np.stack(pages, axis=0)


def read_hdf5_volume(path: str | os.PathLike, dataset: str = "/data") -> np.ndarray:
    """Read a 3-D dataset from an HDF5 file into ``(Nz, Ny, Nx)`` order."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Reading HDF5 files in pure-Python mode requires 'h5py' (`pip install h5py`)."
        ) from exc

    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise KeyError(
                f"Dataset '{dataset}' not found in '{path}'. Available keys: {list(f.keys())}"
            )
        arr = np.asarray(f[dataset])
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D HDF5 dataset '{dataset}', got shape {arr.shape}")
    return arr


def read_dat_volume(path: str | os.PathLike) -> np.ndarray:
    """Read a legacy OpenImpala ``.dat`` file (3x int32 LE header + uint16 LE data)."""
    raw_bytes = Path(path).read_bytes()
    if len(raw_bytes) < 12:
        raise ValueError(f"DAT file '{path}' is too small for its 12-byte header.")
    width, height, depth = struct.unpack_from("<iii", raw_bytes, 0)
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"Invalid DAT header dimensions ({width}, {height}, {depth}) in '{path}'.")
    expected = width * height * depth
    needed_bytes = 12 + expected * 2
    if len(raw_bytes) < needed_bytes:
        raise ValueError(
            f"DAT file '{path}' truncated: expected {needed_bytes} bytes, got {len(raw_bytes)}."
        )
    vox = np.frombuffer(raw_bytes, dtype="<u2", count=expected, offset=12)
    return vox.reshape((depth, height, width)).copy()


def _resolve_raw_dtype(raw_data_type) -> np.dtype:
    if raw_data_type is None:
        return np.dtype("<u1")
    if isinstance(raw_data_type, np.dtype):
        return raw_data_type
    if isinstance(raw_data_type, str):
        key = raw_data_type.strip().lower()
        if key in _RAW_DTYPE_MAP:
            return np.dtype(_RAW_DTYPE_MAP[key])
        return np.dtype(raw_data_type)
    name = getattr(raw_data_type, "name", None)
    if isinstance(name, str) and name.lower() in _RAW_DTYPE_MAP:
        return np.dtype(_RAW_DTYPE_MAP[name.lower()])
    raise ValueError(f"Unsupported raw_data_type: {raw_data_type!r}")


def read_raw_volume(
    path: str | os.PathLike,
    width: int,
    height: int,
    depth: int,
    raw_data_type=None,
) -> np.ndarray:
    """Read a flat binary ``.raw`` volume into ``(depth, height, width)`` order."""
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(
            f"Positive (raw_width, raw_height, raw_depth) are required for RAW files, "
            f"got ({width}, {height}, {depth})."
        )
    dt = _resolve_raw_dtype(raw_data_type)
    expected = width * height * depth
    needed_bytes = expected * dt.itemsize
    raw_bytes = Path(path).read_bytes()
    if len(raw_bytes) < needed_bytes:
        raise ValueError(
            f"RAW file '{path}' is smaller ({len(raw_bytes)} B) than expected ({needed_bytes} B)."
        )
    vox = np.frombuffer(raw_bytes, dtype=dt, count=expected, offset=0)
    return vox.reshape((depth, height, width)).copy()
