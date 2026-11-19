"""Pure-Python 3-D voxel image I/O, Otsu segmentation, OME-Zarr, and AMReX plotfiles.

Supports uncompressed 1/8/16/32/64-bit stripped and tiled TIFF stacks using the
Python standard library (falling back to ``tifffile`` for compressed TIFFs),
HDF5 datasets (via optional ``h5py``), OME-NGFF v0.4 chunked Zarr stores (via
optional ``zarr``), legacy ``.dat`` files, flat binary ``.raw`` volumes, ``.npy``
arrays, zero-dependency Otsu thresholding, and single-level AMReX ``HyperCLaw-V1.1``
plotfile export compatible with ``yt`` and ParaView.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

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

    @classmethod
    def from_grayscale_otsu(
        cls,
        arr: np.ndarray,
        *,
        pore_value: int = 0,
        solid_value: int = 1,
        dark_is_pore: bool = True,
        nbins: Optional[int] = None,
        max_grid_size: int = 32,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        periodic: Sequence[bool] = (False, False, False),
    ) -> "PureVoxelImage":
        """Segment a 3-D grayscale array using Otsu's threshold and return a ``VoxelImage``."""
        del max_grid_size
        seg, _ = binarize_otsu(
            arr,
            pore_value=pore_value,
            solid_value=solid_value,
            dark_is_pore=dark_is_pore,
            nbins=nbins,
        )
        return cls(seg, dx=dx, dy=dy, dz=dz, periodic=periodic)

    @classmethod
    def from_zarr(
        cls,
        path: Union[str, os.PathLike],
        *,
        array_path: Optional[str] = None,
        level: int = 0,
        threshold: Optional[Union[float, str]] = None,
        pore_value: int = 0,
        solid_value: int = 1,
        dark_is_pore: bool = True,
        max_grid_size: int = 32,
        voxel_size: Optional[Sequence[float] | float] = None,
        periodic: Sequence[bool] = (False, False, False),
    ) -> "PureVoxelImage":
        """Load a 3-D Zarr array or OME-Zarr multiscale store into a ``VoxelImage``."""
        del max_grid_size
        arr, meta_vz = read_zarr_volume(
            path,
            array_path=array_path,
            level=level,
            threshold=threshold,
            pore_value=pore_value,
            solid_value=solid_value,
            dark_is_pore=dark_is_pore,
        )
        if voxel_size is not None:
            if isinstance(voxel_size, (int, float, np.floating, np.integer)):
                dz = dy = dx = float(voxel_size)
            else:
                dz, dy, dx = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
        elif meta_vz is not None:
            dz, dy, dx = meta_vz
        else:
            dz = dy = dx = 1.0
        return cls(arr, dx=dx, dy=dy, dz=dz, periodic=periodic)

    def to_zarr(
        self,
        path: Union[str, os.PathLike],
        *,
        chunks: tuple[int, int, int] = (64, 64, 64),
        name: str = "volume",
        unit: str = "micrometer",
        overwrite: bool = True,
    ) -> Path:
        """Export this ``VoxelImage`` to an OME-NGFF v0.4 chunked Zarr store."""
        return write_zarr_volume(
            path,
            self._arr,
            voxel_size=self.voxel_size,
            chunks=chunks,
            name=name,
            unit=unit,
            overwrite=overwrite,
        )

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
        raw_threshold: Union[float, str],
        value_if_true_or_grid: int = 32,
        value_if_false: Optional[int] = None,
        max_grid_size: int = 32,
    ) -> PureVoxelImage:
        del max_grid_size
        if isinstance(raw_threshold, str):
            if raw_threshold.strip().lower() != "otsu":
                raise ValueError(
                    f"Unsupported string threshold {raw_threshold!r}; expected a number or 'otsu'."
                )
            thresh_val = calculate_otsu_threshold(self._raw)
        else:
            thresh_val = float(raw_threshold)
        if value_if_false is None:
            v_true, v_false = 1, 0
        else:
            v_true = int(value_if_true_or_grid)
            v_false = int(value_if_false)
        binary = np.where(self._raw > thresh_val, v_true, v_false).astype(np.int32)
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


# ---------------------------------------------------------------------------
# Zero-dependency Otsu thresholding
# ---------------------------------------------------------------------------

def calculate_otsu_threshold(array: np.ndarray, *, nbins: Optional[int] = None) -> float:
    """Compute Otsu's optimal global threshold maximising between-class variance.

    Parameters
    ----------
    array : np.ndarray
        Grayscale intensity array of arbitrary shape (2-D or 3-D; ``uint8``,
        ``uint16``, integer, or floating-point).
    nbins : int, optional
        Number of histogram bins.  When ``None`` (default), integer volumes whose
        dynamic range ``max - min <= 65535`` (including all ``uint8``, ``int8``,
        ``uint16``, and ``int16`` X-ray CT tomograms) use an exact integer
        histogram with zero quantisation error, while floating-point or wide-range
        integer arrays use ``256`` linearly spaced bins.

    Returns
    -------
    float
        Optimal scalar threshold ``T`` such that ``array > T`` separates the
        bright foreground class from the dark background class.  Tied maxima
        of :math:`\\sigma_B^2` across flat histogram valleys are averaged to
        return the valley midpoint.
    """
    arr = np.asarray(array)
    if arr.size == 0:
        raise ValueError("Cannot compute Otsu threshold on an empty array.")
    if nbins is not None and int(nbins) < 2:
        raise ValueError(f"nbins must be >= 2, got {nbins}")
    if np.issubdtype(arr.dtype, np.bool_):
        arr = arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError("Input array contains NaN or infinite values.")

    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_min == v_max:
        return v_min

    use_exact_int = False
    if np.issubdtype(arr.dtype, np.integer):
        i_min = int(np.min(arr))
        i_max = int(np.max(arr))
        span = i_max - i_min
        if nbins is None and span <= 65535:
            use_exact_int = True
        elif arr.dtype == np.uint8 and nbins == 256:
            i_min, i_max, span = 0, 255, 255
            use_exact_int = True

    if use_exact_int:
        shifted = arr.ravel().astype(np.int64, copy=False) - i_min
        counts = np.bincount(shifted, minlength=span + 1).astype(np.float64)
        bin_centers = np.arange(i_min, i_max + 1, dtype=np.float64)
        split_thresholds = bin_centers[:-1] + 0.5
    else:
        eff_bins = 256 if nbins is None else int(nbins)
        counts_int, bin_edges = np.histogram(arr, bins=eff_bins, range=(v_min, v_max))
        counts = counts_int.astype(np.float64)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        split_thresholds = bin_edges[1:-1]

    probs = counts / float(arr.size)
    omega0 = np.cumsum(probs)[:-1]
    omega1 = 1.0 - omega0
    mu_cum = np.cumsum(probs * bin_centers)
    mu_total = mu_cum[-1]
    mu0_num = mu_cum[:-1]

    denom = omega0 * omega1
    sigma_b2 = np.zeros_like(omega0)
    valid = denom > 0.0
    diff = mu_total * omega0[valid] - mu0_num[valid]
    sigma_b2[valid] = (diff * diff) / denom[valid]

    max_var = float(np.max(sigma_b2))
    if max_var <= 0.0:
        return v_min
    best_indices = np.flatnonzero(np.isclose(sigma_b2, max_var, rtol=1e-12, atol=0.0))
    return float(np.mean(split_thresholds[best_indices]))


otsu_threshold = calculate_otsu_threshold


def binarize_otsu(
    array: np.ndarray,
    *,
    pore_value: int = 0,
    solid_value: int = 1,
    dark_is_pore: bool = True,
    nbins: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """Segment a grayscale volume using Otsu's method and return ``(segmented, threshold)``.

    Parameters
    ----------
    array : np.ndarray
        Grayscale volume (typically ``(Nz, Ny, Nx)``).
    pore_value : int, default 0
        Phase ID assigned to the pore phase.
    solid_value : int, default 1
        Phase ID assigned to the solid matrix phase.
    dark_is_pore : bool, default True
        When ``True`` (standard X-ray attenuation contrast), dark voxels
        (``intensity <= threshold``) map to *pore_value* and bright voxels
        (``intensity > threshold``) map to *solid_value*.  Set ``False`` for
        inverted contrast images where pores are brighter than solid.
    nbins : int, optional
        Histogram bin count passed to :func:`calculate_otsu_threshold` (``None``
        selects exact integer bins for ``uint8``/``uint16`` scans).

    Returns
    -------
    segmented : np.ndarray
        C-contiguous ``int32`` phase array with the same shape as *array*.
    threshold : float
        Computed Otsu threshold value.
    """
    arr = np.asarray(array)
    thresh = calculate_otsu_threshold(arr, nbins=nbins)
    if dark_is_pore:
        seg = np.where(arr > thresh, int(solid_value), int(pore_value)).astype(np.int32)
    else:
        seg = np.where(arr > thresh, int(pore_value), int(solid_value)).astype(np.int32)
    return np.ascontiguousarray(seg, dtype=np.int32), thresh


def _safe_rmtree_store(
    target: Path,
    *,
    kind: str,
    marker_names: tuple[str, ...],
) -> None:
    """Safely remove an existing directory store, refusing CWD, HOME, root, or non-stores."""
    import shutil

    if target.is_symlink():
        raise ValueError(
            f"Refusing to overwrite symlink '{target}' when writing {kind}."
        )
    resolved = target.resolve()
    cwd_resolved = Path.cwd().resolve()
    try:
        home_resolved = Path.home().resolve()
    except Exception:
        home_resolved = None

    if (
        resolved == Path(resolved.anchor)
        or resolved == cwd_resolved
        or (home_resolved is not None and resolved == home_resolved)
        or cwd_resolved.is_relative_to(resolved)
    ):
        raise ValueError(
            f"Refusing to overwrite protected directory '{target}' when writing {kind}; "
            f"pass a dedicated output subdirectory path."
        )

    if not target.exists():
        return
    if not target.is_dir():
        target.unlink()
        return

    entries = [p.name for p in target.iterdir()]
    if entries and not any(m in entries for m in marker_names):
        raise ValueError(
            f"Refusing to recursively delete existing non-{kind} directory '{target}' "
            f"(missing store markers {marker_names}). Specify a dedicated output path."
        )
    shutil.rmtree(target)


# ---------------------------------------------------------------------------
# Cloud-native OME-Zarr (NGFF v0.4 / v0.5) 3-D volume I/O
# ---------------------------------------------------------------------------

def _import_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Zarr support requires the optional 'zarr' package. "
            "Install it with: pip install 'openimpala[zarr]' (or pip install zarr)."
        ) from exc
    return zarr


def _coerce_vz_tuple(voxel_size: Optional[Sequence[float] | float]) -> tuple[float, float, float]:
    if voxel_size is None:
        return (1.0, 1.0, 1.0)
    if isinstance(voxel_size, (int, float, np.floating, np.integer)):
        v = float(voxel_size)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"voxel_size must be positive and finite, got {voxel_size!r}")
        return (v, v, v)
    seq = tuple(float(x) for x in voxel_size)
    if len(seq) != 3 or any((not np.isfinite(v) or v <= 0.0) for v in seq):
        raise ValueError(
            f"voxel_size must have 3 positive finite entries (dz, dy, dx), got {voxel_size!r}"
        )
    return (seq[0], seq[1], seq[2])


def _extract_ome_zyx_scale(
    ds_meta: Mapping[str, Any],
    axes_meta: Sequence[Any],
) -> Optional[tuple[float, float, float]]:
    """Extract ``(dz, dy, dx)`` from an OME-NGFF dataset entry (supports 3-D, 4-D, 5-D)."""
    for tr in ds_meta.get("coordinateTransformations", []):
        if not isinstance(tr, Mapping) or tr.get("type") != "scale":
            continue
        raw_sc = tr.get("scale")
        if not isinstance(raw_sc, (list, tuple)) or len(raw_sc) < 3:
            continue
        if isinstance(axes_meta, (list, tuple)) and len(axes_meta) == len(raw_sc):
            idx_by_name = {
                str(ax.get("name", "")).lower(): idx
                for idx, ax in enumerate(axes_meta)
                if isinstance(ax, Mapping)
            }
            if {"z", "y", "x"}.issubset(idx_by_name):
                return (
                    float(raw_sc[idx_by_name["z"]]),
                    float(raw_sc[idx_by_name["y"]]),
                    float(raw_sc[idx_by_name["x"]]),
                )
        return (float(raw_sc[-3]), float(raw_sc[-2]), float(raw_sc[-1]))
    return None


def write_zarr_volume(
    path: Union[str, os.PathLike],
    array: np.ndarray,
    *,
    voxel_size: Optional[Sequence[float] | float] = (1.0, 1.0, 1.0),
    chunks: tuple[int, int, int] = (64, 64, 64),
    name: str = "volume",
    unit: str = "micrometer",
    overwrite: bool = True,
    zarr_format: int = 2,
) -> Path:
    """Write a 3-D volume ``(Nz, Ny, Nx)`` to an OME-NGFF chunked Zarr store.

    Defaults to ``zarr_format=2`` (OME-NGFF v0.4 on Zarr v2) for out-of-the-box
    compatibility with ``ome-zarr-py``, Napari, Fiji MoBIE, Bio-Formats,
    Neuroglancer, and ``dask``.  Passing ``zarr_format=3`` on ``zarr>=3.0``
    writes an OME-NGFF v0.5 store on Zarr v3.
    """
    zarr = _import_zarr()
    if zarr_format not in (2, 3):
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format!r}")
    arr_c = np.ascontiguousarray(array)
    if arr_c.ndim != 3:
        raise ValueError(f"Expected a 3-D array (Nz, Ny, Nx), got shape {arr_c.shape}")
    dz, dy, dx = _coerce_vz_tuple(voxel_size)
    if len(chunks) != 3 or any(int(c) <= 0 for c in chunks):
        raise ValueError(f"chunks must be 3 positive integers, got {chunks!r}")
    eff_chunks = tuple(min(int(c), int(s)) for c, s in zip(chunks, arr_c.shape))

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Zarr store '{target}' already exists and overwrite=False.")
    _safe_rmtree_store(
        target,
        kind="Zarr store",
        marker_names=(".zgroup", ".zarray", ".zattrs", "zarr.json"),
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        grp = zarr.open_group(str(target), mode="w", zarr_format=zarr_format)
    except TypeError:
        if zarr_format == 3:
            raise ValueError(
                "zarr_format=3 requires zarr>=3.0; installed zarr only supports v2."
            ) from None
        grp = zarr.open_group(str(target), mode="w")

    if hasattr(grp, "create_array"):
        try:
            grp.create_array("0", data=arr_c, chunks=eff_chunks, overwrite=True)
        except TypeError:
            ds = grp.create_array(
                "0", shape=arr_c.shape, dtype=arr_c.dtype, chunks=eff_chunks, overwrite=True
            )
            ds[:] = arr_c
    else:
        grp.create_dataset("0", data=arr_c, chunks=eff_chunks, overwrite=True)

    ms_entry: dict[str, Any] = {
        "name": str(name),
        "axes": [
            {"name": "z", "type": "space", "unit": str(unit)},
            {"name": "y", "type": "space", "unit": str(unit)},
            {"name": "x", "type": "space", "unit": str(unit)},
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(dz), float(dy), float(dx)]}
                ],
            }
        ],
    }
    if zarr_format == 3:
        grp.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [ms_entry],
        }
    else:
        grp.attrs["multiscales"] = [{"version": "0.4", **ms_entry}]
    return target


def read_zarr_volume(
    path: Union[str, os.PathLike],
    *,
    array_path: Optional[str] = None,
    level: int = 0,
    threshold: Optional[Union[float, str]] = None,
    pore_value: int = 0,
    solid_value: int = 1,
    dark_is_pore: bool = True,
) -> tuple[np.ndarray, Optional[tuple[float, float, float]]]:
    """Read a 3-D Zarr array or OME-NGFF (v0.4 / v0.5) multiscale Zarr group.

    Returns ``(volume, voxel_size_zyx)`` where ``voxel_size_zyx`` is extracted
    from OME-NGFF ``coordinateTransformations`` (or ``None`` if absent).
    """
    zarr = _import_zarr()
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Zarr store not found: '{target}'")

    node = zarr.open(str(target), mode="r")
    meta_vz: Optional[tuple[float, float, float]] = None
    axes_meta: Sequence[Any] = ()

    if hasattr(node, "shape") and hasattr(node, "dtype"):
        zarr_arr = node
        raw_vz = node.attrs.get("voxel_size")
        if isinstance(raw_vz, (list, tuple)) and len(raw_vz) == 3:
            meta_vz = (float(raw_vz[0]), float(raw_vz[1]), float(raw_vz[2]))
    else:
        multiscales = node.attrs.get("multiscales")
        if multiscales is None:
            ome_block = node.attrs.get("ome")
            if isinstance(ome_block, Mapping):
                multiscales = ome_block.get("multiscales")

        if isinstance(multiscales, list) and multiscales and isinstance(multiscales[0], Mapping):
            axes_meta = multiscales[0].get("axes", ())

        if array_path is not None:
            zarr_arr = node[array_path]
            if isinstance(multiscales, list) and multiscales:
                for ds_meta in multiscales[0].get("datasets", []):
                    if ds_meta.get("path") == array_path:
                        meta_vz = _extract_ome_zyx_scale(ds_meta, axes_meta)
                        break
        elif isinstance(multiscales, list) and multiscales:
            datasets = multiscales[0].get("datasets", [])
            if not datasets:
                raise ValueError(f"OME-Zarr store '{target}' defines empty multiscales datasets.")
            if level < 0 or level >= len(datasets):
                raise IndexError(
                    f"Requested multiscale level {level}, but store has {len(datasets)} level(s)."
                )
            ds_meta = datasets[level]
            zarr_arr = node[ds_meta["path"]]
            meta_vz = _extract_ome_zyx_scale(ds_meta, axes_meta)
        elif "0" in node:
            zarr_arr = node["0"]
        elif "data" in node:
            zarr_arr = node["data"]
        else:
            raise ValueError(
                f"Could not locate a 3-D array in Zarr group '{target}'. "
                f"Specify array_path=... explicitly."
            )

    arr = np.asarray(zarr_arr)
    # Squeeze leading singleton non-spatial dimensions (e.g. 4-D (1, Z, Y, X) or 5-D (1, 1, Z, Y, X))
    n_squeezed = 0
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
        n_squeezed += 1
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D Zarr volume (Nz, Ny, Nx), got shape {arr.shape}")

    # Reorder spatial axes to (z, y, x) if OME-NGFF axes declare a non-zyx spatial order
    if isinstance(axes_meta, (list, tuple)) and len(axes_meta) - n_squeezed == 3:
        spatial_names = [
            str(ax.get("name", "")).lower() if isinstance(ax, Mapping) else ""
            for ax in axes_meta[n_squeezed:]
        ]
        if set(spatial_names) == {"x", "y", "z"} and spatial_names != ["z", "y", "x"]:
            perm = (
                spatial_names.index("z"),
                spatial_names.index("y"),
                spatial_names.index("x"),
            )
            arr = np.transpose(arr, perm)

    if threshold is not None:
        if isinstance(threshold, str):
            if threshold.strip().lower() != "otsu":
                raise ValueError(
                    f"Unsupported threshold={threshold!r}; expected float or 'otsu'."
                )
            arr, _ = binarize_otsu(
                arr,
                pore_value=pore_value,
                solid_value=solid_value,
                dark_is_pore=dark_is_pore,
            )
        else:
            t_val = float(threshold)
            if dark_is_pore:
                arr = np.where(arr > t_val, int(solid_value), int(pore_value)).astype(np.int32)
            else:
                arr = np.where(arr > t_val, int(pore_value), int(solid_value)).astype(np.int32)

    return np.ascontiguousarray(arr), meta_vz


# ---------------------------------------------------------------------------
# AMReX single-level plotfile writer (BoxLib HyperCLaw-V1.1)
# ---------------------------------------------------------------------------

def write_amrex_plotfile(
    plotfile_dir: Union[str, os.PathLike],
    fields: Mapping[str, np.ndarray],
    *,
    voxel_size: Optional[Sequence[float] | float] = (1.0, 1.0, 1.0),
    time: float = 0.0,
    step: int = 0,
) -> Path:
    """Write a 3-D single-level AMReX ``HyperCLaw-V1.1`` plotfile directory.

    Each field in *fields* must be a 3-D array ordered ``(Nz, Ny, Nx)``; arrays
    are transposed to AMReX column-major ``(Nx, Ny, Nz)`` little-endian IEEE-754
    ``float64`` FAB format so ``yt.load()`` and ParaView read them identically
    to native C++ ``amrex::WriteSingleLevelPlotfile`` outputs.
    """
    if not fields:
        raise ValueError("fields mapping must contain at least one 3-D array.")

    varnames = list(fields.keys())
    field_arrays: list[np.ndarray] = []
    ref_shape: Optional[tuple[int, int, int]] = None
    for name, raw in fields.items():
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(
                f"Field {name!r} must be 3-D (Nz, Ny, Nx), got shape {arr.shape}"
            )
        if ref_shape is None:
            ref_shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Field {name!r} has shape {arr.shape}, expected {ref_shape}"
            )
        field_arrays.append(arr)

    assert ref_shape is not None
    nz, ny, nx = ref_shape
    dz, dy, dx = _coerce_vz_tuple(voxel_size)
    ncomp = len(varnames)

    out_dir = Path(plotfile_dir)
    _safe_rmtree_store(
        out_dir,
        kind="AMReX plotfile",
        marker_names=("Header", "Level_0"),
    )
    level_dir = out_dir / "Level_0"
    level_dir.mkdir(parents=True, exist_ok=True)

    box_str = f"((0,0,0) ({nx - 1},{ny - 1},{nz - 1}) (0,0,0))"
    header_lines = [
        "HyperCLaw-V1.1",
        str(ncomp),
        *varnames,
        "3",
        f"{float(time):.16e}",
        "0",
        "0.0 0.0 0.0",
        f"{nx * dx:.16e} {ny * dy:.16e} {nz * dz:.16e}",
        "",
        f"{box_str} ",
        f"{int(step)}",
        f"{dx:.16e} {dy:.16e} {dz:.16e}",
        "0",
        "0",
        f"0 1 {float(time):.16e}",
        f"{int(step)}",
        f"0.0 {nx * dx:.16e}",
        f"0.0 {ny * dy:.16e}",
        f"0.0 {nz * dz:.16e}",
        "Level_0/Cell",
    ]
    (out_dir / "Header").write_text("\n".join(header_lines) + "\n", encoding="ascii")

    fab_header = (
        f"FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1))){box_str} {ncomp}\n"
    ).encode("ascii")
    payload = b"".join(
        np.transpose(f, (2, 1, 0)).astype("<f8", copy=False).tobytes(order="F")
        for f in field_arrays
    )
    data_file = level_dir / "Cell_D_00000"
    data_file.write_bytes(fab_header + payload)

    cell_h_lines = [
        "1",
        "1",
        str(ncomp),
        "0",
        "(1 0",
        box_str,
        ")",
        "1",
        "FabOnDisk: Cell_D_00000 0",
        "",
        f"1,{ncomp}",
        *(["0.0,"] * ncomp),
        "",
        f"1,{ncomp}",
        *(["0.0,"] * ncomp),
    ]
    (level_dir / "Cell_H").write_text("\n".join(cell_h_lines) + "\n", encoding="ascii")
    return out_dir

