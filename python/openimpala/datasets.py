"""Built-in 3-D porous-media benchmark datasets for tutorials and testing.

All dataset loaders return C-contiguous ``(Nz, Ny, Nx)`` ``int32`` NumPy arrays
(or :class:`openimpala.VoxelImage` handles when ``as_voxel_image=True``) and do
not require an active :class:`openimpala.Session` unless ``as_voxel_image=True``
with the compiled C++ backend.  When running from a repository checkout, files
are read directly from ``data/``; in standalone wheel installations, files are
downloaded once over HTTPS, verified against pinned SHA-256 digests, and cached
under ``$OPENIMPALA_DATA_DIR`` (default ``~/.cache/openimpala``).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from ._io import read_hdf5_volume, read_raw_volume, read_tiff_volume

_GITHUB_REPO_RAW = "https://raw.githubusercontent.com/BASE-Laboratory/OpenImpala"
_GITHUB_DATA_BASE = f"{_GITHUB_REPO_RAW}/master/data"
_ORIG_URLRETRIEVE = urllib.request.urlretrieve

_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "two_phase_stack": {
        "filename": "SampleData_2Phase_stack_3d_1bit.tif",
        "sha256": "0c0a7ea15c1c7cd48d0e6543db505eb10d08111120d22ff72d712e13cde3cde7",
        "format": "tiff",
        "shape": (100, 100, 100),
        "conducting_phase": 1,
        "porosity": 0.398309,
        "description": "100x100x100 isotropic two-phase porous medium (phase 1 = pore, VF ~ 0.3983).",
        "aliases": ("porespy_blobs_100", "sample_2phase"),
    },
    "two_phase_cubic": {
        "filename": "SampleData_2Phase_squared.tif",
        "sha256": "8a084eb76f97080517a730e18ab8d7d270d9e6271b736f2d0b3bbb330c9dcae6",
        "format": "tiff",
        "shape": (64, 64, 64),
        "conducting_phase": 1,
        "porosity": 0.399326,
        "description": "64x64x64 cubic two-phase porous medium (phase 1 = pore, VF ~ 0.3993).",
        "aliases": ("porespy_blobs_64", "sample_2phase_64"),
    },
    "sphere_pack": {
        "filename": "spheres.tif",
        "sha256": "2b743776a9757883172bd75574c903babc2f34bcd9a8834c1765ef473fd01ad9",
        "format": "tiff",
        "shape": (100, 100, 100),
        "conducting_phase": 1,
        "porosity": 0.888024,
        "description": "100x100x100 regular 8-sphere array (R = 15 voxels in phase 0; phase 1 = pore).",
        "aliases": ("spheres_100", "spheres"),
    },
    "two_phase_hdf5": {
        "filename": "SampleData_2Phase_3d.hdf5",
        "sha256": "1b29853b6fbb60ffe639a2dce5148d4a1acc12abb03d8fac073de45f48999229",
        "format": "hdf5",
        "hdf5_dataset": "/image",
        "shape": (100, 100, 100),
        "conducting_phase": 1,
        "porosity": 0.398309,
        "description": "100x100x100 two-phase porous medium stored in HDF5 (/image dataset).",
        "aliases": ("sample_hdf5",),
    },
    "two_phase_raw": {
        "filename": "SampleData_2Phase_stack_3d_uint8.raw",
        "sha256": "95f22f8bac4b7979dae726b86022cb7ba2c36df06961438f7a1fb5f54eb3d1c7",
        "format": "raw",
        "shape": (100, 100, 100),
        "conducting_phase": 1,
        "porosity": 0.398309,
        "description": "100x100x100 two-phase porous medium as flat uint8 binary volume.",
        "aliases": ("sample_raw",),
    },
}

_SHA256_BY_FILENAME: dict[str, str] = {
    meta["filename"]: str(meta["sha256"]) for meta in _DATASET_REGISTRY.values()
}

_ALIAS_MAP: dict[str, str] = {}
for _canon_key, _meta in _DATASET_REGISTRY.items():
    _ALIAS_MAP[_canon_key] = _canon_key
    for _alias in _meta.get("aliases", ()):
        _ALIAS_MAP[_alias] = _canon_key

__all__ = [
    "fetch_path",
    "list_datasets",
    "fetch_example",
    "two_phase_stack",
    "two_phase_cubic",
    "sphere_pack",
]


def _repo_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _cache_dir() -> Path:
    env_dir = os.environ.get("OPENIMPALA_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home() / ".cache" / "openimpala"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def list_datasets() -> dict[str, dict[str, Any]]:
    """Return metadata for all built-in benchmark datasets keyed by canonical name."""
    return {
        name: {
            "filename": meta["filename"],
            "sha256": meta["sha256"],
            "format": meta["format"],
            "shape": meta["shape"],
            "conducting_phase": meta["conducting_phase"],
            "porosity": meta["porosity"],
            "description": meta["description"],
            "aliases": tuple(meta.get("aliases", ())),
        }
        for name, meta in _DATASET_REGISTRY.items()
    }


def _download_to_file(filename: str, dest: Path, *, timeout: float) -> None:
    """Download *filename* to *dest* with a bounded socket timeout.

    Prefers the immutable release tag ``v{__version__}`` when available on
    GitHub and falls back to ``master`` (e.g. on pre-release branches).
    Preserves compatibility with tests that monkeypatch ``urllib.request.urlretrieve``.
    """
    primary_url = f"{_GITHUB_DATA_BASE}/{filename}"
    if urllib.request.urlretrieve is not _ORIG_URLRETRIEVE:
        urllib.request.urlretrieve(primary_url, str(dest))
        return

    from . import __version__

    candidate_urls: list[str] = []
    if __version__ and __version__ != "unknown" and "dev" not in __version__:
        tag = __version__ if __version__.startswith("v") else f"v{__version__}"
        candidate_urls.append(f"{_GITHUB_REPO_RAW}/{tag}/data/{filename}")
    candidate_urls.append(primary_url)

    last_err: Optional[Exception] = None
    for idx, url in enumerate(candidate_urls):
        try:
            with urllib.request.urlopen(url, timeout=float(timeout)) as resp, dest.open(
                "wb"
            ) as out_fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out_fh.write(chunk)
            return
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 404 and idx + 1 < len(candidate_urls):
                continue
            raise
    if last_err is not None:
        raise last_err


def fetch_path(
    filename: str,
    *,
    expected_sha256: Optional[str] = None,
    verify_hash: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Resolve *filename* in the repository ``data/`` folder or verified user cache.

    Parameters
    ----------
    filename : str
        Dataset file name under ``data/`` (e.g. ``'SampleData_2Phase_stack_3d_1bit.tif'``).
    expected_sha256 : str, optional
        Explicit hex SHA-256 digest to enforce.  Defaults to the pinned registry
        digest when *filename* belongs to :func:`list_datasets`.
    verify_hash : bool, default True
        Verify SHA-256 checksum on cached and newly downloaded files.
    timeout : float, default 30.0
        Socket timeout in seconds for HTTPS downloads.

    Returns
    -------
    pathlib.Path
        Local filesystem path to the dataset file.
    """
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError(f"Expected a plain dataset filename, got {filename!r}")
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"timeout must be positive and finite, got {timeout!r}")

    pinned_hash = (
        expected_sha256.lower()
        if expected_sha256 is not None
        else _SHA256_BY_FILENAME.get(filename)
    )

    local_candidate = _repo_data_dir() / filename
    if local_candidate.is_file():
        if verify_hash and pinned_hash is not None:
            actual = _sha256_file(local_candidate)
            if actual != pinned_hash:
                raise ValueError(
                    f"SHA-256 mismatch for local dataset '{local_candidate}': "
                    f"expected {pinned_hash}, got {actual}."
                )
        return local_candidate

    cache_target = _cache_dir() / filename
    if cache_target.is_file() and cache_target.stat().st_size > 0:
        if not verify_hash or pinned_hash is None:
            return cache_target
        if _sha256_file(cache_target) == pinned_hash:
            return cache_target
        # Corrupted cache entry — evict and re-fetch
        cache_target.unlink(missing_ok=True)

    cache_target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".part", dir=str(cache_target.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _download_to_file(filename, tmp_path, timeout=timeout)
        if verify_hash and pinned_hash is not None:
            actual = _sha256_file(tmp_path)
            if actual != pinned_hash:
                raise ValueError(
                    f"SHA-256 checksum mismatch for downloaded '{filename}': "
                    f"expected {pinned_hash}, got {actual}."
                )
        tmp_path.replace(cache_target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return cache_target


def fetch_example(
    name: str = "two_phase_stack",
    *,
    as_voxel_image: bool = False,
    voxel_size: Optional[Union[float, Sequence[float]]] = None,
    max_grid_size: Union[int, str] = "auto",
):
    """Fetch a named benchmark dataset as an ``int32`` NumPy array or ``VoxelImage``.

    Parameters
    ----------
    name : str, default ``'two_phase_stack'``
        Dataset key or alias (see :func:`list_datasets`).
    as_voxel_image : bool, default False
        If ``True``, wrap the volume in an :class:`openimpala.VoxelImage`
        (native AMReX ``VoxelImage`` when ``_core`` is active, or
        ``PureVoxelImage`` in pure-Python mode).
    voxel_size : float or 3-sequence of floats, optional
        Physical voxel spacing ``h`` or ``(dz, dy, dx)`` when *as_voxel_image* is ``True``.
    max_grid_size : int or ``'auto'``, default ``'auto'``
        AMReX grid block size when *as_voxel_image* is ``True``.
    """
    key = name.strip().lower()
    canon = _ALIAS_MAP.get(key)
    if canon is None:
        valid = sorted(_DATASET_REGISTRY.keys())
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {valid}")

    meta = _DATASET_REGISTRY[canon]
    path = fetch_path(meta["filename"], expected_sha256=meta["sha256"])
    fmt = meta["format"]
    if fmt == "tiff":
        arr = np.ascontiguousarray(read_tiff_volume(path), dtype=np.int32)
    elif fmt == "hdf5":
        arr = np.ascontiguousarray(
            read_hdf5_volume(path, dataset=meta.get("hdf5_dataset", "/image")),
            dtype=np.int32,
        )
    elif fmt == "raw":
        nz, ny, nx = meta["shape"]
        arr = np.ascontiguousarray(
            read_raw_volume(path, width=nx, height=ny, depth=nz, raw_data_type="uint8"),
            dtype=np.int32,
        )
    else:
        raise ValueError(f"Unsupported registry format {fmt!r}")

    if not as_voxel_image:
        return arr

    from . import facade
    from ._io import PureVoxelImage
    from .session import Session

    Session.ensure_active()
    dz, dy, dx = facade._normalize_voxel_size(voxel_size)
    if facade._is_pure_python():
        return PureVoxelImage(arr, dx=dx, dy=dy, dz=dz)
    return facade._numpy_to_voxelimage(arr, max_grid_size=max_grid_size, voxel_size=(dz, dy, dx))


def two_phase_stack() -> np.ndarray:
    """Return a ``(100, 100, 100)`` ``int32`` isotropic two-phase porous medium.

    Phase ``1`` is the percolating pore network (:math:`\\varepsilon \\approx 0.3983`);
    phase ``0`` is the solid matrix (:math:`1 - \\varepsilon \\approx 0.6017`).
    """
    return fetch_example("two_phase_stack", as_voxel_image=False)


def two_phase_cubic() -> np.ndarray:
    """Return a ``(64, 64, 64)`` ``int32`` two-phase porous medium.

    Phase ``1`` is the percolating pore network (:math:`\\varepsilon \\approx 0.3993`);
    phase ``0`` is the solid matrix (:math:`1 - \\varepsilon \\approx 0.6007`).
    """
    return fetch_example("two_phase_cubic", as_voxel_image=False)


def sphere_pack() -> np.ndarray:
    """Return a ``(100, 100, 100)`` ``int32`` regular 8-sphere benchmark volume.

    Phase ``0`` consists of 8 non-overlapping spheres of radius :math:`R = 15`
    voxels (:math:`\\phi_0 \\approx 0.1120`); phase ``1`` is the continuous
    surrounding pore space (:math:`\\varepsilon \\approx 0.8880`).
    """
    return fetch_example("sphere_pack", as_voxel_image=False)
