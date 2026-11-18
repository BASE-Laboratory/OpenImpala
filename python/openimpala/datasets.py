"""Built-in 3-D porous-media benchmark datasets for tutorials and testing.

All dataset loaders return C-contiguous ``(Nz, Ny, Nx)`` ``int32`` NumPy arrays
and do not require an active :class:`openimpala.Session`.  When running from a
repository checkout, files are read directly from ``data/``; in standalone wheel
installations, files are downloaded once over HTTPS and cached under
``$OPENIMPALA_DATA_DIR`` (default ``~/.cache/openimpala``).
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from ._io import read_tiff_volume

_GITHUB_DATA_BASE = (
    "https://raw.githubusercontent.com/BASE-Laboratory/OpenImpala/master/data"
)

__all__ = [
    "fetch_path",
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


def fetch_path(filename: str) -> Path:
    """Resolve *filename* in the repository ``data/`` folder or user cache.

    Parameters
    ----------
    filename : str
        Dataset file name under ``data/`` (e.g. ``'SampleData_2Phase_stack_3d_1bit.tif'``).

    Returns
    -------
    pathlib.Path
        Local filesystem path to the dataset file.
    """
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError(f"Expected a plain dataset filename, got {filename!r}")

    local_candidate = _repo_data_dir() / filename
    if local_candidate.is_file():
        return local_candidate

    cache_target = _cache_dir() / filename
    if cache_target.is_file() and cache_target.stat().st_size > 0:
        return cache_target

    cache_target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_GITHUB_DATA_BASE}/{filename}"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".part", dir=str(cache_target.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.replace(cache_target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return cache_target


def two_phase_stack() -> np.ndarray:
    """Return a ``(100, 100, 100)`` ``int32`` isotropic two-phase porous medium.

    Phase ``1`` is the percolating pore network (:math:`\\varepsilon \\approx 0.3983`);
    phase ``0`` is the solid matrix (:math:`1 - \\varepsilon \\approx 0.6017`).
    """
    path = fetch_path("SampleData_2Phase_stack_3d_1bit.tif")
    return np.ascontiguousarray(read_tiff_volume(path), dtype=np.int32)


def two_phase_cubic() -> np.ndarray:
    """Return a ``(64, 64, 64)`` ``int32`` two-phase porous medium.

    Phase ``1`` is the percolating pore network (:math:`\\varepsilon \\approx 0.3993`);
    phase ``0`` is the solid matrix (:math:`1 - \\varepsilon \\approx 0.6007`).
    """
    path = fetch_path("SampleData_2Phase_squared.tif")
    return np.ascontiguousarray(read_tiff_volume(path), dtype=np.int32)


def sphere_pack() -> np.ndarray:
    """Return a ``(100, 100, 100)`` ``int32`` regular 8-sphere benchmark volume.

    Phase ``0`` consists of 8 non-overlapping spheres of radius :math:`R = 15`
    voxels (:math:`\\phi_0 \\approx 0.1120`); phase ``1`` is the continuous
    surrounding pore space (:math:`\\varepsilon \\approx 0.8880`).
    """
    path = fetch_path("spheres.tif")
    return np.ascontiguousarray(read_tiff_volume(path), dtype=np.int32)
