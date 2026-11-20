"""Structural VolumeProtocol and physical length-unit conversion for OpenImpala.

Follows the BASE Laboratory ``ECOSYSTEM.md`` Layer 3 solver boundary contract:
``OpenImpala`` accepts any upstream volume container implementing
:class:`VolumeProtocol` (such as ``basecoat.Volume``) via structural duck-typing
with **zero runtime dependency** on ``basecoat``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .exceptions import UnitCalibrationError

_LENGTH_TO_METRES: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "meters": 1.0,
    "metres": 1.0,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimetre": 1e-2,
    "centimeters": 1e-2,
    "centimetres": 1e-2,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimetre": 1e-3,
    "millimeters": 1e-3,
    "millimetres": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "micrometer": 1e-6,
    "micrometre": 1e-6,
    "micrometers": 1e-6,
    "micrometres": 1e-6,
    "micron": 1e-6,
    "microns": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometre": 1e-9,
    "nanometers": 1e-9,
    "nanometres": 1e-9,
    "angstrom": 1e-10,
    "å": 1e-10,
}

_UNCALIBRATED_UNITS = {"", "none", "voxel", "voxels", "pixel", "pixels", "px", "index"}


@runtime_checkable
class VolumeProtocol(Protocol):
    """Structural protocol for calibrated 3-D voxel volumes (e.g. ``basecoat.Volume``).

    When passed into ``OpenImpala`` high-level functions with ``voxel_size=None``,
    ``spacing_in("m")`` is queried to obtain ``(dz, dy, dx)`` in SI metres.  If the
    volume is uncalibrated (e.g. ``unit='voxel'``), :class:`~openimpala.UnitCalibrationError`
    is raised unless ``voxel_size=...`` is supplied explicitly.
    """

    @property
    def data(self) -> np.ndarray:
        """Underlying NumPy array ordered ``(Nz, Ny, Nx)``."""
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """Array element data type."""
        ...

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        ...

    def spacing_in(self, unit: str) -> Tuple[float, ...]:
        """Return per-axis physical voxel spacing converted to *unit* (e.g. ``'m'``)."""
        ...


def resolve_length_unit_to_metres(unit: Optional[str], *, context: str = "volume") -> float:
    """Return the multiplicative factor mapping *unit* to SI metres, or raise :class:`UnitCalibrationError`."""
    if unit is None:
        raise UnitCalibrationError(
            f"{context} has no physical length unit metadata; calibrate the volume "
            f"(e.g. unit='micrometer') or pass voxel_size=... explicitly."
        )
    key = str(unit).strip().lower()
    if key in _UNCALIBRATED_UNITS:
        raise UnitCalibrationError(
            f"{context} is uncalibrated (unit={unit!r}); calibrate the volume with "
            f"physical units or pass voxel_size=... explicitly."
        )
    factor = _LENGTH_TO_METRES.get(key)
    if factor is None:
        raise UnitCalibrationError(
            f"Unsupported physical length unit {unit!r} on {context}; expected one of "
            f"'nm', 'um'/'micrometer', 'mm', 'cm', or 'm', or pass voxel_size=... explicitly."
        )
    return factor


def convert_ome_zyx_scale(
    scale_zyx: Tuple[float, float, float],
    axes_meta: Sequence[Any],
    target_unit: str,
    *,
    n_squeezed: int = 0,
    context: str = "OME-Zarr store",
) -> Tuple[float, float, float]:
    """Convert ``(dz, dy, dx)`` from OME-NGFF axis units into *target_unit*."""
    target_m = resolve_length_unit_to_metres(target_unit, context="target unit")
    spatial_axes = list(axes_meta[n_squeezed:]) if len(axes_meta) >= n_squeezed + 3 else list(axes_meta)
    by_name: dict[str, Optional[str]] = {}
    for ax in spatial_axes:
        if isinstance(ax, Mapping):
            nm = str(ax.get("name", "")).strip().lower()
            raw_u = ax.get("unit")
            by_name[nm] = str(raw_u) if raw_u is not None else None

    if not {"z", "y", "x"}.issubset(by_name):
        if len(spatial_axes) == 3 and all(isinstance(a, Mapping) for a in spatial_axes):
            units_zyx = [
                str(spatial_axes[i].get("unit")) if spatial_axes[i].get("unit") is not None else None
                for i in range(3)
            ]
        else:
            raise UnitCalibrationError(
                f"{context} does not declare calibrated ('z', 'y', 'x') axis units; "
                f"pass unit=None or voxel_size=... explicitly."
            )
    else:
        units_zyx = [by_name["z"], by_name["y"], by_name["x"]]

    converted: list[float] = []
    for val, ax_name, u_str in zip(scale_zyx, ("z", "y", "x"), units_zyx):
        src_m = resolve_length_unit_to_metres(u_str, context=f"{context} axis '{ax_name}'")
        out_val = float(val) * (src_m / target_m)
        if not math.isfinite(out_val) or out_val <= 0.0:
            raise UnitCalibrationError(
                f"Non-positive or non-finite physical spacing {out_val!r} on {context} axis '{ax_name}'."
            )
        converted.append(out_val)
    return (converted[0], converted[1], converted[2])
