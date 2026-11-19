"""Synthetic 3-D porous-media generators and ML surrogate dataset pipeline.

Provides zero-extra-dependency (NumPy + SciPy) procedural 3-D microstructure
generators and a batch dataset builder that pairs 3-D phase volumes with ground-
truth transport properties (:math:`\\varepsilon`, Cauchy–Crofton SSA, directional
tortuosity :math:`(\\tau_x, \\tau_y, \\tau_z)`, and the full :math:`3 \\times 3`
periodic homogenised diffusivity tensor :math:`\\mathbf{D}_{\\mathrm{eff}}`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import scipy.ndimage as ndimage

__all__ = [
    "gaussian_blobs",
    "overlapping_spheres",
    "multiphase_electrode",
    "generate_ml_dataset",
]


def _normalize_shape(shape: Union[int, Sequence[int]]) -> tuple[int, int, int]:
    if isinstance(shape, (int, np.integer)):
        s = int(shape)
        if s < 2:
            raise ValueError(f"Domain dimensions must be >= 2, got {s}")
        return (s, s, s)
    seq = tuple(int(x) for x in shape)
    if len(seq) != 3 or any(x < 2 for x in seq):
        raise ValueError(
            f"shape must be an int >= 2 or a 3-tuple of ints >= 2 (Nz, Ny, Nx), got {shape!r}"
        )
    return (seq[0], seq[1], seq[2])


def _normalize_vec3(val: Union[float, Sequence[float]], name: str) -> tuple[float, float, float]:
    if isinstance(val, (int, float, np.floating, np.integer)):
        v = float(val)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {val!r}")
        return (v, v, v)
    seq = tuple(float(x) for x in val)
    if len(seq) != 3 or any((not np.isfinite(x) or x <= 0.0) for x in seq):
        raise ValueError(f"{name} must be a positive scalar or 3-tuple, got {val!r}")
    return (seq[0], seq[1], seq[2])


def gaussian_blobs(
    shape: Union[int, Sequence[int]] = (64, 64, 64),
    porosity: float = 0.5,
    blobiness: Union[float, Sequence[float]] = 1.0,
    *,
    anisotropy: Union[float, Sequence[float]] = (1.0, 1.0, 1.0),
    pore_value: int = 0,
    solid_value: int = 1,
    periodic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a 3-D two-phase Gaussian random-field ("blobs") microstructure.

    Parameters
    ----------
    shape : int or 3-tuple of ints, default ``(64, 64, 64)``
        Output grid dimensions ``(Nz, Ny, Nx)``.
    porosity : float, default 0.5
        Target volume fraction of *pore_value* in ``[0, 1]``.
    blobiness : float or 3-tuple of floats, default 1.0
        Feature frequency factor (smaller values yield larger, coarser pores;
        larger values yield finer pores).
    anisotropy : float or 3-tuple of floats, default ``(1.0, 1.0, 1.0)``
        Relative stretch factors along ``(z, y, x)`` (values ``> 1`` elongate
        features along that axis).
    pore_value : int, default 0
        Phase ID assigned to the pore space.
    solid_value : int, default 1
        Phase ID assigned to the solid matrix.
    periodic : bool, default True
        Use periodic wrap-around convolution so opposite domain faces match
        seamlessly for periodic homogenisation cell problems.
    seed : int, optional
        Random seed for deterministic generation.

    Returns
    -------
    np.ndarray
        C-contiguous ``int32`` array of shape ``(Nz, Ny, Nx)``.
    """
    shape_t = _normalize_shape(shape)
    if not (0.0 <= float(porosity) <= 1.0):
        raise ValueError(f"porosity must lie in [0, 1], got {porosity!r}")
    blob_t = _normalize_vec3(blobiness, "blobiness")
    aniso_t = _normalize_vec3(anisotropy, "anisotropy")

    if float(porosity) <= 0.0:
        return np.full(shape_t, int(solid_value), dtype=np.int32)
    if float(porosity) >= 1.0:
        return np.full(shape_t, int(pore_value), dtype=np.int32)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape_t)
    sigmas = tuple(
        max(0.5, (shape_t[d] / (40.0 * blob_t[d])) * aniso_t[d]) for d in range(3)
    )
    mode = "wrap" if periodic else "reflect"
    field = ndimage.gaussian_filter(noise, sigma=sigmas, mode=mode)
    cutoff = float(np.quantile(field, float(porosity)))
    out = np.where(field < cutoff, int(pore_value), int(solid_value)).astype(np.int32)
    return np.ascontiguousarray(out, dtype=np.int32)


def overlapping_spheres(
    shape: Union[int, Sequence[int]] = (64, 64, 64),
    porosity: float = 0.5,
    radius: float = 6.0,
    *,
    pore_value: int = 0,
    solid_value: int = 1,
    periodic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a 3-D Boolean-model packing of randomly overlapping solid spheres.

    Sphere count is calibrated from the analytical Boolean-model relation
    :math:`\\varepsilon = \\exp(-N_s V_{\\mathrm{sphere}} / V_{\\mathrm{domain}})`
    with iterative top-up so achieved porosity closely matches *porosity*.
    """
    shape_t = _normalize_shape(shape)
    if not (0.0 <= float(porosity) <= 1.0):
        raise ValueError(f"porosity must lie in [0, 1], got {porosity!r}")
    r = float(radius)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"radius must be positive and finite, got {radius!r}")

    if float(porosity) <= 0.0:
        return np.full(shape_t, int(solid_value), dtype=np.int32)
    if float(porosity) >= 1.0:
        return np.full(shape_t, int(pore_value), dtype=np.int32)

    nz, ny, nx = shape_t
    total_voxels = float(nz * ny * nx)

    r_int = int(np.ceil(r))
    oz, oy, ox = np.ogrid[-r_int : r_int + 1, -r_int : r_int + 1, -r_int : r_int + 1]
    sphere_mask = (oz * oz + oy * oy + ox * ox) <= (r * r)
    rel_z, rel_y, rel_x = np.nonzero(sphere_mask)
    rel_z = rel_z - r_int
    rel_y = rel_y - r_int
    rel_x = rel_x - r_int
    v_sphere = max(1.0, float(len(rel_z)))

    rng = np.random.default_rng(seed)
    solid_mask = np.zeros(shape_t, dtype=bool)
    target_solid = 1.0 - float(porosity)

    est_spheres = max(1, int(np.round(-total_voxels * np.log(max(porosity, 1e-6)) / v_sphere)))

    def _stamp_batch(n_add: int) -> None:
        cz = rng.integers(0, nz, size=n_add)
        cy = rng.integers(0, ny, size=n_add)
        cx = rng.integers(0, nx, size=n_add)
        for sz, sy, sx in zip(cz, cy, cx):
            zz = sz + rel_z
            yy = sy + rel_y
            xx = sx + rel_x
            if periodic:
                solid_mask[zz % nz, yy % ny, xx % nx] = True
            else:
                valid = (
                    (zz >= 0)
                    & (zz < nz)
                    & (yy >= 0)
                    & (yy < ny)
                    & (xx >= 0)
                    & (xx < nx)
                )
                solid_mask[zz[valid], yy[valid], xx[valid]] = True

    _stamp_batch(est_spheres)
    for _ in range(8):
        cur_porosity = 1.0 - (float(np.count_nonzero(solid_mask)) / total_voxels)
        if cur_porosity <= float(porosity) or cur_porosity <= 1e-4:
            break
        extra = max(
            1,
            int(
                np.ceil(
                    -total_voxels
                    * np.log(max(float(porosity) / cur_porosity, 1e-6))
                    / v_sphere
                )
            ),
        )
        _stamp_batch(extra)
        if (float(np.count_nonzero(solid_mask)) / total_voxels) >= target_solid:
            break

    out = np.where(solid_mask, int(solid_value), int(pore_value)).astype(np.int32)
    return np.ascontiguousarray(out, dtype=np.int32)


def multiphase_electrode(
    shape: Union[int, Sequence[int]] = (64, 64, 64),
    pore_fraction: float = 0.35,
    active_fraction: float = 0.50,
    binder_fraction: float = 0.15,
    *,
    blobiness: float = 1.2,
    pore_value: int = 0,
    active_value: int = 1,
    binder_value: int = 2,
    periodic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a 3-phase composite battery electrode microstructure.

    Produces an active-material backbone (*active_value*), surface-wetting
    carbon-binder domain (*binder_value*) concentrated along active-material
    interfaces, and electrolyte pore network (*pore_value*) matching the
    requested volume fractions.
    """
    shape_t = _normalize_shape(shape)
    pf = float(pore_fraction)
    af = float(active_fraction)
    bf = float(binder_fraction)
    if any(x < 0.0 for x in (pf, af, bf)):
        raise ValueError("Phase fractions must all be >= 0.")
    total_f = pf + af + bf
    if not np.isclose(total_f, 1.0, atol=1e-5):
        raise ValueError(
            f"pore_fraction + active_fraction + binder_fraction must equal 1.0, got {total_f:.6f}"
        )

    rng = np.random.default_rng(seed)
    mode = "wrap" if periodic else "reflect"
    sig_primary = tuple(max(0.8, shape_t[d] / (36.0 * float(blobiness))) for d in range(3))
    primary = ndimage.gaussian_filter(rng.standard_normal(shape_t), sigma=sig_primary, mode=mode)

    out = np.full(shape_t, int(pore_value), dtype=np.int32)
    if af >= 1.0:
        return np.full(shape_t, int(active_value), dtype=np.int32)
    if pf >= 1.0:
        return out

    if af > 0.0:
        active_thresh = float(np.quantile(primary, 1.0 - af))
        active_mask = primary >= active_thresh
        out[active_mask] = int(active_value)
    else:
        active_mask = np.zeros(shape_t, dtype=bool)

    if bf > 0.0:
        non_active = ~active_mask
        n_non_active = int(np.count_nonzero(non_active))
        if n_non_active > 0:
            # Bias binder toward active-material interfaces + spatial heterogeneity
            if np.any(active_mask):
                if periodic:
                    pad = min(max(shape_t) // 2, 24)
                    if pad > 0:
                        padded_na = np.pad(non_active, pad_width=pad, mode="wrap")
                        dist = ndimage.distance_transform_edt(padded_na)[
                            pad:-pad, pad:-pad, pad:-pad
                        ]
                    else:
                        dist = ndimage.distance_transform_edt(non_active)
                else:
                    dist = ndimage.distance_transform_edt(non_active)
            else:
                dist = np.ones(shape_t, dtype=np.float64)
            sig_cbd = tuple(max(0.6, s * 0.5) for s in sig_primary)
            cbd_noise = ndimage.gaussian_filter(
                rng.standard_normal(shape_t), sigma=sig_cbd, mode=mode
            )
            cbd_score = -dist + 0.75 * cbd_noise
            n_binder = min(n_non_active, int(np.round(bf * float(out.size))))
            if n_binder > 0:
                scores_na = cbd_score[non_active]
                part_idx = np.argpartition(scores_na, len(scores_na) - n_binder)[
                    len(scores_na) - n_binder :
                ]
                na_flat = np.flatnonzero(non_active)
                out.ravel()[na_flat[part_idx]] = int(binder_value)

    return np.ascontiguousarray(out, dtype=np.int32)


def _is_io_rank() -> bool:
    """Return True on MPI rank 0 (or in serial / pure-Python runs)."""
    try:
        from .session import _has_core

        if _has_core():
            import openimpala._core as _core

            if _core.amrex_initialized():
                if hasattr(_core, "is_io_rank"):
                    return bool(_core.is_io_rank())
                if _core.mpi_ranks() <= 1:
                    return True
    except Exception:
        pass

    try:
        from mpi4py import MPI  # type: ignore[import-not-found]

        if MPI.Is_initialized():
            return int(MPI.COMM_WORLD.Get_rank()) == 0
    except Exception:
        pass

    for env_var in (
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
    ):
        raw = os.environ.get(env_var)
        if raw is not None:
            try:
                return int(raw) == 0
            except ValueError:
                pass

    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            return int(os.environ["SLURM_PROCID"]) == 0
        except ValueError:
            pass

    return True


def generate_ml_dataset(
    output_path: Union[str, os.PathLike],
    n_samples: int = 16,
    shape: Union[int, Sequence[int]] = (32, 32, 32),
    *,
    generator: str = "blobs",
    porosity_range: tuple[float, float] = (0.30, 0.70),
    blobiness_range: tuple[float, float] = (0.8, 2.0),
    radius_range: tuple[float, float] = (3.0, 7.0),
    seed: int = 42,
    compute_deff: bool = True,
    compute_tortuosity: bool = True,
    solver: str = "auto",
    return_volumes: bool = True,
) -> dict[str, Any]:
    """Generate a labelled 3-D porous-media ML surrogate dataset on disk.

    Each sample contains a binary 3-D microstructure (``0`` = conducting pore,
    ``1`` = insulating solid) alongside ground-truth OpenImpala physics labels:
    porosity, Cauchy–Crofton specific surface area, directional tortuosities
    ``(tau_x, tau_y, tau_z)``, and the symmetric ``3x3`` homogenised effective-
    diffusivity tensor ``D_eff``.

    Supported output formats (auto-detected from *output_path* suffix):
      * ``.npz`` — compressed NumPy archive (zero optional dependencies)
      * ``.h5`` / ``.hdf5`` — chunked HDF5 archive streamed sample-by-sample (requires ``h5py``)
      * ``.zarr`` — chunked Zarr group archive streamed sample-by-sample (requires ``zarr``)

    Set ``return_volumes=False`` when generating large ``.h5`` or ``.zarr``
    datasets (e.g. ``10_000+`` 3-D volumes) to stream directly to disk in
    constant ``O(Nz * Ny * Nx)`` host memory.
    """
    import openimpala as oi

    from ._io import _safe_rmtree_store

    if int(n_samples) < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    n_s = int(n_samples)
    shape_t = _normalize_shape(shape)
    gen_key = generator.strip().lower()
    if gen_key not in ("blobs", "spheres"):
        raise ValueError(f"Unknown generator {generator!r}; expected 'blobs' or 'spheres'.")
    p_lo, p_hi = float(porosity_range[0]), float(porosity_range[1])
    if not (0.0 < p_lo <= p_hi < 1.0):
        raise ValueError(
            f"porosity_range must satisfy 0 < low <= high < 1, got {porosity_range!r}"
        )

    out_p = Path(output_path)
    suffix = out_p.suffix.lower()
    if suffix not in (".npz", ".h5", ".hdf5", ".zarr"):
        raise ValueError(
            f"Unsupported output extension '{suffix}' for '{out_p}'. "
            f"Use .npz, .h5/.hdf5, or .zarr."
        )

    # Fail fast before expensive solver loop if optional writer backend is missing
    h5py_mod = None
    zarr_mod = None
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py as h5py_mod  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "Writing .h5/.hdf5 datasets requires 'h5py' (pip install h5py)."
            ) from exc
    elif suffix == ".zarr":
        from ._io import _import_zarr

        zarr_mod = _import_zarr()

    param_rng = np.random.default_rng(int(seed))
    target_porosities = param_rng.uniform(p_lo, p_hi, size=n_s)
    sample_seeds = param_rng.integers(0, 2**31 - 1, size=n_s, dtype=np.int64)
    feature_scales = (
        param_rng.uniform(float(blobiness_range[0]), float(blobiness_range[1]), size=n_s)
        if gen_key == "blobs"
        else param_rng.uniform(float(radius_range[0]), float(radius_range[1]), size=n_s)
    )

    keep_mem_vols = bool(return_volumes or suffix == ".npz")
    volumes: Optional[np.ndarray] = (
        np.empty((n_s, *shape_t), dtype=np.uint8) if keep_mem_vols else None
    )
    porosity_arr = np.zeros(n_s, dtype=np.float64)
    ssa_arr = np.zeros(n_s, dtype=np.float64)
    tau_xyz = np.full((n_s, 3), np.inf, dtype=np.float64)
    deff_tensors = np.zeros((n_s, 3, 3), dtype=np.float64)

    with oi.Session():
        io_rank = _is_io_rank()
        hf_handle = None
        h5_vol_ds = None
        zarr_grp = None
        zarr_vol_ds = None

        try:
            if io_rank:
                out_p.parent.mkdir(parents=True, exist_ok=True)
                if suffix in (".h5", ".hdf5"):
                    assert h5py_mod is not None
                    hf_handle = h5py_mod.File(out_p, "w")
                    hf_handle.attrs["generator"] = gen_key
                    hf_handle.attrs["conducting_phase"] = 0
                    h5_vol_ds = hf_handle.create_dataset(
                        "volumes",
                        shape=(n_s, *shape_t),
                        chunks=(1, *shape_t),
                        dtype=np.uint8,
                        compression="gzip",
                    )
                elif suffix == ".zarr":
                    assert zarr_mod is not None
                    _safe_rmtree_store(
                        out_p,
                        kind="Zarr store",
                        marker_names=(".zgroup", ".zarray", ".zattrs", "zarr.json"),
                    )
                    try:
                        zarr_grp = zarr_mod.open_group(str(out_p), mode="w", zarr_format=2)
                    except TypeError:
                        zarr_grp = zarr_mod.open_group(str(out_p), mode="w")
                    zarr_grp.attrs["generator"] = gen_key
                    zarr_grp.attrs["conducting_phase"] = 0
                    vol_chunks = (1, *shape_t)
                    if hasattr(zarr_grp, "create_array"):
                        zarr_vol_ds = zarr_grp.create_array(
                            "volumes",
                            shape=(n_s, *shape_t),
                            chunks=vol_chunks,
                            dtype=np.uint8,
                            overwrite=True,
                        )
                    else:
                        zarr_vol_ds = zarr_grp.create_dataset(
                            "volumes",
                            shape=(n_s, *shape_t),
                            chunks=vol_chunks,
                            dtype=np.uint8,
                            overwrite=True,
                        )

            for idx in range(n_s):
                s_seed = int(sample_seeds[idx])
                p_target = float(target_porosities[idx])
                f_scale = float(feature_scales[idx])

                if gen_key == "blobs":
                    vol = gaussian_blobs(
                        shape_t,
                        porosity=p_target,
                        blobiness=f_scale,
                        pore_value=0,
                        solid_value=1,
                        periodic=True,
                        seed=s_seed,
                    )
                else:
                    vol = overlapping_spheres(
                        shape_t,
                        porosity=p_target,
                        radius=f_scale,
                        pore_value=0,
                        solid_value=1,
                        periodic=True,
                        seed=s_seed,
                    )

                vol_u8 = vol.astype(np.uint8)
                if volumes is not None:
                    volumes[idx] = vol_u8
                if io_rank:
                    if h5_vol_ds is not None:
                        h5_vol_ds[idx] = vol_u8
                    elif zarr_vol_ds is not None:
                        zarr_vol_ds[idx] = vol_u8

                vf_res = oi.volume_fraction(vol, phase=0)
                porosity_arr[idx] = vf_res.fraction
                ssa_res = oi.specific_surface_area(vol, phase_a=0, phase_b=1)
                ssa_arr[idx] = ssa_res.ssa_corrected

                if compute_tortuosity:
                    tau_map = oi.tortuosity_all(vol, phase=0, solver=solver, strict=False)
                    for d_i, d_key in enumerate(("x", "y", "z")):
                        t_res = tau_map.get(d_key)
                        tau_xyz[idx, d_i] = t_res.tortuosity if t_res is not None else np.inf

                if compute_deff:
                    hypre_solver = "flexgmres" if solver in ("auto", "mlmg") else solver
                    deff_res = oi.effective_diffusivity(vol, phase=0, solver=hypre_solver)
                    deff_tensors[idx] = deff_res.tensor

            scalar_labels: dict[str, np.ndarray] = {
                "porosity": porosity_arr,
                "specific_surface_area": ssa_arr,
                "tortuosity_xyz": tau_xyz,
                "deff_tensor": deff_tensors,
                "seeds": sample_seeds,
                "feature_scale": feature_scales,
            }

            if io_rank:
                if suffix == ".npz":
                    assert volumes is not None
                    np.savez_compressed(out_p, volumes=volumes, **scalar_labels)
                elif hf_handle is not None:
                    for k, v in scalar_labels.items():
                        hf_handle.create_dataset(k, data=v, compression="gzip")
                elif zarr_grp is not None:
                    for k, v in scalar_labels.items():
                        if hasattr(zarr_grp, "create_array"):
                            try:
                                zarr_grp.create_array(k, data=v, overwrite=True)
                            except TypeError:
                                ds = zarr_grp.create_array(
                                    k, shape=v.shape, dtype=v.dtype, overwrite=True
                                )
                                ds[:] = v
                        else:
                            zarr_grp.create_dataset(k, data=v, overwrite=True)
        finally:
            if hf_handle is not None:
                hf_handle.close()

    result: dict[str, Any] = {
        "output_path": str(out_p),
        "n_samples": n_s,
        "shape": shape_t,
        "generator": gen_key,
        **scalar_labels,
    }
    if return_volumes and volumes is not None:
        result["volumes"] = volumes
    return result
