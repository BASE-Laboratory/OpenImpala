#!/usr/bin/env python3
"""Generate 3-D two-phase porous-media benchmark volumes in TIFF, HDF5/XDMF, and RAW formats.

Uses ``porespy.generators.blobs`` when PoreSpy is installed, and automatically
falls back to :func:`openimpala.synthetic.gaussian_blobs` so sample structures
can be generated in any standard OpenImpala environment without extra packages.

Example::

    python data/create_sample_structure.py --output-dir ./my_samples --shape 64 --porosity 0.40
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time

import numpy as np


def _generate_boolean_solid(
    shape: tuple[int, int, int],
    porosity: float,
    blobiness: float,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Return boolean array where ``False`` = void/pore and ``True`` = solid."""
    try:
        import porespy as ps  # type: ignore[import-not-found]

        try:
            im_void = ps.generators.blobs(
                shape=shape, porosity=porosity, blobiness=blobiness, seed=seed
            )
        except TypeError:
            np.random.seed(seed)
            im_void = ps.generators.blobs(
                shape=shape, porosity=porosity, blobiness=blobiness
            )
        # PoreSpy convention in legacy script: solid=True, void=False when porosity=0.40
        im_bool = np.asarray(im_void, dtype=bool)
        return im_bool, f"porespy {getattr(ps, '__version__', 'unknown')}"
    except ImportError:
        from openimpala.synthetic import gaussian_blobs

        # Match legacy convention: True (1) = phase of fraction `porosity` (~0.40), False (0) = matrix
        vol = gaussian_blobs(
            shape,
            porosity=1.0 - porosity,
            blobiness=blobiness,
            pore_value=0,
            solid_value=1,
            seed=seed,
        )
        return (vol == 1), "openimpala.synthetic.gaussian_blobs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where generated sample files are written (default: current directory)",
    )
    parser.add_argument(
        "--base-name",
        default="SampleData_2Phase",
        help="Base filename prefix (default: SampleData_2Phase)",
    )
    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=[100, 100, 100],
        help="3-D grid dimensions: 1 int (cubic) or 3 ints NZ NY NX (default: 100 100 100)",
    )
    parser.add_argument(
        "--porosity", type=float, default=0.40, help="Target phase-1 fraction (default: 0.40)"
    )
    parser.add_argument(
        "--blobiness", type=float, default=1.2, help="Blobiness scale factor (default: 1.2)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--plot", action="store_true", help="Display interactive 2-D Z-slice figure"
    )
    args = parser.parse_args(argv)

    if len(args.shape) == 1:
        gen_shape = (args.shape[0], args.shape[0], args.shape[0])
    elif len(args.shape) == 3:
        gen_shape = (args.shape[0], args.shape[1], args.shape[2])
    else:
        raise SystemExit("--shape expects 1 integer or 3 integers.")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running on Python {platform.python_version()}")
    t0 = time.perf_counter()
    im_3d, backend_used = _generate_boolean_solid(
        gen_shape, porosity=args.porosity, blobiness=args.blobiness, seed=args.seed
    )
    elapsed = time.perf_counter() - t0
    phase1_frac = float(np.mean(im_3d))
    print(f"Generated {gen_shape} volume via {backend_used} in {elapsed:.2f}s")
    print(f"Phase-1 volume fraction: {phase1_frac:.4f} (phase-0: {1.0 - phase1_frac:.4f})")

    slice_index_z = gen_shape[0] // 2
    slice_to_plot = im_3d[slice_index_z, :, :]

    if args.plot:
        import matplotlib.pyplot as plt

        fig_2d, ax_2d = plt.subplots(1, 1, figsize=(6, 6))
        ax_2d.imshow(slice_to_plot, origin="lower", cmap="gray", interpolation="none")
        ax_2d.set_xlabel("X axis")
        ax_2d.set_ylabel("Y axis")
        ax_2d.set_title(f"Middle Z-Slice (Z={slice_index_z}) — Blobiness={args.blobiness}")
        plt.show()

    base = args.base_name
    tiff_stack_path = out_dir / f"{base}_stack_3d_1bit.tif"
    hdf5_3d_path = out_dir / f"{base}_3d.hdf5"
    xmf_path = out_dir / f"{base}_3d.xmf"
    hdf5_2d_path = out_dir / f"{base}_slice_z{slice_index_z}.hdf5"
    raw_path = out_dir / f"{base}_stack_3d_uint8.raw"
    hdf5_dataset_name = "image"

    # 1. Raw uint8 binary (always available)
    im_u8 = np.ascontiguousarray(im_3d, dtype=np.uint8)
    raw_path.write_bytes(im_u8.tobytes())
    print(f"Saved 3D RAW uint8 volume: {raw_path} ({raw_path.stat().st_size} bytes)")

    # 2. HDF5 + XDMF (when h5py is installed)
    try:
        import h5py

        with h5py.File(hdf5_3d_path, "w") as f:
            f.create_dataset(hdf5_dataset_name, data=im_u8)
        with h5py.File(hdf5_2d_path, "w") as f:
            f.create_dataset(hdf5_dataset_name, data=slice_to_plot.astype(np.uint8))
        nz, ny, nx = im_u8.shape
        xdmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="StructureGrid" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{nz} {ny} {nx}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Name="Origin" Format="XML" Dimensions="3">0 0 0</DataItem>
        <DataItem Name="Spacing" Format="XML" Dimensions="3">1 1 1</DataItem>
      </Geometry>
      <Attribute Name="Phase" AttributeType="Scalar" Center="Cell">
        <DataItem Format="HDF" Dimensions="{nz} {ny} {nx}" NumberType="UChar" Precision="1">
          {hdf5_3d_path.name}:/{hdf5_dataset_name}
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
        xmf_path.write_text(xdmf_content, encoding="utf-8")
        print(f"Saved 3D HDF5 + XDMF: {hdf5_3d_path}, {xmf_path}")
    except ImportError:
        print("Skipping HDF5/XDMF output (h5py not installed).")

    # 3. 3D TIFF stack (when tifffile is installed)
    try:
        import tifffile

        tifffile.imwrite(
            str(tiff_stack_path),
            im_3d,
            photometric="minisblack",
            compression=None,
        )
        print(f"Saved 3D TIFF stack: {tiff_stack_path}")
    except ImportError:
        print("Skipping TIFF stack output (tifffile not installed).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())