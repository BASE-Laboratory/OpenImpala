#!/usr/bin/env python3
"""Generate a high-throughput 3-D porous-media ML surrogate dataset with OpenImpala.

Runs identically on a laptop (pure-Python or single-rank CPU/GPU) or across MPI
ranks on an HPC cluster::

    python examples/generate_ml_dataset.py --output surrogate_train.npz --samples 32 --shape 32
    mpirun -np 8 python examples/generate_ml_dataset.py --output surrogate_64.h5 --samples 128 --shape 64
"""

from __future__ import annotations

import argparse
import json
import numpy as np

import openimpala as oi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="ml_surrogate_dataset.npz",
        help="Output archive (.npz, .h5/.hdf5, or .zarr)",
    )
    parser.add_argument("-n", "--samples", type=int, default=16, help="Number of 3-D volumes")
    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=[32],
        help="Cubic side length L or 3 ints NZ NY NX",
    )
    parser.add_argument(
        "--generator",
        choices=("blobs", "spheres"),
        default="blobs",
        help="Procedural microstructure family",
    )
    parser.add_argument(
        "--porosity-range",
        nargs=2,
        type=float,
        default=[0.30, 0.70],
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shape_arg = args.shape[0] if len(args.shape) == 1 else tuple(args.shape)
    if len(args.shape) not in (1, 3):
        raise SystemExit("--shape expects 1 integer (cubic) or 3 integers (NZ NY NX).")

    with oi.Session():
        ds = oi.synthetic.generate_ml_dataset(
            args.output,
            n_samples=args.samples,
            shape=shape_arg,
            generator=args.generator,
            porosity_range=(args.porosity_range[0], args.porosity_range[1]),
            seed=args.seed,
        )

    summary = {
        "output_path": ds["output_path"],
        "n_samples": ds["n_samples"],
        "shape": list(ds["shape"]),
        "generator": ds["generator"],
        "mean_porosity": float(np.mean(ds["porosity"])),
        "mean_deff_iso": float(np.mean(np.trace(ds["deff_tensor"], axis1=1, axis2=2) / 3.0)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
