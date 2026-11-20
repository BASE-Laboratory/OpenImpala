"""Command-line interface for OpenImpala.

Usage::

    openimpala analyze microstructure.tif --phase 0 --solver mlmg
    openimpala vf microstructure.tif --phase 0
    openimpala percolation microstructure.tif --phase 0 --direction x
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Union

from . import __version__


def _parse_max_grid_size(val: str) -> Union[int, str]:
    """Accept ``'auto'`` or a positive integer for ``--max-grid-size``."""
    if val.strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(val)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected 'auto' or a positive integer, got {val!r}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"max_grid_size must be > 0, got {parsed}")
    return parsed


def _parse_threshold(val: str) -> Union[float, str]:
    """Accept ``'otsu'`` or a finite floating-point number for ``--threshold``."""
    import math

    if val.strip().lower() == "otsu":
        return "otsu"
    try:
        parsed = float(val)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a number or 'otsu', got {val!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"threshold must be finite, got {val!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openimpala",
        description="OpenImpala — transport-property computation on 3-D voxel images.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    def _add_io_args(subparser: argparse.ArgumentParser) -> None:
        grp = subparser.add_mutually_exclusive_group()
        grp.add_argument(
            "--dark-is-pore",
            dest="dark_is_pore",
            action="store_true",
            default=True,
            help="Dark voxels (<= threshold) map to pore phase 0 (default)",
        )
        grp.add_argument(
            "--bright-is-pore",
            dest="dark_is_pore",
            action="store_false",
            help="Inverted contrast: bright voxels (> threshold) map to pore phase 0",
        )
        subparser.add_argument(
            "--hdf5-dataset",
            default="/data",
            help="HDF5 dataset path when reading .h5/.hdf5 files (default: /data)",
        )

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Full analysis pipeline")
    p_analyze.add_argument("image", help="Path to the 3-D image file")
    p_analyze.add_argument("--phase", type=int, default=0, help="Phase ID (default: 0)")
    p_analyze.add_argument(
        "--threshold",
        type=_parse_threshold,
        default=0.5,
        help="Binarisation threshold: float or 'otsu' (default: 0.5)",
    )
    _add_io_args(p_analyze)
    p_analyze.add_argument("--direction", default="x", help="Flow direction (x/y/z, default: x)")
    p_analyze.add_argument(
        "--solver",
        default="auto",
        help="Solver backend: auto, mlmg, pcg, flexgmres, bicgstab, gmres, smg, pfmg (default: auto)",
    )
    p_analyze.add_argument(
        "--preconditioner",
        default="smg",
        help="HYPRE Krylov preconditioner: smg or pfmg (default: smg)",
    )
    p_analyze.add_argument(
        "--voxel-size",
        nargs="+",
        type=float,
        default=None,
        help="Physical voxel size: single float (isotropic) or 3 floats DZ DY DX",
    )
    p_analyze.add_argument(
        "--downsample",
        nargs="+",
        type=int,
        default=None,
        help="Coarsen volume before analysis: 1 int (isotropic) or 3 ints FZ FY FX",
    )
    p_analyze.add_argument(
        "--deff",
        action="store_true",
        help="Also compute the 3x3 periodic homogenised effective diffusivity tensor",
    )
    p_analyze.add_argument(
        "--bpx-electrode",
        choices=("negative", "positive", "separator"),
        default=None,
        help="Embed a Faraday BPX electrode parameterisation fragment in JSON output",
    )
    p_analyze.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    p_analyze.add_argument("--max-grid-size", type=_parse_max_grid_size, default="auto")
    p_analyze.add_argument("--verbose", type=int, default=0)

    # --- vf ---
    p_vf = sub.add_parser("vf", help="Compute volume fraction")
    p_vf.add_argument("image", help="Path to the 3-D image file")
    p_vf.add_argument("--phase", type=int, default=0)
    p_vf.add_argument("--threshold", type=_parse_threshold, default=0.5)
    _add_io_args(p_vf)
    p_vf.add_argument("--max-grid-size", type=_parse_max_grid_size, default="auto")

    # --- percolation ---
    p_perc = sub.add_parser("percolation", help="Percolation connectivity check")
    p_perc.add_argument("image", help="Path to the 3-D image file")
    p_perc.add_argument("--phase", type=int, default=0)
    p_perc.add_argument("--threshold", type=_parse_threshold, default=0.5)
    _add_io_args(p_perc)
    p_perc.add_argument("--direction", default="x")
    p_perc.add_argument("--max-grid-size", type=_parse_max_grid_size, default="auto")
    p_perc.add_argument("--verbose", type=int, default=0)

    # --- generate-dataset ---
    p_gen = sub.add_parser(
        "generate-dataset",
        help="Generate synthetic 3-D porous-media ML surrogate dataset (.npz/.h5/.zarr)",
    )
    p_gen.add_argument("output", help="Output dataset path (.npz, .h5/.hdf5, or .zarr)")
    p_gen.add_argument(
        "-n", "--samples", type=int, default=16, help="Number of 3-D samples (default: 16)"
    )
    p_gen.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=[32],
        help="Grid shape: 1 int (cubic L) or 3 ints NZ NY NX (default: 32)",
    )
    p_gen.add_argument(
        "--generator",
        choices=("blobs", "spheres"),
        default="blobs",
        help="Microstructure generator (default: blobs)",
    )
    p_gen.add_argument(
        "--porosity-range",
        nargs=2,
        type=float,
        default=[0.30, 0.70],
        metavar=("LOW", "HIGH"),
        help="Uniform porosity range (default: 0.30 0.70)",
    )
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_gen.add_argument("--solver", default="auto", help="Solver backend (default: auto)")
    p_gen.add_argument(
        "--no-deff", action="store_true", help="Skip 3x3 periodic homogenisation solve"
    )
    p_gen.add_argument(
        "--no-tortuosity", action="store_true", help="Skip directional tortuosity solves"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``openimpala`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Lazy imports — only pulled in when actually running a command
    import openimpala as oi

    with oi.Session():
        if args.command == "vf":
            return _cmd_vf(args)
        elif args.command == "percolation":
            return _cmd_percolation(args)
        elif args.command == "analyze":
            return _cmd_analyze(args)
        elif args.command == "generate-dataset":
            return _cmd_generate_dataset(args)

    return 0


def _is_cli_io_rank(oi_mod) -> bool:
    try:
        return int(oi_mod.build_info().get("mpi_rank", 0)) == 0
    except Exception:
        return True


def _cmd_vf(args: argparse.Namespace) -> int:
    import openimpala as oi

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
        dark_is_pore=args.dark_is_pore,
        hdf5_dataset=args.hdf5_dataset,
        max_grid_size=args.max_grid_size,
    )
    vf_res = oi.volume_fraction(img, phase=args.phase)
    result = {
        "phase": args.phase,
        "phase_count": vf_res.phase_count,
        "total_count": vf_res.total_count,
        "volume_fraction": vf_res.fraction,
    }
    if _is_cli_io_rank(oi):
        print(json.dumps(result, indent=2))
    return 0


def _cmd_percolation(args: argparse.Namespace) -> int:
    import openimpala as oi

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
        dark_is_pore=args.dark_is_pore,
        hdf5_dataset=args.hdf5_dataset,
        max_grid_size=args.max_grid_size,
    )
    pc_res = oi.percolation_check(
        img, phase=args.phase, direction=args.direction, verbose=args.verbose
    )
    result = {
        "phase": args.phase,
        "direction": args.direction.upper(),
        "percolates": pc_res.percolates,
        "active_volume_fraction": pc_res.active_volume_fraction,
    }
    if _is_cli_io_rank(oi):
        print(json.dumps(result, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    import math
    import openimpala as oi

    vz = None
    if args.voxel_size is not None:
        if len(args.voxel_size) == 1:
            vz = args.voxel_size[0]
        elif len(args.voxel_size) == 3:
            vz = tuple(args.voxel_size)
        else:
            raise SystemExit(
                f"Error: --voxel-size expects 1 value (isotropic) or 3 values (DZ DY DX), got {len(args.voxel_size)}"
            )

    ds_factor: Union[int, tuple[int, int, int], None] = None
    if getattr(args, "downsample", None) is not None:
        if len(args.downsample) == 1:
            ds_factor = int(args.downsample[0])
        elif len(args.downsample) == 3:
            ds_factor = (int(args.downsample[0]), int(args.downsample[1]), int(args.downsample[2]))
        else:
            raise SystemExit(
                f"Error: --downsample expects 1 value (isotropic) or 3 values (FZ FY FX), got {len(args.downsample)}"
            )
        if (isinstance(ds_factor, int) and ds_factor < 1) or (
            isinstance(ds_factor, tuple) and any(f < 1 for f in ds_factor)
        ):
            raise SystemExit(f"Error: --downsample factors must be >= 1, got {args.downsample}")

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
        dark_is_pore=args.dark_is_pore,
        hdf5_dataset=args.hdf5_dataset,
        voxel_size=vz,
        max_grid_size=args.max_grid_size,
    )

    if ds_factor is not None:
        img = img.downsample(factor=ds_factor, mode="majority")

    vf_res = oi.volume_fraction(img, phase=args.phase)

    result: dict = {
        "input_file": args.image,
        "phase": args.phase,
        "direction": args.direction.upper(),
        "volume_fraction": vf_res.fraction,
    }
    if vz is not None or ds_factor is not None:
        if hasattr(img, "voxel_size"):
            result["voxel_size"] = list(img.voxel_size)
    if ds_factor is not None and hasattr(img, "shape"):
        result["shape"] = list(img.shape)

    tau_res = None
    try:
        tau_res = oi.tortuosity(
            img,
            phase=args.phase,
            direction=args.direction,
            solver=args.solver,
            preconditioner=args.preconditioner,
            verbose=args.verbose,
        )
    except oi.PercolationError:
        result["percolates"] = False
        result["active_volume_fraction"] = 0.0
        result["tortuosity"] = None
        result["note"] = "Phase does not percolate — tortuosity is undefined."
    except oi.ConvergenceError as exc:
        pc_res = oi.percolation_check(
            img, phase=args.phase, direction=args.direction, verbose=args.verbose
        )
        result["percolates"] = pc_res.percolates
        result["active_volume_fraction"] = pc_res.active_volume_fraction
        result["error"] = str(exc)
    else:
        result["percolates"] = True
        result["active_volume_fraction"] = tau_res.active_volume_fraction
        result["tortuosity"] = tau_res.tortuosity
        result["effective_diffusivity"] = tau_res.effective_diffusivity
        nm = tau_res.macmullin_number
        result["macmullin_number"] = nm if math.isfinite(nm) else None
        result["bruggeman_exponent"] = tau_res.bruggeman_exponent
        result["solver_converged"] = tau_res.solver_converged
        result["iterations"] = tau_res.iterations
        result["residual_norm"] = tau_res.residual_norm
        result["flux_in"] = tau_res.flux_in
        result["flux_out"] = tau_res.flux_out

    if getattr(args, "deff", False):
        hypre_solver = "flexgmres" if args.solver in ("auto", "mlmg") else args.solver
        deff_res = oi.effective_diffusivity(
            img, phase=args.phase, solver=hypre_solver, verbose=args.verbose
        )
        result["deff_tensor"] = deff_res.tensor.tolist()
        result["deff_diagonal"] = list(deff_res.diagonal)

    bpx_elec = getattr(args, "bpx_electrode", None)
    if bpx_elec is not None:
        sec = "Separator" if bpx_elec == "separator" else f"{bpx_elec.capitalize()} electrode"
        te = (
            float(tau_res.effective_diffusivity)
            if (tau_res is not None and tau_res.effective_diffusivity is not None)
            else 0.0
        )
        b_val = (
            float(math.log(te) / math.log(vf_res.fraction))
            if (0.0 < vf_res.fraction < 1.0 and te > 0.0)
            else 0.0
        )
        result["bpx"] = {
            "_comment": "BPX-compatible fragment. Merge into a full BPX file for PyBaMM import.",
            "electrode": bpx_elec,
            "Parameterisation": {
                sec: {
                    "Porosity": vf_res.fraction,
                    "Transport efficiency": te,
                    "Bruggeman coefficient (electrolyte)": b_val,
                }
            },
        }

    if _is_cli_io_rank(oi):
        output = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output + "\n")
            print(f"Results written to {args.output}")
        else:
            print(output)

    return 0


def _cmd_generate_dataset(args: argparse.Namespace) -> int:
    import numpy as np
    import openimpala as oi

    if len(args.shape) == 1:
        shape_arg: Union[int, tuple[int, int, int]] = int(args.shape[0])
    elif len(args.shape) == 3:
        shape_arg = (int(args.shape[0]), int(args.shape[1]), int(args.shape[2]))
    else:
        raise SystemExit(
            f"Error: --shape expects 1 value (cubic) or 3 values (NZ NY NX), got {len(args.shape)}"
        )

    ds = oi.synthetic.generate_ml_dataset(
        args.output,
        n_samples=args.samples,
        shape=shape_arg,
        generator=args.generator,
        porosity_range=(float(args.porosity_range[0]), float(args.porosity_range[1])),
        seed=args.seed,
        compute_deff=not args.no_deff,
        compute_tortuosity=not args.no_tortuosity,
        solver=args.solver,
        return_volumes=False,
    )
    summary = {
        "output_path": ds["output_path"],
        "n_samples": ds["n_samples"],
        "shape": list(ds["shape"]),
        "generator": ds["generator"],
        "mean_porosity": float(np.mean(ds["porosity"])),
    }
    if not args.no_deff:
        summary["mean_deff_iso"] = float(
            np.mean(np.trace(ds["deff_tensor"], axis1=1, axis2=2) / 3.0)
        )
    if _is_cli_io_rank(oi):
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
