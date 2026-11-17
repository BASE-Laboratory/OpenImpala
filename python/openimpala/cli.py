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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openimpala",
        description="OpenImpala — transport-property computation on 3-D voxel images.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Full analysis pipeline")
    p_analyze.add_argument("image", help="Path to the 3-D image file")
    p_analyze.add_argument("--phase", type=int, default=0, help="Phase ID (default: 0)")
    p_analyze.add_argument("--threshold", type=float, default=0.5, help="Binarisation threshold")
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
    p_analyze.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    p_analyze.add_argument("--max-grid-size", type=_parse_max_grid_size, default="auto")
    p_analyze.add_argument("--verbose", type=int, default=0)

    # --- vf ---
    p_vf = sub.add_parser("vf", help="Compute volume fraction")
    p_vf.add_argument("image", help="Path to the 3-D image file")
    p_vf.add_argument("--phase", type=int, default=0)
    p_vf.add_argument("--threshold", type=float, default=0.5)
    p_vf.add_argument("--max-grid-size", type=_parse_max_grid_size, default="auto")

    # --- percolation ---
    p_perc = sub.add_parser("percolation", help="Percolation connectivity check")
    p_perc.add_argument("image", help="Path to the 3-D image file")
    p_perc.add_argument("--phase", type=int, default=0)
    p_perc.add_argument("--threshold", type=float, default=0.5)
    p_perc.add_argument("--direction", default="x")
    p_perc.add_argument("--max-grid-size", type=_parse_max_grid_size, default="auto")
    p_perc.add_argument("--verbose", type=int, default=0)

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

    return 0


def _cmd_vf(args: argparse.Namespace) -> int:
    import openimpala as oi

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
        max_grid_size=args.max_grid_size,
    )
    vf_res = oi.volume_fraction(img, phase=args.phase)
    result = {
        "phase": args.phase,
        "phase_count": vf_res.phase_count,
        "total_count": vf_res.total_count,
        "volume_fraction": vf_res.fraction,
    }
    print(json.dumps(result, indent=2))
    return 0


def _cmd_percolation(args: argparse.Namespace) -> int:
    import openimpala as oi

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
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
    print(json.dumps(result, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
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

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
        voxel_size=vz,
        max_grid_size=args.max_grid_size,
    )

    vf_res = oi.volume_fraction(img, phase=args.phase)

    result: dict = {
        "input_file": args.image,
        "phase": args.phase,
        "direction": args.direction.upper(),
        "volume_fraction": vf_res.fraction,
    }
    if vz is not None:
        result["voxel_size"] = list(img.voxel_size)

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
        result["solver_converged"] = tau_res.solver_converged
        result["iterations"] = tau_res.iterations
        result["residual_norm"] = tau_res.residual_norm
        result["flux_in"] = tau_res.flux_in
        result["flux_out"] = tau_res.flux_out

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Results written to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
