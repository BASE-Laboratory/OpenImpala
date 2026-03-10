"""Command-line interface for OpenImpala.

Usage::

    openimpala analyze microstructure.tif --phase 0 --solver hypre
    openimpala vf microstructure.tif --phase 0
    openimpala percolation microstructure.tif --phase 0 --direction x
"""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openimpala",
        description="OpenImpala — transport-property computation on 3-D voxel images.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Full analysis pipeline")
    p_analyze.add_argument("image", help="Path to the 3-D image file")
    p_analyze.add_argument("--phase", type=int, default=0, help="Phase ID (default: 0)")
    p_analyze.add_argument("--threshold", type=float, default=0.5, help="Binarisation threshold")
    p_analyze.add_argument("--direction", default="x", help="Flow direction (x/y/z, default: x)")
    p_analyze.add_argument("--solver", default="flexgmres", help="HYPRE solver (default: flexgmres)")
    p_analyze.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    p_analyze.add_argument("--max-grid-size", type=int, default=32)
    p_analyze.add_argument("--verbose", type=int, default=0)

    # --- vf ---
    p_vf = sub.add_parser("vf", help="Compute volume fraction")
    p_vf.add_argument("image", help="Path to the 3-D image file")
    p_vf.add_argument("--phase", type=int, default=0)
    p_vf.add_argument("--threshold", type=float, default=0.5)
    p_vf.add_argument("--max-grid-size", type=int, default=32)

    # --- percolation ---
    p_perc = sub.add_parser("percolation", help="Percolation connectivity check")
    p_perc.add_argument("image", help="Path to the 3-D image file")
    p_perc.add_argument("--phase", type=int, default=0)
    p_perc.add_argument("--threshold", type=float, default=0.5)
    p_perc.add_argument("--direction", default="x")
    p_perc.add_argument("--max-grid-size", type=int, default=32)
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
    vf = oi.core.VolumeFraction(img, args.phase, 0)
    pc, tc = vf.value()
    frac = pc / tc if tc > 0 else 0.0
    result = {"phase": args.phase, "phase_count": pc, "total_count": tc, "volume_fraction": frac}
    print(json.dumps(result, indent=2))
    return 0


def _cmd_percolation(args: argparse.Namespace) -> int:
    import openimpala as oi

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
        max_grid_size=args.max_grid_size,
    )
    d = oi.facade._parse_direction(args.direction)
    pc = oi.core.PercolationCheck(img, args.phase, d, args.verbose)
    result = {
        "phase": args.phase,
        "direction": args.direction.upper(),
        "percolates": pc.percolates,
        "active_volume_fraction": pc.active_volume_fraction,
    }
    print(json.dumps(result, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    import openimpala as oi

    _, img = oi.read_image(
        args.image,
        threshold=args.threshold,
        max_grid_size=args.max_grid_size,
    )

    d = oi.facade._parse_direction(args.direction)

    # Volume fraction
    vf_calc = oi.core.VolumeFraction(img, args.phase, 0)
    vf_val = vf_calc.value_vf()

    # Percolation
    pc = oi.core.PercolationCheck(img, args.phase, d, args.verbose)

    result: dict = {
        "input_file": args.image,
        "phase": args.phase,
        "direction": args.direction.upper(),
        "volume_fraction": vf_val,
        "percolates": pc.percolates,
        "active_volume_fraction": pc.active_volume_fraction,
    }

    if pc.percolates:
        st = oi.facade._parse_solver(args.solver)
        solver = oi.core.TortuosityHypre(
            img, vf_val, args.phase, d, st, ".",
            0.0, 1.0, args.verbose, False,
        )
        try:
            tau = solver.value()
        except RuntimeError as exc:
            result["error"] = str(exc)
        else:
            result["tortuosity"] = tau
            result["solver_converged"] = solver.solver_converged
            result["iterations"] = solver.iterations
            result["residual_norm"] = solver.residual_norm
    else:
        result["tortuosity"] = None
        result["note"] = "Phase does not percolate — tortuosity is undefined."

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
