#!/usr/bin/env python3
"""Sphere Packing Hashin-Shtrikman Bounds Validation for OpenImpala.

This script generates random overlapping sphere packings at varying
porosities, solves for effective diffusivity with OpenImpala, and
validates that every result falls within the Hashin-Shtrikman upper
bound for a binary (pore/solid) composite.

Unlike the analytical_bounds_vv.py script which uses layered/random
voxel structures, this script produces physically realistic isotropic
microstructures via overlapping spheres. The HS upper bound is the
tightest possible bound for such isotropic structures:

    HS+ = 2 * phi / (3 - phi)

where phi is the pore volume fraction (phase with D=1).

The script is fully self-contained — no external datasets or
dependencies beyond openimpala, numpy, and matplotlib are needed.

Usage
-----
    python sphere_packing_vv.py [--grid-size 64] [--output sphere_packing_vv.png]

Requires: openimpala, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Sphere packing generator
# ---------------------------------------------------------------------------

def make_overlapping_spheres(
    N: int,
    target_solid_fraction: float,
    min_radius: int = 3,
    max_radius: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Generate a random overlapping sphere packing.

    Places solid spheres randomly in a 3D domain until the target solid
    volume fraction is approximately reached. The resulting pore space
    (phase 1) is isotropic on average.

    Parameters
    ----------
    N : int
        Domain size (N x N x N).
    target_solid_fraction : float
        Target fraction of solid voxels (1 - porosity).
    min_radius, max_radius : int
        Range of sphere radii (in voxels).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        3D int32 array with 0 = solid, 1 = pore.
    """
    rng = np.random.RandomState(seed)
    data = np.ones((N, N, N), dtype=np.int32)  # start all pore

    # Pre-compute coordinate grids
    coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)

    total_voxels = N ** 3
    target_solid_count = int(target_solid_fraction * total_voxels)

    solid_count = 0
    max_attempts = 10000

    for _ in range(max_attempts):
        if solid_count >= target_solid_count:
            break

        # Random sphere centre and radius
        cx = rng.randint(0, N)
        cy = rng.randint(0, N)
        cz = rng.randint(0, N)
        r = rng.randint(min_radius, max_radius + 1)

        # Distance from centre (periodic wrapping not needed — edges are fine)
        dist_sq = (
            (coords[0] - cx) ** 2
            + (coords[1] - cy) ** 2
            + (coords[2] - cz) ** 2
        )
        mask = dist_sq <= r * r
        data[mask] = 0  # mark as solid
        solid_count = int(np.sum(data == 0))

    return data


# ---------------------------------------------------------------------------
# Analytical bounds
# ---------------------------------------------------------------------------

def hs_upper_binary(phi: float) -> float:
    """Hashin-Shtrikman upper bound for binary (D0=1, D1=0) composite.

    HS+ = 2 * phi / (3 - phi)
    """
    if phi <= 0.0:
        return 0.0
    return 2.0 * phi / (3.0 - phi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate OpenImpala against HS bounds on sphere packings."
    )
    parser.add_argument(
        "--output", default="sphere_packing_vv.png",
        help="Output path for the validation plot (default: sphere_packing_vv.png)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=64,
        help="Voxel grid size N for NxNxN structures (default: 64)",
    )
    args = parser.parse_args()

    N = args.grid_size

    # Solid fractions to test (porosity = 1 - solid_fraction)
    solid_fractions = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

    print("=" * 70)
    print("OpenImpala Sphere Packing V&V (Hashin-Shtrikman Bounds)")
    print("=" * 70)
    print(f"  Grid size:         {N}^3 = {N**3:,} voxels")
    print(f"  Solid fractions:   {solid_fractions}")
    print(f"  Bound:             HS+ = 2*phi / (3 - phi)")
    print()

    try:
        import openimpala as oi
    except ImportError:
        print("ERROR: openimpala not installed.")
        return 1

    results = []
    violations = []

    print("--- Generating sphere packings and solving ---\n")
    t_total = time.time()

    with oi.Session():
        for i, sf in enumerate(solid_fractions):
            phi_target = 1.0 - sf
            print(f"  [{i+1}/{len(solid_fractions)}] "
                  f"target porosity = {phi_target:.2f} "
                  f"(solid fraction = {sf:.2f})")

            # Generate structure
            data = make_overlapping_spheres(
                N, sf, min_radius=max(2, N // 20), max_radius=max(4, N // 10),
                seed=42 + i,
            )

            # Measure actual porosity
            vf = oi.volume_fraction(data, phase=1)
            phi_actual = vf.fraction

            # Check percolation
            perc = oi.percolation_check(data, phase=1, direction="z")
            if not perc.percolates:
                print(f"    phi = {phi_actual:.4f}, does not percolate — skipping")
                continue

            # Solve
            t0 = time.time()
            result = oi.tortuosity(
                data, phase=1, direction="z",
                solver="flexgmres", max_grid_size=min(N, 32),
            )
            dt = time.time() - t0

            d_eff = phi_actual / result.tortuosity
            hs_bound = hs_upper_binary(phi_actual)

            # Validate
            tol = 1e-6
            status = "OK"
            if d_eff > hs_bound + tol:
                msg = (f"  VIOLATION: D_eff={d_eff:.6f} > HS+={hs_bound:.6f} "
                       f"at phi={phi_actual:.4f}")
                violations.append(msg)
                status = "VIOLATION"
            if d_eff < -tol:
                msg = f"  VIOLATION: D_eff={d_eff:.6f} < 0 at phi={phi_actual:.4f}"
                violations.append(msg)
                status = "VIOLATION"

            results.append({
                "phi": phi_actual,
                "tau": result.tortuosity,
                "d_eff": d_eff,
                "hs_upper": hs_bound,
                "margin": hs_bound - d_eff,
                "iterations": result.iterations,
            })

            print(f"    phi = {phi_actual:.4f}  tau = {result.tortuosity:.4f}  "
                  f"D_eff = {d_eff:.4f}  HS+ = {hs_bound:.4f}  "
                  f"margin = {hs_bound - d_eff:.4f}  [{status}]  ({dt:.1f}s)")

    t_total = time.time() - t_total
    print(f"\n  Total time: {t_total:.1f}s")

    # --- Generate plot ---
    if results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

            phi_curve = np.linspace(0.01, 0.99, 500)
            hs_curve = 2.0 * phi_curve / (3.0 - phi_curve)

            # Shaded feasible region
            ax.fill_between(phi_curve, 0, hs_curve, alpha=0.15, color="#2ca02c",
                            label="Feasible region (0 to HS+)")
            ax.plot(phi_curve, hs_curve, "-", color="#2ca02c", lw=2,
                    label=r"HS upper: $2\phi/(3-\phi)$")
            ax.plot(phi_curve, phi_curve, "--", color="gray", lw=1,
                    label=r"Wiener upper: $\phi$")

            # Solver data points
            phis = [r["phi"] for r in results]
            deffs = [r["d_eff"] for r in results]
            ax.scatter(phis, deffs, marker="o", s=80, color="#ff7f0e",
                       edgecolor="black", linewidth=0.8, zorder=5,
                       label="OpenImpala (sphere packings)")

            ax.set_xlabel(r"Porosity $\phi$", fontsize=13)
            ax.set_ylabel(r"$D_\mathrm{eff} / D_0$", fontsize=13)
            ax.set_title(
                f"Sphere Packing V&V: OpenImpala vs. Hashin-Shtrikman Upper Bound\n"
                f"(overlapping spheres, {N}$^3$ grid)",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.8)
            ax.legend(loc="upper left", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nPlot saved to: {args.output}")
        except ImportError:
            print("\nmatplotlib not available — skipping plot generation")

    # --- Summary ---
    print()
    print("=" * 70)
    if violations:
        print("VALIDATION FAILED — HS bound violations detected:")
        for v in violations:
            print(v)
        print("=" * 70)
        return 1
    else:
        print(f"VALIDATION PASSED — {len(results)} sphere packings, "
              f"all within HS upper bound.")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
