#!/usr/bin/env python3
"""Analytical Bounds Verification & Validation for OpenImpala.

This script validates OpenImpala's effective diffusivity calculations against
strict theoretical bounds from composite materials theory. It generates
synthetic microstructures at varying volume fractions, solves for transport
properties, and verifies that all computed values respect the Wiener (Voigt/
Reuss) and Hashin-Shtrikman bounds.

The script produces a publication-ready plot (validation_bounds.png) showing
the bounds as shaded regions with OpenImpala data points overlaid.

Theory
------
For a two-phase 3D composite with phase diffusivities D0 and D1, the
effective diffusivity D_eff is bounded by:

    Wiener (widest):
        Reuss  (series):   D_R  = 1 / (phi/D1 + (1-phi)/D0)
        Voigt  (parallel): D_V  = phi*D1 + (1-phi)*D0

    Hashin-Shtrikman (tightest without geometric information):
        HS-: D1 + (1-phi) / (1/(D0-D1) + phi/(3*D1))
        HS+: D0 + phi / (1/(D1-D0) + (1-phi)/(3*D0))

    where phi is the volume fraction of phase 1, and D0 > D1 > 0.

For binary structures (D0=1, D1=0), the solver runs in single-phase mode
and the effective diffusivity is recovered from tortuosity: D_eff = VF / tau.
The Wiener upper bound reduces to D_V = phi, and the HS upper bound reduces
to HS+ = 2*phi / (3 - phi).

Usage
-----
    python analytical_bounds_vv.py [--output validation_bounds.png]

Requires: openimpala, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Analytical bound functions (pure math, no solver dependency)
# ---------------------------------------------------------------------------

def wiener_upper(phi: np.ndarray, D0: float, D1: float) -> np.ndarray:
    """Voigt / parallel / arithmetic mean bound (upper Wiener).

    D_V = phi * D1 + (1 - phi) * D0
    """
    return phi * D1 + (1.0 - phi) * D0


def wiener_lower(phi: np.ndarray, D0: float, D1: float) -> np.ndarray:
    """Reuss / series / harmonic mean bound (lower Wiener).

    D_R = 1 / (phi / D1 + (1 - phi) / D0)

    Returns 0 where D1 == 0 (degenerate case).
    """
    if D1 == 0.0:
        return np.zeros_like(phi)
    return 1.0 / (phi / D1 + (1.0 - phi) / D0)


def hashin_shtrikman_upper(phi: np.ndarray, D0: float, D1: float) -> np.ndarray:
    """Hashin-Shtrikman upper bound (assumes D0 > D1).

    HS+ = D0 + phi / (1/(D1 - D0) + (1 - phi)/(3 * D0))

    For D1 = 0: reduces to HS+ = 2*phi / (3 - phi)  [with phi = porosity].
    """
    if D1 == 0.0:
        # Derivation for D0=1, D1=0, phi = porosity (vol. frac. of D0 phase):
        #   HS+ = D0 + phi_solid / (1/(D1-D0) + phi_pore/(3*D0))
        # where phi_solid = 1 - phi, phi_pore = phi:
        #   = 1 + (1-phi) / (-1 + phi/3)
        #   = 2*phi / (3 - phi)
        return 2.0 * phi * D0 / (3.0 * D0 - phi * D0)
    denom = 1.0 / (D1 - D0) + (1.0 - phi) / (3.0 * D0)
    return D0 + phi / denom


def hashin_shtrikman_lower(phi: np.ndarray, D0: float, D1: float) -> np.ndarray:
    """Hashin-Shtrikman lower bound (assumes D0 > D1 > 0).

    HS- = D1 + (1 - phi) / (1/(D0 - D1) + phi/(3 * D1))

    Returns 0 where D1 == 0 (degenerate case: lower bound is trivially 0).
    """
    if D1 == 0.0:
        return np.zeros_like(phi)
    denom = 1.0 / (D0 - D1) + phi / (3.0 * D1)
    return D1 + (1.0 - phi) / denom


# ---------------------------------------------------------------------------
# Structure generators
# ---------------------------------------------------------------------------

def make_parallel_layers(N: int, phi: float) -> np.ndarray:
    """Create parallel layers along the Z-axis (flow in Z).

    Layers of phase 1 (pore) stacked with layers of phase 0 (solid).
    The number of pore layers is approximately phi * N.
    All pore layers are contiguous and span the full Z extent, ensuring
    percolation in Z.

    For a binary structure (D_pore=1, D_solid=0), the effective diffusivity
    in Z equals phi — the Voigt / Wiener upper bound.
    """
    n_pore = max(1, min(N - 1, int(round(phi * N))))
    data = np.zeros((N, N, N), dtype=np.int32)
    # Pore layers along Y: first n_pore rows
    data[:, :n_pore, :] = 1
    return data


def make_series_layers(N: int, phi: float) -> np.ndarray:
    """Create series layers perpendicular to the Z-axis (flow in Z).

    Alternating layers of pore (phase 1) and solid (phase 0) along Z.
    For binary (D_solid=0), this geometry does NOT percolate in Z because
    solid layers block transport entirely.

    For multi-phase (D_solid > 0), this gives D_eff = Reuss/harmonic bound.
    """
    n_pore = max(1, min(N - 1, int(round(phi * N))))
    data = np.zeros((N, N, N), dtype=np.int32)
    # Distribute pore layers evenly along Z
    pore_indices = np.linspace(0, N - 1, n_pore, dtype=int)
    data[pore_indices, :, :] = 1
    return data


def make_random_mixture(N: int, phi: float, seed: int = 42) -> np.ndarray:
    """Create a random uniform mixture of phase 0 and phase 1.

    Each voxel is independently assigned phase 1 (pore) with probability phi.
    The resulting structure is isotropic on average.
    """
    rng = np.random.RandomState(seed)
    return rng.choice([0, 1], size=(N, N, N), p=[1 - phi, phi]).astype(np.int32)


# ---------------------------------------------------------------------------
# OpenImpala solver execution
# ---------------------------------------------------------------------------

def compute_deff_binary(data: np.ndarray, direction: str = "z") -> dict | None:
    """Run the OpenImpala tortuosity solver on a binary structure.

    In single-phase mode (which the Python API uses), phase 1 has D=1 and
    phase 0 has D=0.  The effective diffusivity is D_eff = VF / tau.

    Returns a dict with solver results, or None if the phase doesn't percolate.
    """
    import openimpala as oi

    phase = 1  # pore phase

    # Check percolation first
    perc = oi.percolation_check(data, phase=phase, direction=direction)
    if not perc.percolates:
        return None

    # Volume fraction
    vf = oi.volume_fraction(data, phase=phase)

    # Solve for tortuosity
    result = oi.tortuosity(
        data, phase=phase, direction=direction,
        solver="flexgmres", max_grid_size=32,
    )

    d_eff = vf.fraction / result.tortuosity

    return {
        "phi": vf.fraction,
        "tau": result.tortuosity,
        "d_eff": d_eff,
        "converged": result.solver_converged,
        "iterations": result.iterations,
        "residual": result.residual_norm,
    }


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_result(d_eff: float, phi: float, D0: float, D1: float,
                    label: str) -> list[str]:
    """Check that a computed D_eff respects all applicable bounds.

    Returns a list of violation messages (empty if all bounds satisfied).
    """
    violations = []
    tol = 1e-6  # numerical tolerance

    dv = wiener_upper(np.array([phi]), D0, D1)[0]
    dr = wiener_lower(np.array([phi]), D0, D1)[0]
    hs_plus = hashin_shtrikman_upper(np.array([phi]), D0, D1)[0]
    hs_minus = hashin_shtrikman_lower(np.array([phi]), D0, D1)[0]

    if d_eff > dv + tol:
        violations.append(
            f"  VIOLATION [{label}]: D_eff={d_eff:.6f} > Wiener upper={dv:.6f}"
        )
    if d_eff < dr - tol:
        violations.append(
            f"  VIOLATION [{label}]: D_eff={d_eff:.6f} < Wiener lower={dr:.6f}"
        )
    if d_eff > hs_plus + tol:
        violations.append(
            f"  VIOLATION [{label}]: D_eff={d_eff:.6f} > HS upper={hs_plus:.6f}"
        )
    if d_eff < hs_minus - tol:
        violations.append(
            f"  VIOLATION [{label}]: D_eff={d_eff:.6f} < HS lower={hs_minus:.6f}"
        )

    return violations


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plot(
    phi_range: np.ndarray,
    D0: float,
    D1: float,
    solver_data: list[dict],
    output_path: str,
) -> None:
    """Generate a publication-ready validation plot.

    Shows Wiener and HS bounds as shaded regions, with OpenImpala solver
    results overlaid as data points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    # --- Compute bound curves for the multi-phase reference case ---
    dv = wiener_upper(phi_range, D0, D1)
    dr = wiener_lower(phi_range, D0, D1)
    hs_plus = hashin_shtrikman_upper(phi_range, D0, D1)
    hs_minus = hashin_shtrikman_lower(phi_range, D0, D1)

    # --- Shaded regions ---
    # Wiener region (widest)
    ax.fill_between(phi_range, dr, dv, alpha=0.10, color="#1f77b4",
                    label="Wiener bounds (Reuss-Voigt)")
    # HS region (tighter)
    ax.fill_between(phi_range, hs_minus, hs_plus, alpha=0.20, color="#2ca02c",
                    label="Hashin-Shtrikman bounds")

    # --- Bound curves ---
    ax.plot(phi_range, dv, "-", color="#1f77b4", lw=1.5, label="Voigt (parallel)")
    ax.plot(phi_range, dr, "--", color="#1f77b4", lw=1.5, label="Reuss (series)")
    ax.plot(phi_range, hs_plus, "-", color="#2ca02c", lw=1.5, label="HS upper")
    ax.plot(phi_range, hs_minus, "--", color="#2ca02c", lw=1.5, label="HS lower")

    # --- Binary bounds for reference (D0=1, D1=0) ---
    hs_plus_binary = hashin_shtrikman_upper(phi_range, 1.0, 0.0)
    ax.plot(phi_range, hs_plus_binary, ":", color="gray", lw=1.0,
            label="HS upper (binary, $D_1=0$)")

    # --- Solver data points ---
    # Group by geometry type
    markers = {"parallel": ("^", "#d62728", 90), "random": ("o", "#ff7f0e", 60)}
    for geom_type, (marker, color, size) in markers.items():
        pts = [d for d in solver_data if d["type"] == geom_type]
        if pts:
            phis = [p["phi"] for p in pts]
            deffs = [p["d_eff"] for p in pts]
            ax.scatter(phis, deffs, marker=marker, s=size, color=color,
                       edgecolor="black", linewidth=0.5, zorder=5,
                       label=f"OpenImpala ({geom_type})")

    # --- Formatting ---
    ax.set_xlabel(r"Volume Fraction $\phi$ (pore phase)", fontsize=13)
    ax.set_ylabel(r"Effective Diffusivity $D_\mathrm{eff} / D_0$", fontsize=13)
    ax.set_title(
        f"Verification: OpenImpala vs. Analytical Bounds\n"
        f"($D_0 = {D0}$, $D_1 = {D1}$; binary solver: $D_0 = 1$, $D_1 = 0$)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate OpenImpala against analytical effective diffusivity bounds."
    )
    parser.add_argument(
        "--output", default="validation_bounds.png",
        help="Output path for the validation plot (default: validation_bounds.png)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=32,
        help="Voxel grid size N for NxNxN structures (default: 32)",
    )
    args = parser.parse_args()

    # Reference multi-phase diffusivities (for bound curves)
    D0 = 1.0  # high-diffusivity phase
    D1 = 0.1  # low-diffusivity phase

    # Volume fractions to test
    phi_test = np.arange(0.1, 0.95, 0.1)
    phi_plot = np.linspace(0.001, 0.999, 500)
    N = args.grid_size

    print("=" * 70)
    print("OpenImpala Analytical Bounds V&V")
    print("=" * 70)
    print(f"  Reference bounds: D0={D0}, D1={D1}")
    print(f"  Binary solver:    D0=1.0, D1=0.0 (single-phase mode)")
    print(f"  Grid size:        {N}^3 = {N**3:,} voxels")
    print(f"  Volume fractions: {phi_test}")
    print()

    # --- Self-test: verify bound ordering at a few points ---
    print("--- Self-test: bound ordering ---")
    for phi_val in [0.2, 0.5, 0.8]:
        phi_arr = np.array([phi_val])
        dv = wiener_upper(phi_arr, D0, D1)[0]
        dr = wiener_lower(phi_arr, D0, D1)[0]
        hs_p = hashin_shtrikman_upper(phi_arr, D0, D1)[0]
        hs_m = hashin_shtrikman_lower(phi_arr, D0, D1)[0]
        assert dr <= hs_m <= hs_p <= dv, (
            f"Bound ordering violated at phi={phi_val}: "
            f"DR={dr:.4f} HS-={hs_m:.4f} HS+={hs_p:.4f} DV={dv:.4f}"
        )
        print(f"  phi={phi_val:.1f}: Reuss={dr:.4f} <= HS-={hs_m:.4f} "
              f"<= HS+={hs_p:.4f} <= Voigt={dv:.4f}  OK")
    print()

    # --- Run OpenImpala solver on synthetic structures ---
    solver_data = []
    all_violations = []

    try:
        import openimpala as oi
    except ImportError:
        print("ERROR: openimpala not installed. Generating bounds plot only.")
        generate_plot(phi_plot, D0, D1, [], args.output)
        return 1

    print("--- Running OpenImpala solver ---")
    t_total_start = time.time()

    with oi.Session():
        for phi_target in phi_test:
            print(f"\n  phi = {phi_target:.2f}:")

            # --- Parallel layers ---
            data_par = make_parallel_layers(N, phi_target)
            t0 = time.time()
            res = compute_deff_binary(data_par, direction="z")
            dt = time.time() - t0
            if res is not None:
                solver_data.append({**res, "type": "parallel"})
                v = validate_result(res["d_eff"], res["phi"], 1.0, 0.0,
                                    f"parallel phi={phi_target:.2f}")
                all_violations.extend(v)
                print(f"    Parallel:  phi={res['phi']:.4f}  tau={res['tau']:.4f}  "
                      f"D_eff={res['d_eff']:.4f}  ({dt:.2f}s, {res['iterations']} iter)")
            else:
                print(f"    Parallel:  did not percolate (expected for binary series)")

            # --- Random mixture ---
            data_rnd = make_random_mixture(N, phi_target, seed=int(phi_target * 100))
            t0 = time.time()
            res = compute_deff_binary(data_rnd, direction="z")
            dt = time.time() - t0
            if res is not None:
                solver_data.append({**res, "type": "random"})
                v = validate_result(res["d_eff"], res["phi"], 1.0, 0.0,
                                    f"random phi={phi_target:.2f}")
                all_violations.extend(v)
                print(f"    Random:    phi={res['phi']:.4f}  tau={res['tau']:.4f}  "
                      f"D_eff={res['d_eff']:.4f}  ({dt:.2f}s, {res['iterations']} iter)")
            else:
                print(f"    Random:    did not percolate at phi={phi_target:.2f}")

    t_total = time.time() - t_total_start
    print(f"\n  Total solver time: {t_total:.1f}s")

    # --- Generate plot ---
    print()
    generate_plot(phi_plot, D0, D1, solver_data, args.output)

    # --- Report ---
    print()
    print("=" * 70)
    if all_violations:
        print("VALIDATION FAILED — bound violations detected:")
        for v in all_violations:
            print(v)
        print("=" * 70)
        return 1
    else:
        n_points = len(solver_data)
        print(f"VALIDATION PASSED — {n_points} data points, all within bounds.")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
