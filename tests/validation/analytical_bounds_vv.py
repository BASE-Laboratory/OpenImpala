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

    Here ``phi`` is the volume fraction of phase 1 (diffusivity ``D0``) and
    ``1 - phi`` is the volume fraction of phase 0 (diffusivity ``D1``, with
    ``D0 >= D1 >= 0``):

        D_V = phi * D0 + (1 - phi) * D1
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    return phi_arr * D0 + (1.0 - phi_arr) * D1


def wiener_lower(phi: np.ndarray, D0: float, D1: float) -> np.ndarray:
    """Reuss / series / harmonic mean bound (lower Wiener).

        D_R = 1 / (phi / D0 + (1 - phi) / D1)

    Returns 0 where ``D1 == 0`` and ``phi < 1``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if D1 == 0.0:
        return np.where(phi_arr >= 1.0, D0, 0.0)
    return 1.0 / (phi_arr / D0 + (1.0 - phi_arr) / D1)


def hashin_shtrikman_upper(phi: np.ndarray, D0: float, D1: float) -> np.ndarray:
    """3-D Hashin-Shtrikman isotropic upper bound (``D0 >= D1 >= 0``).

    High-diffusivity ``D0`` matrix (volume fraction ``phi``) enclosing
    spherical inclusions of ``D1`` (volume fraction ``1 - phi``):

        HS+ = D0 + (1 - phi) / (1 / (D1 - D0) + phi / (3 * D0))

    For ``D1 == 0`` this simplifies to ``2 * phi * D0 / (3 - phi)``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if D0 == D1:
        return np.full_like(phi_arr, D0)
    if D1 == 0.0:
        return 2.0 * phi_arr * D0 / (3.0 - phi_arr)
    denom = 1.0 / (D1 - D0) + phi_arr / (3.0 * D0)
    return D0 + (1.0 - phi_arr) / denom


def hashin_shtrikman_lower(phi: np.ndarray, D0: float, D1: float) -> np.ndarray:
    """3-D Hashin-Shtrikman isotropic lower bound (``D0 >= D1 >= 0``).

    Low-diffusivity ``D1`` matrix (volume fraction ``1 - phi``) enclosing
    spherical inclusions of ``D0`` (volume fraction ``phi``):

        HS- = D1 + phi / (1 / (D0 - D1) + (1 - phi) / (3 * D1))

    Returns 0 where ``D1 == 0`` and ``phi < 1``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    if D0 == D1:
        return np.full_like(phi_arr, D0)
    if D1 == 0.0:
        return np.where(phi_arr >= 1.0, D0, 0.0)
    denom = 1.0 / (D0 - D1) + (1.0 - phi_arr) / (3.0 * D1)
    return D1 + phi_arr / denom


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


def make_resolved_isotropic_media(N: int, phi: float, seed: int = 42) -> np.ndarray:
    """Create a spatially resolved 3-D isotropic random medium at volume fraction ``phi``."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)
    field = rng.randn(N, N, N)
    sigma = max(1.5, N / 14.0)
    smoothed = gaussian_filter(field, sigma=sigma, mode="wrap")
    thresh = np.quantile(smoothed, 1.0 - phi)
    return (smoothed >= thresh).astype(np.int32)


# ---------------------------------------------------------------------------
# OpenImpala solver execution
# ---------------------------------------------------------------------------

def compute_deff_binary(data: np.ndarray, direction: str = "z") -> dict | None:
    """Run the OpenImpala tortuosity solver on a binary structure (D_1=1, D_0=0)."""
    import openimpala as oi

    phase = 1  # pore phase

    perc = oi.percolation_check(data, phase=phase, direction=direction)
    if not perc.percolates:
        return None

    vf = oi.volume_fraction(data, phase=phase)
    result = oi.tortuosity(
        data, phase=phase, direction=direction,
        solver="mlmg", max_grid_size=32,
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


def compute_deff_multiphase(
    data: np.ndarray, D0: float, D1: float, direction: str = "z"
) -> dict:
    """Run OpenImpala multi-phase transport (phase 1 = D0, phase 0 = D1 > 0)."""
    import openimpala as oi

    vf = oi.volume_fraction(data, phase=1)
    result = oi.tortuosity(
        data,
        phase=1,
        direction=direction,
        phase_diffusivities={1: D0, 0: D1},
        solver="mlmg",
        max_grid_size=32,
    )
    return {
        "phi": vf.fraction,
        "tau": result.tortuosity,
        "d_eff": float(result.effective_diffusivity),
        "converged": result.solver_converged,
        "iterations": result.iterations,
        "residual": result.residual_norm,
    }


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate_result(
    d_eff: float,
    phi: float,
    D0: float,
    D1: float,
    label: str,
    *,
    isotropic: bool = True,
    exact_expected: float | None = None,
) -> list[str]:
    """Check that a computed D_eff respects all applicable physical bounds.

    Anisotropic laminates (parallel / series layers) are checked against the
    Wiener [Reuss, Voigt] bounds (and optional exact analytical equality),
    whereas statistically isotropic 3-D media are additionally checked against
    the tighter 3-D Hashin-Shtrikman [HS-, HS+] bounds.
    """
    violations: list[str] = []
    tol = 1e-4

    dv = float(wiener_upper(np.array([phi]), D0, D1)[0])
    dr = float(wiener_lower(np.array([phi]), D0, D1)[0])
    hs_plus = float(hashin_shtrikman_upper(np.array([phi]), D0, D1)[0])
    hs_minus = float(hashin_shtrikman_lower(np.array([phi]), D0, D1)[0])

    if d_eff > dv + tol:
        violations.append(
            f"  VIOLATION [{label}]: D_eff={d_eff:.6f} > Wiener upper={dv:.6f}"
        )
    if d_eff < dr - tol:
        violations.append(
            f"  VIOLATION [{label}]: D_eff={d_eff:.6f} < Wiener lower={dr:.6f}"
        )
    if isotropic:
        if d_eff > hs_plus + tol:
            violations.append(
                f"  VIOLATION [{label}]: D_eff={d_eff:.6f} > HS upper={hs_plus:.6f}"
            )
        if d_eff < hs_minus - tol:
            violations.append(
                f"  VIOLATION [{label}]: D_eff={d_eff:.6f} < HS lower={hs_minus:.6f}"
            )
    if exact_expected is not None and abs(d_eff - exact_expected) > 1e-4:
        violations.append(
            f"  VIOLATION [{label}]: D_eff={d_eff:.6f} != analytical={exact_expected:.6f}"
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
    """Generate a publication-ready validation plot if matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plot generation)")
        return

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    dv = wiener_upper(phi_range, D0, D1)
    dr = wiener_lower(phi_range, D0, D1)
    hs_plus = hashin_shtrikman_upper(phi_range, D0, D1)
    hs_minus = hashin_shtrikman_lower(phi_range, D0, D1)

    ax.fill_between(phi_range, dr, dv, alpha=0.10, color="#1f77b4",
                    label="Wiener bounds (Reuss-Voigt, 2-phase)")
    ax.fill_between(phi_range, hs_minus, hs_plus, alpha=0.20, color="#2ca02c",
                    label="Hashin-Shtrikman bounds (2-phase)")

    ax.plot(phi_range, dv, "-", color="#1f77b4", lw=1.5, label="Voigt (parallel)")
    ax.plot(phi_range, dr, "--", color="#1f77b4", lw=1.5, label="Reuss (series)")
    ax.plot(phi_range, hs_plus, "-", color="#2ca02c", lw=1.5, label="HS upper")
    ax.plot(phi_range, hs_minus, "--", color="#2ca02c", lw=1.5, label="HS lower")

    hs_plus_binary = hashin_shtrikman_upper(phi_range, 1.0, 0.0)
    ax.plot(phi_range, hs_plus_binary, ":", color="gray", lw=1.2,
            label="HS upper (binary, $D_1=0$)")

    markers = {
        "multiphase_parallel": ("^", "#1f77b4", 80, "2-phase parallel (Voigt)"),
        "multiphase_series": ("v", "#9467bd", 80, "2-phase series (Reuss)"),
        "multiphase_random": ("s", "#2ca02c", 65, "2-phase isotropic random"),
        "binary_parallel": ("D", "#d62728", 55, "Binary parallel ($D_1=0$)"),
        "binary_random": ("o", "#ff7f0e", 55, "Binary random ($D_1=0$)"),
    }
    for geom_type, (marker, color, size, lbl) in markers.items():
        pts = [d for d in solver_data if d["type"] == geom_type]
        if pts:
            phis = [p["phi"] for p in pts]
            deffs = [p["d_eff"] for p in pts]
            ax.scatter(phis, deffs, marker=marker, s=size, color=color,
                       edgecolor="black", linewidth=0.5, zorder=5, label=lbl)

    ax.set_xlabel(r"Volume Fraction $\phi$ (conducting phase 1)", fontsize=13)
    ax.set_ylabel(r"Effective Diffusivity $D_\mathrm{eff} / D_0$", fontsize=13)
    ax.set_title(
        f"Verification: OpenImpala vs. Analytical Wiener & Hashin-Shtrikman Bounds\n"
        f"(2-phase: $D_0 = {D0}$, $D_1 = {D1}$; binary: $D_0 = 1$, $D_1 = 0$)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
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
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation (useful in headless CI containers)",
    )
    args = parser.parse_args()

    D0 = 1.0  # high-diffusivity phase (phase 1)
    D1 = 0.1  # low-diffusivity phase (phase 0)

    phi_test = np.arange(0.2, 0.85, 0.2)
    phi_plot = np.linspace(0.001, 0.999, 500)
    N = args.grid_size

    print("=" * 70)
    print("OpenImpala Analytical Bounds V&V")
    print("=" * 70)
    print(f"  Two-phase bounds: D0={D0} (phase 1), D1={D1} (phase 0)")
    print(f"  Binary bounds:    D0=1.0 (phase 1), D1=0.0 (phase 0)")
    print(f"  Grid size:        {N}^3 = {N**3:,} voxels")
    print(f"  Volume fractions: {phi_test}")
    print()

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

    solver_data: list[dict] = []
    all_violations: list[str] = []

    try:
        import openimpala as oi
    except ImportError:
        print("ERROR: openimpala not installed.")
        return 1

    print("--- Running OpenImpala solver ---")
    t_total_start = time.time()

    with oi.Session():
        for phi_target in phi_test:
            print(f"\n  phi_target = {phi_target:.2f}:")
            data_par = make_parallel_layers(N, phi_target)
            data_ser = make_series_layers(N, phi_target)
            data_iso = make_resolved_isotropic_media(N, phi_target, seed=int(phi_target * 100) + 7)

            # 1. Binary parallel layers -> exact Voigt bound D_eff == phi
            res_bp = compute_deff_binary(data_par, direction="z")
            if res_bp is not None:
                solver_data.append({**res_bp, "type": "binary_parallel"})
                exact_dv = float(wiener_upper(np.array([res_bp["phi"]]), 1.0, 0.0)[0])
                all_violations.extend(
                    validate_result(
                        res_bp["d_eff"], res_bp["phi"], 1.0, 0.0,
                        f"binary parallel phi={res_bp['phi']:.3f}",
                        isotropic=False, exact_expected=exact_dv,
                    )
                )
                print(f"    Binary parallel:   phi={res_bp['phi']:.4f}  D_eff={res_bp['d_eff']:.4f} (Voigt={exact_dv:.4f})")

            # 2. Binary isotropic medium -> 0 <= trace(D_eff)/3 <= HS+(1, 0)
            iso_bin_dirs = [
                r["d_eff"]
                for d_ax in ("x", "y", "z")
                if (r := compute_deff_binary(data_iso, direction=d_ax)) is not None
            ]
            if len(iso_bin_dirs) == 3:
                phi_iso = float(oi.volume_fraction(data_iso, phase=1).fraction)
                d_iso_bin = float(np.mean(iso_bin_dirs))
                solver_data.append({"phi": phi_iso, "d_eff": d_iso_bin, "type": "binary_random"})
                all_violations.extend(
                    validate_result(
                        d_iso_bin, phi_iso, 1.0, 0.0,
                        f"binary isotropic phi={phi_iso:.3f}",
                        isotropic=True,
                    )
                )
                hs_b = float(hashin_shtrikman_upper(np.array([phi_iso]), 1.0, 0.0)[0])
                print(f"    Binary isotropic:  phi={phi_iso:.4f}  D_eff={d_iso_bin:.4f} <= HS+={hs_b:.4f}")

            # 3. Two-phase parallel layers -> exact Voigt bound
            res_mp = compute_deff_multiphase(data_par, D0, D1, direction="z")
            solver_data.append({**res_mp, "type": "multiphase_parallel"})
            exact_mp_dv = float(wiener_upper(np.array([res_mp["phi"]]), D0, D1)[0])
            all_violations.extend(
                validate_result(
                    res_mp["d_eff"], res_mp["phi"], D0, D1,
                    f"2-phase parallel phi={res_mp['phi']:.3f}",
                    isotropic=False, exact_expected=exact_mp_dv,
                )
            )
            print(f"    2-phase parallel:  phi={res_mp['phi']:.4f}  D_eff={res_mp['d_eff']:.4f} (Voigt={exact_mp_dv:.4f})")

            # 4. Two-phase series layers -> exact Reuss bound
            res_ms = compute_deff_multiphase(data_ser, D0, D1, direction="z")
            solver_data.append({**res_ms, "type": "multiphase_series"})
            exact_ms_dr = float(wiener_lower(np.array([res_ms["phi"]]), D0, D1)[0])
            all_violations.extend(
                validate_result(
                    res_ms["d_eff"], res_ms["phi"], D0, D1,
                    f"2-phase series phi={res_ms['phi']:.3f}",
                    isotropic=False, exact_expected=exact_ms_dr,
                )
            )
            print(f"    2-phase series:    phi={res_ms['phi']:.4f}  D_eff={res_ms['d_eff']:.4f} (Reuss={exact_ms_dr:.4f})")

            # 5. Two-phase isotropic medium -> HS- <= trace(D_eff)/3 <= HS+
            iso_mp_dirs = [
                compute_deff_multiphase(data_iso, D0, D1, direction=d_ax)
                for d_ax in ("x", "y", "z")
            ]
            phi_mp_iso = iso_mp_dirs[0]["phi"]
            d_mp_iso = float(np.mean([r["d_eff"] for r in iso_mp_dirs]))
            solver_data.append({"phi": phi_mp_iso, "d_eff": d_mp_iso, "type": "multiphase_random"})
            all_violations.extend(
                validate_result(
                    d_mp_iso, phi_mp_iso, D0, D1,
                    f"2-phase isotropic phi={phi_mp_iso:.3f}",
                    isotropic=True,
                )
            )
            hs_lo = float(hashin_shtrikman_lower(np.array([phi_mp_iso]), D0, D1)[0])
            hs_hi = float(hashin_shtrikman_upper(np.array([phi_mp_iso]), D0, D1)[0])
            print(f"    2-phase isotropic: phi={phi_mp_iso:.4f}  HS-={hs_lo:.4f} <= D_eff={d_mp_iso:.4f} <= HS+={hs_hi:.4f}")

    t_total = time.time() - t_total_start
    print(f"\n  Total solver time: {t_total:.2f}s")

    if not args.no_plot:
        print()
        generate_plot(phi_plot, D0, D1, solver_data, args.output)

    print()
    print("=" * 70)
    if all_violations:
        print("VALIDATION FAILED — bound violations detected:")
        for v in all_violations:
            print(v)
        print("=" * 70)
        return 1

    n_points = len(solver_data)
    print(f"VALIDATION PASSED — {n_points} data points, all within analytical bounds.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
