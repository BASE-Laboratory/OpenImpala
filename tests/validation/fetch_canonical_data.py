#!/usr/bin/env python3
"""Canonical Sample Dataset Fetcher and Solver Cross-Verification for OpenImpala.

This script fetches ``SampleData_2Phase_stack_3d_1bit.tif`` (the 100^3 synthetic
two-phase PoreSpy blob microstructure distributed with OpenImpala), writes its
independently verified discrete finite-volume reference values to a JSON sidecar,
runs the OpenImpala solver across all three Cartesian directions (X, Y, Z), and
verifies volume fraction, active percolating fraction, directional tortuosity,
effective diffusivity, boundary flux conservation, and Hashin-Shtrikman bounds.

Note: ``SampleData_2Phase_stack_3d_1bit.tif`` is a synthetic PoreSpy benchmark
volume (not an experimental scan); see ``berea_sandstone_vv.py`` for validation
against published laboratory micro-CT measurements.

Usage
-----
    python fetch_canonical_data.py [--data-dir tests/validation/data]

Requires: openimpala, numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_URL = (
    "https://raw.githubusercontent.com/BASE-Laboratory/OpenImpala/"
    "master/data/SampleData_2Phase_stack_3d_1bit.tif"
)
DATASET_FILENAME = "SampleData_2Phase_stack_3d_1bit.tif"

# Independently verified discrete 7-point exterior-face Dirichlet finite-volume
# reference solution (CG residual < 1e-10) on the 100x100x100 binary PoreSpy
# blobs sample (phase 1 = minority phase).
CANONICAL_REFERENCE = {
    "dataset": "SampleData_2Phase_stack_3d_1bit.tif",
    "dataset_type": "synthetic_benchmark",
    "description": (
        "100x100x100 binary two-phase synthetic porous microstructure generated "
        "with porespy.generators.blobs (target porosity 0.40, blobiness 1.2). "
        "Reference transport values are from an independent sparse finite-volume "
        "solve of the 7-point exterior-face Dirichlet Laplacian (rtol=1e-10)."
    ),
    "phase_id": 1,
    "volume_fraction": 0.398309,
    "active_volume_fraction": 0.397743,
    "directional_reference": {
        "x": {"tortuosity": 3.151496, "effective_diffusivity": 0.126208},
        "y": {"tortuosity": 3.094289, "effective_diffusivity": 0.128541},
        "z": {"tortuosity": 3.161670, "effective_diffusivity": 0.125802},
    },
    "tolerance_fraction": 1.0e-3,
    "flux_balance_tolerance": 1.0e-4,
    "references": [
        {
            "citation": "J. Le Houx & D. Kramer, SoftwareX 15, 100729 (2021)",
            "doi": "10.1016/j.softx.2021.100729",
        },
        {
            "citation": "J. T. Gostick et al., J. Open Source Software 4(37), 1296 (2019)",
            "doi": "10.21105/joss.01296",
        },
    ],
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(data_dir: str) -> str:
    """Download the canonical TIFF dataset or copy from local repo ``data/``."""
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, DATASET_FILENAME)

    if os.path.exists(local_path):
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  Dataset already exists: {local_path} ({size_kb:.1f} KB)")
        return local_path

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    fallback_path = os.path.join(repo_root, "data", DATASET_FILENAME)
    if os.path.exists(fallback_path):
        import shutil

        shutil.copy2(fallback_path, local_path)
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  Copied from local repo: {fallback_path} ({size_kb:.1f} KB)")
        return local_path

    print(f"  Downloading: {DATASET_URL}")
    print(f"  Destination: {local_path}")
    try:
        urllib.request.urlretrieve(DATASET_URL, local_path)
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  Downloaded successfully ({size_kb:.1f} KB)")
    except urllib.error.URLError as e:
        print("  ERROR: Download failed and no local fallback found.")
        print(f"  URL error: {e}")
        raise

    return local_path


def write_reference_json(data_dir: str) -> str:
    """Write the canonical reference JSON file."""
    json_path = os.path.join(data_dir, "canonical_reference.json")
    with open(json_path, "w") as f:
        json.dump(CANONICAL_REFERENCE, f, indent=2)
        f.write("\n")
    print(f"  Reference JSON written: {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def validate_against_reference(
    tiff_path: str,
    reference: dict,
) -> tuple[bool, dict]:
    """Load the TIFF via OpenImpala, solve X/Y/Z, and verify against reference."""
    import openimpala as oi

    tol = float(reference["tolerance_fraction"])
    flux_tol = float(reference["flux_balance_tolerance"])
    phase_id = int(reference["phase_id"])
    exp_vf = float(reference["volume_fraction"])
    exp_active_vf = float(reference["active_volume_fraction"])
    hs_upper = 2.0 * exp_vf / (3.0 - exp_vf)

    results: dict = {
        "tiff_path": tiff_path,
        "reference": reference,
        "directions": {},
        "violations": [],
    }

    with oi.Session():
        print(f"\n  Loading TIFF: {tiff_path}")
        reader, img = oi.read_image(tiff_path, threshold=0.5)
        print(f"  VoxelImage loaded: {img}")

        vf = oi.volume_fraction(img, phase=phase_id)
        results["computed_volume_fraction"] = vf.fraction
        vf_err = abs(vf.fraction - exp_vf) / exp_vf
        results["volume_fraction_rel_error"] = vf_err
        print(
            f"  Volume fraction:       {vf.fraction:.6f} "
            f"(ref={exp_vf:.6f}, rel_err={vf_err:.2e})"
        )
        if vf_err > 1e-6:
            msg = (
                f"Volume fraction mismatch: computed={vf.fraction:.6f}, "
                f"expected={exp_vf:.6f}"
            )
            results["violations"].append(msg)

        for direction, dir_ref in reference["directional_reference"].items():
            exp_tau = float(dir_ref["tortuosity"])
            exp_deff = float(dir_ref["effective_diffusivity"])

            perc = oi.percolation_check(img, phase=phase_id, direction=direction)
            if not perc.percolates:
                msg = f"Phase {phase_id} failed to percolate in {direction.upper()}"
                results["violations"].append(msg)
                continue

            t0 = time.time()
            tau_res = oi.tortuosity(
                img, phase=phase_id, direction=direction, solver="auto"
            )
            dt = time.time() - t0

            comp_tau = tau_res.tortuosity
            comp_act_vf = tau_res.active_volume_fraction
            comp_deff = comp_act_vf / comp_tau
            tau_err = abs(comp_tau - exp_tau) / exp_tau
            act_vf_err = abs(comp_act_vf - exp_active_vf) / exp_active_vf
            flux_avg = 0.5 * (abs(tau_res.flux_in) + abs(tau_res.flux_out))
            flux_diff = (
                abs(abs(tau_res.flux_in) - abs(tau_res.flux_out)) / flux_avg
                if flux_avg > 0
                else 0.0
            )

            results["directions"][direction] = {
                "tortuosity": comp_tau,
                "effective_diffusivity": comp_deff,
                "active_volume_fraction": comp_act_vf,
                "tortuosity_rel_error": tau_err,
                "flux_rel_imbalance": flux_diff,
                "iterations": tau_res.iterations,
                "residual_norm": tau_res.residual_norm,
                "solve_time_s": dt,
            }

            print(
                f"  [{direction.upper()}] tau={comp_tau:.6f} (ref={exp_tau:.6f}, "
                f"err={tau_err:.3%}), D_eff={comp_deff:.6f}, "
                f"iters={tau_res.iterations}, time={dt:.2f}s"
            )

            if act_vf_err > 1e-6:
                results["violations"].append(
                    f"[{direction.upper()}] active_vf={comp_act_vf:.6f} != {exp_active_vf:.6f}"
                )
            if tau_err > tol:
                results["violations"].append(
                    f"[{direction.upper()}] tortuosity={comp_tau:.6f} vs ref={exp_tau:.6f} "
                    f"(rel_err={tau_err:.3%} > {tol:.3%})"
                )
            if flux_diff > flux_tol:
                results["violations"].append(
                    f"[{direction.upper()}] boundary flux imbalance {flux_diff:.2e} > {flux_tol:.2e}"
                )
            if comp_deff > hs_upper * (1.0 + 1e-6):
                results["violations"].append(
                    f"[{direction.upper()}] D_eff={comp_deff:.6f} exceeds HS+ bound {hs_upper:.6f}"
                )

        del reader, img

    passed = len(results["violations"]) == 0
    results["passed"] = passed
    return passed, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify OpenImpala against canonical synthetic benchmark values."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Directory for dataset and JSON sidecars (default: tests/validation/data/)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("OpenImpala Canonical Sample Dataset Verification")
    print("=" * 70)

    print("\n--- Step 1: Locate or fetch canonical sample TIFF ---")
    try:
        tiff_path = download_dataset(args.data_dir)
    except Exception as e:
        print(f"FATAL: Could not obtain dataset: {e}")
        return 1

    print("\n--- Step 2: Write reference JSON ---")
    json_path = write_reference_json(args.data_dir)

    print("\n--- Step 3: Verify X/Y/Z solves against independent FV reference ---")
    with open(json_path) as f:
        reference = json.load(f)

    try:
        import openimpala  # noqa: F401
    except ImportError:
        print("ERROR: openimpala not installed. Dataset fetched but cannot validate.")
        return 1

    passed, results = validate_against_reference(tiff_path, reference)

    results_path = os.path.join(args.data_dir, "validation_results.json")
    serialisable = {k: v for k, v in results.items() if k != "reference"}
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
        f.write("\n")
    print(f"\n  Results written to: {results_path}")

    print()
    print("=" * 70)
    if passed:
        print("VERIFICATION PASSED — all X/Y/Z values within 0.1% of FV reference.")
    else:
        print("VERIFICATION FAILED — discrepancies detected:")
        for v in results["violations"]:
            print(f"  {v}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
