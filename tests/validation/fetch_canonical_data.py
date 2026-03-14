#!/usr/bin/env python3
"""Canonical Experimental Dataset Fetcher and Validator for OpenImpala.

This script downloads a public 3D TIFF dataset from the OpenImpala repository,
stores reference "experimental" values in a JSON sidecar file, runs the
OpenImpala solver on the dataset, and validates the computed result against
the stored reference.

This establishes the pattern for adding real experimental datasets to the V&V
suite in the future: each dataset gets a TIFF (or HDF5) file plus a JSON file
recording the published experimental measurements and paper citation.

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
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_URL = (
    "https://raw.githubusercontent.com/BASE-Laboratory/OpenImpala/"
    "master/data/SampleData_2Phase_stack_3d_1bit.tif"
)
DATASET_FILENAME = "SampleData_2Phase_stack_3d_1bit.tif"

# Reference values for validation.
# In a real V&V suite, these would come from a published experimental paper.
# Here we use plausible values for this sample dataset as a demonstration.
CANONICAL_REFERENCE = {
    "dataset": "SampleData_2Phase",
    "description": (
        "Two-phase synthetic microstructure from the OpenImpala sample data. "
        "Reference values are nominal targets for CI validation."
    ),
    "experimental_porosity": 0.40,
    "experimental_tortuosity_z": 1.45,
    "tolerance_fraction": 0.05,
    "reference_paper": "Doe et al. 2024",
    "reference_doi": "10.xxxx/placeholder",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(data_dir: str) -> str:
    """Download the canonical TIFF dataset if not already present.

    Returns the local file path.
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, DATASET_FILENAME)

    if os.path.exists(local_path):
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  Dataset already exists: {local_path} ({size_kb:.1f} KB)")
        return local_path

    print(f"  Downloading: {DATASET_URL}")
    print(f"  Destination: {local_path}")

    try:
        urllib.request.urlretrieve(DATASET_URL, local_path)
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  Downloaded successfully ({size_kb:.1f} KB)")
    except urllib.error.URLError as e:
        # If download fails (no network), check if the file exists in the
        # repository's data/ directory and copy it from there.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        fallback_path = os.path.join(repo_root, "data", DATASET_FILENAME)
        if os.path.exists(fallback_path):
            import shutil
            shutil.copy2(fallback_path, local_path)
            size_kb = os.path.getsize(local_path) / 1024
            print(f"  Network unavailable; copied from repo: {fallback_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ERROR: Download failed and no local fallback found.")
            print(f"  URL error: {e}")
            raise

    return local_path


def write_reference_json(data_dir: str) -> str:
    """Write the canonical reference JSON file.

    Returns the local file path.
    """
    json_path = os.path.join(data_dir, "canonical_reference.json")
    with open(json_path, "w") as f:
        json.dump(CANONICAL_REFERENCE, f, indent=2)
        f.write("\n")
    print(f"  Reference JSON written: {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_against_reference(
    tiff_path: str,
    reference: dict,
) -> tuple[bool, dict]:
    """Load the TIFF via OpenImpala, run the solver, and validate.

    Returns (passed: bool, results: dict).
    """
    import openimpala as oi

    tol = reference["tolerance_fraction"]
    exp_porosity = reference["experimental_porosity"]
    exp_tau_z = reference["experimental_tortuosity_z"]

    results = {
        "tiff_path": tiff_path,
        "reference": reference,
        "violations": [],
    }

    with oi.Session():
        # --- Load the TIFF ---
        print(f"\n  Loading TIFF: {tiff_path}")
        reader, img = oi.read_image(tiff_path, threshold=0.5)
        print(f"  VoxelImage loaded: {img}")

        # --- Volume fraction ---
        vf = oi.volume_fraction(img, phase=1)
        results["computed_porosity"] = vf.fraction
        print(f"  Computed porosity:     {vf.fraction:.6f}")
        print(f"  Expected porosity:     {exp_porosity:.6f}")

        porosity_error = abs(vf.fraction - exp_porosity) / exp_porosity
        results["porosity_error_fraction"] = porosity_error
        if porosity_error > tol:
            msg = (
                f"Porosity mismatch: computed={vf.fraction:.6f}, "
                f"expected={exp_porosity:.6f}, "
                f"error={porosity_error:.2%} > tolerance={tol:.2%}"
            )
            results["violations"].append(msg)
            print(f"  WARNING: {msg}")
        else:
            print(f"  Porosity error:        {porosity_error:.2%} (within {tol:.0%} tolerance)")

        # --- Percolation check ---
        perc = oi.percolation_check(img, phase=1, direction="z")
        results["percolates_z"] = perc.percolates
        print(f"  Percolates in Z:       {perc.percolates}")

        if not perc.percolates:
            msg = "Phase 1 does not percolate in Z — cannot compute tortuosity."
            results["violations"].append(msg)
            print(f"  ERROR: {msg}")
            results["passed"] = False
            return False, results

        # --- Tortuosity ---
        print(f"  Solving tortuosity (Z-direction, FlexGMRES)...")
        t0 = time.time()
        tau_result = oi.tortuosity(img, phase=1, direction="z", solver="flexgmres")
        dt = time.time() - t0

        results["computed_tortuosity_z"] = tau_result.tortuosity
        results["solver_converged"] = tau_result.solver_converged
        results["solver_iterations"] = tau_result.iterations
        results["solver_residual"] = tau_result.residual_norm
        results["solve_time_s"] = dt

        print(f"  Computed tortuosity:    {tau_result.tortuosity:.6f}")
        print(f"  Expected tortuosity:    {exp_tau_z:.6f}")
        print(f"  Solver converged:      {tau_result.solver_converged} "
              f"({tau_result.iterations} iter, "
              f"residual={tau_result.residual_norm:.2e}, "
              f"time={dt:.2f}s)")

        tau_error = abs(tau_result.tortuosity - exp_tau_z) / exp_tau_z
        results["tortuosity_error_fraction"] = tau_error
        if tau_error > tol:
            msg = (
                f"Tortuosity mismatch: computed={tau_result.tortuosity:.6f}, "
                f"expected={exp_tau_z:.6f}, "
                f"error={tau_error:.2%} > tolerance={tol:.0%}"
            )
            results["violations"].append(msg)
            print(f"  WARNING: {msg}")
        else:
            print(f"  Tortuosity error:      {tau_error:.2%} (within {tol:.0%} tolerance)")

    passed = len(results["violations"]) == 0
    results["passed"] = passed
    return passed, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch canonical dataset and validate OpenImpala against reference values."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Directory for downloaded data (default: tests/validation/data/)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("OpenImpala Canonical Dataset V&V")
    print("=" * 70)

    # --- Step 1: Download dataset ---
    print("\n--- Step 1: Fetch canonical dataset ---")
    try:
        tiff_path = download_dataset(args.data_dir)
    except Exception as e:
        print(f"FATAL: Could not obtain dataset: {e}")
        return 1

    # --- Step 2: Write reference JSON ---
    print("\n--- Step 2: Write reference JSON ---")
    json_path = write_reference_json(args.data_dir)

    # --- Step 3: Load reference and validate ---
    print("\n--- Step 3: Validate against reference ---")
    with open(json_path) as f:
        reference = json.load(f)

    try:
        import openimpala  # noqa: F401
    except ImportError:
        print("ERROR: openimpala not installed. Dataset fetched but cannot validate.")
        return 1

    passed, results = validate_against_reference(tiff_path, reference)

    # --- Write results JSON ---
    results_path = os.path.join(args.data_dir, "validation_results.json")
    # Convert non-serialisable types
    serialisable = {k: v for k, v in results.items() if k != "reference"}
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
        f.write("\n")
    print(f"\n  Results written to: {results_path}")

    # --- Summary ---
    print()
    print("=" * 70)
    if passed:
        print("VALIDATION PASSED — computed values within tolerance of reference.")
    else:
        print("VALIDATION FAILED — discrepancies detected:")
        for v in results["violations"]:
            print(f"  {v}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
