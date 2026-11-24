#!/usr/bin/env python3
"""Analytical & cross-backend wheel verification suite for OpenImpala.

Verifies installed wheels (`pure-python`, `cpp-cpu`, and `cpp-cuda`) against
closed-form analytical solutions (uniform domain, parallel/perpendicular
channels, Voigt/Reuss two-phase laminates, anisotropic voxels), fundamental
identities (`tau == phi / D_eff`), optional C++ vs pure-Python parity on random
3-D media (`--compare-pure`), and clean zero-clutter stdout on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

import openimpala as oi


def _run_analytical_suite(grid_n: int = 32) -> dict:
    """Run all analytical checks on the active OpenImpala backend and return metrics."""
    # 1. Uniform domain: tau == 1.0
    uniform = np.ones((16, 16, 16), dtype=np.int32)
    tau_uni = oi.tortuosity(uniform, phase=1, direction="x")
    assert abs(tau_uni.tortuosity - 1.0) < 1e-6, f"Uniform tau={tau_uni.tortuosity}"

    # 2. Straight channels along 'x' (layers stacked along y): tau == 1.0
    channels = np.zeros((16, 16, 16), dtype=np.int32)
    channels[:, ::2, :] = 1
    tau_par = oi.tortuosity(channels, phase=1, direction="x")
    assert abs(tau_par.tortuosity - 1.0) < 1e-6, f"Parallel channel tau={tau_par.tortuosity}"

    # 3. Straight channels across 'y': non-percolating -> PercolationError
    raised_perc = False
    try:
        oi.tortuosity(channels, phase=1, direction="y")
    except oi.PercolationError:
        raised_perc = True
    assert raised_perc, "Expected PercolationError across non-percolating layers"

    # 4 & 5. Two-phase laminate (phi_0 = 0.25 with D_0 = 1.0; phi_1 = 0.75 with D_1 = 0.1)
    # Layers stacked along y (axis 1): flow along 'x' is parallel (Voigt = 0.325),
    # flow along 'y' is series (Reuss = 1 / (0.25/1.0 + 0.75/0.1) = 1 / 7.75).
    lam = np.ones((16, 16, 16), dtype=np.int32)
    lam[:, :4, :] = 0
    pmap = {0: 1.0, 1: 0.1}
    d_voigt_expected = 0.25 * 1.0 + 0.75 * 0.1  # 0.325
    d_reuss_expected = 1.0 / (0.25 / 1.0 + 0.75 / 0.1)  # 0.12903225806451613

    res_voigt = oi.tortuosity(lam, direction="x", phase_diffusivities=pmap)
    res_reuss = oi.tortuosity(lam, direction="y", phase_diffusivities=pmap)
    assert res_voigt.effective_diffusivity is not None
    assert res_reuss.effective_diffusivity is not None
    assert abs(res_voigt.effective_diffusivity - d_voigt_expected) < 1e-5, (
        f"Voigt D_eff={res_voigt.effective_diffusivity} != {d_voigt_expected}"
    )
    assert abs(res_reuss.effective_diffusivity - d_reuss_expected) < 1e-4, (
        f"Reuss D_eff={res_reuss.effective_diffusivity} != {d_reuss_expected}"
    )

    # 6. Anisotropic voxels (dz, dy, dx) = (1, 1, 3), straight channels along x and z
    tau_aniso_x = oi.tortuosity(channels, phase=1, direction="x", voxel_size=(1.0, 1.0, 3.0))
    tau_aniso_z = oi.tortuosity(channels, phase=1, direction="z", voxel_size=(1.0, 1.0, 3.0))
    assert abs(tau_aniso_x.tortuosity - 1.0) < 1e-6, f"Aniso x tau={tau_aniso_x.tortuosity}"
    assert abs(tau_aniso_z.tortuosity - 1.0) < 1e-6, f"Aniso z tau={tau_aniso_z.tortuosity}"

    # 7 & 8. Random 3-D porous microstructure: tau = phi / D_eff & full 3x3 tensor
    blobs = oi.synthetic.gaussian_blobs(
        grid_n, porosity=0.52, blobiness=1.2, pore_value=0, solid_value=1, seed=42
    )
    tau_blobs = oi.tortuosity(blobs, phase=0, direction="x")
    assert tau_blobs.effective_diffusivity is not None
    assert (
        abs(tau_blobs.tortuosity * tau_blobs.effective_diffusivity - tau_blobs.active_volume_fraction)
        < 1e-10
    ), (
        f"Identity tau * D_eff == active_vf violated: "
        f"{tau_blobs.tortuosity} * {tau_blobs.effective_diffusivity} != {tau_blobs.active_volume_fraction}"
    )

    info = oi.build_info()
    deff_blobs = oi.effective_diffusivity(blobs, phase=0)
    assert deff_blobs.tensor.shape == (3, 3)
    assert np.allclose(deff_blobs.tensor, deff_blobs.tensor.T, atol=1e-5)
    tensor_list = deff_blobs.tensor.tolist()

    return {
        "backend": info["backend"],
        "mpi_ranks": info["mpi_ranks"],
        "uniform_tau": float(tau_uni.tortuosity),
        "channel_tau": float(tau_par.tortuosity),
        "voigt_deff": float(res_voigt.effective_diffusivity),
        "reuss_deff": float(res_reuss.effective_diffusivity),
        "aniso_tau": float(tau_aniso_x.tortuosity),
        "random_tau": float(tau_blobs.tortuosity),
        "random_deff_dir": float(tau_blobs.effective_diffusivity),
        "random_phi": float(tau_blobs.active_volume_fraction),
        "random_tensor": tensor_list,
    }


def _check_ux_contracts() -> None:
    """Verify v5.0 UX contracts: explicit phase requirement & downsample truncation warning."""
    arr = np.zeros((5, 6, 7), dtype=np.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = oi.downsample_volume(arr, factor=2)
    assert ds.shape == (2, 3, 3), f"Unexpected downsampled shape {ds.shape}"
    assert any("truncating trailing slices" in str(w.message) for w in caught), (
        f"Expected truncation UserWarning, got {caught}"
    )

    # Required phase argument on single-phase calls
    cube = np.ones((8, 8, 8), dtype=np.int32)
    for fn_name in ("volume_fraction", "tortuosity", "effective_diffusivity"):
        fn = getattr(oi, fn_name)
        raised = False
        try:
            fn(cube)
        except TypeError:
            raised = True
        assert raised, f"{fn_name}(cube) should require explicit phase argument"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-backend", default=None)
    parser.add_argument("--expected-version", default=None)
    parser.add_argument("--expected-ranks", type=int, default=None)
    parser.add_argument("--grid-n", type=int, default=32)
    parser.add_argument("--compare-pure", action="store_true")
    parser.add_argument("--check-silent-exit", action="store_true")
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args()

    if args.expected_version is not None:
        assert oi.__version__ == args.expected_version, (
            f"Version mismatch: {oi.__version__} != {args.expected_version}"
        )

    _check_ux_contracts()

    with oi.Session():
        info = oi.build_info()
        if args.expected_backend is not None:
            assert info["backend"] == args.expected_backend, (
                f"Backend mismatch: {info['backend']} != {args.expected_backend} ({info})"
            )
        if args.expected_ranks is not None:
            assert info["mpi_ranks"] == args.expected_ranks, (
                f"MPI ranks mismatch: {info['mpi_ranks']} != {args.expected_ranks}"
            )
        metrics = _run_analytical_suite(grid_n=args.grid_n)

    if args.json_out:
        if info["mpi_rank"] == 0:
            print("VV_JSON:" + json.dumps(metrics))
        return 0

    if args.compare_pure and info["mpi_rank"] == 0:
        env = os.environ.copy()
        env["OPENIMPALA_BACKEND"] = "pure"
        out = subprocess.check_output(
            [
                sys.executable,
                __file__,
                "--expected-backend",
                "pure-python",
                "--grid-n",
                str(args.grid_n),
                "--json-out",
            ],
            env=env,
            text=True,
        )
        pure_line = next(l for l in out.splitlines() if l.startswith("VV_JSON:"))
        pure = json.loads(pure_line[len("VV_JSON:") :])
        tau_diff = abs(metrics["random_tau"] - pure["random_tau"])
        assert tau_diff < 1e-4, (
            f"C++ vs pure-Python random tau mismatch: {metrics['random_tau']} vs {pure['random_tau']}"
        )
        if metrics["random_tensor"] is not None and pure["random_tensor"] is not None:
            t_cpp = np.asarray(metrics["random_tensor"])
            t_pure = np.asarray(pure["random_tensor"])
            assert np.allclose(t_cpp, t_pure, atol=1e-4), (
                f"C++ vs pure-Python 3x3 tensor mismatch:\n{t_cpp}\nvs\n{t_pure}"
            )

    if args.check_silent_exit and info["mpi_rank"] == 0:
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import numpy as np, openimpala as oi\n"
                "vol = np.ones((8, 8, 8), dtype=np.int32)\n"
                "with oi.Session():\n"
                "    r1 = oi.tortuosity(vol, phase=1, direction='x')\n"
                "with oi.Session():\n"
                "    r2 = oi.tortuosity(vol, phase=1, direction='z')\n"
                "vf = oi.volume_fraction(vol, phase=1)\n"
                "print('OK', round(r1.tortuosity, 4), round(r2.tortuosity, 4), round(vf.fraction, 4))\n",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        for noisy in ("TinyProfiler", "Initializing AMReX", "Memory Usage", "AMReX (", "MPI_FINALIZE"):
            assert noisy not in out, f"Unexpected AMReX/MPI noise in default stdout:\n{out}"
        assert out.strip() == "OK 1.0 1.0 1.0", (
            f"Expected single line 'OK 1.0 1.0 1.0', got:\n{out!r}"
        )

    if info["mpi_rank"] == 0:
        print(
            f"[PASS] wheel_analytical_vv ({metrics['backend']}, ranks={metrics['mpi_ranks']}): "
            f"uniform={metrics['uniform_tau']:.6f}, Voigt={metrics['voigt_deff']:.6f}, "
            f"Reuss={metrics['reuss_deff']:.6f}, random_tau={metrics['random_tau']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
