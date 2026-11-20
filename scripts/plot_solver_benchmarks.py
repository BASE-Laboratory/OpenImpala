#!/usr/bin/env python3
"""Generate publication-quality solver scaling and memory benchmark figure (`scaling_and_memory.png`).

Plots empirically measured wall-clock solve times and peak solver RAM (`VmHWM` delta
after `/proc/self/clear_refs = 5`) across 3-D Gaussian random-field porous media
(porosity = 0.50, seed = 42) from 32^3 (3.28e4 voxels) to 256^3 (1.68e7 voxels).
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Exact empirical measurements on compiled OpenImpala C++ (AMReX 25.03 + HYPRE 2.31,
# 2 OpenMP threads, Linux x86_64) and pure-Python SciPy CG fallback.
DATA = {
    "AMReX MLMG (matrix-free)": {
        "n": [32, 64, 96, 128, 160, 192, 256],
        "time_s": [0.0563, 0.3250, 0.9811, 4.0580, 8.2743, 17.0448, 33.1035],
        "mem_mb": [11.37, 75.25, 246.37, 638.05, 1232.88, 2333.61, 4855.70],
        "color": "#0072B2",
        "marker": "o",
        "ls": "-",
        "lw": 2.3,
    },
    "HYPRE PCG + PFMG": {
        "n": [32, 64, 96, 128, 160],
        "time_s": [0.1771, 1.7617, 7.5199, 21.9067, 51.9297],
        "mem_mb": [14.54, 95.02, 302.39, 710.29, 1463.56],
        "color": "#009E73",
        "marker": "s",
        "ls": "-",
        "lw": 1.9,
    },
    "HYPRE FlexGMRES + PFMG": {
        "n": [64, 96, 128],
        "time_s": [2.5739, 13.4988, 33.3379],
        "mem_mb": [268.34, 865.91, 2031.01],
        "color": "#E69F00",
        "marker": "^",
        "ls": "--",
        "lw": 1.9,
    },
    "HYPRE FlexGMRES + SMG": {
        "n": [64, 96, 128],
        "time_s": [5.1026, 24.3740, 66.8705],
        "mem_mb": [346.93, 1193.31, 2779.49],
        "color": "#D55E00",
        "marker": "D",
        "ls": "--",
        "lw": 1.9,
    },
    "Pure-Python (SciPy CG)": {
        "n": [32, 64, 96, 128],
        "time_s": [0.1644, 1.8881, 10.0217, 34.8898],
        "mem_mb": [10.18, 81.95, 279.00, 699.38],
        "color": "#7F7F7F",
        "marker": "x",
        "ls": ":",
        "lw": 1.8,
    },
}


def main() -> int:
    out_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "scaling_and_memory.png"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
        }
    )

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(10.6, 4.35), dpi=300)

    # --- Panel (a): Wall-clock solve time vs total voxels ---
    for label, d in DATA.items():
        vox = np.asarray(d["n"], dtype=np.float64) ** 3
        ax_time.loglog(
            vox,
            d["time_s"],
            label=label,
            color=d["color"],
            marker=d["marker"],
            linestyle=d["ls"],
            linewidth=d["lw"],
            markersize=5.8,
        )

    # Linear O(N) reference slope
    ref_vox = np.array([2.6e5, 1.68e7])
    ref_time = 0.22 * (ref_vox / 2.6e5) ** 1.0
    ax_time.loglog(ref_vox, ref_time, color="#222222", linestyle="-.", linewidth=1.1, alpha=0.65)
    ax_time.text(
        2.7e6,
        1.45,
        r"Ideal $\mathcal{O}(N_{\mathrm{vox}})$",
        fontsize=8.3,
        color="#222222",
        rotation=29,
    )

    ax_time.annotate(
        "8.6× faster than SciPy\n16.5× faster than SMG",
        xy=(128**3, 4.058),
        xytext=(5.2e5, 0.085),
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.0},
        fontsize=8.0,
        color="#005587",
        fontweight="bold",
    )

    ax_time.set_title("(a) Algorithmic Solve-Time Scaling", fontweight="bold", pad=8)
    ax_time.set_xlabel(r"Total Grid Size $N_{\mathrm{vox}} = N^3$ (voxels)")
    ax_time.set_ylabel("Wall-Clock Solve Time (s)")
    ax_time.grid(True, which="major", linestyle="-", alpha=0.28)
    ax_time.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax_time.legend(loc="upper left", frameon=True, framealpha=0.92)

    # Secondary top tick labels showing N^3
    ax_top_t = ax_time.twiny()
    ax_top_t.set_xscale("log")
    ax_top_t.set_xlim(ax_time.get_xlim())
    cube_ns = [32, 64, 128, 256]
    ax_top_t.set_xticks([n**3 for n in cube_ns])
    ax_top_t.set_xticklabels([rf"${n}^3$" for n in cube_ns], fontsize=8.3)
    ax_top_t.minorticks_off()

    # --- Panel (b): Peak solver RAM vs total voxels ---
    for label, d in DATA.items():
        if label == "Pure-Python (SciPy CG)":
            continue
        vox = np.asarray(d["n"], dtype=np.float64) ** 3
        ax_mem.loglog(
            vox,
            d["mem_mb"],
            label=label,
            color=d["color"],
            marker=d["marker"],
            linestyle=d["ls"],
            linewidth=d["lw"],
            markersize=5.8,
        )

    ax_mem.annotate(
        "3.2×–4.4× lower RAM\nat $128^3$ (638 MB vs 2.0–2.8 GB)",
        xy=(128**3, 638.05),
        xytext=(3.7e4, 920.0),
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.0},
        fontsize=8.0,
        color="#005587",
        fontweight="bold",
    )
    ax_mem.annotate(
        "$256^3$ (16.8M vox)\n4.86 GB (~304 B/vox)",
        xy=(256**3, 4855.7),
        xytext=(1.25e6, 5400.0),
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.0},
        fontsize=8.0,
        color="#005587",
    )

    ax_mem.set_title("(b) Measured Peak Solver Memory Footprint", fontweight="bold", pad=8)
    ax_mem.set_xlabel(r"Total Grid Size $N_{\mathrm{vox}} = N^3$ (voxels)")
    ax_mem.set_ylabel("Peak Solver Memory ΔVmHWM (MB)")
    ax_mem.set_ylim(8.0, 9500.0)
    ax_mem.grid(True, which="major", linestyle="-", alpha=0.28)
    ax_mem.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax_mem.legend(loc="lower right", frameon=True, framealpha=0.92)

    ax_top_m = ax_mem.twiny()
    ax_top_m.set_xscale("log")
    ax_top_m.set_xlim(ax_mem.get_xlim())
    ax_top_m.set_xticks([n**3 for n in cube_ns])
    ax_top_m.set_xticklabels([rf"${n}^3$" for n in cube_ns], fontsize=8.3)
    ax_top_m.minorticks_off()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
