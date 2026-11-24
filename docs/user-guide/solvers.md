# Linear Solvers

OpenImpala provides two compiled C++ multigrid solver families—matrix-free **AMReX MLMG** and **HYPRE** structured Krylov/multigrid solvers—plus a pure-Python **SciPy / CuPy** Jacobi-preconditioned Conjugate Gradient fallback for platforms without compiled C++ wheels.

## AMReX MLMG (`solver="auto"` or `"mlmg"`)

By default, `oi.tortuosity(..., solver="auto")`, `oi.effective_diffusivity(..., solver="auto")`, and `oi.rev_study(..., solver="auto")` select AMReX's matrix-free geometric multigrid solver (`MLABecLaplacian` + `GMRESMLMG` / V-cycle multigrid):

- **Matrix-free**: evaluates the 7-point harmonic-face stencil on the fly without assembling an explicit 7-band sparse matrix or storing large HYPRE structured hierarchies.
- **3.2×–4.4× lower memory footprint than FlexGMRES+PFMG/SMG**: requires roughly **~304–319 bytes per voxel** across the full multigrid hierarchy (including 3-axis face conductances, ghost halos, and V-cycle levels), compared to **~1,016 bytes per voxel** for HYPRE `FlexGMRES + PFMG` and **~1,390 bytes per voxel** for `FlexGMRES + SMG`.
- **CPU + MPI + CUDA native**: executes identical AMReX `ParallelFor` kernels across OpenMP threads, MPI ranks, and NVIDIA GPUs—including full $3\times 3$ periodic homogenisation on `openimpala-cuda` with exact per-component SPD nullspace pinning.

```python
import openimpala as oi

vol = oi.datasets.two_phase_cubic()
res = oi.tortuosity(vol, phase=1, direction="z", solver="mlmg", mlmg_eps=1e-9)
deff = oi.effective_diffusivity(vol, phase=1, solver="mlmg")
```

### Measured Memory & Runtime Comparison

Measured on 3D Gaussian random porous media ($\phi = 0.50$, two `x86_64` OpenMP threads; all solvers agree to $|\Delta\tau| < 4\times 10^{-8}$):

| Solver Backend | $64^3$ Peak RAM | $128^3$ Time | $128^3$ Peak RAM | Footprint | $256^3$ ($16.8\text{M}$ vox) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **AMReX `mlmg` (default)** | **75 MB** | **4.06 s** | **638 MB** | **~304–319 B/vox** | **33.1 s / 4,856 MB** |
| HYPRE `pcg` + `pfmg` | 95 MB | 21.91 s | 710 MB | ~355–380 B/vox | — |
| HYPRE `flexgmres` + `pfmg` | 268 MB | 33.34 s | 2,031 MB | ~1,016–1,073 B/vox | — |
| HYPRE `flexgmres` + `smg` | 347 MB | 66.87 s | 2,779 MB | ~1,388–1,414 B/vox | — |
| Pure-Python (`SciPy CG`) | 82 MB | 34.89 s | 699 MB | ~328–350 B/vox | — |

## HYPRE Structured Solvers

For extreme diffusivity contrasts ($\kappa \ll 10^{-2}$ in multi-phase media) or explicit cross-verification on CPU/MPI clusters, OpenImpala also dispatches to [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)'s structured-grid Krylov solvers preconditioned by Semicoarsening Multigrid (`SMG`) or Pointwise Semicoarsening Multigrid (`PFMG`):

| Python `solver=` | Algorithm | Preconditioner | Recommended Use Case |
|------------------|-----------|----------------|----------------------|
| `"mlmg"` / `"auto"` | AMReX geometric multigrid | Native V-cycle (`GMRESMLMG`) | Default fast path for `tortuosity()`, `effective_diffusivity()`, and `rev_study()` (CPU & CUDA) |
| `"pcg"` | Conjugate Gradient | `"smg"` or `"pfmg"` | SPD diffusion / homogenisation on CPU MPI decompositions |
| `"flexgmres"` | Flexible GMRES | `"smg"` or `"pfmg"` | High-contrast multi-phase transport and HYPRE periodic reference |
| `"bicgstab"` | BiCGSTAB | `"smg"` or `"pfmg"` | Alternative Krylov solver when GMRES memory grows |
| `"gmres"` | GMRES | `"smg"` or `"pfmg"` | General structured linear systems |
| `"pfmg"` | Standalone PFMG | — | Fast pointwise multigrid on large regular grids |
| `"smg"` | Standalone SMG | — | Highly anisotropic voxel aspect ratios ($h_z \gg h_x$) |

Select the HYPRE preconditioner in `oi.tortuosity()` via `preconditioner="pfmg"` (lower memory, faster setup on GPUs) or `preconditioner="smg"` (stronger plane-smoothing on high-aspect-ratio grids):

```python
res = oi.tortuosity(
    vol,
    phase=1,
    direction="z",
    solver="pcg",
    preconditioner="pfmg",
)
```

## Periodic Cell-Problem Solver (`oi.effective_diffusivity`)

`oi.effective_diffusivity()` solves three periodic cell problems using matrix-free `EffectiveDiffusivityMLMG` (`solver="auto"` / `"mlmg"`, sharing a single connected-component nullspace pin across $X$, $Y$, and $Z$), or HYPRE structured solvers (`solver="flexgmres" | "pcg" | ...` on CPU):

```python
deff_mlmg = oi.effective_diffusivity(vol, phase=1)                      # default: matrix-free MLMG (CPU + CUDA)
deff_hypre = oi.effective_diffusivity(vol, phase=1, solver="flexgmres") # explicit HYPRE reference
```

## Convergence Tolerances in C++ Input Files

When running the standalone C++ `Diffusion` executable or tuning solvers via `ParmParse`:

```ini
mlmg.eps      = 1.0e-8    # Matrix-free AMReX MLMG relative residual tolerance
mlmg.maxiter  = 200       # Maximum outer iterations / V-cycles
hypre.eps     = 1.0e-9    # HYPRE relative residual convergence tolerance
hypre.maxiter = 500       # HYPRE maximum Krylov iterations
```
