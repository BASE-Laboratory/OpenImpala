# Linear Solvers

OpenImpala provides two compiled C++ multigrid solver families—matrix-free **AMReX MLMG** and **HYPRE** structured Krylov/multigrid solvers—plus a pure-Python **SciPy / CuPy** Jacobi-preconditioned Conjugate Gradient fallback for platforms without compiled C++ wheels.

## AMReX MLMG (`solver="auto"` or `"mlmg"`)

By default, `oi.tortuosity(..., solver="auto")` selects AMReX's matrix-free geometric multigrid solver (`MLABecLaplacian` + V-cycle multigrid):

- **Matrix-free**: evaluates the 7-point harmonic-face stencil on the fly without allocating a 7-band sparse matrix.
- **~3x lower memory footprint**: requires roughly 24–30 bytes per voxel compared to ~70 bytes per voxel for HYPRE `StructMatrix`, enabling larger single-GPU and single-node volumes.
- **CPU + CUDA native**: executes identical AMReX `ParallelFor` kernels across OpenMP threads and NVIDIA GPUs.

```python
import openimpala as oi

vol = oi.datasets.two_phase_cubic()
res = oi.tortuosity(vol, phase=1, direction="z", solver="mlmg", mlmg_eps=1e-9)
```

## HYPRE Structured Solvers

For extreme diffusivity contrasts ($\kappa \ll 10^{-2}$ in multi-phase media) or periodic homogenisation cell problems (`oi.effective_diffusivity()`), OpenImpala dispatches to [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)'s structured-grid Krylov solvers preconditioned by Semicoarsening Multigrid (`SMG`) or Pointwise Semicoarsening Multigrid (`PFMG`):

| Python `solver=` | Algorithm | Preconditioner | Recommended Use Case |
|------------------|-----------|----------------|----------------------|
| `"mlmg"` / `"auto"` | AMReX geometric multigrid | Native V-cycle | Default fast path for `tortuosity()` (lowest memory) |
| `"pcg"` | Conjugate Gradient | `"smg"` or `"pfmg"` | SPD flow-through diffusion on large MPI decompositions |
| `"flexgmres"` | Flexible GMRES | `"smg"` or `"pfmg"` | Default for `effective_diffusivity()` and high-contrast multi-phase |
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

`oi.effective_diffusivity()` solves three periodic cell problems using HYPRE (`solver="flexgmres"` by default with `PFMG` preconditioning on the periodic structured grid):

```python
deff = oi.effective_diffusivity(vol, phase=1, solver="flexgmres")
```

## Convergence Tolerances in C++ Input Files

When running the standalone C++ `Diffusion` executable or tuning HYPRE via `ParmParse`:

```ini
hypre.eps     = 1.0e-9    # Relative residual convergence tolerance
hypre.maxiter = 500       # Maximum Krylov iterations
```
