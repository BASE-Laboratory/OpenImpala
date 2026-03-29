# Solvers

OpenImpala provides two solver backends for computing tortuosity, plus a
legacy solver retained for comparison.

## HYPRE solvers (default)

The primary backend uses [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)
structured-grid solvers. Available algorithms:

| Solver | Type | Best for | Python name |
|--------|------|----------|-------------|
| **PCG** | Krylov (CG) | Single-phase diffusion (SPD systems) | `"pcg"` or `"auto"` |
| FlexGMRES | Krylov | Multi-phase, non-symmetric problems | `"flexgmres"` |
| GMRES | Krylov | General sparse systems | `"gmres"` |
| BiCGSTAB | Krylov | Non-symmetric, when GMRES stalls | `"bicgstab"` |
| SMG | Multigrid | Small grids, direct-like convergence | `"smg"` |
| PFMG | Multigrid | Large grids, low memory | `"pfmg"` |

**Default:** `"auto"` selects PCG, which is optimal for the single-phase
steady-state diffusion problem (the Laplacian with harmonic-mean face
coefficients is symmetric positive-definite).

```python
# Use the default (PCG)
result = oi.tortuosity(data, phase=1, direction="z")

# Explicitly choose a solver
result = oi.tortuosity(data, phase=1, direction="z", solver="flexgmres")
```

## AMReX MLMG solver

The matrix-free geometric multigrid solver uses AMReX's native
`MLABecLaplacian` operator. Advantages:

- **No matrix assembly** — the operator is applied matrix-free
- **Lower memory** — approximately 3x less than HYPRE's `StructMatrix`
- **Faster setup** — no algebraic multigrid (AMG) construction

Best for small-to-medium grids on shared-memory systems.

```python
result = oi.tortuosity(data, phase=1, direction="z", solver="mlmg")
```

## When to use which

| Scenario | Recommended solver |
|----------|--------------------|
| Quick desktop analysis (<256^3) | `"mlmg"` |
| Single-phase, any size | `"auto"` (PCG) |
| Multi-phase with varying D | `"flexgmres"` |
| Large distributed MPI runs | `"pcg"` or `"pfmg"` |
| Debugging / comparison | `"smg"` (most robust) |

## Effective diffusivity tensor

The `EffectiveDiffusivityHypre` solver uses the same HYPRE backends but solves
the cell problem with periodic boundary conditions. This is accessed via the
C++ API or the command-line interface, not yet exposed in the high-level
Python facade.

## Solver parameters

When using the C++ interface or input files, solver behaviour is controlled via:

```
hypre.eps = 1.0e-9       # Convergence tolerance
hypre.maxiter = 200      # Maximum iterations
```
