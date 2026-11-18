# User Guide

This section combines **Explanation** pages covering OpenImpala's mathematical discretisation and linear solvers with **How-To Guides** for GPU acceleration, distributed MPI/SLURM clusters, C++ `ParmParse` inputs, and out-of-core visualisation.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Mathematical Formulation
:link: concepts
:link-type: doc

Voxel grid conventions `(Nz, Ny, Nx)`, half-cell harmonic Dirichlet BCs, periodic homogenisation cell problems, multi-phase conductances, and Cauchy–Crofton stereology.
:::

:::{grid-item-card} Linear Solvers
:link: solvers
:link-type: doc

Memory and convergence trade-offs between matrix-free AMReX `MLMG` and HYPRE structured Krylov/multigrid solvers (`PCG`, `FlexGMRES`, `BiCGSTAB`, `SMG`, `PFMG`).
:::

:::{grid-item-card} CUDA GPU Acceleration
:link: gpu
:link-type: doc

Running native AMReX + HYPRE CUDA kernels via `openimpala-cuda`, checking device visibility with `oi.build_info()`, and sizing GPU memory.
:::

:::{grid-item-card} MPI & HPC Clusters
:link: hpc
:link-type: doc

Multi-rank Python (`mpi4py`) and C++ (`Diffusion`) execution, SLURM + Apptainer batch scripts, and `max_grid_size` domain decomposition tuning.
:::

:::{grid-item-card} C++ Input Files
:link: input-files
:link-type: doc

Configuring standalone `/usr/bin/Diffusion` runs via AMReX `ParmParse` key-value files for batch HPC pipelines.
:::

:::{grid-item-card} Out-of-Core Plotfiles with `yt`
:link: yt-plotfiles
:link-type: doc

Inspecting multi-gigabyte AMReX plotfiles (`plt_*`) lazily with `yt` when full 3D fields exceed workstation RAM.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

concepts
solvers
gpu
hpc
input-files
yt-plotfiles
```
