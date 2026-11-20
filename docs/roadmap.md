# Development Roadmap

This document summarises shipped milestones across the `v4.x`–`v5.0.0` series, near-term engineering priorities (`v5.1`–`v5.2`), and longer-term multi-physics research tracks (`v6.0+`). Live tracking is maintained on GitHub in [Issue #40 — Strategic Development Roadmap](https://github.com/BASE-Laboratory/OpenImpala/issues/40).

## Shipped Milestones (`v4.0` – `v5.0.0`)

- **Zero-Fortran C++17 & CUDA Architecture (`v4.0`–`v4.5`)** — Complete migration of all finite-volume stencil, flux, connected-component, and boundary-condition kernels to portable AMReX `ParallelFor` lambdas (`CPU` + `CUDA` via `openimpala-cuda`), exact outer-face harmonic Dirichlet boundary conditions, and matrix-free `TortuosityMLMG` (`~310 B/voxel`, ~3.3–4.6× lower RAM than HYPRE structured Krylov solvers).
- **Symmetric $3\times 3$ Homogenisation, Multi-Phase & Stereology (`v4.6`)** — Full periodic cell-problem effective diffusivity tensor `effective_diffusivity()`, arbitrary per-phase conductivity maps (`phase_diffusivities`), anisotropic physical `voxel_size`, Cauchy–Crofton specific surface area, 6-connected components, equivalent-sphere PSD, through-thickness profiles, and REV convergence studies across both C++ and pure-Python backends.
- **Cloud OME-Zarr, Zero-Dependency Bridges & Surrogate Generators (`v4.7`)** — OME-NGFF v0.4 Zarr I/O (`read_zarr_volume`, `write_zarr_volume`), Otsu segmentation (`threshold="otsu"`), majority-vote coarsening (`downsample_volume`), out-of-core AMReX plotfiles for `yt` and ParaView, direct ingestion of STFC CIL `ImageData` and `xarray.DataArray` handles, SHA-256-verified benchmark datasets, and synthetic porous-media ML dataset generation (`openimpala.synthetic`).
- **`v5.0.0` Relaunch, PyBaMM / Faraday BPX Interoperability & JOSS 2026** — Explicit `phase` API consistency across all solvers, automatic OpenMPI runtime discovery (`DT_RUNPATH` + `openmpi` wheel preloading) with loud fallback diagnostics, quiet-by-default AMReX session lifecycle, `.macmullin_number` and `.bruggeman_exponent` on `TortuosityResult`, one-call `to_pybamm_parameters()` and `export_bpx_json()` helpers, CLI parity (`--downsample`, `--deff`, `--bpx-electrode`), and automated 8-case analytical wheel verification in CI.

## Near-Term Engineering (`v5.1` – `v5.2`)

1. **Matrix-Free Periodic Homogenisation (`EffectiveDiffusivityMLMG`)** — Extend the matrix-free AMReX `MLABecLaplacian` operator to solve the three periodic cell problems ($\nabla\cdot(D(\nabla\chi_k + \mathbf{e}_k)) = 0$) on CPU and CUDA without HYPRE `SStruct` matrix assembly, bringing the ~3.3× memory reduction of `TortuosityMLMG` to full $3\times 3$ tensor calculations.
2. **HPC & Conda Packaging Ecosystem**
   - `conda-forge` feedstock for `openimpala` ([Issue #147](https://github.com/BASE-Laboratory/OpenImpala/issues/147)).
   - Upstream `Spack` package recipe for national Tier-1/Tier-0 supercomputers ([Issue #220](https://github.com/BASE-Laboratory/OpenImpala/issues/220)).
3. **Extreme-Scale I/O & Mixed-Precision Solvers**
   - Collective MPI-IO in `RawReader` ([Issue #18](https://github.com/BASE-Laboratory/OpenImpala/issues/18)) and scalable parallel plotfile/checkpoint I/O ([Issue #32](https://github.com/BASE-Laboratory/OpenImpala/issues/32), [Issue #33](https://github.com/BASE-Laboratory/OpenImpala/issues/33)).
   - Single-precision (`float32`) and mixed-precision V-cycle options in `TortuosityMLMG` to halve GPU device memory on $1024^3+$ tomograms ([Issue #13](https://github.com/BASE-Laboratory/OpenImpala/issues/13), [Issue #30](https://github.com/BASE-Laboratory/OpenImpala/issues/30)).
   - Optional 3D local diffusive flux-vector field export and active-pore interfacial surface area ([Issue #34](https://github.com/BASE-Laboratory/OpenImpala/issues/34)).

## Multi-Physics & Numerical Research Tracks (`v6.0+`)

1. **Transient Diffusion & Electrochemical Impedance Tortuosity** — Time-stepping $\partial_t c = \nabla\cdot(D\nabla c)$ and frequency-domain blocking-electrode EIS transmission-line tortuosity for dead-end pore characterisation ([Issue #12](https://github.com/BASE-Laboratory/OpenImpala/issues/12), [Issue #35](https://github.com/BASE-Laboratory/OpenImpala/issues/35)).
2. **Sub-Voxel Grayscale & Cut-Cell Geometry** — Direct-from-grayscale partial-volume transport mapping without hard binarisation ([Issue #213](https://github.com/BASE-Laboratory/OpenImpala/issues/213)) and AMReX Embedded Boundary (EB) cut-cell discretisation for curved phase interfaces ([Issue #69](https://github.com/BASE-Laboratory/OpenImpala/issues/69)), alongside adaptive mesh refinement ([Issue #6](https://github.com/BASE-Laboratory/OpenImpala/issues/6)).
3. **Generalised Multi-Physics, Operando & Differentiable Solvers** — Runtime per-face Dirichlet/Neumann/Robin/Periodic boundary-condition composition ([Issue #5](https://github.com/BASE-Laboratory/OpenImpala/issues/5), [Issue #16](https://github.com/BASE-Laboratory/OpenImpala/issues/16)), modular multi-physics operator coupling ([Issue #15](https://github.com/BASE-Laboratory/OpenImpala/issues/15)), 4D *operando* digital volume correlation (DVC) strain-coupled transport ([Issue #222](https://github.com/BASE-Laboratory/OpenImpala/issues/222)), random-walk `ParticleContainer` tracers ([Issue #70](https://github.com/BASE-Laboratory/OpenImpala/issues/70)), and adjoint/differentiable sensitivity operators ([Issue #66](https://github.com/BASE-Laboratory/OpenImpala/issues/66)).

## Core Repository Scope Boundary

To keep core maintenance focused on numerical accuracy, HPC scalability, and zero-overhead Python/C++ interoperability, **standalone desktop GUI plugins and facility-specific workflow wrappers** (such as Fiji/ImageJ macros, `napari` dock widgets, `Orange Canvas` / Tomwer add-ons, `Trame` dashboards, and `AiiDA` plugins) live outside the core `OpenImpala` repository. Upstream reconstruction and analysis tools (`CIL`, `TomoPy`, `HTTomo`, `TIGRE`, `PoreSpy`, `xarray`, `PyBaMM`) integrate directly in memory via OpenImpala's zero-dependency array, metadata, and Faraday BPX interfaces.
