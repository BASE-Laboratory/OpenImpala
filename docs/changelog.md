# Changelog

## v4.7.0 (2026-11-19)

Cloud-native OME-Zarr I/O, zero-dependency Otsu segmentation, out-of-core `yt`
plotfile integration, synthetic porous-media ML dataset generators, and JOSS
release polish.

- **Cloud-native OME-NGFF v0.4 Zarr I/O** — `read_zarr_volume()`, `write_zarr_volume()`, `VoxelImage.from_zarr()` / `.to_zarr()`, and transparent `.zarr` store support in `read_image()` and all high-level solvers (`pip install 'openimpala[zarr]'`)
- **Zero-dependency Otsu thresholding** — `calculate_otsu_threshold()` / `binarize_otsu()`, `VoxelImage.from_grayscale_otsu()`, `read_image(..., threshold="otsu")`, and `--threshold otsu` in the `openimpala` CLI
- **High-level AMReX plotfiles & `yt` integration** — `write_plotfile=True` on `tortuosity()`, `tortuosity_all()`, and `effective_diffusivity()` plus `write_amrex_plotfile()` for out-of-core slicing and profiling with `yt` and ParaView
- **Synthetic microstructures & ML surrogate generator (`openimpala.synthetic`)** — `gaussian_blobs()`, `overlapping_spheres()`, `multiphase_electrode()`, `generate_ml_dataset()`, and `openimpala generate-dataset` CLI subcommand
- **Verified benchmark dataset registry** — `datasets.list_datasets()`, `datasets.fetch_example()`, and pinned SHA-256 integrity checks in `datasets.fetch_path()`
- **Multi-component periodic homogenisation** — exact per-component pinning in `EffectiveDiffusivityHypre` (CPU/MPI/CUDA) and `_solver.py` eliminating near-singular modes on isolated pore pockets

See the [v4.7.0 release notes](https://github.com/BASE-Laboratory/OpenImpala/releases/tag/v4.7.0).

## v4.6.0 (2026-04-17)

Feature parity across the Python facade and C++ backends, full 3D field export,
and multi-phase / non-cubic voxel support.

- **3D solution and corrector fields** — `return_field=True` on `tortuosity()` (`result.solution_field`) and `return_fields=True` on `effective_diffusivity()` (`result.corrector_fields`)
- **Python `effective_diffusivity()`** — full symmetric $3\times 3$ homogenisation tensor exposed in the high-level Python API across both C++ HYPRE and pure-Python solvers
- **Multi-phase transport (`phase_diffusivities`)** — arbitrary per-phase conductivity/diffusivity dictionaries in `tortuosity()` and `effective_diffusivity()`
- **Anisotropic & physical `voxel_size`** — scalar or `(dz, dy, dx)` voxel pitches propagated consistently through `tortuosity()`, `specific_surface_area()`, `particle_size_distribution()`, `through_thickness_profile()`, and `rev_study()`
- **Complete microstructural suite in pure Python & C++** — `specific_surface_area` (with Cauchy–Crofton correction), `connected_components`, `particle_size_distribution`, `through_thickness_profile`, and `rev_study`

See the [v4.6.0 release notes](https://github.com/BASE-Laboratory/OpenImpala/releases/tag/v4.6.0).

## v4.5.0 (2026-04-17)

Boundary-condition accuracy upgrade and universal PyPI wheel distribution.

- **Exact half-cell harmonic Dirichlet BCs** — inlet and outlet boundary conditions enforced at outer voxel faces ($z = 0$ and $z = L$) across `TortuosityMLMG`, `TortuosityHypre`, and the pure-Python solver, eliminating the cell-centred $(N-1)/N$ discretisation offset and matching Voigt/Reuss laminates to machine precision
- **`TortuosityMLMG` default fast path** — matrix-free AMReX geometric multigrid is now the default solver across CPU and CUDA backends
- **Universal PyPI fallback wheel** — `pip install openimpala` provides pre-compiled AMReX/HYPRE wheels on Linux `x86_64` alongside a pure-Python SciPy/CuPy wheel for macOS, Windows, ARM64, and MPI-free environments

See the [v4.5.0 release notes](https://github.com/BASE-Laboratory/OpenImpala/releases/tag/v4.5.0).

## v4.0.0 (2026-03-29)

Major release introducing GPU acceleration, matrix-free geometric multigrid,
and C++ kernel modernisation.

- **CUDA GPU acceleration** via `openimpala-cuda` PyPI package
- **`TortuosityMLMG` solver** — matrix-free AMReX geometric multigrid
- **Microstructural parameterisation engine** — SSA, REV study, PSD, connected components
- **Fortran-to-C++ kernel migration** — all compute kernels native C++17 AMReX lambdas

See the [v4.0.0 release notes](https://github.com/BASE-Laboratory/OpenImpala/releases/tag/v4.0.0).

## v3.1.0 (2026-03-10)

- Replaced `pyamrex` dependency with native C++ NumPy ingestion via `VoxelImage`
- Self-contained PyPI wheels — `pip install openimpala` with zero compilation
- Memory-safe workflows: ingest data, free Python array, then solve

## v3.0.0 — v3.0.2

- Python bindings via pybind11
- CMake build system modernisation
- scikit-build-core + cibuildwheel integration
- Multi-phase transport support

## v2.0.0 — v2.1.1

- AMReX upgrade and CI/CD pipeline
- Catch2 test framework integration
- Code coverage with Codecov
- clang-format and clang-tidy enforcement

## v1.0.0 — v1.1.1

- Initial public release
- HYPRE-based tortuosity and effective diffusivity solvers
- TIFF, HDF5, RAW, DAT image readers
- Apptainer container builds
