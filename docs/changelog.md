# Changelog

## v5.0.0 (2026-11-19)

Major API consistency, runtime diagnostics, and JOSS 2026 release.

- **Breaking API clean-up — explicit `phase` parameter** — `volume_fraction()`, `percolation_check()`, `tortuosity()`, `tortuosity_all()`, `effective_diffusivity()`, `connected_components()`, `particle_size_distribution()`, `through_thickness_profile()`, and `rev_study()` now require `phase` explicitly on single-phase calls (eliminating the `phase=0` vs `phase=1` default split across functions; multi-phase `phase_diffusivities={...}` calls continue to require no `phase`)
- **Automatic OpenMPI discovery & loud fallback diagnostics** — compiled Linux wheels embed `$ORIGIN/../../..` ELF `DT_RUNPATH` and preload `pip`/`uv`/Conda `openmpi` libraries automatically (`pip install "openimpala[openmpi]"`), while HPC `module load openmpi` (`LD_LIBRARY_PATH`) retains precedence; if `_core` exists on disk yet fails to load, `import openimpala` now emits a one-time `RuntimeWarning` with exact `dlopen` diagnostics and remediation commands
- **Quiet default AMReX lifecycle** — `AMReX` startup banners, `TinyProfiler` timing tables, and Arena memory summaries are now silenced by default so scripts and notebooks remain clutter-free; pass `oi.Session(verbose=True)` or set `OPENIMPALA_PROFILE=1` to enable profiling output
- **Explicit `downsample_volume()` warnings & tie-breaking** — `downsample_volume()` emits a `UserWarning` when truncating non-divisible trailing slices and adds `tie_break="lowest" | "highest"` for deterministic modal tie resolution
- **Empirically calibrated `estimate_memory()` & wheel analytical V&V** — calibrated `estimate_memory()` against measured `MLMG` vs `HYPRE` peak RAM (`~310` vs `~1,030`–`1,400` B/voxel) and added automated 8-case analytical & cross-backend parity verification on every CI and PyPI wheel build
- **PyBaMM & Faraday BPX parameterisation helpers** — added `.macmullin_number` and `.bruggeman_exponent` properties on `TortuosityResult`, in-memory `to_pybamm_parameters()`, BattINFO-aligned `export_bpx_json()`, and CLI parity flags (`openimpala analyze --downsample --deff --bpx-electrode`), plus root `.zenodo.json` archive metadata

See the [v5.0.0 release notes](https://github.com/BASE-Laboratory/OpenImpala/releases/tag/v5.0.0).

## v4.7.0 (2026-11-19)

Cloud-native OME-Zarr I/O, zero-dependency Otsu segmentation, out-of-core `yt`
plotfile integration, synthetic porous-media ML dataset generators, and JOSS
release polish.

- **Cloud-native OME-NGFF v0.4 Zarr I/O** — `read_zarr_volume()`, `write_zarr_volume()`, `VoxelImage.from_zarr()` / `.to_zarr()`, and transparent `.zarr` store support in `read_image()` and all high-level solvers (`pip install 'openimpala[zarr]'`)
- **Zero-dependency Otsu thresholding** — `calculate_otsu_threshold()` / `binarize_otsu()`, `VoxelImage.from_grayscale_otsu()`, `read_image(..., threshold="otsu")`, and `--threshold otsu` in the `openimpala` CLI
- **High-level AMReX plotfiles & `yt` integration** — `write_plotfile=True` on `tortuosity()`, `tortuosity_all()`, and `effective_diffusivity()` plus `write_amrex_plotfile()` for out-of-core slicing and profiling with `yt` and ParaView
- **Synthetic microstructures & ML surrogate generator (`openimpala.synthetic`)** — `gaussian_blobs()`, `overlapping_spheres()`, `multiphase_electrode()`, `generate_ml_dataset()`, and `openimpala generate-dataset` CLI subcommand
- **Verified benchmark dataset registry** — `datasets.list_datasets()`, `datasets.fetch_example()`, and pinned SHA-256 integrity checks in `datasets.fetch_path()`
- **Discrete & grayscale volume coarsening** — `downsample_volume()` and `VoxelImage.downsample()` with exact majority-vote phase pooling and automatic `(dz, dy, dx)` scaling
- **Zero-dependency CIL & `xarray` ingestion** — pass STFC Core Imaging Library `ImageData` (with automatic `.geometry` voxel pitch extraction), `xarray.DataArray`, `zarr.Array`, or `h5py.Dataset` handles directly to any solver
- **Multi-component periodic homogenisation** — exact per-component pinning in `EffectiveDiffusivityHypre` (CPU/MPI/CUDA) and `_solver.py` eliminating near-singular modes on isolated pore pockets
- **Runtime & packaging hardening** — PEP 561 `py.typed` marker, `OPENIMPALA_BACKEND=auto|cpp|pure` override, `core_load_error` diagnostics in `build_info()`, and graceful pure-Python fallback when `openimpala-cuda` runs on GPU-less login/Colab nodes

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
