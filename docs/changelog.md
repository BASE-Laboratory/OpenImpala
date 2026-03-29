# Changelog

## v4.0.0 (2026-03-29)

Major release introducing GPU acceleration, a new matrix-free solver,
comprehensive architectural refactoring, and expanded tutorials.

### Highlights

- **CUDA GPU acceleration** via `openimpala-cuda` PyPI package
- **TortuosityMLMG solver** — matrix-free AMReX geometric multigrid
- **Microstructural parameterisation engine** — SSA, REV study, PSD, connected components
- **Fortran-to-C++ kernel migration** — all compute kernels now native C++ AMReX lambdas
- **7-part tutorial series** with Google Colab support

See the full [release notes on GitHub](https://github.com/BASE-Laboratory/OpenImpala/releases/tag/v4.0.0).

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
