# Contributing to OpenImpala

Thanks for your interest in contributing — bug reports, fixes, documentation
improvements, and new features are all welcome. This file describes the
practical workflow.

## Reporting bugs and requesting features

Please open an issue at
[github.com/BASE-Laboratory/OpenImpala/issues](https://github.com/BASE-Laboratory/OpenImpala/issues).
Issue templates are provided for both bug reports and feature requests; please
use them so the maintainers have enough context to act on the report.

For bug reports, include:

* OpenImpala version (`oi.__version__`) and platform (CPU vs GPU wheel, OS).
* For wheels: output of `oi.build_info()`.
* A minimal reproducer — ideally a small Python script with synthetic data,
  not a 4 GB tomogram.
* The exception traceback or the silent-crash banner from Colab if relevant.

## Asking questions

Open a GitHub Discussion in the
[OpenImpala Discussions board](https://github.com/BASE-Laboratory/OpenImpala/discussions)
for "how do I…" or "is this expected?" questions. Use issues for bugs and
features, discussions for everything else.

## Development setup

### Option 1 — Apptainer container (recommended)

The repository ships a definition file for an Apptainer/Singularity container
with all build dependencies pre-installed (AMReX, HYPRE, HDF5, libtiff,
gcc-toolset-13, libomp, OpenMPI). This is the same container CI uses, so
your local builds will behave identically to the CI build.

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala.git
cd OpenImpala
# Build the container (one-off, ~20 min)
sudo apptainer build dependency_image.sif containers/dependency.def
# Configure and build
apptainer exec --bind "$(pwd):/src" dependency_image.sif \
    bash -c "cd /src && mkdir -p build && cd build && cmake .. && make -j"
# Run tests
apptainer exec --bind "$(pwd):/src" dependency_image.sif \
    bash -c "cd /src/build && ctest --output-on-failure"
```

### Option 2 — Native build

If you have AMReX, HYPRE, HDF5, and libtiff installed natively, point CMake
at them via `CMAKE_PREFIX_PATH`:

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/amrex;/path/to/hypre;/path/to/hdf5;/path/to/libtiff"
make -j
ctest --output-on-failure
```

For GPU support, add `-DGPU_BACKEND=CUDA`. The build will require a matching
HYPRE built with `--with-cuda` and AMReX built with `-DAMReX_GPU_BACKEND=CUDA`.

## Code style

C++ and Python both have automated style enforcement. CI will reject PRs
that fail either check.

* **C++**: clang-format with the rules in `.clang-format` (LLVM-based,
  100 column limit, 4-space indent). Run `clang-format -i src/**/*.{cpp,H}`
  before committing.
* **C++**: clang-tidy with the rules in `.clang-tidy` (bugprone-*,
  performance-*, modernize-use-nullptr). CI runs this against every
  changed file.
* **Python**: PEP 8 via `ruff` is recommended but not enforced. Public
  API functions in `python/openimpala/facade.py` should have NumPy-style
  docstrings.
* **Fortran files** are not processed by clang-format or clang-tidy.

All C++ code lives in `namespace OpenImpala`. Headers use `#ifndef`
include guards rather than `#pragma once`.

## Testing

The test suite has three layers:

1. **C++ unit and integration tests** (under `tests/`) — Catch2-based,
   register with CTest. Run with `ctest --output-on-failure`. Each test
   is a small standalone solve with an analytical solution or a tightly
   bounded numerical reference.
2. **Python tests** (under `python/tests/`) — pytest-based, exercise the
   facade and binding layers. Run with `pytest python/tests/`.
3. **Regression benchmarks** (under `tests/benchmarks/`) — Python scripts
   generating synthetic datasets with known analytical tortuosity values
   (uniform block, series layers, parallel layers).

When adding a new feature, please include at least one test that exercises
it. When fixing a bug, please add a regression test that fails on the bug
and passes on your fix.

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes, with tests.
3. Run the local CI loop:
   ```bash
   clang-format -i src/**/*.{cpp,H} python/bindings/*.cpp
   apptainer exec --bind "$(pwd):/src" dependency_image.sif \
       bash -c "cd /src/build && make -j && ctest --output-on-failure"
   ```
4. Open a pull request against `main`. The PR description should explain
   *why* the change is needed, not just *what* it changes.
5. CI runs clang-format check, clang-tidy static analysis, the C++ test
   suite, Python tests, code-coverage reporting (via Codecov), and wheel
   builds for both CPU and GPU. All checks must pass before merge.
6. A maintainer will review. Please respond to review comments by pushing
   additional commits to the same branch (not by force-pushing) so the
   review history stays readable.

## Code of conduct

By participating in this project you agree to abide by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under
the [BSD-3-Clause license](LICENSE) that covers the rest of the project.
