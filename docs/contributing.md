# Contributing

Contributions to OpenImpala are welcome. This guide summarises the C++, Python,
and documentation development workflows; see `CONTRIBUTING.md` in the
repository root for issue reporting guidelines.

## Development Setup

### Apptainer Container (C++ + MPI + HYPRE)

The repository provides `containers/Singularity.deps.def` with all compiled
dependencies pre-installed (AMReX, HYPRE, HDF5, libtiff, OpenMPI):

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala.git
cd OpenImpala
git checkout -b my-feature

# Build the dependency container once (~20 min)
sudo apptainer build dependency_image.sif containers/Singularity.deps.def

# Configure and compile
apptainer exec --bind "$(pwd):/src" dependency_image.sif \
    bash -c "cd /src && cmake -B build && cmake --build build -j"

# Run C++ Catch2 tests
apptainer exec --bind "$(pwd):/src" dependency_image.sif \
    bash -c "cd /src/build && ctest --output-on-failure"
```

### Pure-Python Environment

High-level Python API changes, solver tests, and documentation can be developed
in a standard virtual environment without compiling C++ extensions:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[docs]" pytest scipy tifffile h5py
PYTHONPATH=python pytest python/tests/ -v
```

## Code Style

- **C++17**: 100-column line limit, 4-space indentation, enforced by `.clang-format` and `.clang-tidy`
- All C++ declarations live in `namespace OpenImpala` and use `#ifndef` header guards
- Doxygen `@file`, `@brief`, and `@param` comments on public C++ classes and functions
- **Python**: PEP 8 with NumPy-style docstrings on all public functions in `python/openimpala/`

### Formatting Check

```bash
find src/ tests/ python/bindings/ -type f \( -name "*.cpp" -o -name "*.H" \) \
  | xargs clang-format --dry-run --Werror
```

## Building Documentation & Tutorials

Tutorial notebooks in `tutorials/` are rendered directly into the Sphinx site
via `myst-nb`:

```bash
pip install -r docs/requirements.txt

# Execute and refresh tutorial outputs in place
PYTHONPATH="$(pwd)/python" jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=180 --inplace tutorials/0*.ipynb

# Generate Doxygen XML for C++ API pages
doxygen Doxyfile

# Build Sphinx HTML with warnings treated as errors
PYTHONPATH="$(pwd)/python" sphinx-build -W --keep-going -b html docs/ docs/_build/html
```

## Pull Request Workflow

1. Fork the repository and create a feature branch from `master`.
2. Add or update unit/analytical regression tests covering your change.
3. Run `clang-format`, `pytest`, and `ctest` locally.
4. Open a pull request against `master`.
