# Contributing

Contributions to OpenImpala are welcome. This guide covers the development
workflow and coding standards.

## Development setup

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala.git
cd OpenImpala
git checkout -b my-feature

# Build (inside the dependency container)
apptainer exec --bind "$(pwd):/src" dependency_image.sif bash -c "cd /src && make all -j"

# Run tests
apptainer exec --bind "$(pwd):/src" dependency_image.sif bash -c "cd /src && make test"
```

## Pull request workflow

1. Fork the repository and create a feature branch
2. Make your changes, ensuring tests pass
3. Run `clang-format` on modified files
4. Submit a pull request against `master`

## Code style

- **C++17**, 100-column line limit, 4-space indentation
- LLVM-based style enforced by `.clang-format`
- All code in `namespace OpenImpala`
- Headers use `#ifndef` include guards (not `#pragma once`)
- Doxygen `@file` / `@brief` / `@param` comments on all public APIs
- Fortran files are **not** processed by clang-format

### Formatting check

```bash
# Check formatting (CI runs this automatically)
find src/ tests/ python/bindings/ -type f \( -name "*.cpp" -o -name "*.H" \) \
  | xargs clang-format --dry-run --Werror

# Auto-format
find src/ tests/ python/bindings/ -type f \( -name "*.cpp" -o -name "*.H" \) \
  | xargs clang-format -i
```

## Testing

- **C++ tests:** CTest with Catch2 (run via `ctest --output-on-failure`)
- **Python tests:** pytest (`python -m pytest python/tests/`)
- **Analytical benchmarks:** Uniform block, series layers, parallel layers with
  exact solutions

## Building documentation

```bash
# Install doc dependencies
pip install -r docs/requirements.txt

# Generate Doxygen XML (needed by Breathe)
doxygen Doxyfile

# Build Sphinx HTML
sphinx-build -b html docs/ docs/_build/html

# View locally
open docs/_build/html/index.html
```
