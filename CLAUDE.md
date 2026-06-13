# CLAUDE.md — OpenImpala Developer Context

OpenImpala is a high-performance framework for computing effective transport
properties (diffusivity, conductivity, tortuosity) directly on 3D voxel images
of porous microstructures. It uses finite differences on the voxel grid (no
mesh generation), parallelized via MPI through the AMReX library, with HYPRE
for linear solves.

## Architecture

```
                        ┌─────────────────────────┐
                        │   Diffusion.cpp (main)   │  ← Application layer
                        │  Orchestrates pipeline:  │     Parses inputs, calls
                        │  read → solve → output   │     solvers, writes results
                        └────────┬────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼──────┐  ┌───────▼────────┐  ┌──────▼───────────┐
    │   I/O Readers  │  │ Physics Solvers │  │   Output Layer   │
    │   (src/io/)    │  │  (src/props/)   │  │   (src/props/)   │
    └────────────────┘  └────────────────┘  └──────────────────┘

    TiffReader.H/cpp     TortuosityHypre      ResultsJSON.H
    HDF5Reader.H/cpp     TortuosityMLMG        → BPX, BattINFO
    RawReader.H/cpp      EffDiffusivityHypre   → JSON + text output
    DatReader.H/cpp      VolumeFraction
                         PercolationCheck
                               │
                    ┌──────────▼──────────┐
                    │  C++ Kernels        │  ← Computational hot path
                    │  (TortuosityKernels │     Matrix fill, flux calc
                    │   .H, AMReX GPU)    │     (AMReX ParallelFor/Array4)
                    └─────────────────────┘

A Python layer (`python/`) wraps the C++ libraries via pybind11 and adds a
NumPy-native high-level API plus a pure-Python SciPy/CuPy fallback.
```

### Module Relationships

**I/O layer** (`src/io/`) reads 3D images into `amrex::iMultiFab` (integer
MultiFab). All readers produce the same output type — a phase-labeled voxel
grid. The readers are independent of the physics layer.

**Physics solvers** (`src/props/`) operate on `iMultiFab` phase data:
- `TortuosityHypre` — Solves ∇·(D∇φ) = 0 with Dirichlet BCs to get τ, using
  HYPRE structured-grid Krylov solvers + SMG/PFMG multigrid preconditioners
- `TortuosityMLMG` — Same PDE via AMReX's matrix-free MLMG geometric multigrid
  (embedded boundary for inactive cells); ~3× less memory, fastest on GPU
- `EffectiveDiffusivityHypre` — Solves cell problem ∇·(D∇χ) = -∇·(Dê) for
  the effective diffusivity tensor via homogenization
- `PercolationCheck` — Flood-fill connectivity check (no solver needed)
- `VolumeFraction` — Phase counting with MPI reduction
- `TortuosityDirect` — Legacy iterative solver (Forward Euler, not HYPRE)

`TortuosityHypre`, `TortuosityMLMG`, and `TortuosityDirect` share
`TortuositySolverBase`, which factors out the solver-independent work:
building the diffusion-coefficient field, removing isolated cells, the
flood-fill activity mask, and post-solve boundary-flux integration →
conservation check → τ. Each backend just implements `solve()`.

**Native C++ kernels** — The HYPRE/MLMG matrix fill and flux calculations were
historically Fortran 90 but have been **migrated to native C++**
(`TortuosityKernels.H`, using AMReX `ParallelFor`/`Array4`) so the hot path
runs on CPU and GPU (CUDA/HIP) from a single source. There are no `.F90` /
`*_F.H` files in the tree; CMake still lists Fortran as a language for AMReX
compatibility only.

**Configuration** is via AMReX `ParmParse` (text `inputs` files). Key types:
- `PhysicsConfig.H` — Maps solver output to physical quantities (diffusion,
  electrical conductivity, thermal conductivity, etc.)
- `Tortuosity.H` — Enums: `Direction` (X/Y/Z), `CellType`, `SolverType`

### Key Data Flow

`Diffusion.cpp` selects one of four modes via the `calculation_method` /
`dry_run` / `rev.do_study` inputs: `dry_run` (percolation + volume fraction
only), REV study, `homogenization` (D_eff tensor), or `flow_through`
(tortuosity).

```
TIFF/HDF5/RAW/DAT file
  → Reader.threshold() → iMultiFab (phase IDs: 0, 1, ...)
    → PercolationCheck (is phase connected inlet→outlet?)
    → VolumeFraction (what fraction is this phase?)
    → TortuosityHypre / TortuosityMLMG  (flow_through)
      or EffectiveDiffusivityHypre + DeffTensor  (homogenization)
      → C++ kernel fills system (harmonic mean face coefficients)
      → solve (HYPRE FlexGMRES/PCG/… or AMReX MLMG V-cycle)
      → Flux integration → D_eff, tortuosity
        → ResultsJSON → results.json + results.txt
```

## Key Conventions

### Face Coefficients (Harmonic Mean)
Inter-cell face diffusivities use the harmonic mean of adjacent cell values:
```
D_face = 2 * D_left * D_right / (D_left + D_right)
```
This is physically correct for series resistance and appears in the C++
matrix-fill paths of `TortuosityHypre`, `TortuosityMLMG`, and
`EffectiveDiffusivityHypre`.

### Tortuosity Definition
```
τ = active_volume_fraction / D_eff
```
where `D_eff = |average_flux| / (cross_section_area × |∇φ_imposed|)`.
On a discrete N-cell grid with Dirichlet BCs at cell centers:
`D_eff = D_bulk × N/(N-1)`, so `τ = (N-1)/N` for a uniform medium.

### Boundary Conditions
- **Flow direction**: Dirichlet (φ=vlo at inlet face, φ=vhi at outlet face)
- **Lateral directions**: Zero-flux Neumann (no transport across sides)
- **Inactive cells**: Decoupled from system (A_ii=1, A_ij=0, rhs=0)

### Phase Data
Images are segmented into integer phase IDs stored in `amrex::iMultiFab`.
Phase 0 typically = pore/void, Phase 1 = solid, but this is configurable
via `phase_id` and `threshold_value` in the inputs file.

## Build & Test

```bash
# Build inside dependency container (recommended)
apptainer exec --bind "$(pwd):/src" dependency_image.sif bash -c "cd /src && make all -j"

# Run tests
apptainer exec --bind "$(pwd):/src" dependency_image.sif bash -c "cd /src && make test"

# Native build (if dependencies are installed)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_COMPILER=$(which mpicxx) \
         -DCMAKE_Fortran_COMPILER=$(which mpif90)
make -j$(nproc) && ctest --output-on-failure
```

### CI Pipeline
- **Format check**: `clang-format` enforces 100-column LLVM-based style
- **Static analysis**: `clang-tidy` (bugprone-*, performance-*, modernize-use-nullptr)
- **Build & CTest**: Full test suite in dependency container
- **Code coverage**: gcovr on PRs, posted to Codecov

## File Reference

### Source — I/O (`src/io/`)
| File | Purpose |
|------|---------|
| `TiffReader.H/cpp` | Reads single/multi-directory TIFFs and TIFF sequences |
| `HDF5Reader.H/cpp` | Reads HDF5 datasets with parallel I/O |
| `RawReader.H/cpp` | Reads flat binary files (UINT8, INT16, FLOAT32, etc.) |
| `DatReader.H/cpp` | Reads legacy DAT binary format |
| `CathodeWrite.H/cpp` | Output struct for battery electrode parameters |

### Source — Physics (`src/props/`)
| File | Purpose |
|------|---------|
| `Diffusion.cpp` | **Main application entry point** — orchestrates full pipeline |
| `Tortuosity.H` | Base interface + enums (Direction, CellType, SolverType) |
| `TortuositySolverBase.H/cpp` | Shared solver setup + flux/τ post-processing (base class) |
| `TortuosityHypre.H/cpp` | HYPRE structured-grid tortuosity solver |
| `TortuosityMLMG.H/cpp` | Matrix-free AMReX MLMG tortuosity solver (EB, GPU-native) |
| `TortuosityDirect.H/cpp` | Legacy iterative tortuosity solver (Forward Euler) |
| `TortuosityKernels.H` | C++ hot-path kernels (isolated-cell removal, face flux, flux integration) |
| `HypreStructSolver.H/cpp` | HYPRE lifecycle/infrastructure shared by HYPRE solvers |
| `EffectiveDiffusivityHypre.H/cpp` | Cell-problem solver for χ fields (homogenization) |
| `DeffTensor.H/cpp` | Assembles 3×3 D_eff tensor from χ gradients |
| `VolumeFraction.H/cpp` | Phase volume fraction calculator |
| `PercolationCheck.H/cpp` | Flood-fill percolation connectivity check |
| `FloodFill.H/cpp` | GPU-parallel flood-fill (used by percolation/components/mask) |
| `ConnectedComponents.H/cpp` | Labels contiguous regions of a phase |
| `SpecificSurfaceArea.H/cpp` | Interface area via Cauchy–Crofton stereology |
| `ParticleSizeDistribution.H/cpp` | Equivalent-sphere radii from components |
| `ThroughThicknessProfile.H/cpp` | Per-slice volume fraction along a direction |
| `REVStudy.H/cpp` | Representative Elementary Volume convergence study |
| `PhysicsConfig.H` | Physics type mapping (diffusion ↔ conductivity ↔ thermal) |
| `SolverConfig.H` | String↔enum parsing (solver type, direction) |
| `BoundaryCondition.H` | BC enums/abstraction (Dirichlet/Neumann/Periodic) |
| `MacroGeometry.H` | Electrode thickness / cross-section / volume helpers |
| `ResultsJSON.H` | Structured JSON output (BPX/BattINFO compatible) |

> **Note:** The HYPRE/MLMG matrix-fill and flux kernels were historically
> Fortran 90 (`*_F.H` ↔ `*.F90`) but have been migrated to native C++ in
> `TortuosityKernels.H` (AMReX `ParallelFor`/`Array4`, CPU+GPU). No Fortran
> source files remain in the repository.

### Source — Python layer (`python/`)
| File | Purpose |
|------|---------|
| `openimpala/facade.py` | High-level NumPy-native API (tortuosity, volume_fraction, …) |
| `openimpala/session.py` | `Session` context manager — AMReX init/finalize (re-entrant) |
| `openimpala/_solver.py` | Pure-Python SciPy/CuPy fallback backend (no compiled `_core`) |
| `openimpala/cli.py` | `openimpala` CLI entry point |
| `bindings/*.cpp` | pybind11 `_core` module (module/io/props/solvers/config/enums) |
| `bindings/VoxelImage.H` | NumPy → AMReX `iMultiFab` bridge struct |

### Tests (`tests/`)
| File | Purpose |
|------|---------|
| `tTortuosity.cpp` | Tortuosity on real TIFF data |
| `tEffectiveDiffusivity.cpp` | D_eff tensor on real TIFF data |
| `tVolumeFraction.cpp` | Volume fraction validation |
| `tPercolationCheck.cpp` | Flood-fill connectivity check |
| `tTortuosityMLMG.cpp` | Matrix-free MLMG solver (directional/geometry variants) |
| `tMultiPhaseTransport.cpp` | Synthetic geometry tests (analytical validation) |
| `tSynthetic*.cpp` | Synthetic VF / percolation / D_eff / microstructure / REV tests |
| `tTiffReader.cpp` / `tHDF5Reader.cpp` / `tRawReader.cpp` / `tDatReader.cpp` | I/O correctness |
| `tests/inputs/tDiffusion_*.inputs` | End-to-end `Diffusion` integration tests (dry-run, tiff, hdf5, REV, …) |
| `tests/unit/` | Catch2 unit tests (PhysicsConfig, ResultsJSON, MacroGeometry) |
| `tests/validation/` | V&V scripts (Hashin–Shtrikman bounds, sphere packing, Berea sandstone) |
| `tests/benchmarks/` | Python scripts for generating benchmark datasets |
| `python/tests/` | pytest suite for the Python API and bindings |

### Regression Benchmarks
Three CTest benchmarks with exact analytical solutions on discrete grids:
- **Uniform block**: τ = (N-1)/N — basic solver sanity check
- **Series layers**: τ = (N-1)(D₀+D₁)/(2N·D₀·D₁) — Reuss/harmonic bound
- **Parallel layers**: τ = 2(N-1)/(N(D₀+D₁)) — Voigt/arithmetic bound

## External Dependencies
- **AMReX** — Parallel mesh infrastructure (BoxArray, iMultiFab, Geometry, MPI)
- **HYPRE** — Structured-grid linear solvers (GMRES, FlexGMRES, PCG, BiCGSTAB, SMG, PFMG)
- **HDF5** — Hierarchical Data Format I/O
- **LibTIFF** — TIFF image reading
- **nlohmann/json** — JSON output (fetched via CMake FetchContent)
- **Catch2 v3** — Test framework (fetched via CMake FetchContent)
- **pybind11** — C++ ↔ Python bindings for the `_core` extension (Python wheels)
- **NumPy / SciPy** (+ optional **CuPy**) — Python API and pure-Python fallback solver

## Code Style
- C++17, 100-column limit, 4-space indent (LLVM-based via `.clang-format`)
- All code in `namespace OpenImpala`
- Headers use `#ifndef` include guards (not `#pragma once`)
- Doxygen `@file` / `@brief` / `@param` comments on all public APIs
- Hot-path kernels are native C++ (AMReX `ParallelFor`/`Array4`) for CPU+GPU;
  there are no Fortran source files (the historical `.F90` kernels were ported)
