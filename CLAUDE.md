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
    HDF5Reader.H/cpp     EffDiffusivityHypre   → BPX, BattINFO
    RawReader.H/cpp      VolumeFraction        → JSON + text output
    DatReader.H/cpp      PercolationCheck
                         TortuosityDirect
                               │
                    ┌──────────▼──────────┐
                    │  Fortran Kernels    │  ← Computational hot path
                    │  (*_F.H ↔ *.F90)   │     Matrix fill, flux calc
                    └─────────────────────┘
```

### Module Relationships

**I/O layer** (`src/io/`) reads 3D images into `amrex::iMultiFab` (integer
MultiFab). All readers produce the same output type — a phase-labeled voxel
grid. The readers are independent of the physics layer.

**Physics solvers** (`src/props/`) operate on `iMultiFab` phase data:
- `TortuosityHypre` — Solves ∇·(D∇φ) = 0 with Dirichlet BCs to get τ
- `EffectiveDiffusivityHypre` — Solves cell problem ∇·(D∇χ) = -∇·(Dê) for
  the effective diffusivity tensor via homogenization
- `PercolationCheck` — Flood-fill connectivity check (no solver needed)
- `VolumeFraction` — Phase counting with MPI reduction
- `TortuosityDirect` — Legacy iterative solver (Forward Euler, not HYPRE)

**Fortran interop** — The HYPRE matrix fill and flux calculations are in
Fortran 90 for performance. C interface headers (`*_F.H`) bridge C++ ↔ Fortran
with explicit documentation of index convention differences (C: 0-based,
Fortran: 1-based).

**Configuration** is via AMReX `ParmParse` (text `inputs` files). Key types:
- `PhysicsConfig.H` — Maps solver output to physical quantities (diffusion,
  electrical conductivity, thermal conductivity, etc.)
- `Tortuosity.H` — Enums: `Direction` (X/Y/Z), `CellType`, `SolverType`

### Key Data Flow

```
TIFF/HDF5/RAW file
  → Reader.threshold() → iMultiFab (phase IDs: 0, 1, ...)
    → PercolationCheck (is phase connected inlet→outlet?)
    → VolumeFraction (what fraction is this phase?)
    → TortuosityHypre or EffDiffusivityHypre
      → Fortran kernel fills HYPRE matrix (harmonic mean face coefficients)
      → HYPRE solve (FlexGMRES, PCG, etc.)
      → Flux integration → D_eff, tortuosity
        → ResultsJSON → results.json + results.txt
```

## Key Conventions

### Face Coefficients (Harmonic Mean)
Inter-cell face diffusivities use the harmonic mean of adjacent cell values:
```
D_face = 2 * D_left * D_right / (D_left + D_right)
```
This is physically correct for series resistance and appears in both
`TortuosityHypreFill.F90` and `EffDiffFillMtx.F90`.

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
| `Tortuosity.H` | Base class + enums (Direction, CellType, SolverType) |
| `TortuosityHypre.H/cpp` | HYPRE-based tortuosity solver (primary solver) |
| `TortuosityDirect.H/cpp` | Legacy iterative tortuosity solver (Forward Euler) |
| `EffectiveDiffusivityHypre.H/cpp` | Effective diffusivity tensor via homogenization |
| `VolumeFraction.H/cpp` | Phase volume fraction calculator |
| `PercolationCheck.H/cpp` | Flood-fill percolation connectivity check |
| `PhysicsConfig.H` | Physics type mapping (diffusion ↔ conductivity ↔ thermal) |
| `ResultsJSON.H` | Structured JSON output (BPX/BattINFO compatible) |

### Source — Fortran Kernels (`src/props/`)
| File | Purpose |
|------|---------|
| `TortuosityHypreFill_F.H` / `.F90` | HYPRE matrix fill for tortuosity (7-pt stencil) |
| `EffDiffFillMtx_F.H` / `.F90` | HYPRE matrix fill for effective diffusivity cell problem |
| `Tortuosity_filcc_F.H` / `.F90` | Cell type ID, ghost cell fill, initial conditions |
| `Tortuosity_poisson_3d_F.H` / `.F90` | Flux calculation and Forward Euler update (legacy solver) |

### Tests (`tests/`)
| File | Purpose |
|------|---------|
| `tTortuosity.cpp` | Tortuosity on real TIFF data |
| `tEffectiveDiffusivity.cpp` | D_eff tensor on real TIFF data |
| `tVolumeFraction.cpp` | Volume fraction validation |
| `tPercolationCheck.cpp` | Flood-fill connectivity check |
| `tMultiPhaseTransport.cpp` | Synthetic geometry tests (analytical validation) |
| `tTiffReader.cpp` | TIFF I/O correctness |
| `tHDF5Reader.cpp` | HDF5 I/O correctness |
| `tRawReader.cpp` | RAW binary I/O correctness |
| `tests/unit/` | Catch2 unit tests (PhysicsConfig, ResultsJSON) |
| `tests/benchmarks/` | Python scripts for generating benchmark datasets |

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

## Code Style
- C++17, 100-column limit, 4-space indent (LLVM-based via `.clang-format`)
- All code in `namespace OpenImpala`
- Headers use `#ifndef` include guards (not `#pragma once`)
- Doxygen `@file` / `@brief` / `@param` comments on all public APIs
- Fortran files are NOT processed by clang-format or clang-tidy
