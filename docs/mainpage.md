# OpenImpala API Reference {#mainpage}

OpenImpala is a high-performance framework for computing effective transport
properties (diffusivity, conductivity, tortuosity) directly on 3D voxel images
of porous microstructures.

## Architecture

The codebase is organized into three layers:

| Layer | Location | Purpose |
|-------|----------|---------|
| **I/O Readers** | `src/io/` | Read TIFF, HDF5, RAW, DAT images into `amrex::iMultiFab` |
| **Physics Solvers** | `src/props/` | Solve transport equations, compute effective properties |
| **Fortran Kernels** | `src/props/*.F90` | Performance-critical matrix fill and flux calculations |

## Key Classes

### I/O Layer
- @ref OpenImpala::TiffReader — Single/multi-directory TIFF and TIFF sequence reader
- @ref OpenImpala::HDF5Reader — HDF5 dataset reader with parallel I/O
- @ref OpenImpala::RawReader — Flat binary file reader (UINT8, INT16, FLOAT32, etc.)
- @ref OpenImpala::DatReader — Legacy DAT binary format reader

### Physics Solvers
- @ref OpenImpala::Tortuosity — Abstract base class defining the solver interface
- @ref OpenImpala::TortuosityHypre — Primary solver: HYPRE-based tortuosity via Laplace equation
- @ref OpenImpala::TortuosityDirect — Legacy Forward Euler iterative solver
- @ref OpenImpala::EffectiveDiffusivityHypre — Effective diffusivity tensor via homogenization
- @ref OpenImpala::VolumeFraction — Phase volume fraction calculator
- @ref OpenImpala::PercolationCheck — Flood-fill percolation connectivity check

### Configuration & Output
- @ref OpenImpala::PhysicsConfig — Maps solver output to physical quantities
- @ref OpenImpala::ResultsJSON — Structured JSON output (BPX/BattINFO compatible)

## Data Flow

```
TIFF/HDF5/RAW file
  -> Reader.threshold() -> iMultiFab (phase IDs)
    -> PercolationCheck (connectivity?)
    -> VolumeFraction (phase fraction?)
    -> TortuosityHypre or EffDiffusivityHypre
      -> Fortran kernel fills HYPRE matrix
      -> HYPRE solve
      -> Flux integration -> D_eff, tortuosity
        -> ResultsJSON -> results.json
```

## Dependency Graph

Use the **Include Dependency** graphs on each file page to trace how
modules depend on AMReX (`amrex::iMultiFab`, `amrex::Geometry`, etc.)
and HYPRE.

## Further Information

- [GitHub Repository](https://github.com/BASE-Laboratory/OpenImpala)
- [Software Paper (SoftwareX 2021)](https://doi.org/10.1016/j.softx.2021.100729)
