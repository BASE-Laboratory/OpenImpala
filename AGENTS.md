# AGENTS.md — `OpenImpala` (Layer 3: Physics Solvers)

Read [`ECOSYSTEM.md`](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md) before modifying repository boundaries, I/O contracts, or public interfaces.

## Architectural Role & Hard Rules
- **Layer:** Layer 3 — Physics solvers (`BASE-Laboratory` 4-layer software architecture).
- **Licence:** `BSD-3-Clause`.
- **Owns:** Steady-state diffusive transport, tortuosity (`tortuosity`, `tortuosity_all`), periodic homogenisation (`effective_diffusivity`), geometric characterisation (`volume_fraction`, `percolation_check`, `specific_surface_area`, `connected_components`, `particle_size_distribution`, `through_thickness_profile`, `rev_study`), battery parameterisation (`to_pybamm_parameters`, `export_bpx_json`), AMReX/HYPRE C++ solvers, and standalone pure-Python fallback readers/solvers.
- **Zero `basecoat` Runtime Dependency:** `OpenImpala` keeps its lean `numpy` + `scipy` runtime footprint and **never** imports `basecoat` at runtime (`"basecoat"` is banned in `[tool.basecoat.layers]` alongside Layer 2 and Layer 4 packages).
- **Structural `VolumeProtocol` & OME-Zarr Contract (`v5.1`):**
  - High-level Python functions accept any object satisfying `openimpala.VolumeProtocol` (including `basecoat.Volume` and `basecoat.VolumeSet` with `labels["phase"]`), converting physical spacing to **SI metres** via `spacing_in("m")` by default and raising `openimpala.UnitCalibrationError` when uncalibrated unless `voxel_size=` is supplied explicitly.
  - `read_zarr_volume` and `VoxelImage.from_zarr` accept `label_key=` (e.g. `"phase"`, `"phase_lower"`, `"phase_upper"`) and `unit=` (e.g. `"m"`, `"um"`), and solver entry points automatically resolve `labels/phase` when an OME-Zarr v0.4 store's root image is floating-point grayscale.
- **Public API & C++ Stability:** Do not modify C++ headers/sources (`src/`, `python/bindings/`) or breaking Python signatures without James Le Houx's explicit approval.
