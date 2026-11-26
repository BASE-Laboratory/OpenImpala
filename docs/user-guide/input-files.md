# C++ Input Files

When running the standalone C++ driver (`Diffusion`, installed at `/usr/bin/Diffusion` inside the release container or `build/Diffusion` when built from source), simulation parameters are supplied via an AMReX `ParmParse` configuration file:

```bash
/usr/bin/Diffusion inputs
```

Any parameter can also be overridden on the command line after the filename:

```bash
/usr/bin/Diffusion inputs tortuosity.direction=2 amr.max_grid_size=128
```

## Example `inputs` File

```ini
# --- Image Input ---
image.filename      = microstructure.tif
image.threshold     = 128

# --- Transport Solver Configuration ---
tortuosity.direction       = 2          # 0 = X, 1 = Y, 2 = Z
tortuosity.phase_id        = 1          # Conducting phase label
tortuosity.solver_type     = PCG        # PCG, FlexGMRES, GMRES, BiCGSTAB, SMG, PFMG

# --- HYPRE Convergence Controls ---
hypre.eps           = 1.0e-9
hypre.maxiter       = 500

# --- AMReX Domain Decomposition ---
amr.max_grid_size   = 64

# --- Output ---
results.path              = ./results
tortuosity.write_plotfile = false       # Write AMReX plotfile directory (plt_*)
tortuosity.verbose        = 1           # 0 = silent, 1 = summary, 2 = detailed
```

## Multi-Phase Transport Parameters

To solve composite conduction across multiple conducting phases, specify matching lists of phase IDs and diffusivities:

```ini
tortuosity.active_phases       = 1 2
tortuosity.phase_diffusivities = 1.0 0.15
```

All unlisted phase IDs are treated as insulating ($D = 0$).

## Key Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image.filename` | string | — | Input volume path (`.tif`, `.tiff`, `.h5`, `.hdf5`, `.raw`, `.dat`) |
| `image.threshold` | float | — | Optional grayscale binarisation threshold (`pixel > threshold` $\to 1$) |
| `image.hdf5_dataset` | string | `/data` | Internal dataset path for HDF5 inputs |
| `tortuosity.direction` | int | `0` | Transport direction (`0` = X, `1` = Y, `2` = Z) |
| `tortuosity.phase_id` | int | `0` | Conducting phase ID (single-phase mode) |
| `tortuosity.solver_type` | string | `PCG` | Linear solver (`PCG`, `FlexGMRES`, `GMRES`, `BiCGSTAB`, `SMG`, `PFMG`) |
| `tortuosity.vlo` | float | `0.0` | Inlet face Dirichlet potential |
| `tortuosity.vhi` | float | `1.0` | Outlet face Dirichlet potential |
| `hypre.eps` | float | `1e-9` | Relative residual convergence tolerance |
| `hypre.maxiter` | int | `200` | Maximum solver iterations |
| `amr.max_grid_size` | int | `32` | Maximum AMReX patch edge length |
| `raw.width` / `raw.height` / `raw.depth` | int | — | Dimensions $(N_x, N_y, N_z)$ when `image.filename` is `.raw` |
| `raw.datatype` | string | `UINT8` | Raw binary element type (`UINT8`, `UINT16_LE`, `INT16_LE`, `FLOAT32_LE`, …) |
| `mlmg.precision` | string | `float64` | Matrix-free MLMG precision (`float64`, `float32`, `mixed`) |
| `post_process.write_flux_field` | bool | `false` | Append 4 flux components (`flux_x`, `flux_y`, `flux_z`, `flux_magnitude`) to tortuosity plotfiles |
| `post_process.calc_flux_stats` | bool | `false` | Compute and write per-direction `flux_stats` (mean, max, std, histogram) to JSON |
| `microstructure.compute_active_ssa` | bool | `false` | Compute percolation-filtered participating interfacial SSA alongside total SSA |
| `plotfile.nfiles` | int | AMReX default | Concurrent `VisMF` file streams per plotfile directory |
| `checkpoint.dir` / `checkpoint.restart` | string / bool | `""` / `false` | Write / resume solved directional potential or corrector `VisMF` checkpoints |
| `rev.resume` | bool | `false` | Resume an interrupted REV study from existing CSV output |
| `results.path` | string | `.` | Output directory for JSON/text summaries and plotfiles |
| `tortuosity.write_plotfile` | bool | `false` | Write distributed AMReX plotfile for `yt` / ParaView / VisIt |

