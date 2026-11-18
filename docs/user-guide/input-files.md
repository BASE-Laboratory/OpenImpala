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
| `results.path` | string | `.` | Output directory for JSON/text summaries and plotfiles |
| `tortuosity.write_plotfile` | bool | `false` | Write distributed AMReX plotfile for `yt` / ParaView / VisIt |
