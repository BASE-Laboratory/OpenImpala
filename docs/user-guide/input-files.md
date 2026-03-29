# Input Files

When running OpenImpala from the command line (C++ executable), configuration
is specified via AMReX `ParmParse` text files. The file is passed as the first
argument:

```bash
./Diffusion3d inputs
```

## Example input file

```ini
# --- Image Input ---
image.filename    = microstructure.tiff
image.threshold   = 128

# --- Solver Configuration ---
tortuosity.direction      = 2          # 0=X, 1=Y, 2=Z
tortuosity.phase_id       = 0          # Phase to solve for
tortuosity.solver_type    = PCG        # PCG, FlexGMRES, GMRES, BiCGSTAB, SMG, PFMG

# --- HYPRE Solver Parameters ---
hypre.eps      = 1.0e-9                # Convergence tolerance
hypre.maxiter  = 200                   # Maximum iterations

# --- AMReX Grid Configuration ---
amr.max_grid_size = 64                 # Box decomposition size

# --- Output ---
results.path = ./results               # Output directory
tortuosity.write_plotfile = false       # Write AMReX plotfile of solution
tortuosity.verbose = 1                 # 0=silent, 1=basic, 2+=detailed
```

## Key parameters

### Image input

| Parameter | Type | Description |
|-----------|------|-------------|
| `image.filename` | string | Path to 3D image (TIFF, HDF5, RAW, DAT) |
| `image.threshold` | float | Binarisation threshold value |
| `image.hdf5_dataset` | string | HDF5 dataset path (default: `/data`) |

### Solver

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tortuosity.direction` | int | 0 | Flow direction: 0=X, 1=Y, 2=Z |
| `tortuosity.phase_id` | int | 0 | Phase ID of the conducting phase |
| `tortuosity.solver_type` | string | PCG | HYPRE solver algorithm |
| `tortuosity.vlo` | float | 0.0 | Dirichlet BC at inlet |
| `tortuosity.vhi` | float | 1.0 | Dirichlet BC at outlet |

### Multi-phase transport

```ini
tortuosity.active_phases      = 0 2       # Phase IDs with non-zero D
tortuosity.phase_diffusivities = 1.0 0.5  # Corresponding D values
```

### Grid decomposition

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `amr.max_grid_size` | int | 32 | Maximum box size for domain decomposition |

Smaller values create more boxes (better MPI load balance); larger values
reduce communication overhead. Powers of 2 that divide the domain dimensions
evenly are recommended.

## Output files

| File | Description |
|------|-------------|
| `results.json` | Structured JSON with all computed properties |
| `results.txt` | Human-readable summary |
| `plt_*/` | AMReX plotfile (if `write_plotfile = true`) |
