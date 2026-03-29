# HPC Usage

OpenImpala is designed for distributed-memory parallelism via MPI, making it
suitable for large-scale simulations on HPC clusters.

## Running with MPI

### Python

```bash
# Install mpi4py
pip install openimpala mpi4py

# Run on 4 MPI ranks
mpirun -np 4 python my_script.py
```

### C++ executable

```bash
mpirun -np 16 ./Diffusion3d inputs
```

### Apptainer on a cluster

```bash
mpirun -np 16 apptainer exec openimpala-v4.0.0.sif /opt/OpenImpala/build/Diffusion3d inputs
```

## SLURM batch script

```bash
#!/bin/bash
#SBATCH --job-name=openimpala
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --partition=compute

module load mpi

srun apptainer exec openimpala-v4.0.0.sif \
    /opt/OpenImpala/build/Diffusion3d inputs
```

## Domain decomposition

AMReX decomposes the 3D domain into boxes distributed across MPI ranks. The
`max_grid_size` parameter controls the maximum box size:

```ini
amr.max_grid_size = 64
```

- **Smaller values** create more boxes, improving load balance across many ranks
- **Larger values** reduce inter-rank communication but may cause load imbalance
- Choose a power of 2 that evenly divides your domain dimensions

## Scaling guidelines

| Domain size | Recommended ranks | max_grid_size |
|-------------|-------------------|---------------|
| 128^3 | 1-4 | 64 |
| 256^3 | 4-16 | 64 |
| 512^3 | 16-64 | 64 |
| 1024^3 | 64-256 | 128 |

## Memory estimates

Approximate memory per rank for a tortuosity solve:

- Phase data: ~4 bytes/voxel (int32)
- Solution field: ~8 bytes/voxel (float64)
- HYPRE matrix: ~56 bytes/voxel (7-point stencil)
- **Total: ~70 bytes/voxel**

For a 512^3 domain on 64 ranks: ~140 MB per rank.
