# MPI & HPC Clusters

OpenImpala decomposes 3D voxel domains into distributed AMReX `BoxArray` patches across MPI ranks, combining MPI inter-rank halo exchanges with OpenMP host threading or multi-GPU CUDA execution.

## Multi-Rank Python Scripts (`mpi4py`)

Import and run the exact same `openimpala` facade under `mpirun` or `srun`. Always wrap multi-rank scripts in `with oi.Session():` so `amrex::Finalize()` runs cleanly before `MPI_Finalize()`, and guard console/file output on rank 0:

```python
# hpc_tortuosity.py
from mpi4py import MPI
import openimpala as oi

comm = MPI.COMM_WORLD

with oi.Session():
    _, img = oi.read_image("electrode_1024cubed.tif", threshold=128, max_grid_size=128)
    vf = oi.volume_fraction(img, phase=1)
    res = oi.tortuosity(img, phase=1, direction="z", solver="mlmg")

    if comm.Get_rank() == 0:
        print(f"Porosity:   {vf.fraction:.5f}")
        print(f"Tortuosity: {res.tortuosity:.5f} ({res.iterations} MLMG iterations)")
```

Launch across MPI ranks:

```bash
mpirun -np 16 python hpc_tortuosity.py
```

:::{tip}
Prefer `oi.read_image("volume.tif", ...)` or `oi.read_image("volume.raw", raw_width=..., raw_height=..., raw_depth=..., raw_data_type="UINT8", ...)` over loading a full NumPy array on every rank—C++ `TiffReader`, `Hdf5Reader`, and collective MPI-IO `RawReader` (`MPI_File_read_all` + `MPI_Type_create_subarray`) populate distributed AMReX `iMultiFab` boxes without allocating a full-volume host buffer on any rank.
:::

## Running the C++ Executable via Apptainer & SLURM

Each GitHub Release provides a self-contained Apptainer image (`openimpala-v5.2.0.sif`) containing `/usr/bin/Diffusion` built with OpenMPI, OpenMP, AMReX, HYPRE, HDF5, and `libtiff`:

```bash
#!/bin/bash
#SBATCH --job-name=openimpala
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=compute

module load openmpi apptainer
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --mpi=pmix apptainer exec -B "${PWD}:/data" openimpala-v5.2.0.sif \
    /usr/bin/Diffusion /data/inputs
```

## Parallel Plotfile Striping & Checkpoint / Restart

On Lustre or GPFS parallel filesystems, control the number of concurrent `VisMF` plotfile streams via `plotfile_nfiles=` in Python (`oi.tortuosity(..., write_plotfile=True, plotfile_nfiles=64)`) or `plotfile.nfiles = 64` in C++ `inputs` files.

For long multi-direction or REV studies subject to SLURM walltime limits:

- **Python / C++ REV resume**: pass `oi.rev_study(..., csv_path="rev.csv", resume=True)` (or `rev.resume = true` in C++ `inputs`) to skip `(SampleNo, REV_Size_Target)` pairs already recorded in `rev.csv` while preserving the deterministic sub-volume RNG sequence.
- **C++ `VisMF` field checkpoints**: set `checkpoint.dir = ./chk` and `checkpoint.restart = true` in `Diffusion` `inputs` files to persist each solved directional potential or periodic corrector `MultiFab` on completion and skip already-checkpointed directions on restart.

## Tuning `max_grid_size`

AMReX subdivides an $(N_z, N_y, N_x)$ domain into cubic patches of at most `max_grid_size` voxels per side (`amr.max_grid_size` in input files, or `max_grid_size=` in Python):

- **Too large** (e.g. `max_grid_size=512` on a $512^3$ volume with 16 ranks): produces only 1 box, leaving 15 MPI ranks idle.
- **Too small** (e.g. `max_grid_size=16` on $512^3$): creates $32^3 = 32{,}768$ tiny boxes, inflating halo-exchange latency.
- **Rule of thumb**: aim for 1–8 boxes per MPI rank (or 1–4 boxes per GPU), using powers of two that divide the domain size cleanly:

| Domain Size | Typical CPU Ranks | Recommended `max_grid_size` | Approx. Total Memory (`mlmg` `float64` / `mixed` / `hypre`) |
|-------------|-------------------|-----------------------------|-------------------------------------------------------------|
| $128^3$ | 1–4 | `64` | 0.65 GB / 0.49 GB / 2.1 GB |
| $256^3$ | 4–16 | `64` | 5.2 GB / 3.9 GB / 17 GB |
| $512^3$ | 16–64 | `64` or `128` | 41 GB / 31 GB / 138 GB |
| $1024^3$ | 64–256 | `128` | 332 GB / 252 GB / 1.1 TB |

