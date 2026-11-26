# Getting Started

## Installation

::::{tab-set}

:::{tab-item} Linux (CPU + MPI)
```bash
pip install "openimpala[openmpi]"   # installs compiled wheel + OpenMPI runtime
```
On HPC clusters that already provide OpenMPI via environment modules, load your cluster module first and install `openimpala` + `mpi4py`:
```bash
module load openmpi
pip install openimpala mpi4py
```
:::

:::{tab-item} Linux (CUDA 12 GPU)
```bash
pip install openimpala-cuda openmpi
```
`openimpala-cuda` is a drop-in replacement compiled with native CUDA 12 support for AMReX and HYPRE structured solvers (`sm_75` Turing or newer; pre-installed CUDA drivers on Google Colab and GPU cluster nodes work out of the box). Do not install `openimpala` and `openimpala-cuda` in the same virtual environment; both provide the `openimpala` import namespace.
:::

:::{tab-item} macOS & Windows
```bash
pip install openimpala
```
On macOS (Apple Silicon and Intel), Windows, and Linux ARM64, `openimpala` installs a universal pure-Python wheel backed by NumPy and SciPy (with optional CuPy GPU acceleration if installed). Every function in the Python API accepts the same arguments and returns the same result dataclasses.
:::

:::{tab-item} HPC Container (Apptainer)
Pre-built Apptainer (`.sif`) images are attached to each [GitHub Release](https://github.com/BASE-Laboratory/OpenImpala/releases):
```bash
wget https://github.com/BASE-Laboratory/OpenImpala/releases/download/v5.2.0/openimpala-v5.2.0.sif

apptainer exec -B "$(pwd):/data" openimpala-v5.2.0.sif \
    /usr/bin/Diffusion /data/inputs
```
:::

:::{tab-item} Conda & Spack (HPC)
Build or install via the bundled `conda-forge` and `Spack` recipes under `packaging/`:
```bash
# Conda build from repository recipe
conda build packaging/conda-forge -c conda-forge

# Spack custom repo on Tier-1/Tier-0 HPC clusters (CPU MPI or +cuda)
spack repo add packaging/spack
spack install openimpala@5.2.0 +mpi +openmp +python
```
:::

:::{tab-item} From Source
See {doc}`contributing` for containerised and native CMake builds against AMReX, HYPRE, HDF5, and `libtiff`.
:::

::::

Verify which backend is active at runtime with `oi.build_info()`:

```python
import openimpala as oi

print(oi.build_info())
```

## Five-Line Quickstart

OpenImpala ships sample segmented 3D datasets in `openimpala.datasets` so you can verify your installation immediately:

```python
import openimpala as oi

vol = oi.datasets.two_phase_cubic()  # (64, 64, 64) int32 array, (Nz, Ny, Nx)
vf = oi.volume_fraction(vol, phase=1)
perc = oi.percolation_check(vol, phase=1, direction="z")
res = oi.tortuosity(vol, phase=1, direction="z", voxel_size=0.5e-6)

print(f"eps = {vf.fraction:.4f}, tau_z = {res.tortuosity:.4f}, D_eff/D0 = {res.effective_diffusivity:.4f}")
```

:::{note}
**Session Lifecycle & Profiling (`oi.Session`)**
In interactive Python sessions and Jupyter notebooks, OpenImpala automatically starts a quiet process-level session on the first API call and cleans up at interpreter exit. In multi-rank `mpirun` batch scripts, wrap your workflow in `with oi.Session():` so `AMReX::Finalize()` executes deterministically before `MPI_Finalize()`. Pass `with oi.Session(verbose=True):` (or set `OPENIMPALA_PROFILE=1`) to print AMReX `TinyProfiler` timing and Arena memory tables on exit.
:::

## Reading 3D Image Files

`oi.read_image()` reads 3D `.tif`/`.tiff` stacks, `.h5`/`.hdf5` datasets, `.zarr` OME-NGFF stores, `.raw` binary volumes, `.npy` arrays, and `.dat` grids—or you can pass a file path, CIL `ImageData`, or `xarray.DataArray` directly to any analysis function:

```python
import openimpala as oi

tiff_path = oi.datasets.fetch_path("SampleData_2Phase_squared.tif")

# Option 1: explicit reader handle (reusable across multiple analyses)
_, img = oi.read_image(tiff_path, voxel_size=(0.5e-6, 0.5e-6, 0.5e-6))
taus = oi.tortuosity_all(img, phase=1)

# Option 2: pass path directly
ssa = oi.specific_surface_area(tiff_path, phase_a=0, phase_b=1, voxel_size=0.5e-6)
```

For unsegmented grayscale reconstructions, pass `threshold="otsu"` for automatic Otsu segmentation or a numeric threshold such as `threshold=128` (`dark_is_pore=True` maps dark pores to `0` and bright solid to `1` when using Otsu).

## Ecosystem Interoperability (`basecoat`, CIL, PoreSpy & PyBaMM)

OpenImpala operates as a zero-dependency **Layer 3** solver in the `BASE-Laboratory` stack: it accepts calibrated `basecoat.Volume` / `basecoat.VolumeSet` objects (via structural `oi.VolumeProtocol` duck-typing, automatically converting spacing to SI metres), OME-Zarr v0.4 stores with `labels/phase` (plus conformal `phase_lower` / `phase_upper` envelopes), [STFC CIL](https://tomographicimaging.github.io/CIL/) `ImageData`, and [PoreSpy](https://porespy.org/) arrays, and exports continuum battery parameters directly to [PyBaMM](https://www.pybamm.org/) and [Faraday BPX](https://bpxstandard.com/):

```python
import openimpala as oi

# 1. PoreSpy / NumPy arrays, CIL ImageData, basecoat.VolumeSet, or .ome.zarr stores
vol = oi.synthetic.multiphase_electrode(shape=(64, 64, 64), seed=7)

# 2. Extract in-memory PyBaMM DFN/SPM parameters (porosity, Bruggeman, thickness, R_p)
params = oi.to_pybamm_parameters(
    vol,
    electrolyte_phase=0,
    solid_phase=1,
    electrode="negative",
    direction="x",
    voxel_size=0.5e-6,  # metres (inferred automatically from basecoat / calibrated .ome.zarr)
)
# pybamm.ParameterValues("Chen2020").update(params, check_already_exists=False)

# 3. Or export a Faraday BPX & BattINFO JSON record across X, Y, Z
bpx_doc = oi.export_bpx_json(
    vol,
    "negative_electrode_bpx.json",
    electrolyte_phase=0,
    solid_phase=1,
    electrode="negative",
    voxel_size=0.5e-6,
)

# 4. Inspect conformal segmentation bounds from a basecoat .ome.zarr store:
# lo_arr, vz_m = oi.read_zarr_volume("catalyst.ome.zarr", label_key="phase_lower", unit="m")
# hi_arr, _    = oi.read_zarr_volume("catalyst.ome.zarr", label_key="phase_upper", unit="m")
```

## Estimating Memory Before Large Runs

Before solving on large tomograms ($512^3$ and above), check peak host/device memory requirements with `oi.estimate_memory()`:

```python
est = oi.estimate_memory((512, 512, 512), solver="mlmg", mpi_ranks=4)
print(f"Estimated per-rank peak: {est['per_rank_gb']:.2f} GB")
```

## Next Steps

- {doc}`tutorials/index` — Work through the six executed tutorial notebooks
- {doc}`user-guide/concepts` — Finite-volume discretisation, boundary conditions, and homogenisation theory
- {doc}`user-guide/solvers` — Choosing between AMReX MLMG and HYPRE Krylov/multigrid solvers
- {doc}`roadmap` — Planned `v5.1`–`v6.0` solver, packaging, and multi-physics milestones
