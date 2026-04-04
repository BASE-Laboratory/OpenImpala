# Getting Started

## Installation

### Python (recommended)

OpenImpala is available on PyPI — no compilation required.

```bash
pip install openimpala
```

**GPU acceleration** is automatic. If you have an NVIDIA GPU and
[CuPy](https://cupy.dev/) installed, OpenImpala detects it at runtime and
offloads compute kernels to the GPU. No separate package is needed:

```bash
# Optional: install CuPy for automatic GPU acceleration
pip install cupy-cuda12x   # match your CUDA toolkit version
```

If CuPy is not available, OpenImpala falls back to SciPy on the CPU.

**Requirements:** Python 3.8+ and NumPy. Optional: `mpi4py` for MPI parallelism.

#### Advanced / HPC: compiled HYPRE backend

For HPC clusters that need the compiled C++ HYPRE solvers, a separate package
is available:

```bash
pip install openimpala-cuda --find-links \
  https://github.com/BASE-Laboratory/OpenImpala/releases/expanded_assets/v4.0.6
```

This package bundles AMReX + HYPRE compiled with CUDA and is a drop-in
replacement for the pure-Python `openimpala` package.

### Container (HPC)

For HPC clusters, download the pre-built Apptainer/Singularity container from
[GitHub Releases](https://github.com/BASE-Laboratory/OpenImpala/releases):

```bash
# Download the latest .sif file
wget https://github.com/BASE-Laboratory/OpenImpala/releases/expanded_assets/v4.0.6openimpala-v4.0.0.sif

# Run interactively
apptainer shell openimpala-v4.0.0.sif

# Run a simulation
apptainer exec openimpala-v4.0.0.sif /opt/OpenImpala/build/Diffusion3d inputs
```

### From source (developers)

```bash
git clone https://github.com/BASE-Laboratory/OpenImpala.git
cd OpenImpala
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_COMPILER=$(which mpicxx) \
         -DCMAKE_Fortran_COMPILER=$(which mpif90)
make -j$(nproc)
ctest --output-on-failure
```

Dependencies: AMReX, HYPRE, HDF5, LibTIFF. See the
[README](https://github.com/BASE-Laboratory/OpenImpala#native-installation-advanced)
for full details.

## Your first simulation

```python
import numpy as np
import openimpala as oi

# Create a simple porous medium (random 50/50 mix)
data = np.random.choice([0, 1], size=(64, 64, 64), dtype=np.int32)

with oi.Session():
    # Volume fraction
    vf = oi.volume_fraction(data, phase=1)
    print(f"Volume fraction: {vf.fraction:.4f}")

    # Percolation check
    perc = oi.percolation_check(data, phase=1, direction="z")
    print(f"Percolates: {perc.percolates}")

    # Tortuosity (only if phase percolates)
    if perc.percolates:
        result = oi.tortuosity(data, phase=1, direction="z")
        print(f"Tortuosity: {result.tortuosity:.4f}")
```

All computation happens inside the `oi.Session()` context manager, which
manages the AMReX and MPI lifecycle.

## Working with real images

OpenImpala reads TIFF stacks, HDF5, and raw binary files:

```python
import openimpala as oi

with oi.Session():
    reader, img = oi.read_image("sample.tiff", threshold=128)
    result = oi.tortuosity(img, phase=1, direction="z")
```

## Memory-safe workflows

For large datasets, free the Python array before solving:

```python
import gc
import numpy as np
import openimpala as oi

with oi.Session():
    arr = np.load("large_volume.npy")
    dataset = oi.core.VoxelImage.from_numpy(arr)

    del arr
    gc.collect()  # Free Python memory

    result = oi.tortuosity(dataset, phase=1, direction="z")
```

## Next steps

- {doc}`user-guide/concepts` — Understand tortuosity, effective diffusivity, and the mathematics
- {doc}`user-guide/solvers` — Choose the right solver for your problem
- {doc}`tutorials/index` — Interactive Colab notebooks
