# GPU Acceleration

OpenImpala automatically accelerates computations on NVIDIA GPUs when
[CuPy](https://cupy.dev/) is available. No code changes or separate packages
are required for the default path.

## Quick setup

```bash
pip install openimpala

# Install CuPy for your CUDA toolkit version
pip install cupy-cuda12x   # CUDA 12.x
# or
pip install cupy-cuda11x   # CUDA 11.x
```

When CuPy is detected, OpenImpala offloads solver kernels, flood fills, and
flux integrations to the GPU. If CuPy is not installed, it falls back to
SciPy on the CPU transparently.

## Verifying GPU support

```python
import openimpala as oi

print(oi.backend())  # "cupy" if GPU is active, "scipy" otherwise
```

## Usage

No code changes are needed. The same Python scripts work on both CPU and GPU:

```python
import openimpala as oi
import numpy as np

data = np.random.choice([0, 1], size=(256, 256, 256), dtype=np.int32)

with oi.Session():
    result = oi.tortuosity(data, phase=1, direction="z")
```

## What runs on GPU

- Phase data lookup and coefficient field construction
- Flood-fill percolation checks
- Linear solver iterations
- Solution extraction and flux integration
- Through-thickness profile computation
- Connected components labelling

## Performance considerations

- GPU acceleration provides the most benefit for large domains (>128^3)
- For small problems, CPU may be faster due to kernel launch overhead
- Data stays on the device throughout the solve to minimise transfers

## Advanced / HPC: openimpala-cuda

For HPC clusters that need the compiled C++ HYPRE linear solvers with native
CUDA support (AMReX + HYPRE compiled with CUDA), a separate package is
available on PyPI:

```bash
pip install openimpala-cuda
```

The `openimpala-cuda` package is a drop-in replacement for `openimpala` and
provides:

- HYPRE matrix assembly and linear solves on the GPU
- AMReX `ParallelFor` kernel offloading
- MPI + CUDA multi-GPU support

**Requirements for openimpala-cuda:**
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA runtime libraries (typically provided by the NVIDIA driver)
