# GPU Acceleration

OpenImpala supports NVIDIA GPU acceleration via CUDA. All compute kernels,
flood fills, and solver loops are GPU-compatible.

## Installation

```bash
# GPU wheels are distributed via GitHub Releases due to their size (~300 MB).
pip install openimpala-cuda --find-links \
  https://github.com/BASE-Laboratory/OpenImpala/releases/latest/download/
```

The GPU wheel requires:
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA runtime libraries (typically provided by the NVIDIA driver)

The `openimpala-cuda` package is a drop-in replacement for `openimpala` — the
Python API is identical.

## Usage

No code changes are needed. The same Python scripts work on both CPU and GPU:

```python
import openimpala as oi
import numpy as np

data = np.random.choice([0, 1], size=(256, 256, 256), dtype=np.int32)

with oi.Session():
    result = oi.tortuosity(data, phase=1, direction="z")
```

When a GPU is available, AMReX automatically offloads `ParallelFor` kernels
and HYPRE solver operations to the device.

## What runs on GPU

- Phase data lookup and coefficient field construction
- Flood-fill percolation checks (atomic scatter-add)
- HYPRE matrix assembly and linear solves
- Solution extraction and flux integration
- Through-thickness profile computation
- Connected components labelling

## Performance considerations

- GPU acceleration provides the most benefit for large domains (>128^3)
- For small problems, CPU may be faster due to kernel launch overhead
- The MLMG solver currently runs on CPU only; use HYPRE solvers for GPU
- Data transfer between host and device is minimised by keeping AMReX
  data structures on the device throughout the solve
