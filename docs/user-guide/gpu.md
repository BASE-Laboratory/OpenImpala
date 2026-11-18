# CUDA GPU Acceleration

OpenImpala accelerates 3D PDE solves, percolation flood-fills, and microstructural reductions on NVIDIA GPUs primarily through the compiled **`openimpala-cuda`** wheel, which builds AMReX and HYPRE with native CUDA 12 device kernels.

## Installing `openimpala-cuda` (Linux `x86_64`)

```bash
pip install openimpala-cuda
```

`openimpala-cuda` is a drop-in replacement for `openimpala` (both expose `import openimpala as oi`). Install **only one** of `openimpala` or `openimpala-cuda` in a given environment.

**Hardware & driver requirements:**
- NVIDIA GPU with compute capability `sm_75` or newer (Turing T4/RTX 20-series, Ampere A100, Hopper H100, Ada L4/L40).
- Linux `x86_64` with an NVIDIA driver supporting CUDA 12 (CUDA runtime libraries are pulled automatically via PyPI `nvidia-*-cu12` wheels).

## Verifying GPU Detection

Use `oi.build_info()` to confirm that the CUDA C++ backend is active and check how many GPU devices AMReX sees:

```python
import openimpala as oi

with oi.Session():
    info = oi.build_info()
    print(info)
    # {'backend': 'cpp', 'cuda_enabled': True, 'gpu_device_count': 1, ...}
```

:::{note}
`gpu_device_count` is queried from AMReX after runtime initialisation—call `oi.build_info()` inside `with oi.Session():` (or after your first solver call) to inspect the live device count.
:::

## GPU Memory Sizing

AMReX initializes its device arena lazily (`amrex.the_arena_init_size=0`) so OpenImpala coexists cleanly with other GPU libraries on shared GPUs (such as 15 GB Colab T4 instances). Use `oi.estimate_memory()` before running large volumes:

```python
est = oi.estimate_memory((512, 512, 512), solver="mlmg")
print(f"Estimated peak memory: {est['total_gb']:.2f} GB")
```

For single-GPU runs above $384^3$, prefer `solver="mlmg"` over HYPRE structured matrices—matrix-free geometric multigrid cuts device VRAM usage by roughly $3\times$.

## Pure-Python CuPy Fallback

On platforms where `openimpala-cuda` is not installed and OpenImpala runs in pure-Python fallback mode (`info["backend"] == "pure-python"`), installing [CuPy](https://cupy.dev/) (`pip install cupy-cuda12x`) automatically offloads the pure-Python sparse Conjugate Gradient solver to the GPU. For production $256^3+$ workloads on Linux, `openimpala-cuda` is strongly preferred because multigrid preconditioning (`MLMG` / `PFMG`) scales as $O(N)$ iterations rather than $O(N^{4/3})$ for Jacobi-preconditioned CG.
