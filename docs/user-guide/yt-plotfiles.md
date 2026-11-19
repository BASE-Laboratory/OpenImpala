# Out-of-Core Plotfiles with `yt`

OpenImpala supports two ways to inspect 3D potential and homogenisation corrector fields, depending on domain size:

| Domain Scale | Recommended Approach | Memory Behaviour |
|--------------|----------------------|------------------|
| Up to ~$256^3$ ($\le 150\;\text{MB}$ per `float64` array) | `oi.tortuosity(..., return_field=True)` + NumPy / Matplotlib (see {doc}`../tutorials/01_getting_started`) | Materialises full `(Nz, Ny, Nx)` NumPy array on rank 0 |
| $512^3$ to $2048^3+$ (multi-GB distributed HPC runs) | AMReX plotfiles (`write_plotfile=True`) + [`yt`](https://yt-project.org/) / ParaView (see {doc}`../tutorials/06_visualization_yt_and_zarr`) | Parallel per-patch disk output; lazy out-of-core slice reading |

## Writing an AMReX Plotfile

Enable plotfile output directly from the high-level Python facade (works across both the MPI/CUDA AMReX backend and the pure-Python solver):

```python
import openimpala as oi

vol = oi.datasets.fetch_example("two_phase_cubic")

with oi.Session():
    tau_res = oi.tortuosity(
        vol,
        phase=1,
        direction="z",
        write_plotfile=True,
        results_path="./results",
    )
    print("Tortuosity plotfile:", tau_res.plotfile_path)

    deff_res = oi.effective_diffusivity(
        vol,
        phase=1,
        write_plotfile=True,
        results_path="./results",
    )
    print("Corrector plotfiles:", deff_res.plotfile_paths)
```

Or in a C++ `Diffusion` input file:

```ini
results.path              = ./results
tortuosity.write_plotfile = true
```

OpenImpala writes BoxLib `HyperCLaw-V1.1` plotfile directories containing a `Header` file and binary `Level_0/Cell_D_*` patch files:

| Solver | Plotfile Directory (`idir = 0, 1, 2` for `x, y, z`) | Stored `boxlib` Fields |
|--------|------------------------------------------------------|------------------------|
| `tortuosity(..., solver="mlmg")` | `<results_path>/tortuosity_mlmg_<idir>/` | `"solution_potential"`, `"phase_id"`, `"active_mask"` |
| `tortuosity(..., solver="pcg" \| "flexgmres" \| ...)` | `<results_path>/tortuosity_solution_<idir>/` | `"solution_potential"`, `"phase_id"`, `"active_mask"` |
| `effective_diffusivity(...)` | `<results_path>/effdiff_chi_dir<idir>/` | `"chi_k"`, `"active_mask_from_solver"` |

## Lazy Slicing and 1D Profiles with `yt`

Install `yt` (`pip install 'openimpala[viz]'`) to slice or profile multi-gigabyte AMReX plotfiles—`yt` reads only the `BoxArray` patches intersecting your slice plane:

```python
import yt

ds = yt.load("results/tortuosity_mlmg_2")
print("Available fields:", ds.field_list)

# Render an out-of-core slice normal to Y through the domain centre
slc = yt.SlicePlot(ds, "y", ("boxlib", "solution_potential"))
slc.set_cmap(("boxlib", "solution_potential"), "viridis")
slc.save("phi_slice_y.png")
```

To extract a fixed-resolution 2D NumPy buffer from an out-of-core slice for custom Matplotlib figures without loading the full 3D grid:

```python
import matplotlib.pyplot as plt
import numpy as np

slice_obj = ds.slice("y", ds.domain_center[1])
frb = slice_obj.to_frb(width=ds.domain_width[0], resolution=(512, 512))

phi_2d = np.array(frb[("boxlib", "solution_potential")])
mask_2d = np.array(frb[("boxlib", "active_mask")])
phi_masked = np.where(mask_2d > 0.5, phi_2d, np.nan)

fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
im = ax.imshow(phi_masked, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
fig.colorbar(im, ax=ax, label=r"Potential $\phi$")
plt.show()
```
