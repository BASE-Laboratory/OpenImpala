# Out-of-Core Plotfiles with `yt`

OpenImpala supports two ways to inspect 3D potential and corrector fields, depending on domain size:

| Domain Scale | Recommended Approach | Memory Behaviour |
|--------------|----------------------|------------------|
| Up to ~$256^3$ ($\le 150\;\text{MB}$ per `float64` array) | `oi.tortuosity(..., return_field=True)` + NumPy / Matplotlib (see {doc}`../tutorials/01_getting_started`) | Materialises full `(Nz, Ny, Nx)` NumPy array on rank 0 |
| $512^3$ to $2048^3+$ (multi-GB distributed HPC runs) | AMReX plotfiles (`write_plotfile = true`) + [`yt`](https://yt-project.org/) / ParaView | Parallel per-patch disk output; lazy out-of-core slice reading |

## Writing an AMReX Plotfile

In a C++ `Diffusion` input file, enable plotfile output:

```ini
results.path              = ./results
tortuosity.write_plotfile = true
```

Or from the low-level compiled C++ bindings in an MPI Python script without gathering the full 3D array onto rank 0:

```python
import openimpala as oi
from openimpala import _core

with oi.Session():
    _, img = oi.read_image("large_tomogram.tif", max_grid_size=128)
    solver = _core.TortuosityMLMG(
        img,
        phase_id=1,
        direction=_core.Direction.Z,
        results_path="./results",
        verbose=1,
        write_plotfile=True,
    )
    solver.solve()
```

AMReX writes a directory such as `./results/plt_tortuosity_z/` containing a `Header` file and per-rank binary `Level_0/Cell_D_*` patch files storing `"phi"` (potential) and `"phase"` (active mask).

## Lazy Slicing and 1D Profiles with `yt`

Install `yt` (`pip install openimpala[viz]`) to slice or profile multi-gigabyte AMReX plotfiles—`yt` reads only the `BoxArray` patches intersecting your slice plane:

```python
import yt

ds = yt.load("results/plt_tortuosity_z")
print("Available fields:", ds.field_list)

# Render an out-of-core slice normal to Y through the domain centre
slc = yt.SlicePlot(ds, "y", ("boxlib", "phi"))
slc.set_cmap(("boxlib", "phi"), "viridis")
slc.save("phi_slice_y.png")
```

To extract a fixed-resolution 2D NumPy buffer from an out-of-core slice for custom Matplotlib figures without loading the 3D grid:

```python
import matplotlib.pyplot as plt
import numpy as np

slice_obj = ds.slice("y", ds.domain_center[1])
frb = slice_obj.to_frb(width=ds.domain_width[0], resolution=(512, 512))

phi_2d = np.array(frb[("boxlib", "phi")])
mask_2d = np.array(frb[("boxlib", "phase")])
phi_masked = np.where(mask_2d > 0.5, phi_2d, np.nan)

fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
im = ax.imshow(phi_masked, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
fig.colorbar(im, ax=ax, label=r"Potential $\phi$")
plt.show()
```
