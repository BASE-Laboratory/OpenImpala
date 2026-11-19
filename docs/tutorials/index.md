# Tutorials

Each tutorial below is executed automatically during documentation builds—all tables, diagnostics, and Matplotlib figures reflect live `openimpala` outputs. Click any card to read the rendered notebook with full equations and outputs, or launch it interactively in Google Colab.

:::::{grid} 1 2 2 2
:gutter: 3

::::{grid-item-card} 01 — Steady-State Diffusion & 3D Fields
:link: 01_getting_started
:link-type: doc

Verify exact analytical slab transport ($\tau = 1$), load a 3D TIFF tomogram, check percolation, solve $\nabla \cdot (D_0 \nabla \phi) = 0$, and inspect 3D potential and local flux bottleneck maps.

:::{card-footer}
`tortuosity` · `volume_fraction` · `percolation_check` · `read_image`
:::
::::

::::{grid-item-card} 02 — Microstructure Metrics & Non-Cubic Voxels
:link: 02_microstructure_and_anisotropy
:link-type: doc

Verify the $2/3$ Cauchy–Crofton specific surface area correction against analytical spheres, extract connected components and particle radii, and evaluate directional tortuosity on non-cubic FIB-SEM voxels.

:::{card-footer}
`specific_surface_area` · `connected_components` · `particle_size_distribution` · `through_thickness_profile` · `tortuosity_all`
:::
::::

::::{grid-item-card} 03 — Periodic Homogenisation & $3\times 3$ $\mathbf{D}_{\text{eff}}$
:link: 03_homogenisation_tensor
:link-type: doc

Solve the periodic corrector cell problems on a tilted laminate to recover off-diagonal coupling ($D_{xz} = D_{zx}$) and principal transport axes, and compare periodic homogenisation with Dirichlet flow-through.

:::{card-footer}
`effective_diffusivity` · `corrector_fields` · `tortuosity_all`
:::
::::

::::{grid-item-card} 04 — Multi-Phase Transport in Composite Media
:link: 04_multiphase_transport
:link-type: doc

Benchmark harmonic interface conductances against exact Voigt and Reuss bounds, then model ionic bridging across nanoporous binder shells when primary macropores lose percolation.

:::{card-footer}
`phase_diffusivities` · `tortuosity` · `return_field=True`
:::
::::

::::{grid-item-card} 05 — Representative Elementary Volume (REV)
:link: 05_rev_and_uncertainty
:link-type: doc

Quantify scale convergence and spatial uncertainty across cubic subvolume windows $L \in [16, 64]$ voxels with bootstrap 95% confidence intervals and the 5% coefficient-of-variation criterion.

:::{card-footer}
`rev_study` · `REVStudyResult`
:::
::::

::::{grid-item-card} 06 — Cloud-Native OME-Zarr & Out-of-Core `yt` Plotfiles
:link: 06_visualization_yt_and_zarr
:link-type: doc

Export chunked OME-NGFF v0.4 Zarr tomograms with physical scale metadata, write AMReX `HyperCLaw-V1.1` plotfiles directly from `tortuosity` and `effective_diffusivity`, and slice fields lazily with `yt`.

:::{card-footer}
`write_zarr_volume` · `VoxelImage.from_zarr` · `write_plotfile=True` · `yt.load`
:::
::::

:::::

## Running Locally or on Google Colab

| # | Notebook | Colab | Source |
|---|----------|-------|--------|
| 01 | {doc}`01_getting_started` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/OpenImpala/blob/master/tutorials/01_getting_started.ipynb) | [`01_getting_started.ipynb`](https://github.com/BASE-Laboratory/OpenImpala/blob/master/tutorials/01_getting_started.ipynb) |
| 02 | {doc}`02_microstructure_and_anisotropy` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/OpenImpala/blob/master/tutorials/02_microstructure_and_anisotropy.ipynb) | [`02_microstructure_and_anisotropy.ipynb`](https://github.com/BASE-Laboratory/OpenImpala/blob/master/tutorials/02_microstructure_and_anisotropy.ipynb) |
| 03 | {doc}`03_homogenisation_tensor` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/OpenImpala/blob/master/tutorials/03_homogenisation_tensor.ipynb) | [`03_homogenisation_tensor.ipynb`](https://github.com/BASE-Laboratory/OpenImpala/blob/master/tutorials/03_homogenisation_tensor.ipynb) |
| 04 | {doc}`04_multiphase_transport` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/OpenImpala/blob/master/tutorials/04_multiphase_transport.ipynb) | [`04_multiphase_transport.ipynb`](https://github.com/BASE-Laboratory/OpenImpala/blob/master/tutorials/04_multiphase_transport.ipynb) |
| 05 | {doc}`05_rev_and_uncertainty` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/OpenImpala/blob/master/tutorials/05_rev_and_uncertainty.ipynb) | [`05_rev_and_uncertainty.ipynb`](https://github.com/BASE-Laboratory/OpenImpala/blob/master/tutorials/05_rev_and_uncertainty.ipynb) |
| 06 | {doc}`06_visualization_yt_and_zarr` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/OpenImpala/blob/master/tutorials/06_visualization_yt_and_zarr.ipynb) | [`06_visualization_yt_and_zarr.ipynb`](https://github.com/BASE-Laboratory/OpenImpala/blob/master/tutorials/06_visualization_yt_and_zarr.ipynb) |

To run the notebooks locally:

```bash
pip install "openimpala[zarr,viz]" jupyter
jupyter lab tutorials/
```

```{toctree}
:hidden:
:maxdepth: 1

01_getting_started
02_microstructure_and_anisotropy
03_homogenisation_tensor
04_multiphase_transport
05_rev_and_uncertainty
06_visualization_yt_and_zarr
```
