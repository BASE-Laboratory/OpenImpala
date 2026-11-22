---
title: 'OpenImpala: Scalable Transport Property Computation on 3D Voxel Images of Porous Media'
tags:
  - C++
  - Python
  - high-performance computing
  - porous media
  - battery modelling
  - AMReX
  - tortuosity
  - effective diffusivity
authors:
  - given-names: James
    surname: Le Houx
    orcid: 0000-0002-1576-0673
    corresponding: true
    email: j.lehoux@greenwich.ac.uk
    affiliation: "1, 2, 3"
affiliations:
 - name: University of Greenwich, Old Royal Naval College, Park Row, London, SE10 9LS, United Kingdom
   index: 1
 - name: ISIS Neutron & Muon Source, Rutherford Appleton Laboratory, Didcot, OX11 0QX, United Kingdom
   index: 2
 - name: The Faraday Institution, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
   index: 3
date: 20 November 2026
bibliography: paper.bib
---

# Summary

`OpenImpala` is an open-source, high-performance computing framework for evaluating effective transport properties — tortuosity, symmetric $3\times 3$ effective diffusivity tensors, and effective conductivity — directly from three-dimensional microstructural imaging data such as X-ray computed tomography (CT) and focused ion beam scanning electron microscopy (FIB-SEM) reconstructions [@withers2021xray]. Built upon the AMReX block-structured adaptive mesh refinement library [@amrex2019], `OpenImpala` formulates steady-state conservation laws directly on the Cartesian voxel grid, eliminating the error-prone unstructured mesh-generation step that bottlenecks conventional finite element workflows. The software couples a distributed-memory C++17 solver engine backed by AMReX and HYPRE [@falgout2002hypre] with an ergonomic Python interface via pybind11 [@jakob2017pybind11], enabling researchers across energy storage, geosciences, catalysis, and materials science to translate raw tomograms into continuum model parameters on hardware ranging from laptops to multi-node GPU supercomputers.

An initial CPU-only C++/Fortran command-line prototype (`v1.0`, scalar axis-aligned tortuosity on TIFF/DAT volumes) was described in @LeHoux2021OpenImpala. Version 5 (`v5.0`–`v5.1`) represents a complete ground-up architectural rewrite constituting substantial new work: all legacy Fortran 90 kernels have been replaced by portable C++17 AMReX `ParallelFor` CPU/GPU kernels with native NVIDIA CUDA acceleration; a matrix-free geometric multigrid solver (`TortuosityMLMG`) cuts peak solver memory by $3.2\times$–$4.4\times$; periodic cell-problem homogenisation computes full $3\times 3$ anisotropic diffusivity tensors alongside multi-phase composite transport; a full stereological characterisation suite, synthetic microstructure and machine-learning surrogate dataset generators, zero-dependency `VolumeProtocol` and cloud-native OME-Zarr v0.4 (`labels/phase`) [@moore2021omengff] plus Core Imaging Library [@jorgensen2021cil] bridges, and pre-compiled PyPI wheels (`openimpala`, `openimpala-cuda`) with a pure-Python fallback now make the entire workflow accessible without source compilation.

# Statement of need

Modern synchrotron beamlines and laboratory micro-CT instruments routinely acquire multi-gigavoxel three-dimensional reconstructions of porous battery electrodes, membranes, soils, and reservoir rocks [@withers2021xray; @LeHoux2020Tomography]. Translating these volumetric images into physically rigorous bulk transport descriptors — most notably the tortuosity factor [@epstein1989tortuosity; @tjaden2016origin] and the full effective diffusivity tensor $\mathbf{D}_{\text{eff}}$ — requires solving elliptic partial differential equations across tens of millions to billions of voxel degrees of freedom. Researchers parameterising continuum-scale electrochemical simulators such as PyBaMM [@sulzer2021python] or guiding live synchrotron imaging campaigns need a single reproducible package that runs identically on desktop subvolumes, cloud notebooks, and distributed institutional HPC clusters without manual meshing.

# State of the field

Several open-source packages address image-based transport modelling at different points in the usability–scalability space. TauFactor [@cooper2016taufactor] popularised voxel-grid tortuosity calculation in MATLAB, and TauFactor 2 [@kench2023taufactor2] introduced fast PyTorch single-GPU acceleration in Python, yet both remain bounded by single-device memory and axis-aligned scalar solves. PoreSpy [@gostick2019porespy] provides rich morphological analysis alongside single-process finite-difference transport routines, while PuMA [@ferguson2018puma] offers shared-memory OpenMP C++ solvers; however, neither supports distributed-memory MPI domain decomposition for out-of-core multi-hundred-million-voxel tomograms or periodic homogenisation for off-diagonal anisotropic tensors. Extending single-process array tools to distributed halo exchanges and algebraic/geometric multigrid preconditioning across MPI ranks and GPUs requires a parallel grid substrate, motivating `OpenImpala`'s native integration with AMReX and HYPRE.

# Software design

`OpenImpala` is structured into three decoupled layers (\autoref{fig:architecture}): (1) an ingestion and preprocessing layer supporting TIFF, HDF5, OME-Zarr [@moore2021omengff], raw binary volumes, Otsu thresholding [@otsu1979], majority-vote coarsening, and zero-copy bridges for upstream reconstruction frameworks such as CIL [@jorgensen2021cil], partitioned across MPI ranks as distributed AMReX `iMultiFab` grids; (2) a physics and microstructure layer; and (3) an output layer exporting out-of-core AMReX plotfiles for `yt` [@turk2011yt] and ParaView alongside structured JSON aligned with BPX and BattINFO battery data standards.

![Three-stage architecture of `OpenImpala`. **Input** (left): 3D voxel images are ingested through TIFF, HDF5, OME-Zarr, RAW, or CIL/NumPy handles (with optional Otsu thresholding and discrete coarsening) and partitioned across MPI ranks as an AMReX `iMultiFab` distributed grid. **Compute** (centre): tortuosity, $3\times 3$ periodic homogenisation, multi-phase composite transport, and stereological metrics are evaluated using matrix-free AMReX MLMG or HYPRE Krylov/multigrid solvers across MPI, OpenMP, and CUDA backends. **Output** (right): $\tau$, $\mathbf{D}_{\text{eff}}$, and microstructural descriptors are exported as AMReX plotfiles (`yt`, ParaView), BPX/BattINFO JSON, and Python dataclasses for downstream simulators such as PyBaMM.\label{fig:architecture}](Overview.png)

A central design choice is pairing dual linear solver families over the same seven-point finite-volume discretisation ($\nabla \cdot (D \nabla \phi) = 0$ with exact exterior-face harmonic Dirichlet boundary conditions): **AMReX MLMG**, a matrix-free geometric multigrid V-cycle solver that avoids assembling explicit sparse matrices, and **HYPRE** [@falgout2002hypre] (`PCG`, `FlexGMRES`, `BiCGSTAB` with `PFMG`/`SMG` preconditioning) for extreme-contrast composites and periodic homogenisation cell problems with per-component nullspace pinning. As measured across isotropic Gaussian random porous media ($\phi = 0.50$, \autoref{fig:scaling} and \autoref{tab:memory}), matrix-free `MLMG` exhibits grid-independent iteration counts ($11$–$18$ V-cycles from $32^3$ to $256^3$), near-linear $\mathcal{O}(N_{\text{vox}})$ runtime scaling, and $3.2\times$–$4.4\times$ lower peak memory than HYPRE `FlexGMRES` with `PFMG`/`SMG` (`~304`–`319` B/voxel vs `1,016`–`1,390` B/voxel), solving $256^3$ ($1.68\times 10^7$ voxels) inside $4.86\text{ GB}$ RAM where explicit Krylov–multigrid solvers exceed $16\text{ GB}$. Distributed MPI strong scaling across multi-node clusters up to $600^3$ ($2.16\times 10^8$ voxels) is preserved via AMReX domain decomposition [@LeHoux2021OpenImpala], while pre-compiled Linux CPU and CUDA 12 wheels automatically discover system or `pip`-installed OpenMPI and warn explicitly if falling back to the pure-Python SciPy solver.

![Empirical algorithmic scaling and memory footprint across $32^3$ ($3.3\times 10^4$ voxels) to $256^3$ ($1.68\times 10^7$ voxels) porous media ($\phi = 0.50$). **(a)** Matrix-free `AMReX MLMG` scales linearly ($\mathcal{O}(N_{\mathrm{vox}})$), outperforming `SciPy CG` by $8.6\times$ and `HYPRE FlexGMRES+SMG` by $16.5\times$ at $128^3$. **(b)** Measured peak solver memory ($\Delta\mathrm{VmHWM}$) confirms a $3.2\times$–$4.4\times$ reduction relative to `FlexGMRES+PFMG`/`SMG`, enabling $256^3$ solves within $4.86\text{ GB}$.\label{fig:scaling}](scaling_and_memory.png)

| Solver Backend | $128^3$ Time | $128^3$ RAM | B / voxel | $256^3$ RAM |
| :--- | ---: | ---: | ---: | ---: |
| **AMReX MLMG (matrix-free)** | **4.06 s** | **638 MB** | **319** | **4,856 MB** |
| HYPRE `PCG + PFMG` | 21.91 s | 710 MB | 355 | >6,000 MB |
| HYPRE `FlexGMRES + PFMG` | 33.34 s | 2,031 MB | 1,016 | >16,000 MB |
| HYPRE `FlexGMRES + SMG` | 66.87 s | 2,779 MB | 1,390 | >22,000 MB |
| Pure-Python (`SciPy CG`) | 34.89 s | 699 MB | 350 | — |

: Measured wall-clock solve time and peak solver memory ($\Delta\text{VmHWM}$) on 3D Gaussian random porous media ($\phi = 0.50$, two x86\_64 OpenMP threads; solutions agree to $|\Delta\tau| < 4\times 10^{-8}$; at $256^3$, `MLMG` uses $304$ B/voxel).\label{tab:memory}

# Research impact statement

`OpenImpala` has delivered direct research impact across synchrotron beamline operations, multi-scale battery modelling, and digital porous media workflows:

- **Live synchrotron tomography pipelines:** Deployed within the on-site semi-automated processing architecture at Diamond Light Source's Dual Imaging and Diffraction (DIAD / K11) beamline and validated on synchrotron micro-CT reconstructions from the TOMCAT beamline at the Paul Scherrer Institute [@le2023statistical], `OpenImpala` reduced 3D homogenisation cell-problem turnaround on 40 CPU cores (IRIDIS5) from $102\text{ min}$ in commercial finite-element software (COMSOL Multiphysics) to $15\text{ s}$ — a $>400\times$ speedup with $<2\%$ homogenisation error — enabling experimentalists to evaluate Representative Elementary Volume (REV) convergence while a sample is still on the beamstage.
- **Facility & battery modelling ecosystems:** Developed through the STFC Ada Lovelace Centre (`CANVAS-NXtomo`) at Rutherford Appleton Laboratory and the Faraday Institution, `OpenImpala` bridges tomographic reconstruction in CIL [@jorgensen2021cil] and OME-Zarr directly to continuum battery simulation in PyBaMM [@sulzer2021python] for lithium-ion electrode characterisation [@LeHoux2020Tomography] and digital rock physics sandstone benchmarks [@andra2013benchmarks2].
- **Verified community distribution:** Over 125 Python/MPI tests, 100+ C++ Catch2 assertions, and automated CI wheel verification enforce exact analytical Voigt/Reuss laminates, 3D Hashin–Shtrikman bounds, and cross-backend parity (`<1e-8`) on every release, paired with six Google Colab tutorials (<https://base-laboratory.github.io/OpenImpala/>) and a synthetic ML dataset generator (`openimpala generate-dataset`) for training 3D surrogate models.

# AI usage disclosure

Generative AI coding assistants (including Anthropic's Claude and Google Gemini) were used to assist with specific software development tasks, including Fortran-to-C++ kernel translation, boilerplate test generation, and documentation formatting. The author reviewed, edited, and validated all AI-assisted outputs against analytical and numerical benchmarks, made all core architectural and scientific decisions, and assumes full responsibility for the accuracy, correctness, and originality of the codebase and manuscript.

# Acknowledgements

The author thanks Denis Kramer for supervision and initial conceptual direction during the early development of OpenImpala during the author's doctoral studies.

This work was financially supported by the EPSRC Centre for Doctoral Training in Energy Storage and its Applications [EP/R021295/1]; the Ada Lovelace Centre (STFC) project CANVAS-NXtomo; the EPSRC prosperity partnership with Imperial College, INFUSE [EP/V038044/1]; the Rutherford Appleton Laboratory; the Faraday Institution Emerging Leader Fellowship [FIELF001]; and Research England's *Expanding Excellence in England* grant at the University of Greenwich via the M34Impact programme. The author acknowledges the use of the IRIDIS HPC facility, Diamond Light Source's Wilson cluster, STFC SCARF, and the University of Greenwich M34Impact cluster, and thanks the developers of AMReX, HYPRE, libtiff, and HDF5.

# References
