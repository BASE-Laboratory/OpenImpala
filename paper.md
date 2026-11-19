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
    email: james.le-houx@gre.ac.uk
    affiliation: "1, 2, 3"
affiliations:
 - name: University of Greenwich, Old Royal Naval College, Park Row, London, SE10 9LS, United Kingdom
   index: 1
 - name: ISIS Neutron & Muon Source, Rutherford Appleton Laboratory, Didcot, OX11 0QX, United Kingdom
   index: 2
 - name: The Faraday Institution, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
   index: 3
date: 19 November 2026
bibliography: paper.bib
---

# Summary

`OpenImpala` is an open-source, high-performance computing framework for evaluating effective transport properties — tortuosity, symmetric $3\times 3$ effective diffusivity tensors, and effective conductivity — directly from three-dimensional microstructural imaging data such as X-ray computed tomography (CT) and focused ion beam scanning electron microscopy (FIB-SEM) reconstructions [@withers2021xray]. Built upon the AMReX block-structured adaptive mesh refinement library [@amrex2019], `OpenImpala` formulates steady-state conservation laws directly on the Cartesian voxel grid, eliminating the error-prone unstructured mesh-generation step that bottlenecks conventional finite element workflows. The software couples a distributed-memory C++ solver engine backed by AMReX and HYPRE [@falgout2002hypre] with an ergonomic Python interface via pybind11 [@jakob2017pybind11], enabling researchers across energy storage, geosciences, catalysis, and materials science to turn raw tomograms into continuum model parameters on hardware ranging from laptops to multi-GPU supercomputers.

An earlier prototype of the software was introduced in @LeHoux2021OpenImpala. Since that publication, the codebase has undergone a complete architectural redesign introducing native CUDA GPU acceleration, a matrix-free geometric multigrid solver (`TortuosityMLMG`), periodic homogenisation for full $3\times 3$ anisotropic diffusivity tensors, multi-phase composite transport, stereological characterisation modules, synthetic microstructure and machine-learning surrogate dataset generators, cloud-native OME-Zarr I/O, and pre-compiled Python wheels on PyPI.

# Statement of need

Modern synchrotron beamlines and laboratory micro-CT instruments routinely acquire multi-gigavoxel three-dimensional reconstructions of porous electrodes, membranes, soils, and reservoir rocks [@withers2021xray]. Translating these volumetric images into physically rigorous bulk transport descriptors — most notably the tortuosity factor [@epstein1989tortuosity; @tjaden2016origin] and the full effective diffusivity tensor $\mathbf{D}_{\text{eff}}$ — requires solving elliptic partial differential equations with hundreds of millions to billions of degrees of freedom. Researchers working with continuum-scale simulators such as PyBaMM [@sulzer2021python] need an accessible, reproducible workflow that scales seamlessly from rapid prototyping on desktop subvolumes to distributed batch campaigns on institutional HPC clusters without rewriting analysis scripts or manually meshing complex pore topologies.

# State of the field

Several established open-source packages address image-based transport modelling, each targeting a distinct point in the usability–scalability space. TauFactor [@cooper2016taufactor] popularised voxel-grid tortuosity calculation in MATLAB, and TauFactor 2 [@kench2023taufactor2] introduced fast PyTorch-based single-GPU acceleration in Python, yet both remain restricted to single-node memory capacities and axis-aligned scalar solves. PoreSpy [@gostick2019porespy] provides a rich Python toolkit for morphological image analysis alongside single-process finite-difference transport routines, while PuMA [@ferguson2018puma] offers shared-memory OpenMP C++ solvers for porous media properties; however, neither provides distributed-memory MPI domain decomposition for out-of-core multi-billion-voxel tomograms or periodic cell-problem homogenisation for off-diagonal anisotropic tensors. Extending single-process array libraries to support distributed halo exchanges and algebraic/geometric multigrid preconditioning across MPI ranks and GPUs is infeasible without a dedicated parallel grid substrate, motivating `OpenImpala`'s native integration with AMReX and HYPRE.

# Software design

`OpenImpala` is structured into three decoupled layers (\autoref{fig:architecture}): (1) an ingestion and preprocessing layer supporting TIFF, HDF5, cloud-native OME-Zarr [@moore2021omengff], and raw binary volumes — complete with built-in Otsu thresholding [@otsu1979], discrete majority-vote coarsening, and zero-copy array bridges for upstream reconstruction frameworks — partitioned across MPI ranks as distributed AMReX `iMultiFab` grids; (2) a physics and microstructure layer; and (3) an output layer exporting out-of-core AMReX plotfiles for `yt` [@turk2011yt] and ParaView alongside structured JSON aligned with BPX and BattINFO battery data standards.

![Three-stage architecture of `OpenImpala`. **Input** (left): 3D voxel images are ingested through TIFF, HDF5, OME-Zarr, RAW binary, or DAT readers (with optional Otsu thresholding and voxel coarsening) and partitioned across MPI ranks as an AMReX `iMultiFab` distributed grid. **Compute** (centre): four physics modules — tortuosity, effective diffusivity tensor via $3\times3$ homogenisation, multi-phase transport for composite electrodes, and microstructural metrics — are solved on the voxel grid using either HYPRE Krylov methods with algebraic multigrid preconditioning, or the AMReX MLMG matrix-free geometric multigrid solver, with MPI, OpenMP, and CUDA backends. **Output** (right): the tortuosity factor $\tau$, the symmetric $3\times3$ effective diffusivity tensor $\mathbf{D}_{\text{eff}}$, and microstructural descriptors (volume fraction, percolation, specific surface area) are exported as AMReX plotfiles (`yt`, ParaView), BPX / BattINFO-compatible JSON, and NumPy arrays through the Python interface, providing direct parameterisation for downstream continuum-scale simulators such as PyBaMM.\label{fig:architecture}](Overview.png)

A central design choice is offering dual complementary linear solver backends over the same seven-point finite-volume discretisation ($\nabla \cdot (D \nabla \phi) = 0$ with exact exterior-face harmonic Dirichlet inlet/outlet conditions and zero-flux lateral boundaries): **AMReX MLMG**, a matrix-free geometric multigrid solver that avoids assembling explicit sparse matrices — cutting peak RAM/VRAM footprint by roughly $3\times$ as the default fast path on both CPUs and GPUs — and **HYPRE** [@falgout2002hypre] (`PCG`, `FlexGMRES`, `BiCGSTAB` with `SMG`/`PFMG` preconditioning), which handles high-contrast multi-phase composites and periodic homogenisation cell problems with isolated pore pockets. To prevent platform lock-in, pre-compiled Linux `x86_64` CPU (`openimpala`) and CUDA 12 (`openimpala-cuda`) wheels are paired with an automatic pure-Python SciPy solver fallback so identical scripts execute on macOS, Windows, ARM64, and MPI-free cloud notebooks.

# Research impact statement

`OpenImpala` has enabled published synchrotron tomography workflows for statistical effective diffusivity and Representative Elementary Volume (REV) estimation at Diamond Light Source [@le2023statistical], demonstrating agreement within a few percent of experimental and numerical benchmarks while keeping computational turnaround commensurate with beamtime acquisition. Accuracy is continuously enforced in CI across over 100 C++ Catch2 tests, Python/MPI suites, and an automated Verification and Validation (V&V) suite verifying machine-precision agreement on analytical Voigt/Reuss laminates, strict compliance with 3D Hashin–Shtrikman bounds across isotropic sphere packings, and validation against digital rock physics benchmarks [@andra2013benchmarks2]. Six executable tutorial notebooks runnable directly in Google Colab and full documentation (<https://base-laboratory.github.io/OpenImpala/>) support immediate community adoption across academic and facility user programmes.

# AI usage disclosure

Generative AI coding assistants (including Anthropic's Claude and Google Gemini) were used to assist with specific software development tasks, including Fortran-to-C++ kernel translation, boilerplate test generation, and documentation formatting. The author reviewed, edited, and validated all AI-assisted outputs against analytical and numerical benchmarks, made all core architectural and scientific decisions, and assumes full responsibility for the accuracy, correctness, and originality of the codebase and manuscript.

# Acknowledgements

The author thanks Denis Kramer for supervision and initial conceptual direction during the early development of OpenImpala during the author's doctoral studies.

This work was financially supported by the EPSRC Centre for Doctoral Training in Energy Storage and its Applications [EP/R021295/1]; the Ada Lovelace Centre (STFC) project CANVAS-NXtomo; the EPSRC prosperity partnership with Imperial College, INFUSE [EP/V038044/1]; the Rutherford Appleton Laboratory; the Faraday Institution Emerging Leader Fellowship [FIELF001]; and Research England's *Expanding Excellence in England* grant at the University of Greenwich via the M34Impact programme. The author acknowledges the use of the IRIDIS HPC facility, Diamond Light Source's Wilson cluster, STFC SCARF, and the University of Greenwich M34Impact cluster, and thanks the developers of AMReX, HYPRE, libtiff, and HDF5.

# References
