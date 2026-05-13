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
  - name: James Le Houx
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
date: 12 May 2026
bibliography: paper.bib
---

# Summary

`OpenImpala` is a high-performance computing framework for evaluating effective transport properties — tortuosity, effective diffusivity tensors, and effective conductivity — directly from three-dimensional microstructural imaging data such as X-ray CT reconstructions [@withers2021xray]. Built upon the AMReX adaptive mesh refinement library [@amrex2019], `OpenImpala` formulates the governing partial differential equations on the Cartesian voxel grid, eliminating the mesh-generation step that limits conventional finite element approaches. The framework couples a scalable, distributed-memory C++ backend with the HYPRE linear solver library [@falgout2002hypre] and a Python interface via pybind11 [@jakob2017pybind11], enabling automated microstructural parameterisation for downstream continuum modelling.

An earlier version of the software was described in @LeHoux2021OpenImpala. Since that publication the codebase has been redesigned with the following major additions: GPU acceleration via CUDA, a matrix-free geometric multigrid solver (AMReX MLMG), full effective diffusivity tensor computation via homogenisation, multi-phase transport, microstructural characterisation modules, and a high-level Python API distributed through PyPI.

# Statement of Need

High-resolution three-dimensional microstructural imaging is increasingly used across battery research, geosciences, and materials engineering to characterise internal transport phenomena [@withers2021xray]. Extracting bulk effective parameters — particularly the tortuosity factor [@epstein1989tortuosity; @tjaden2016origin] — from billion-voxel datasets is a significant computational challenge.

Several open-source tools address this problem. TauFactor [@cooper2016taufactor] pioneered accessible tortuosity computation in MATLAB and has seen widespread adoption in the battery community; TauFactor 2 [@kench2023taufactor2] subsequently added single-GPU PyTorch acceleration but remains single-node. PoreSpy [@gostick2019porespy] provides a comprehensive Python toolkit for morphological image analysis and includes single-process finite-difference transport simulations, but no MPI-distributed solvers for out-of-core volumes. PuMA [@ferguson2018puma] offers voxel-based effective property computation in C++ with multi-threading, but lacks distributed-memory MPI scalability.

`OpenImpala` addresses these limitations by combining MPI, OpenMP, and CUDA parallelism in a single solver backend capable of scaling across hundreds of compute cores and GPU accelerators. Through its Python API (`import openimpala`), the framework serves as an upstream microstructural parameterisation engine: it ingests raw or segmented tomographic data and exports effective properties to downstream continuum models such as PyBaMM [@sulzer2021python]. The methodology has been validated against synchrotron data for statistical effective diffusivity estimation [@le2023statistical], with computed tortuosity factors agreeing to within a few percent of published benchmark values.

# Software Architecture and Capabilities

`OpenImpala` is organised into three layers (\autoref{fig:architecture}): an I/O layer that reads TIFF, HDF5, and raw binary volumetric images into AMReX distributed data structures; a physics solver layer; and an output layer that exports results in structured JSON formats compatible with battery parameterisation standards such as BPX and BattINFO.

![Three-stage architecture of `OpenImpala`. **Input** (left): 3D voxel images are ingested through TIFF, HDF5, RAW binary, or DAT readers and partitioned across MPI ranks as an AMReX `iMultiFab` distributed grid. **Compute** (centre): four physics modules — tortuosity, effective diffusivity tensor via $3\times3$ homogenisation, multi-phase transport for composite electrodes, and microstructural metrics — are solved on the voxel grid using either HYPRE Krylov methods with algebraic multigrid preconditioning, or the AMReX MLMG matrix-free geometric multigrid solver, with MPI, OpenMP, and CUDA backends. **Output** (right): the tortuosity factor $\tau$, the symmetric $3\times3$ effective diffusivity tensor $\mathbf{D}_{\text{eff}}$, and microstructural descriptors (volume fraction, percolation, specific surface area) are exported as BPX / BattINFO-compatible JSON and as NumPy arrays through the `import openimpala` Python interface, providing direct parameterisation for downstream continuum-scale simulators such as PyBaMM.\label{fig:architecture}](Overview.png)

The primary solver discretises the steady-state diffusion equation $\nabla \cdot (D \nabla \phi) = 0$ on a seven-point finite-difference stencil with harmonic-mean face diffusivities, Dirichlet inlet/outlet conditions, and zero-flux Neumann lateral boundaries. Two backends are available: **HYPRE** [@falgout2002hypre] (PCG / FlexGMRES / BiCGSTAB with SMG or PFMG multigrid preconditioning, CUDA-accelerated) and **AMReX MLMG**, a matrix-free geometric multigrid solver that reduces memory consumption by approximately $3\times$ over the HYPRE structured matrix approach and is typically the fastest option on GPU hardware.

Transport properties computed include the tortuosity factor $\tau = \varepsilon / D_{\text{eff}}$, the full $3\times 3$ symmetric effective diffusivity tensor $\mathbf{D}_{\text{eff}}$ via a homogenisation cell problem solved independently along each Cartesian direction, and multi-phase transport with arbitrary per-phase transport coefficients for composite electrode microstructures. The framework additionally provides microstructural characterisation (volume fraction, percolation, connected components, particle size distribution, specific surface area via Cauchy–Crofton stereology, through-thickness profiles, and REV convergence studies).

The Python interface exposes the core capabilities through a high-level facade:

```python
import openimpala as oi
with oi.Session():
    result = oi.tortuosity(data, phase=0, direction="z")
    print(f"Tortuosity: {result.tortuosity:.4f}")
```

A pure-Python package with optional CuPy GPU acceleration (`pip install openimpala`) and a compiled CUDA wheel with HYPRE/AMReX solvers (`pip install openimpala-cuda`) are distributed via PyPI. Interactive tutorial notebooks for Google Colab cover workflows from basic tortuosity computation to digital twin parameterisation with PyBaMM. Documentation is at <https://base-laboratory.github.io/OpenImpala/>. The test suite includes analytical regression benchmarks (uniform block, series layers / Reuss bound, parallel layers / Voigt bound) verified to machine precision, plus integration tests on real tomographic data via CTest; CI enforces `clang-format`, `clang-tidy`, and code coverage via Codecov.

# Acknowledgements

The author thanks Denis Kramer for supervision and initial conceptual direction during the early development of OpenImpala during the author's doctoral studies.

This work was financially supported by the EPSRC Centre for Doctoral Training in Energy Storage and its Applications [EP/R021295/1]; the Ada Lovelace Centre (STFC) project CANVAS-NXtomo; the EPSRC prosperity partnership with Imperial College, INFUSE [EP/V038044/1]; the Rutherford Appleton Laboratory; the Faraday Institution Emerging Leader Fellowship [FIELF001]; and Research England's *Expanding Excellence in England* grant at the University of Greenwich via the M34Impact programme. The author acknowledges the use of the IRIDIS HPC facility, Diamond Light Source's Wilson cluster, STFC SCARF, and the University of Greenwich M34Impact cluster, and thanks the developers of AMReX, HYPRE, libtiff, and HDF5.

# AI Usage Disclosure

Generative AI (Anthropic's Claude) was used to assist with specific software development tasks, including Fortran-to-C++ kernel translation, boilerplate test generation, and documentation formatting. The author reviewed, edited, and validated all AI-assisted outputs, made all core architectural decisions, and assumes full responsibility for the accuracy, correctness, and originality of the codebase.
