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
    email: james.le-houx@stfc.ac.uk
    affiliation: "1, 2, 3"
affiliations:
 - name: University of Greenwich, Old Royal Naval College, Park Row, London, SE10 9LS, United Kingdom
   index: 1
 - name: ISIS Neutron & Muon Source, Rutherford Appleton Laboratory, Didcot, OX11 0QX, United Kingdom
   index: 2
 - name: The Faraday Institution, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
   index: 3
date: 29 March 2026
bibliography: paper.bib
---

# Summary

`OpenImpala` is a high-performance computing framework for evaluating effective transport properties — tortuosity, effective diffusivity tensors, and effective conductivity — directly from three-dimensional microstructural imaging data such as X-ray CT reconstructions [@withers2021xray]. Built upon the AMReX adaptive mesh refinement library [@amrex2019], `OpenImpala` formulates the governing partial differential equations on the Cartesian voxel grid, eliminating the mesh-generation step that limits conventional finite element approaches. The framework couples a scalable, distributed-memory C++ backend with the HYPRE linear solver library [@falgout2002hypre] and a Python interface via pybind11 [@jakob2017pybind11], enabling automated microstructural parameterisation for downstream continuum modelling.

An earlier version of the software was described in @LeHoux2021OpenImpala. Since that publication the codebase has been redesigned with the following major additions: GPU acceleration via CUDA, a matrix-free geometric multigrid solver (AMReX MLMG), full effective diffusivity tensor computation via homogenisation, multi-phase transport, microstructural characterisation modules, and a high-level Python API distributed through PyPI.

# Statement of Need

High-resolution three-dimensional microstructural imaging is increasingly utilised across battery research, geosciences, and materials engineering to characterise internal transport phenomena [@withers2021xray]. Extracting bulk effective physical parameters — particularly the tortuosity factor [@epstein1989tortuosity; @tjaden2016origin] — from billion-voxel datasets represents a significant computational challenge.

Several open-source tools address this problem. TauFactor [@cooper2016taufactor] pioneered accessible tortuosity computation through an intuitive MATLAB interface and has seen widespread adoption in the battery community. However, TauFactor operates on a single compute node with an iterative relaxation scheme that does not leverage distributed-memory parallelism. PoreSpy [@gostick2019porespy] provides a comprehensive Python toolkit for morphological image analysis but does not include PDE-based transport solvers. PuMA [@ferguson2018puma] offers voxel-based effective property computation in C++ with multi-threading support, but lacks distributed-memory (MPI) scalability for out-of-core datasets.

`OpenImpala` addresses these limitations by combining MPI, OpenMP, and CUDA parallelism in a single solver backend capable of scaling across hundreds of compute cores and GPU accelerators. Through its Python API (`import openimpala`), the framework serves as an upstream microstructural parameterisation engine: it ingests raw or segmented tomographic data and exports computed effective properties to downstream continuum models such as PyBaMM [@sulzer2021python]. The methodology has been validated in synchrotron imaging workflows for statistical effective diffusivity estimation [@le2023statistical], and the AMReX-based infrastructure opens a pathway toward multi-scale battery simulation approaches [@lu2025immersed].

# Software Architecture and Capabilities

\autoref{fig:architecture} illustrates the architecture of `OpenImpala`. The framework is organised into three layers: an I/O layer that reads TIFF, HDF5, and raw binary volumetric images into AMReX distributed data structures; a physics solver layer that computes transport properties on the voxel grid; and an output layer that exports structured results in JSON format compatible with battery parameterisation standards such as BPX and BattINFO.

![High-level architecture of OpenImpala. The I/O layer reads 3D voxel images into AMReX distributed data structures. The physics layer solves steady-state diffusion equations via HYPRE (Krylov + algebraic multigrid) or AMReX MLMG (geometric multigrid). The output layer exports tortuosity, effective diffusivity tensors, and microstructural metrics in structured JSON format.\label{fig:architecture}](figure.png)

## Solver Infrastructure

The primary physics solver discretises the steady-state diffusion equation $\nabla \cdot (D \nabla \phi) = 0$ on a seven-point finite difference stencil, with Dirichlet boundary conditions at the inlet and outlet faces and zero-flux Neumann conditions on the lateral boundaries. Inter-cell face diffusivities are computed as the harmonic mean of adjacent cell values, which is physically correct for the series resistance analogy. Two solver backends are available:

- **HYPRE** [@falgout2002hypre]: Krylov solvers (PCG, FlexGMRES, BiCGSTAB) with algebraic multigrid preconditioning (SMG, PFMG). Supports CUDA-accelerated solves via HYPRE's device execution policy.
- **AMReX MLMG**: A matrix-free geometric multigrid solver that operates without explicit matrix assembly, reducing memory consumption by approximately $3\times$ compared to the HYPRE structured matrix approach.

## Transport Properties

`OpenImpala` computes the following effective transport properties:

- **Tortuosity factor**: Defined as $\tau = \varepsilon / D_{\text{eff}}$, where $\varepsilon$ is the connected volume fraction and $D_{\text{eff}}$ is the normalised effective diffusivity obtained from the flux integral across the domain.
- **Effective diffusivity tensor**: The full $3\times 3$ symmetric tensor $\mathbf{D}_{\text{eff}}$ is computed via a homogenisation cell problem $\nabla \cdot (D \nabla \chi_j) = -\nabla \cdot (D \hat{e}_j)$, solved independently for each Cartesian direction $j \in \{x, y, z\}$.
- **Multi-phase transport**: Arbitrary numbers of solid and pore phases can be assigned distinct transport coefficients, enabling simulation of composite electrode microstructures with heterogeneous material properties.

## Microstructural Characterisation

Beyond transport properties, the framework includes modules for:

- **Volume fraction** and **percolation checking** via parallel flood-fill with MPI ghost-cell exchange.
- **Connected component labelling** for identifying distinct pore or particle clusters.
- **Particle size distribution** computed from equivalent sphere radii of labelled connected components.
- **Specific surface area** via voxel face counting with Cauchy–Crofton stereological correction.
- **Through-thickness profiles** of phase fraction along any Cartesian direction.
- **Representative elementary volume (REV) convergence studies** that extract random sub-volumes of increasing size and track tensor convergence.

## Python API and Distribution

The Python interface exposes the core solver capabilities through a high-level facade:

```python
import openimpala as oi
with oi.Session():
    result = oi.tortuosity(data, phase=0, direction="z")
    print(f"Tortuosity: {result.tortuosity:.4f}")
```

A pure-Python package is distributed via PyPI (`pip install openimpala`) with automatic GPU acceleration via CuPy when available, and compiled CUDA GPU wheels with HYPRE solvers are available via GitHub Releases (`pip install openimpala-cuda`) for HPC deployments. Interactive tutorial notebooks are provided for Google Colab, covering workflows from basic tortuosity computation to digital twin parameterisation with PyBaMM. API reference documentation, installation guides, and interactive tutorial notebooks are available at https://base-laboratory.github.io/OpenImpala/

## Testing and Quality Assurance

The test suite includes three analytical regression benchmarks — uniform block, series layers (Reuss bound), and parallel layers (Voigt bound) — each with exact discrete solutions that verify solver correctness to machine precision. Integration tests exercise the full pipeline on real tomographic data via CTest, and Catch2 unit tests cover configuration parsing and JSON output modules. Continuous integration enforces `clang-format` style checking, `clang-tidy` static analysis, and code coverage reporting via Codecov.

# Future Directions

Active development is focused on three areas. First, GPU-accelerated solves via CUDA are now available through dedicated PyPI wheels, and profiling infrastructure (AMReX TinyProfiler integration, NVIDIA Nsight Systems workflows) has been established to guide further kernel-level optimisation. Second, embedded boundary (cut-cell) methods via AMReX's EB2 infrastructure are being investigated to achieve sub-voxel geometric accuracy without mesh generation. Third, direct memory-coupling with PyBaMM [@sulzer2021python] is planned to enable researchers to perform 3D microstructural parameterisation and 1D electrochemical simulation in a single, zero-copy Python script.

# Acknowledgements
The author thanks Denis Kramer for his supervision and initial conceptual direction during the early development of the OpenImpala framework during the author's doctoral studies.

This work was financially supported by the EPSRC Centre for Doctoral Training (CDT) in Energy Storage and its Applications [grant ref: EP/R021295/1]; the Ada Lovelace Centre (ALC) STFC project, CANVAS-NXtomo; the EPSRC prosperity partnership with Imperial College, INFUSE [grant ref: EP/V038044/1]; the Rutherford Appleton Laboratory; The Faraday Institution through James Le Houx's Emerging Leader Fellowship [Grant No. FIELF001]; and Research England's 'Expanding Excellence in England' grant at the University of Greenwich via the "Multi-scale Multi-disciplinary Modelling for Impact" (M34Impact) programme.

The author acknowledge the use of the IRIDIS High Performance Computing Facility, Diamond Light Source's Wilson HPC cluster, STFC's SCARF cluster, and the University of Greenwich's M34Impact HPC Cluster. We also thank the developers of AMReX, HYPRE, libtiff, and HDF5, upon which OpenImpala relies.

# AI Usage Disclosure
Generative AI (Anthropic's Claude) was used to assist with specific software development tasks, including Fortran-to-C++ kernel translation, boilerplate test generation, and documentation formatting. The author reviewed, edited, and validated all AI-assisted outputs, made all core architectural decisions, and assumes full responsibility for the accuracy, correctness, and originality of the codebase.
