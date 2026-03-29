---
title: 'OpenImpala: A High-Performance, Image-Based Microstructural Parameterisation Engine for Porous Media'
tags:
  - C++
  - Python
  - high-performance computing
  - porous media
  - battery modelling
  - AMReX
  - tortuosity
authors:
  - name: James Le Houx
    orcid: 0000-0002-1576-0673
    corresponding: true
    email: james.le-houx@stfc.ac.uk
    affiliation: "1, 2, 3"
  - name: Denis Kramer
    orcid: 0000-0003-0605-1047 
    affiliation: 4
affiliations:
 - name: University of Greenwich, Old Royal Naval College, Park Row, London, SE10 9LS, United Kingdom
   index: 1
 - name: ISIS Neutron & Muon Source, Rutherford Appleton Laboratory, Didcot, OX11 0QX, United Kingdom
   index: 2
 - name: The Faraday Institution, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
   index: 3
 - name: Helmut Schmidt University, Hamburg, Germany
   index: 4
date: 29 March 2026
bibliography: paper.bib
---
# Summary

`OpenImpala` is a high-performance computing (HPC) framework built upon the AMReX library, designed for the direct evaluation of effective transport properties—such as tortuosity and effective electrical conductivity—from three-dimensional microstructural imaging data (e.g., X-ray Computed Tomography). By formulating the discrete governing equations directly on Cartesian voxel grids, `OpenImpala` circumvents the computationally prohibitive mesh-generation bottlenecks typically associated with conventional finite element methods (FEM). The framework couples a highly scalable, distributed-memory C++ computational backend with a flexible Python interface, facilitating the automated parameterisation of complex porous media for downstream systems-level continuum modelling.

# Statement of Need

High-resolution three-dimensional microstructural imaging is increasingly utilized across battery research, geosciences, and materials engineering to characterize internal transport phenomena. However, the extraction of bulk effective physical parameters from billion-voxel datasets represents a significant computational challenge. Conventional finite element tools often encounter tractability limits during the mesh-generation phase when applied to complex, highly tortuous microstructures. Conversely, existing open-source voxel-based solvers, while highly accessible, generally lack the distributed-memory Message Passing Interface (MPI) scalability required to process massive, out-of-core datasets on HPC architectures. 

`OpenImpala` addresses this methodological gap by providing a natively parallelised (MPI, OpenMP, and CUDA) matrix-free AMReX backend capable of scaling across thousands of compute cores. Through its modern Python application programming interface (API), `OpenImpala` serves as a robust upstream microstructural parameterisation engine. It is capable of ingesting raw or segmented tomographic data and seamlessly exporting computed effective macroscopic properties to downstream continuum models, such as the widely adopted PyBaMM battery modelling framework.

# Software Architecture and Capabilities

`OpenImpala` is engineered to achieve exascale readiness on heterogeneous hardware, enable seamless interoperability with the broader scientific Python ecosystem, and ensure long-term software sustainability. To address the computational demands of increasingly massive tomographic datasets, the core architecture is built upon a pure C++ infrastructure utilizing native AMReX Lambdas. This design facilitates native GPU acceleration via CUDA, allowing the solver to fully leverage modern heterogeneous supercomputing architectures. Furthermore, to optimize memory utilization during massive out-of-core computations, the framework utilizes intelligent solver routing driven by AMReX’s Matrix-Free Multi-Level Multi-Grid (MLMG) infrastructure, all managed by a modern, modular CMake build system.

Beyond raw computational performance, the framework functions as an accessible, upstream parameterisation engine for modern continuum modelling workflows. Recognizing that the battery and geoscience modelling communities operate predominantly within Python, `OpenImpala` acts as a fully importable Python library (`import openimpala`) via `pybind11`. Distribution is handled using `pyproject.toml` and `cibuildwheel` to enable standard package manager installations (e.g., `pip`). This community-accessible library is supported by a comprehensive Sphinx and ReadTheDocs infrastructure, featuring extensive tutorial series and Doxygen-generated API references. 

Finally, the physical fidelity and scientific rigor of the framework are a primary focus. `OpenImpala` supports complex multi-phase computations, allowing users to map specific phase identifiers to unique transport coefficients. It also features an integrated microstructural parameterisation engine capable of extracting macroscopic metrics such as Specific Surface Area (SSA), Representative Elementary Volumes (REV), and Pore Size Distributions (PSD). To guarantee the mathematical correctness of these physics modules, rigorous software engineering practices are enforced. This includes comprehensive input validation to prevent silent numerical failures in complex HPC environments, synthetic analytical benchmarking, integrated Catch2 testing, and automated CI/CD GitHub Actions pipelines for code formatting (`clang-format`), static analysis (`clang-tidy`), and test coverage reporting via Codecov.

# Future Directions

Active development is focused on deepening integration with the broader scientific Python ecosystem. This includes the implementation of a direct memory-coupling API for PyBaMM, enabling researchers to perform 3D microstructural parameterisation and 1D electrochemical simulation in a single, zero-copy Python script.

# Acknowledgements

This work was financially supported by the EPSRC Centre for Doctoral Training (CDT) in Energy Storage and its Applications [grant ref: EP/R021295/1]; the Ada Lovelace Centre (ALC) STFC project, CANVAS-NXtomo; the EPSRC prosperity partnership with Imperial College, INFUSE [grant ref: EP/V038044/1]; the Rutherford Appleton Laboratory; The Faraday Institution through James Le Houx's Emerging Leader Fellowship [Grant No. FIELF001]; and Research England’s ‘Expanding Excellence in England’ grant at the University of Greenwich via the “Multi-scale Multi-disciplinary Modelling for Impact” (M34Impact) programme.

The authors acknowledge the use of the IRIDIS High Performance Computing Facility, Diamond Light Source's Wilson HPC cluster, STFC's SCARF cluster, and the University of Greenwich's M34Impact HPC Cluster. We also thank the developers of AMReX, HYPRE, libtiff, and HDF5, upon which OpenImpala relies.

# References
