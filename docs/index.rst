OpenImpala
==========

**OpenImpala** computes effective transport properties—tortuosity factors, symmetric :math:`3 \times 3` effective diffusivity tensors, multi-phase effective conductivities, and stereological microstructure metrics—directly on 3D segmented voxel images from X-ray CT, FIB-SEM, or synthetic reconstructions.

Built on `AMReX <https://amrex-codes.github.io/amrex/>`_ and `HYPRE <https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods>`_, OpenImpala discretises steady-state conservation laws directly on the Cartesian voxel grid without unstructured mesh generation, scaling from interactive Python notebooks on a laptop to multi-node MPI + CUDA clusters.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting-started
      :link-type: doc

      Install pre-built wheels on Linux (CPU or CUDA 12), macOS, or Windows, load a 3D TIFF stack, and run your first tortuosity solve.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Six executed notebooks with full derivations, plots, and Google Colab launchers—from basic diffusion to :math:`3\times 3` homogenisation, REV convergence, and OME-Zarr / ``yt`` plotfiles.

   .. grid-item-card:: User Guide
      :link: user-guide/index
      :link-type: doc

      Finite-volume discretisation, solver selection (AMReX MLMG vs. HYPRE), CUDA GPUs, MPI/SLURM clusters, ``ParmParse`` inputs, and ``yt`` plotfiles.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete documentation for the ``openimpala`` Python facade, ``openimpala.datasets``, ``openimpala.synthetic``, result dataclasses, and the underlying C++ classes.

Quick Install & Example
-----------------------

.. tab-set::

   .. tab-item:: Linux / macOS / Windows (CPU)

      .. code-block:: bash

         pip install openimpala

      On Linux ``x86_64``, ``openimpala`` installs compiled AMReX + HYPRE C++ bindings (install OpenMPI for multi-rank execution). On macOS, Windows, and ARM64, it uses the built-in pure-Python SciPy solver with the identical API.

   .. tab-item:: Linux (CUDA 12 GPU)

      .. code-block:: bash

         pip install openimpala-cuda

      Drop-in GPU package with AMReX ``ParallelFor`` and HYPRE structured multigrid compiled natively for NVIDIA Volta+ GPUs (``sm_75`` and newer).

   .. tab-item:: HPC Container (Apptainer)

      .. code-block:: bash

         apptainer exec -B "$(pwd):/data" openimpala-v4.7.0.sif \
             /usr/bin/Diffusion /data/inputs

.. code-block:: python

   import openimpala as oi

   vol = oi.datasets.two_phase_cubic()          # (64, 64, 64) segmented porous medium
   res = oi.tortuosity(vol, phase=1, direction="z")
   print(f"Porosity: {res.active_volume_fraction:.4f}, Tortuosity: {res.tortuosity:.4f}")

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   tutorials/index
   user-guide/index
   api/index
   contributing
   changelog
