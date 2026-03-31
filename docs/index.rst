OpenImpala Documentation
========================

**OpenImpala** is a high-performance framework for computing effective transport
properties (diffusivity, conductivity, tortuosity) directly on 3D voxel images
of porous microstructures.

It solves steady-state transport equations on the voxel grid using finite
differences, parallelised via MPI through the `AMReX <https://amrex-codes.github.io/amrex/>`_
library, with `HYPRE <https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods>`_
or AMReX MLMG for linear solves.

.. code-block:: python

   import numpy as np
   import openimpala as oi

   data = np.random.choice([0, 1], size=(64, 64, 64), dtype=np.int32)

   with oi.Session():
       result = oi.tortuosity(data, phase=1, direction="z")
       print(f"Tortuosity: {result.tortuosity:.4f}")

Install from PyPI
-----------------

.. code-block:: bash

   # CPU version
   pip install openimpala

   # GPU version (NVIDIA CUDA) — distributed via GitHub Releases
   pip install openimpala-cuda --find-links \
     https://github.com/BASE-Laboratory/OpenImpala/releases/latest/download/

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/concepts
   user-guide/solvers
   user-guide/input-files
   user-guide/gpu
   user-guide/hpc

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/python
   api/cpp

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
