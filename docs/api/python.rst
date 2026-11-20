Python API Reference
====================

High-level NumPy-native interface for computing transport and microstructural
properties on segmented 3D voxel images. All functions listed below are
re-exported at the top level of ``openimpala`` (e.g. ``oi.tortuosity(...)``)
and operate identically across the compiled AMReX/HYPRE C++ backend (CPU and
CUDA) and the pure-Python SciPy/CuPy solver.


Transport & Phase Analysis
--------------------------

.. autofunction:: openimpala.facade.volume_fraction

.. autofunction:: openimpala.facade.percolation_check

.. autofunction:: openimpala.facade.tortuosity

.. autofunction:: openimpala.facade.tortuosity_all

.. autofunction:: openimpala.facade.effective_diffusivity


Microstructural Characterisation
--------------------------------

.. autofunction:: openimpala.facade.specific_surface_area

.. autofunction:: openimpala.facade.connected_components

.. autofunction:: openimpala.facade.particle_size_distribution

.. autofunction:: openimpala.facade.through_thickness_profile

.. autofunction:: openimpala.facade.rev_study


Battery Parameterisation (PyBaMM & Faraday BPX)
-----------------------------------------------

.. autofunction:: openimpala.facade.to_pybamm_parameters

.. autofunction:: openimpala.facade.export_bpx_json


I/O, Segmentation, Cloud Zarr & Plotfiles
-----------------------------------------

.. autofunction:: openimpala.facade.read_image

.. autofunction:: openimpala._io.calculate_otsu_threshold

.. autofunction:: openimpala._io.binarize_otsu

.. autofunction:: openimpala._io.downsample_volume

.. autofunction:: openimpala._io.read_zarr_volume

.. autofunction:: openimpala._io.write_zarr_volume

.. autofunction:: openimpala._io.write_amrex_plotfile

.. autofunction:: openimpala.facade.estimate_memory

.. autofunction:: openimpala.facade.build_info

Sample Datasets (``openimpala.datasets``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: openimpala.datasets
   :members: list_datasets, fetch_example, two_phase_stack, two_phase_cubic, sphere_pack, fetch_path

Synthetic Microstructures & ML Datasets (``openimpala.synthetic``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: openimpala.synthetic
   :members: gaussian_blobs, overlapping_spheres, multiphase_electrode, generate_ml_dataset


Result Dataclasses
------------------

All result objects are frozen Python dataclasses that render structured HTML
summary tables automatically in Jupyter notebooks and MyST-NB documentation.

.. autoclass:: openimpala.facade.VolumeFractionResult
   :members:

.. autoclass:: openimpala.facade.PercolationResult
   :members:

.. autoclass:: openimpala.facade.TortuosityResult
   :members:

.. autoclass:: openimpala.facade.EffectiveDiffusivityResult
   :members:

.. autoclass:: openimpala.facade.SpecificSurfaceAreaResult
   :members:

.. autoclass:: openimpala.facade.ConnectedComponentsResult
   :members:

.. autoclass:: openimpala.facade.ParticleSizeDistributionResult
   :members:

.. autoclass:: openimpala.facade.ThroughThicknessResult
   :members:

.. autoclass:: openimpala.facade.REVStudyResult
   :members:


Session Management
------------------

.. autoclass:: openimpala.Session
   :members:
   :special-members: __enter__, __exit__


Exceptions
----------

.. autoclass:: openimpala.OpenImpalaError
   :members:

.. autoclass:: openimpala.ConvergenceError
   :members:

.. autoclass:: openimpala.PercolationError
   :members:
