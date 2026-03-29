Python API Reference
====================

High-level API
--------------

The recommended interface for most users. These functions accept NumPy arrays
and return Python dataclasses.

.. autofunction:: openimpala.facade.volume_fraction

.. autofunction:: openimpala.facade.percolation_check

.. autofunction:: openimpala.facade.tortuosity

.. autofunction:: openimpala.facade.read_image


Result types
~~~~~~~~~~~~

.. autoclass:: openimpala.facade.VolumeFractionResult
   :members:

.. autoclass:: openimpala.facade.PercolationResult
   :members:

.. autoclass:: openimpala.facade.TortuosityResult
   :members:


Session management
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
