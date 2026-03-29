# C++ API Reference

The C++ API reference is generated from Doxygen comments in the source code
using [Breathe](https://breathe.readthedocs.io/).

## Namespace

All OpenImpala classes live in the `OpenImpala` namespace.

## Key classes

### I/O Readers

```{eval-rst}
.. doxygenclass:: OpenImpala::TiffReader
   :members:
   :outline:

.. doxygenclass:: OpenImpala::HDF5Reader
   :members:
   :outline:

.. doxygenclass:: OpenImpala::RawReader
   :members:
   :outline:
```

### Transport Solvers

```{eval-rst}
.. doxygenclass:: OpenImpala::TortuosityHypre
   :members:
   :outline:

.. doxygenclass:: OpenImpala::TortuosityMLMG
   :members:
   :outline:

.. doxygenclass:: OpenImpala::EffectiveDiffusivityHypre
   :members:
   :outline:
```

### Utilities

```{eval-rst}
.. doxygenclass:: OpenImpala::VolumeFraction
   :members:
   :outline:

.. doxygenclass:: OpenImpala::PercolationCheck
   :members:
   :outline:

.. doxygenclass:: OpenImpala::TortuositySolverBase
   :members:
   :outline:

.. doxygenclass:: OpenImpala::HypreStructSolver
   :members:
   :outline:
```

### Configuration

```{eval-rst}
.. doxygenstruct:: OpenImpala::PhysicsConfig
   :members:
   :outline:
```

## Full Doxygen output

For the complete class hierarchy, include dependency graphs, and file-level
documentation, see the [Doxygen pages](../doxygen/html/index.html).
