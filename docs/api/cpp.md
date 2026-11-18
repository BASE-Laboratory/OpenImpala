# C++ API Reference

The C++ API reference is generated from Doxygen comments in `src/` via
[Breathe](https://breathe.readthedocs.io/). All public symbols live in
`namespace OpenImpala`.

## I/O Readers

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

.. doxygenclass:: OpenImpala::DatReader
   :members:
   :outline:
```

## Transport Solvers

```{eval-rst}
.. doxygenclass:: OpenImpala::TortuosityMLMG
   :members:
   :outline:

.. doxygenclass:: OpenImpala::TortuosityHypre
   :members:
   :outline:

.. doxygenclass:: OpenImpala::EffectiveDiffusivityHypre
   :members:
   :outline:

.. doxygenclass:: OpenImpala::TortuositySolverBase
   :members:
   :outline:

.. doxygenclass:: OpenImpala::HypreStructSolver
   :members:
   :outline:
```

## Microstructural Characterisation

```{eval-rst}
.. doxygenclass:: OpenImpala::VolumeFraction
   :members:
   :outline:

.. doxygenclass:: OpenImpala::PercolationCheck
   :members:
   :outline:

.. doxygenclass:: OpenImpala::SpecificSurfaceArea
   :members:
   :outline:

.. doxygenclass:: OpenImpala::ConnectedComponents
   :members:
   :outline:

.. doxygenclass:: OpenImpala::ParticleSizeDistribution
   :members:
   :outline:

.. doxygenclass:: OpenImpala::ThroughThicknessProfile
   :members:
   :outline:

.. doxygenstruct:: OpenImpala::REVConfig
   :members:
   :outline:
```

## Configuration

```{eval-rst}
.. doxygenstruct:: OpenImpala::PhysicsConfig
   :members:
   :outline:
```

## Full Doxygen Output

For the complete C++ class hierarchy, include graphs, and file-level
documentation, see the
[standalone Doxygen reference](https://base-laboratory.github.io/OpenImpala/doxygen/index.html).
