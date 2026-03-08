"""Shared pytest fixtures for OpenImpala Python binding tests.

Ensures AMReX is initialised exactly once per test session.
"""

import pytest
import numpy as np
import gc  

@pytest.fixture(scope="session", autouse=True)
def amrex_session():
    """Initialise AMReX for the entire test session."""
    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError:
        pass

    import amrex.space3d as amrex

    if not amrex.initialized():
        amrex.initialize([])
    
    yield
    
    # Force Python to destroy all C++ Pybind11 objects NOW
    gc.collect()
    
    # Safely shut down AMReX and MPI before Python exits
    if amrex.initialized():
        amrex.finalize()


@pytest.fixture
def uniform_block():
    """A 16x16x16 block filled entirely with phase 0."""
    return np.zeros((16, 16, 16), dtype=np.int32)


@pytest.fixture
def two_phase_block():
    """A 16x16x16 block: phase 0 for x < 8, phase 1 for x >= 8."""
    data = np.zeros((16, 16, 16), dtype=np.int32)
    data[:, :, 8:] = 1
    return data


@pytest.fixture
def connected_channel():
    """A 16x16x16 block with a connected channel (phase 0) along X.

    Phase 1 everywhere except a 4-wide channel in the centre (y=6..9, z=6..9).
    """
    data = np.ones((16, 16, 16), dtype=np.int32)
    data[6:10, 6:10, :] = 0  # open channel along X
    return data


@pytest.fixture
def disconnected_phase():
    """A 16x16x16 block with phase 0 isolated in the interior (not percolating)."""
    data = np.ones((16, 16, 16), dtype=np.int32)
    data[6:10, 6:10, 6:10] = 0  # island of phase 0
    return data
