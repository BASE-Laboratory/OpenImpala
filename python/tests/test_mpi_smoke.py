"""Multi-rank MPI smoke tests for OpenImpala Python bindings.

Designed to run under ``mpirun -np 2 python -m pytest python/tests/test_mpi_smoke.py -p no:cacheprovider``
(as well as single-rank pytest). Exercises domain decomposition across multiple boxes
and ranks for NumPy ingestion, parallel TIFF I/O, volume fraction reduction,
two-sided flood-fill percolation, TortuosityMLMG, and TortuosityHypre.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import openimpala as oi


def test_mpi_rank_count_matches_expectation():
    """Verify AMReX initialized with the expected MPI communicator size."""
    info = oi.build_info()
    assert info["mpi_ranks"] >= 1
    expected = os.environ.get("OPENIMPALA_EXPECT_MPI_RANKS")
    if expected is not None:
        assert info["mpi_ranks"] == int(expected)


@pytest.mark.parametrize("solver", ["auto", "pcg"])
def test_mpi_porous_volume_multibox(solver: str):
    """24^3 porous volume split into eight 12^3 boxes across MPI ranks."""
    fixture_path = Path(__file__).parent / "data" / "porous24.npz"
    with np.load(fixture_path) as fixture:
        image = fixture["pore"].astype(np.int32)

    vf = oi.volume_fraction(image, phase=1, max_grid_size=12)
    assert vf.total_count == 24**3
    assert 0.0 < vf.fraction < 1.0

    perc = oi.percolation_check(image, phase=1, direction="z", max_grid_size=12)
    assert perc.percolates is True

    result = oi.tortuosity(
        image,
        phase=1,
        direction="z",
        solver=solver,
        max_grid_size=12,
    )
    assert result.solver_converged is True
    assert result.tortuosity == pytest.approx(2.10421252, rel=1e-5)
    assert abs(result.flux_in) == pytest.approx(abs(result.flux_out), rel=1e-4)


def test_mpi_distributed_tiff_reader():
    """Verify distributed TiffReader + multi-box volume_fraction across MPI ranks."""
    repo_root = Path(__file__).resolve().parents[2]
    sample_tiff = repo_root / "data" / "SampleData_2Phase_stack_3d_1bit.tif"
    if not sample_tiff.exists():
        pytest.skip("Sample TIFF not present")

    reader, img = oi.read_image(str(sample_tiff), threshold=0.5, max_grid_size=50)
    vf = oi.volume_fraction(img, phase=1)
    assert vf.total_count == 1_000_000
    assert vf.fraction == pytest.approx(0.398309, rel=1e-6)
    del reader, img
