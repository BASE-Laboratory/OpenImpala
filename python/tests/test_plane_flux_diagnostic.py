"""Diagnostic test for plane-flux NaN on EB channel geometry.

Reproduces the computePlaneFluxes NaN observed in tTortuosityMLMG_channel
and reports exactly which plane fluxes are NaN, plus the boundary fluxes
and solution values at key cells. This helps diagnose whether the NaN
originates in the flux computation, the solution copy, or the ghost-cell
fill.
"""

import numpy as np
import pytest

import openimpala as oi
from openimpala import _core


@pytest.fixture(scope="module")
def channel_with_island():
    """32^3 domain: 4-wide channel (phase 0) through solid (phase 1),
    plus a 2^3 isolated pore island at (2..3, 2..3, 2..3).

    NumPy shape is (Z, Y, X). The channel must span the full Z range
    (flow direction) at lateral positions x=14..17, y=14..17.
    """
    N = 32
    data = np.ones((N, N, N), dtype=np.int32)  # all solid
    ch_lo, ch_hi = N // 2 - 2, N // 2 + 1  # 14..17
    # Channel: all Z, y=14..17, x=14..17  (runs along Z in AMReX coords)
    data[:, ch_lo : ch_hi + 1, ch_lo : ch_hi + 1] = 0
    # Isolated island at AMReX (i=2..3, j=2..3, k=2..3) = numpy [2:4, 2:4, 2:4]
    data[2:4, 2:4, 2:4] = 0
    return data


class TestPlaneFluxDiagnostic:
    def test_channel_boundary_fluxes_are_finite(self, channel_with_island):
        """Boundary fluxes (inlet/outlet) must be finite on the channel."""
        img = _core.VoxelImage.from_numpy(channel_with_island, 16)
        vf = _core.VolumeFraction(img, 0, 0).value_vf()
        solver = _core.TortuosityMLMG(
            img, vf, 0, _core.Direction.Z, ".", 0.0, 1.0, 2, False,
        )
        # Use value_raw to avoid exception on NaN
        tau = solver.value_raw()

        flux_in = solver.flux_in
        flux_out = solver.flux_out
        print(f"\nBoundary fluxes: in={flux_in:.6e}, out={flux_out:.6e}")
        print(f"Converged: {solver.solver_converged}, iters={solver.iterations}")
        print(f"Tau (raw): {tau}")

        assert np.isfinite(flux_in), f"flux_in is {flux_in}"
        assert np.isfinite(flux_out), f"flux_out is {flux_out}"
        assert abs(flux_in) > 1e-10, f"flux_in is near-zero: {flux_in}"

    def test_channel_hypre_for_comparison(self, channel_with_island):
        """Run HYPRE (pcg+smg) on the same channel to check if globalFluxes
        works at all on partial-domain geometries. If HYPRE also gives NaN,
        the bug is in globalFluxes, not in the EB code path."""
        res = oi.tortuosity(
            channel_with_island, phase=0, direction="z",
            solver="pcg", preconditioner="smg", verbose=1,
        )
        print(f"\nHYPRE channel: tau={res.tortuosity}  "
              f"converged={res.solver_converged}  iters={res.iterations}")
        assert np.isfinite(res.tortuosity), (
            f"HYPRE also NaN on channel geometry — globalFluxes bug, not EB"
        )
        """Dump all 31 plane fluxes to identify which ones are NaN."""
        img = _core.VoxelImage.from_numpy(channel_with_island, 16)
        vf = _core.VolumeFraction(img, 0, 0).value_vf()
        solver = _core.TortuosityMLMG(
            img, vf, 0, _core.Direction.Z, ".", 0.0, 1.0, 2, False,
        )
        solver.value_raw()

        plane_fluxes = solver.plane_fluxes
        n_nan = sum(1 for f in plane_fluxes if not np.isfinite(f))
        n_zero = sum(1 for f in plane_fluxes if f == 0.0)
        n_finite_nonzero = sum(1 for f in plane_fluxes if np.isfinite(f) and f != 0.0)

        print(f"\nPlane fluxes ({len(plane_fluxes)} faces):")
        print(f"  NaN/Inf: {n_nan}")
        print(f"  Zero:    {n_zero}")
        print(f"  Finite non-zero: {n_finite_nonzero}")
        for i, f in enumerate(plane_fluxes):
            marker = " <-- NaN!" if not np.isfinite(f) else ""
            marker = " <-- ZERO" if f == 0.0 and not marker else marker
            print(f"  face[{i:2d}] = {f:+.6e}{marker}")

        # The actual assertion: report what we found
        if n_nan > 0:
            nan_indices = [i for i, f in enumerate(plane_fluxes) if not np.isfinite(f)]
            pytest.fail(
                f"{n_nan} of {len(plane_fluxes)} plane fluxes are NaN "
                f"at face indices {nan_indices}. "
                f"Boundary fluxes: in={solver.flux_in:.6e}, out={solver.flux_out:.6e}. "
                f"See stdout for full dump."
            )
