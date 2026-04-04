"""Context manager for deterministic AMReX / MPI lifecycle management."""

from __future__ import annotations


def _has_core() -> bool:
    """Return True if the compiled C++ backend is importable."""
    try:
        import openimpala._core  # noqa: F401
        return True
    except ImportError:
        return False


class Session:
    """Ensures AMReX ``initialize()`` / ``finalize()`` are called exactly once
    and in the correct order, even when *mpi4py* is also in use.

    Usage::

        import openimpala as oi

        with oi.Session():
            vf = oi.volume_fraction(data, phase=0)

    The session is re-entrant: nested ``with Session()`` blocks share the same
    underlying AMReX state; only the outermost block triggers init/finalize.

    When the compiled C++ backend (``_core``) is not available, the session
    activates the pure-Python solver backend (SciPy / CuPy) automatically.
    """

    _depth: int = 0  # nesting counter (class-level)
    _pure_python: bool = False  # True when _core is unavailable

    def __enter__(self) -> "Session":
        if Session._depth == 0:
            self._do_initialize()
        Session._depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        Session._depth -= 1
        if Session._depth == 0:
            self._do_finalize()
        return None  # do not suppress exceptions

    # ------------------------------------------------------------------
    @staticmethod
    def _do_initialize() -> None:
        if _has_core():
            # Import mpi4py first so MPI_Init happens before AMReX touches MPI.
            try:
                from mpi4py import MPI  # noqa: F401
            except ImportError:
                pass

            # Initialise AMReX natively via our C++ bindings — no pyamrex needed.
            from openimpala._core import init_amrex

            init_amrex()
            Session._pure_python = False

            # NOTE: HYPRE initialisation is handled automatically by the C++
            # solver constructors (TortuosityHypre, EffectiveDiffusivityHypre)
            # via std::call_once, so no Python-side HYPRE_Init() is needed.
        else:
            # No compiled backend — use pure-Python solver path.
            Session._pure_python = True

            # Verify we have at least scipy (required for the pure-Python solver)
            try:
                import scipy  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Neither the compiled C++ backend nor SciPy is available. "
                    "Install scipy (pip install scipy) for the pure-Python solver, "
                    "or install the full package (pip install openimpala-cuda) for "
                    "the C++ backend."
                )

            from . import _solver
            backend = _solver.backend_name()
            import sys
            print(f"OpenImpala: using pure-Python solver backend ({backend})",
                  file=sys.stderr)

    @staticmethod
    def _do_finalize() -> None:
        if Session._pure_python:
            return  # Nothing to tear down for pure-Python backend

        import gc

        from openimpala._core import amrex_initialized, finalize_amrex

        if amrex_initialized():
            # Force Python to destroy all orphaned C++ pybind11 objects NOW
            gc.collect()

            # Now it is safe to shut down the C++ backend
            finalize_amrex()
