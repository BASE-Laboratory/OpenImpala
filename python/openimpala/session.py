"""Context manager for deterministic AMReX / MPI lifecycle management."""

from __future__ import annotations


class Session:
    """Ensures AMReX ``initialize()`` / ``finalize()`` are called exactly once
    and in the correct order, even when *mpi4py* is also in use.

    Usage::

        import openimpala as oi

        with oi.Session():
            vf = oi.volume_fraction(data, phase=0)

    The session is re-entrant: nested ``with Session()`` blocks share the same
    underlying AMReX state; only the outermost block triggers init/finalize.
    """

    _depth: int = 0  # nesting counter (class-level)

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
        # Import mpi4py first so MPI_Init happens before AMReX touches MPI.
        try:
            from mpi4py import MPI  # noqa: F401
        except ImportError:
            pass

        # Initialise AMReX natively via our C++ bindings — no pyamrex needed.
        from openimpala._core import init_amrex

        init_amrex()

        # NOTE: HYPRE initialisation is handled automatically by the C++
        # solver constructors (TortuosityHypre, EffectiveDiffusivityHypre)
        # via std::call_once, so no Python-side HYPRE_Init() is needed.

    @staticmethod
    def _do_finalize() -> None:
        import gc

        from openimpala._core import amrex_initialized, finalize_amrex

        if amrex_initialized():
            # Force Python to destroy all orphaned C++ pybind11 objects NOW
            gc.collect()

            # Now it is safe to shut down the C++ backend
            finalize_amrex()
