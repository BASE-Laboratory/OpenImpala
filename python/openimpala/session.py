"""Context manager for deterministic AMReX / MPI lifecycle management."""

from __future__ import annotations


class Session:
    """Ensures ``amrex.initialize()`` / ``amrex.finalize()`` are called exactly
    once and in the correct order, even when *mpi4py* is also in use.

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
        import os
        import sys

        # Import mpi4py first so MPI_Init happens before AMReX touches MPI.
        try:
            from mpi4py import MPI  # noqa: F401
        except ImportError:
            pass

        # CRITICAL: Set RTLD_GLOBAL *before* the first import of amrex so that
        # pyAMReX's C++ type registry is visible to openimpala._core.so.
        # If amrex is loaded with RTLD_LOCAL (the default), pybind11 cross-module
        # type casts will segfault.
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
        try:
            import amrex.space3d as amrex
        finally:
            sys.setdlopenflags(old_flags)

        if not amrex.initialized():
            amrex.initialize([])

        # Initialise HYPRE (required before any HYPRE-based solver is used).
        # Guarded with try/except so that pure-Python usage (e.g. CLI parser
        # tests) still works when _core.so is not available.
        try:
            import importlib
            _core = importlib.import_module("openimpala._core")
            _core.hypre_init()
        except (ImportError, ModuleNotFoundError):
            pass

    @staticmethod
    def _do_finalize() -> None:
        import gc
        import amrex.space3d as amrex

        if amrex.initialized():
            # Force Python to destroy all orphaned C++ pybind11 objects NOW
            gc.collect()

            # Shut down HYPRE before AMReX finalises MPI.
            try:
                import importlib
                _core = importlib.import_module("openimpala._core")
                _core.hypre_finalize()
            except (ImportError, ModuleNotFoundError):
                pass

            # Now it is safe to shut down the C++ backend
            amrex.finalize()
