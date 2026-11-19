"""Context manager for deterministic AMReX / MPI lifecycle management."""

from __future__ import annotations

import os
import warnings


def _requested_backend() -> str:
    """Normalise ``OPENIMPALA_BACKEND`` environment variable to ``'auto'``, ``'pure'``, or ``'cpp'``."""
    raw = os.environ.get("OPENIMPALA_BACKEND", "auto").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    if raw in ("pure", "pure-python", "pure_python", "scipy"):
        return "pure"
    if raw in ("cpp", "c++", "core", "_core"):
        return "cpp"
    raise ValueError(
        f"Invalid OPENIMPALA_BACKEND={raw!r}. Expected 'auto', 'cpp', or 'pure'."
    )


def _probe_core() -> tuple[bool, str | None]:
    """Check whether ``openimpala._core`` can be imported, recording any load error."""
    try:
        import openimpala._core  # noqa: F401

        return True, None
    except (ImportError, OSError) as exc:
        err = f"{type(exc).__name__}: {exc}"
        Session._core_load_error = err
        return False, err


def _has_core() -> bool:
    """Return True if the compiled C++ backend is importable and not disabled."""
    if _requested_backend() == "pure":
        return False
    ok, _ = _probe_core()
    return ok


class Session:
    """Ensures AMReX ``initialize()`` / ``finalize()`` are called exactly once
    and in the correct order, even when *mpi4py* is also in use.

    Usage::

        import openimpala as oi

        with oi.Session():
            vf = oi.volume_fraction(data, phase=0)

    The session is re-entrant: nested ``with Session()`` blocks share the same
    underlying AMReX state; only the outermost block triggers init/finalize.

    When the compiled C++ backend (``_core``) is not available (or when
    ``OPENIMPALA_BACKEND=pure`` is set, or ``openimpala-cuda`` runs on a node
    with no visible GPU), the session activates the pure-Python solver backend
    (SciPy / CuPy) automatically unless ``OPENIMPALA_BACKEND=cpp`` is set.
    """

    _depth: int = 0  # nesting counter (class-level)
    _pure_python: bool = False  # True when _core is unavailable or bypassed
    _core_load_error: "str | None" = None
    _auto_session: "Session | None" = None

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

    @classmethod
    def ensure_active(cls) -> None:
        """Lazily start a process-level session if not already inside ``with Session():``."""
        if cls._depth == 0:
            import atexit

            cls._auto_session = cls()
            cls._auto_session.__enter__()
            atexit.register(cls._cleanup_auto)

    @classmethod
    def _cleanup_auto(cls) -> None:
        if cls._auto_session is not None and cls._depth > 0:
            sess = cls._auto_session
            cls._auto_session = None
            sess.__exit__(None, None, None)

    # ------------------------------------------------------------------
    @staticmethod
    def _do_initialize() -> None:
        req = _requested_backend()
        if req == "pure":
            Session._core_load_error = "Disabled via OPENIMPALA_BACKEND=pure"
        else:
            ok, err = _probe_core()
            if ok:
                try:
                    from mpi4py import MPI  # noqa: F401
                except ImportError:
                    pass

                from openimpala._core import init_amrex

                try:
                    init_amrex()
                except RuntimeError as exc:
                    Session._core_load_error = f"{type(exc).__name__}: {exc}"
                    if req == "cpp":
                        raise
                    warnings.warn(
                        f"{exc} Falling back to pure-Python solver.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    Session._pure_python = False
                    Session._core_load_error = None
                    return
            elif req == "cpp":
                raise ImportError(
                    f"OPENIMPALA_BACKEND=cpp was requested, but openimpala._core "
                    f"could not be loaded ({err})."
                )

        # Pure-Python solver path
        Session._pure_python = True

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise ImportError(
                "Neither the compiled C++ backend nor SciPy is available. "
                "Install scipy (pip install scipy) for the pure-Python solver, "
                "or install the full package (pip install openimpala-cuda) for "
                "the C++ backend."
            )

        import logging

        from . import _solver

        logging.getLogger(__name__).debug(
            "OpenImpala: using pure-Python solver backend (%s)",
            _solver.backend_name(),
        )

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
