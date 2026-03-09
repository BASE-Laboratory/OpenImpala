"""Custom exception hierarchy for OpenImpala Python bindings."""


class OpenImpalaError(Exception):
    """Base exception for all OpenImpala errors."""


class ConvergenceError(OpenImpalaError):
    """Raised when a linear solver fails to converge."""


class PercolationError(OpenImpalaError):
    """Raised when the target phase does not percolate across the domain."""
