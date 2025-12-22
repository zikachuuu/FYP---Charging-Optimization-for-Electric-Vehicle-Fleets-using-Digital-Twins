"""
Custom exception types for the EV fleet charging optimization project.

Use these to signal well-defined failure modes from optimization routines
so callers can handle them explicitly (e.g., logging, retries, fallbacks).
"""
from typing import Any, Optional

__all__ = ["OptimizationFailed"]


class OptimizationError(Exception):
    """Raised when an optimization run fails to produce a usable solution.

    Typical scenarios include:
    - Infeasible model
    - Unbounded model
    - Numerical issues
    - Terminated without incumbent solution

    Parameters
    ----------
    message: Optional[str]
        Human-friendly explanation. If omitted, a default will be constructed.
    status: Optional[Any]
        Solver-specific status code or enum (e.g., from Gurobi).
    details: Optional[Any]
        Any extra diagnostic payload (dict/str) to aid debugging.
    """

    def __init__(self, message: Optional[str] = None, *, status: Optional[Any] = None, details: Optional[Any] = None):
        if message is None:
            base = "Optimization failed"
            if status is not None:
                base += f" (status: {status})"
            message = base
        super().__init__(message)
        self.status = status
        self.details = details

    def __str__(self) -> str:
        base = super().__str__()
        if self.details is None:
            return base
        return f"{base} | details: {self.details}"