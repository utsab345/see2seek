"""Evaluation loop and navigation metrics (SR, SPL)."""

from see2seek.evaluation.metrics import NavigationMetrics

__all__ = ["NavigationMetrics", "Evaluator"]


def __getattr__(name: str) -> object:
    """Lazily import the evaluator so importing metrics does not require ai2thor."""
    if name == "Evaluator":
        from see2seek.evaluation.evaluator import Evaluator

        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
