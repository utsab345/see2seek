"""Unit tests for navigation metrics (see2seek.evaluation.metrics)."""

import pytest
from see2seek.evaluation.metrics import NavigationMetrics


def test_empty_metrics_default_to_zero() -> None:
    m = NavigationMetrics()
    assert m.sr == 0.0
    assert m.spl == 0.0
    assert m.num_episodes == 0
    assert m.summary() == {"sr": 0.0, "spl": 0.0, "num_episodes": 0.0}


def test_sr_and_spl_accumulation() -> None:
    m = NavigationMetrics()
    m.update(success=True, path_length=5.0, shortest_path=10.0)  # spl 1.0
    m.update(success=False, path_length=5.0, shortest_path=10.0)  # spl 0.0
    m.update(success=True, path_length=10.0, shortest_path=10.0)  # spl 1.0
    m.update(success=True, path_length=20.0, shortest_path=10.0)  # spl 0.5
    assert m.num_episodes == 4
    assert m.sr == pytest.approx(0.75)
    assert m.spl == pytest.approx((1.0 + 0.0 + 1.0 + 0.5) / 4)


def test_spl_penalizes_longer_paths() -> None:
    m = NavigationMetrics()
    m.update(success=True, path_length=10.0, shortest_path=5.0)
    assert m.spl == pytest.approx(0.5)


def test_degenerate_zero_path_both_counts_as_success() -> None:
    m = NavigationMetrics()
    m.update(success=True, path_length=0.0, shortest_path=0.0)
    assert m.spl == pytest.approx(1.0)


def test_reset_clears_state() -> None:
    m = NavigationMetrics()
    m.update(success=True, path_length=3.0, shortest_path=3.0)
    m.reset()
    assert m.num_episodes == 0
    assert m.sr == 0.0
    assert m.spl == 0.0


def test_repr_roundtrip() -> None:
    m = NavigationMetrics()
    m.update(success=True, path_length=3.0, shortest_path=3.0)
    assert "NavigationMetrics" in repr(m)
    assert "SR=1.000" in repr(m)
