"""Tests for metrics calculators."""

import pytest

from tracking_metrics.calculators import MetricsCalculator
from tracking_metrics.collectors import TrackingMetricsCollector


def test_metrics_calculator_init():
    """Test calculator initialization."""
    collector = TrackingMetricsCollector()
    calculator = MetricsCalculator(collector)
    assert calculator.collector is collector


def test_compute_confidence_metrics():
    """Test confidence metrics computation."""
    # TODO: Implement test
    pass


def test_compute_track_metrics():
    """Test track metrics computation."""
    # TODO: Implement test
    pass
