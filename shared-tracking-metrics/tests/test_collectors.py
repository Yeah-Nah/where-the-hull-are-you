"""Tests for tracking metrics collectors."""

from tracking_metrics.collectors import TrackingMetricsCollector


def test_tracking_metrics_collector_init() -> None:
    """Test collector initialization."""
    collector = TrackingMetricsCollector()
    assert collector.frame_count == 0
    assert len(collector.detections) == 0
    assert len(collector.tracks) == 0


def test_add_frame_detections() -> None:
    """Test adding frame detections."""
    # collector = TrackingMetricsCollector()
    # TODO: Implement test
    pass


def test_reset() -> None:
    """Test resetting collector."""
    collector = TrackingMetricsCollector()
    collector.frame_count = 10
    collector.reset()
    assert collector.frame_count == 0
