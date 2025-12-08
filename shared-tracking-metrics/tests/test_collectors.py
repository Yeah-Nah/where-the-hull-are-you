"""Tests for tracking metrics collectors."""

import pytest

from tracking_metrics.collectors import Detection, Track, TrackingMetricsCollector


def test_tracking_metrics_collector_init():
    """Test collector initialization."""
    collector = TrackingMetricsCollector()
    assert collector.frame_count == 0
    assert len(collector.detections) == 0
    assert len(collector.tracks) == 0


def test_add_frame_detections():
    """Test adding frame detections."""
    collector = TrackingMetricsCollector()
    # TODO: Implement test
    pass


def test_reset():
    """Test resetting collector."""
    collector = TrackingMetricsCollector()
    collector.frame_count = 10
    collector.reset()
    assert collector.frame_count == 0
