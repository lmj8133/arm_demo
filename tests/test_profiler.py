"""Unit tests for PipelineProfiler.

Test cases:
1. Happy path: context manager correctly measures timing
2. Happy path: start/stop manual timing
3. Edge case: disabled profiler is a no-op
4. Edge case: rolling window statistics are correct
5. Edge case: stop without start returns None
"""

import time

import pytest

from piper_demo.profiler import PipelineProfiler, RollingTimingStats


class TestRollingTimingStats:
    """Tests for RollingTimingStats class."""

    def test_add_single_sample(self):
        """Single sample should set all stats correctly."""
        stats = RollingTimingStats(window_size=10)
        stats.add(5.0)

        assert stats.count == 1
        assert stats.total_count == 1
        assert stats.avg_ms == 5.0
        assert stats.min_ms == 5.0
        assert stats.max_ms == 5.0
        assert stats.last_ms == 5.0

    def test_rolling_window_eviction(self):
        """Old samples should be evicted when window is full."""
        stats = RollingTimingStats(window_size=3)

        # Add 5 samples (exceeds window)
        for i in range(1, 6):
            stats.add(float(i))

        # Window should only contain last 3: [3, 4, 5]
        assert stats.count == 3
        assert stats.total_count == 5
        assert stats.avg_ms == pytest.approx(4.0)
        assert stats.min_ms == 3.0
        assert stats.max_ms == 5.0
        assert stats.last_ms == 5.0

    def test_std_calculation(self):
        """Standard deviation should be calculated correctly."""
        stats = RollingTimingStats(window_size=100)

        # Add samples: [2, 4, 4, 4, 5, 5, 7, 9]
        samples = [2, 4, 4, 4, 5, 5, 7, 9]
        for s in samples:
            stats.add(float(s))

        # Mean = 5.0, variance = 4.0, std = 2.0
        assert stats.avg_ms == pytest.approx(5.0)
        assert stats.std_ms == pytest.approx(2.0)

    def test_reset_clears_all(self):
        """Reset should clear all statistics."""
        stats = RollingTimingStats(window_size=10)
        stats.add(10.0)
        stats.add(20.0)

        stats.reset()

        assert stats.count == 0
        assert stats.total_count == 0
        assert stats.avg_ms == 0.0


class TestPipelineProfiler:
    """Tests for PipelineProfiler class."""

    def test_context_manager_timing(self):
        """Context manager should correctly measure elapsed time."""
        profiler = PipelineProfiler(enabled=True)

        with profiler.stage("test_stage"):
            time.sleep(0.01)  # 10ms

        avg = profiler.get_avg_ms("test_stage")
        # Allow tolerance for timing jitter (8-20ms)
        assert 8.0 <= avg <= 50.0

    def test_start_stop_manual_timing(self):
        """Manual start/stop should correctly measure elapsed time."""
        profiler = PipelineProfiler(enabled=True)

        profiler.start("manual_stage")
        time.sleep(0.01)  # 10ms
        elapsed = profiler.stop("manual_stage")

        assert elapsed is not None
        assert 8.0 <= elapsed <= 50.0
        assert 8.0 <= profiler.get_avg_ms("manual_stage") <= 50.0

    def test_disabled_profiler_is_noop(self):
        """Disabled profiler should be a no-op with minimal overhead."""
        profiler = PipelineProfiler(enabled=False)

        # Context manager should not record anything
        with profiler.stage("noop_stage"):
            time.sleep(0.001)

        # Manual timing should not record anything
        profiler.start("noop_manual")
        elapsed = profiler.stop("noop_manual")

        assert elapsed is None
        assert profiler.get_avg_ms("noop_stage") == 0.0
        assert profiler.get_avg_ms("noop_manual") == 0.0
        assert profiler.format_report() == ""
        assert profiler.format_osd() == []

    def test_stop_without_start_returns_none(self):
        """Stopping a stage that was never started should return None."""
        profiler = PipelineProfiler(enabled=True)

        result = profiler.stop("never_started")

        assert result is None
        assert profiler.get_avg_ms("never_started") == 0.0

    def test_fps_calculation(self):
        """FPS should be calculated from average time."""
        profiler = PipelineProfiler(enabled=True)

        # Simulate 10ms per frame = 100 FPS
        for _ in range(5):
            with profiler.stage(PipelineProfiler.STAGE_TOTAL):
                time.sleep(0.01)

        fps = profiler.get_fps(PipelineProfiler.STAGE_TOTAL)
        # Allow wide tolerance due to timing jitter
        assert 20.0 <= fps <= 125.0

    def test_format_report_includes_all_stages(self):
        """Format report should include all measured stages."""
        profiler = PipelineProfiler(enabled=True)

        with profiler.stage("capture"):
            pass
        with profiler.stage("detect"):
            pass

        report = profiler.format_report()

        assert "capture" in report
        assert "detect" in report
        assert "Profiler" in report

    def test_format_osd_returns_formatted_lines(self):
        """Format OSD should return properly formatted lines."""
        profiler = PipelineProfiler(enabled=True)

        with profiler.stage(PipelineProfiler.STAGE_HSV):
            time.sleep(0.001)
        with profiler.stage(PipelineProfiler.STAGE_TOTAL):
            time.sleep(0.001)

        lines = profiler.format_osd()

        assert len(lines) >= 2
        assert any("hsv_detect" in line for line in lines)
        assert any("FPS" in line for line in lines)

    def test_get_summary_dict(self):
        """Summary dict should contain all stage statistics."""
        profiler = PipelineProfiler(enabled=True)

        with profiler.stage("test"):
            time.sleep(0.001)

        summary = profiler.get_summary_dict()

        assert "test" in summary
        assert "avg_ms" in summary["test"]
        assert "min_ms" in summary["test"]
        assert "max_ms" in summary["test"]
        assert "count" in summary["test"]

    def test_reset_clears_all_stats(self):
        """Reset should clear all statistics and active timers."""
        profiler = PipelineProfiler(enabled=True)

        with profiler.stage("test"):
            pass
        profiler.start("active")

        profiler.reset()

        assert profiler.get_avg_ms("test") == 0.0
        # Active timer should be cleared
        assert profiler.stop("active") is None


class TestPipelineProfilerConstants:
    """Tests for predefined stage constants."""

    def test_stage_constants_are_strings(self):
        """Stage constants should be strings."""
        assert isinstance(PipelineProfiler.STAGE_CAPTURE, str)
        assert isinstance(PipelineProfiler.STAGE_HSV, str)
        assert isinstance(PipelineProfiler.STAGE_IK, str)
        assert isinstance(PipelineProfiler.STAGE_MOTION, str)
        assert isinstance(PipelineProfiler.STAGE_TOTAL, str)
