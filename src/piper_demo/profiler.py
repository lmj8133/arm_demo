"""Pipeline profiler for timing measurements.

Provides a unified framework to measure processing times for each stage
of the vision-guided robotic arm pipeline.

Usage:
    from piper_demo.profiler import PipelineProfiler

    profiler = PipelineProfiler(enabled=True, report_interval_sec=5.0)

    # Context manager usage
    with profiler.stage("hsv_detect"):
        target = tracker.detect(frame)

    # Manual start/stop for spanning multiple blocks
    profiler.start("total")
    # ... do work ...
    profiler.stop("total")

    # Periodic console report
    profiler.maybe_print_report()

    # OSD display data
    lines = profiler.format_osd()  # ["hsv_detect: 4.9ms (avg:4.8)", ...]
"""

import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

# Predefined stage names
STAGE_CAPTURE = "capture"
STAGE_HSV = "hsv_detect"
STAGE_IK = "ik_solve"
STAGE_MOTION = "motion_cmd"
STAGE_TOTAL = "total"


@dataclass
class TimingStats:
    """Cumulative timing statistics for a single stage."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    sum_sq_ms: float = 0.0  # For std calculation

    def add(self, elapsed_ms: float) -> None:
        """Add a timing sample."""
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)
        self.sum_sq_ms += elapsed_ms * elapsed_ms

    @property
    def avg_ms(self) -> float:
        """Average time in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def std_ms(self) -> float:
        """Standard deviation in milliseconds."""
        if self.count < 2:
            return 0.0
        variance = (self.sum_sq_ms / self.count) - (self.avg_ms**2)
        return variance**0.5 if variance > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0
        self.sum_sq_ms = 0.0


class RollingTimingStats:
    """Rolling window timing statistics to avoid dilution from old data.

    Args:
        window_size: Number of samples to keep in the rolling window.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._samples: deque = deque(maxlen=window_size)
        self._total_count = 0  # Total samples ever added

    def add(self, elapsed_ms: float) -> None:
        """Add a timing sample."""
        self._samples.append(elapsed_ms)
        self._total_count += 1

    @property
    def count(self) -> int:
        """Number of samples in current window."""
        return len(self._samples)

    @property
    def total_count(self) -> int:
        """Total samples ever added."""
        return self._total_count

    @property
    def avg_ms(self) -> float:
        """Average time in milliseconds."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        return min(self._samples) if self._samples else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum time in milliseconds."""
        return max(self._samples) if self._samples else 0.0

    @property
    def std_ms(self) -> float:
        """Standard deviation in milliseconds."""
        if len(self._samples) < 2:
            return 0.0
        avg = self.avg_ms
        variance = sum((x - avg) ** 2 for x in self._samples) / len(self._samples)
        return variance**0.5 if variance > 0 else 0.0

    @property
    def last_ms(self) -> float:
        """Most recent sample in milliseconds."""
        return self._samples[-1] if self._samples else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self._samples.clear()
        self._total_count = 0


class PipelineProfiler:
    """Pipeline profiler for measuring stage execution times.

    Provides context manager and manual timing methods with minimal overhead
    when disabled.

    Args:
        enabled: Whether profiling is active. When False, all methods are no-ops.
        report_interval_sec: Interval between automatic console reports.
        window_size: Number of samples for rolling statistics.
    """

    # Class-level stage constants for convenience
    STAGE_CAPTURE = STAGE_CAPTURE
    STAGE_HSV = STAGE_HSV
    STAGE_IK = STAGE_IK
    STAGE_MOTION = STAGE_MOTION
    STAGE_TOTAL = STAGE_TOTAL

    def __init__(
        self,
        enabled: bool = False,
        report_interval_sec: float = 5.0,
        window_size: int = 100,
    ):
        # Check environment variable override
        env_enabled = os.environ.get("PROFILER_ENABLED", "").lower()
        if env_enabled in ("1", "true", "yes"):
            enabled = True

        self._enabled = enabled
        self._report_interval_sec = report_interval_sec
        self._window_size = window_size
        self._stats: Dict[str, RollingTimingStats] = {}
        self._active_starts: Dict[str, float] = {}
        self._last_report_time = time.time()

    @property
    def enabled(self) -> bool:
        """Whether profiling is enabled."""
        return self._enabled

    def _get_or_create_stats(self, stage: str) -> RollingTimingStats:
        """Get or create statistics for a stage."""
        if stage not in self._stats:
            self._stats[stage] = RollingTimingStats(self._window_size)
        return self._stats[stage]

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context manager for timing a stage.

        Args:
            name: Stage identifier.

        Example:
            with profiler.stage("hsv_detect"):
                target = tracker.detect(frame)
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._get_or_create_stats(name).add(elapsed_ms)

    def start(self, name: str) -> None:
        """Start timing a stage manually.

        Use stop() to record the elapsed time. Useful when timing
        spans multiple code blocks.

        Args:
            name: Stage identifier.
        """
        if not self._enabled:
            return
        self._active_starts[name] = time.perf_counter()

    def stop(self, name: str) -> Optional[float]:
        """Stop timing a stage and record the elapsed time.

        Args:
            name: Stage identifier (must match a previous start() call).

        Returns:
            Elapsed time in milliseconds, or None if not enabled or not started.
        """
        if not self._enabled:
            return None

        start = self._active_starts.pop(name, None)
        if start is None:
            return None

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._get_or_create_stats(name).add(elapsed_ms)
        return elapsed_ms

    def get_avg_ms(self, stage: str) -> float:
        """Get average time for a stage in milliseconds."""
        if stage not in self._stats:
            return 0.0
        return self._stats[stage].avg_ms

    def get_last_ms(self, stage: str) -> float:
        """Get most recent time for a stage in milliseconds."""
        if stage not in self._stats:
            return 0.0
        return self._stats[stage].last_ms

    def get_fps(self, stage: str = STAGE_TOTAL) -> float:
        """Calculate FPS based on average time for a stage.

        Args:
            stage: Stage to use for FPS calculation (default: total).

        Returns:
            Frames per second, or 0.0 if no data.
        """
        avg = self.get_avg_ms(stage)
        return 1000.0 / avg if avg > 0 else 0.0

    def format_report(self) -> str:
        """Format a full statistics report.

        Returns:
            Multi-line string with timing statistics for all stages.
        """
        if not self._enabled or not self._stats:
            return ""

        lines = [
            "[Profiler] Timing Summary (ms):",
            "-" * 60,
            f"{'Stage':<20} {'Avg':>8} {'Min':>8} {'Max':>8} {'Std':>8} {'N':>7}",
            "-" * 60,
        ]

        # Sort stages: predefined ones first, then alphabetical
        predefined = [STAGE_CAPTURE, STAGE_HSV, STAGE_IK, STAGE_MOTION, STAGE_TOTAL]
        stage_order = [s for s in predefined if s in self._stats]
        stage_order += sorted(s for s in self._stats if s not in predefined)

        for stage in stage_order:
            stats = self._stats[stage]
            if stats.count == 0:
                continue
            lines.append(
                f"{stage:<20} {stats.avg_ms:>8.2f} {stats.min_ms:>8.2f} "
                f"{stats.max_ms:>8.2f} {stats.std_ms:>8.2f} {stats.count:>7}"
            )

        lines.append("-" * 60)

        # Add FPS if total stage exists
        if STAGE_TOTAL in self._stats:
            fps = self.get_fps(STAGE_TOTAL)
            lines.append(f"Pipeline FPS: {fps:.1f}")

        return "\n".join(lines)

    def maybe_print_report(self) -> bool:
        """Print report if interval has elapsed.

        Returns:
            True if report was printed, False otherwise.
        """
        if not self._enabled:
            return False

        now = time.time()
        if now - self._last_report_time >= self._report_interval_sec:
            report = self.format_report()
            if report:
                print(report)
            self._last_report_time = now
            return True
        return False

    def format_osd(self, stages: Optional[List[str]] = None) -> List[str]:
        """Format timing data for on-screen display.

        Args:
            stages: List of stages to include. If None, shows all stages
                    with data in predefined order.

        Returns:
            List of formatted strings like ["hsv_detect: 4.9ms (avg:4.8)", ...].
        """
        if not self._enabled or not self._stats:
            return []

        if stages is None:
            # Default order: predefined stages that have data
            predefined = [STAGE_HSV, STAGE_IK, STAGE_MOTION, STAGE_TOTAL]
            stages = [s for s in predefined if s in self._stats]

        lines = []
        for stage in stages:
            if stage not in self._stats:
                continue
            stats = self._stats[stage]
            if stats.count == 0:
                continue
            lines.append(f"{stage}: {stats.last_ms:.1f}ms (avg:{stats.avg_ms:.1f})")

        # Add FPS line
        if STAGE_TOTAL in self._stats and self._stats[STAGE_TOTAL].count > 0:
            fps = self.get_fps(STAGE_TOTAL)
            lines.append(f"FPS: {fps:.1f}")

        return lines

    def reset(self) -> None:
        """Reset all statistics."""
        for stats in self._stats.values():
            stats.reset()
        self._active_starts.clear()
        self._last_report_time = time.time()

    def get_summary_dict(self) -> Dict[str, Dict[str, float]]:
        """Get statistics as a dictionary for programmatic access.

        Returns:
            Dict mapping stage names to stats dicts with keys:
            avg_ms, min_ms, max_ms, std_ms, count.
        """
        result = {}
        for stage, stats in self._stats.items():
            result[stage] = {
                "avg_ms": stats.avg_ms,
                "min_ms": stats.min_ms,
                "max_ms": stats.max_ms,
                "std_ms": stats.std_ms,
                "count": stats.count,
            }
        return result


# Global profiler instance for convenience
_global_profiler: Optional[PipelineProfiler] = None


def get_profiler() -> Optional[PipelineProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def set_profiler(profiler: Optional[PipelineProfiler]) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler
