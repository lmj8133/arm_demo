"""DVS event-based laser spot tracking module.

Detects and tracks a single laser spot on a DVS (Dynamic Vision Sensor) using
a 5-stage pipeline optimised for sparse event data:

  1. Binary event extraction (baseline=127, events=255)
  2. Hot pixel removal (static noise mask or auto-learned)
  3. Multi-frame accumulation (sliding window ring buffer)
  4. Spatial clustering via weighted centroid
  5. Temporal smoothing with EMA + lost-target coasting

Returns the same ``DVSTarget`` dataclass used by ``DVSTracker``, so the two
trackers are drop-in interchangeable.

Example:
    tracker = DVSLaserTracker(width=164, height=160)
    tracker.load_noise_mask("recordings/no_signal.npy")

    target = tracker.detect_from_events(event_frame)
    if target:
        print(target.center)
"""

from collections import deque
from typing import Optional

import numpy as np

from dvs_tracker import DVSTarget


class DVSLaserTracker:
    """Laser spot tracker for DVS event frames.

    Args:
        width: Sensor width in pixels.
        height: Sensor height in pixels.
        baseline: Background pixel value (default 127).
        noise_mask_path: Path to a ``.npy`` recording of noise-only frames
            for hot-pixel calibration.  When *None*, falls back to auto-learn.
        auto_learn_frames: Number of initial frames used to build a hot-pixel
            mask automatically (only when *noise_mask_path* is ``None``).
        hot_pixel_threshold: A pixel that fires in >= this many frames during
            the learning / noise-mask window is flagged as hot.
        accumulation_window: Number of frames kept in the ring buffer.
        min_event_count: Minimum active pixels (after accumulation) to accept
            a detection.
        max_cluster_radius: Maximum spatial spread (std-dev of distances to
            centroid) for a valid cluster.
        ema_alpha: Exponential moving average weight for new detections.
        lost_timeout: How many consecutive misses before reporting *None*
            (coasting on last known position until then).
    """

    def __init__(
        self,
        width: int = 164,
        height: int = 160,
        baseline: int = 127,
        noise_mask_path: Optional[str] = None,
        auto_learn_frames: int = 30,
        hot_pixel_threshold: int = 10,
        accumulation_window: int = 5,
        min_event_count: int = 8,
        max_cluster_radius: float = 25.0,
        ema_alpha: float = 0.4,
        lost_timeout: int = 10,
    ):
        self.width = width
        self.height = height
        self.baseline = baseline
        self.auto_learn_frames = auto_learn_frames
        self.hot_pixel_threshold = hot_pixel_threshold
        self.accumulation_window = accumulation_window
        self.min_event_count = min_event_count
        self.max_cluster_radius = max_cluster_radius
        self.ema_alpha = ema_alpha
        self.lost_timeout = lost_timeout

        # Hot-pixel mask (True = hot pixel to suppress)
        self._hot_pixel_mask: Optional[np.ndarray] = None

        # Auto-learn accumulator
        self._learn_acc: Optional[np.ndarray] = None
        self._learn_count: int = 0

        # Ring buffer for multi-frame accumulation
        self._ring: deque = deque(maxlen=accumulation_window)

        # EMA state
        self._ema_cx: Optional[float] = None
        self._ema_cy: Optional[float] = None

        # Lost-target counter
        self._lost_frames: int = 0

        # Last event frame for display
        self._last_event_frame: Optional[np.ndarray] = None

        # Remember path for reload after reset()
        self._noise_mask_path = noise_mask_path

        # Load external noise mask if provided
        if noise_mask_path is not None:
            self.load_noise_mask(noise_mask_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_events(self, event_frame: np.ndarray) -> Optional[DVSTarget]:
        """Detect laser spot from a DVS event frame.

        Args:
            event_frame: Grayscale event frame (H, W), dtype uint8.

        Returns:
            DVSTarget if a laser spot is detected, otherwise None.
        """
        self._last_event_frame = event_frame

        # Stage 1 — binary event extraction
        event_mask = event_frame > self.baseline

        # Stage 2 — hot-pixel removal (or auto-learn)
        if self._hot_pixel_mask is None and self._learn_count < self.auto_learn_frames:
            self._auto_learn_step(event_mask)
            return None  # still learning
        if self._hot_pixel_mask is not None:
            event_mask = event_mask & ~self._hot_pixel_mask

        # Stage 3 — multi-frame accumulation
        self._ring.append(event_mask)
        if len(self._ring) < self.accumulation_window:
            return None  # not enough frames yet

        accumulated = np.sum(self._ring, axis=0)  # (H, W), values 0..window

        # Stage 4 — spatial clustering (weighted centroid)
        mask = accumulated >= 2  # present in at least 2 / window frames
        ys, xs = np.where(mask)

        if len(ys) < self.min_event_count:
            return self._handle_lost()

        weights = accumulated[ys, xs].astype(np.float64)
        total_w = weights.sum()
        cx = float(np.dot(xs, weights) / total_w)
        cy = float(np.dot(ys, weights) / total_w)

        # Validate spatial spread
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        weighted_std = float(np.sqrt(np.dot(dists ** 2, weights) / total_w))
        if weighted_std > self.max_cluster_radius:
            return self._handle_lost()

        # Stage 5 — temporal smoothing (EMA)
        if self._ema_cx is None:
            self._ema_cx = cx
            self._ema_cy = cy
        else:
            a = self.ema_alpha
            self._ema_cx = a * cx + (1.0 - a) * self._ema_cx
            self._ema_cy = a * cy + (1.0 - a) * self._ema_cy

        self._lost_frames = 0

        return self._make_target(self._ema_cx, self._ema_cy, len(ys))

    def load_noise_mask(self, path: str) -> None:
        """Build hot-pixel mask from a pre-recorded noise ``.npy`` file.

        Args:
            path: Path to a ``.npy`` file containing an array of shape
                (N, H, W) with noise-only DVS frames.
        """
        data = np.load(path)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        events = data > self.baseline  # (N, H, W)
        hit_count = events.sum(axis=0)  # (H, W)
        self._hot_pixel_mask = hit_count >= self.hot_pixel_threshold
        # Reset auto-learn since we have an explicit mask
        self._learn_acc = None
        self._learn_count = self.auto_learn_frames

    def reset(self) -> None:
        """Reset tracking state (EMA, ring buffer, learning).

        If a noise mask was loaded from file, it is preserved.
        Otherwise the auto-learn phase restarts from scratch.
        """
        self._ring.clear()
        self._ema_cx = None
        self._ema_cy = None
        self._lost_frames = 0
        self._last_event_frame = None
        if self._noise_mask_path is not None:
            # Preserve externally loaded mask — only reset tracking state
            pass
        else:
            # No external mask — restart auto-learn
            self._learn_acc = None
            self._learn_count = 0
            self._hot_pixel_mask = None

    @property
    def last_event_frame(self) -> Optional[np.ndarray]:
        """Get last DVS event frame for preview display."""
        return self._last_event_frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_learn_step(self, event_mask: np.ndarray) -> None:
        """Accumulate one frame for auto hot-pixel learning."""
        if self._learn_acc is None:
            self._learn_acc = np.zeros(
                (self.height, self.width), dtype=np.int32
            )
        self._learn_acc += event_mask.astype(np.int32)
        self._learn_count += 1
        if self._learn_count >= self.auto_learn_frames:
            self._hot_pixel_mask = self._learn_acc >= self.hot_pixel_threshold
            self._learn_acc = None  # free memory

    def _handle_lost(self) -> Optional[DVSTarget]:
        """Handle a frame with no valid detection (coasting logic)."""
        self._lost_frames += 1
        if self._lost_frames <= self.lost_timeout and self._ema_cx is not None:
            return self._make_target(self._ema_cx, self._ema_cy, 0)
        # Fully lost
        self._ema_cx = None
        self._ema_cy = None
        return None

    @staticmethod
    def _make_target(cx: float, cy: float, event_count: int) -> DVSTarget:
        """Build a DVSTarget from centroid coordinates."""
        # Approximate bbox as a small square around centroid
        half = 10
        x = max(0, int(cx) - half)
        y = max(0, int(cy) - half)
        size = half * 2
        return DVSTarget(
            cx=cx,
            cy=cy,
            area=float(event_count),
            bbox=(x, y, size, size),
            active_ratio=1.0 if event_count > 0 else 0.0,
        )

    def __repr__(self) -> str:
        mask_status = (
            "loaded" if self._hot_pixel_mask is not None
            else f"learning {self._learn_count}/{self.auto_learn_frames}"
        )
        return (
            f"DVSLaserTracker({self.width}x{self.height}, "
            f"window={self.accumulation_window}, "
            f"mask={mask_status})"
        )
