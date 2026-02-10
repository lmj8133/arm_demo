"""DVS event-based hand tracking module.

Provides DVSTracker class for detecting and tracking hand motion using
real DVS (Dynamic Vision Sensor) event frames via raw binary input.

Based on the ROI-based tracking method from:
"Event-based tracking of human hands" (Duarte et al., 2021)

Example:
    tracker = DVSTracker(width=640, height=480, consecutive=3)

    # From raw bytes (e.g. DVS camera buffer)
    target = tracker.detect(raw_bytes)

    # From numpy array directly
    target = tracker.detect_from_events(event_frame)

    # Preview DVS event frame
    event_frame = tracker.last_event_frame
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class DVSTarget:
    """Detected target from DVS event-based tracking.

    Attributes:
        cx: Center X coordinate (pixels)
        cy: Center Y coordinate (pixels)
        area: ROI area (pixels^2)
        bbox: Bounding box as (x, y, width, height)
        active_ratio: Ratio of active pixels within ROI (0.0-1.0)
    """

    cx: float
    cy: float
    area: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    active_ratio: float

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (cx, cy) in pixels."""
        return (self.cx, self.cy)

    def normalized_center(
        self, img_width: int, img_height: int
    ) -> Tuple[float, float]:
        """Get center normalized to 0-1 range.

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            (norm_x, norm_y) where both values are in [0, 1]
        """
        return (self.cx / img_width, self.cy / img_height)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (x1, y1, x2, y2)."""
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def __repr__(self) -> str:
        return (
            f"DVSTarget(center=({self.cx:.0f}, {self.cy:.0f}), "
            f"area={self.area:.0f}, active={self.active_ratio:.2f})"
        )


class DVSTracker:
    """Event-based hand tracking using real DVS camera.

    Accepts raw binary event frames from DVS sensor and detects ROI
    (Region of Interest) using Algorithm 3 from the paper
    "Event-based tracking of human hands".

    Example:
        tracker = DVSTracker(width=640, height=480)

        # From raw bytes buffer
        target = tracker.detect(raw_bytes)

        # From numpy array
        target = tracker.detect_from_events(np_frame)
    """

    def __init__(
        self,
        width: int,
        height: int,
        consecutive: int = 3,
        min_active_ratio: float = 0.2,
        min_roi_size: int = 10,
        dtype: np.dtype = np.uint8,
    ):
        """Initialize DVSTracker.

        Args:
            width: DVS sensor width in pixels.
            height: DVS sensor height in pixels.
            consecutive: Number of consecutive cols/rows above average
                        for ROI boundary detection (default: 3).
            min_active_ratio: Minimum ratio of active pixels in ROI
                             for valid detection (default: 0.2).
            min_roi_size: Minimum ROI size in pixels (default: 10).
            dtype: Pixel data type of the raw buffer (default: uint8).
        """
        self.width = width
        self.height = height
        self.consecutive = consecutive
        self.min_active_ratio = min_active_ratio
        self.min_roi_size = min_roi_size
        self.dtype = np.dtype(dtype)

        self._last_event_frame: Optional[np.ndarray] = None

    def detect(self, raw_data: Union[bytes, bytearray, memoryview]) -> Optional[DVSTarget]:
        """Detect hand from raw DVS frame buffer.

        Args:
            raw_data: Raw binary data from DVS sensor, length must be
                      width * height * dtype.itemsize bytes.

        Returns:
            DVSTarget for the detected hand, or None if no valid detection.

        Raises:
            ValueError: If raw_data size does not match expected frame size.
        """
        expected_size = self.width * self.height * self.dtype.itemsize
        if len(raw_data) != expected_size:
            raise ValueError(
                f"Raw data size mismatch: got {len(raw_data)}, "
                f"expected {expected_size} ({self.width}x{self.height}x{self.dtype.itemsize})"
            )
        event_frame = np.frombuffer(raw_data, dtype=self.dtype).reshape(
            self.height, self.width
        )
        self._last_event_frame = event_frame
        return self._detect_roi(event_frame)

    def detect_from_events(self, event_frame: np.ndarray) -> Optional[DVSTarget]:
        """Detect hand from a numpy event frame directly.

        Args:
            event_frame: Grayscale event frame as numpy array (H, W).

        Returns:
            DVSTarget for the detected hand, or None if no valid detection.
        """
        self._last_event_frame = event_frame
        return self._detect_roi(event_frame)

    @property
    def last_event_frame(self) -> Optional[np.ndarray]:
        """Get last DVS event frame for preview display."""
        return self._last_event_frame

    def reset(self) -> None:
        """Reset internal state."""
        self._last_event_frame = None

    def _detect_roi(self, event_frame: np.ndarray) -> Optional[DVSTarget]:
        """Detect ROI from DVS event frame using Algorithm 3.

        Args:
            event_frame: Grayscale event frame (H, W)

        Returns:
            DVSTarget if valid ROI detected, else None
        """
        height, width = event_frame.shape

        # Compute column and row intensity sums
        col_sums = np.sum(event_frame, axis=0)
        row_sums = np.sum(event_frame, axis=1)

        # Find ROI boundaries
        x_min, x_max = self._find_roi_boundaries(col_sums)
        y_min, y_max = self._find_roi_boundaries(row_sums)

        # Validate boundaries
        if x_max <= x_min:
            x_min, x_max = 0, width - 1
        if y_max <= y_min:
            y_min, y_max = 0, height - 1

        # Make ROI square by selecting most active region
        x_min, x_max, y_min, y_max = self._make_square_roi(
            event_frame, x_min, x_max, y_min, y_max
        )

        # Clamp to image bounds
        x_min = max(0, x_min)
        x_max = min(width - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(height - 1, y_max)

        # Calculate ROI properties
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        roi_size = max(roi_width, roi_height)

        if roi_size < self.min_roi_size:
            return None

        # Calculate active pixel ratio
        roi_region = event_frame[y_min : y_max + 1, x_min : x_max + 1]
        total_pixels = roi_region.size
        active_pixels = np.sum(roi_region > 0)
        active_ratio = active_pixels / max(1, total_pixels)

        # Check minimum active ratio
        if active_ratio < self.min_active_ratio:
            return None

        # Calculate center and area
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        area = roi_width * roi_height

        return DVSTarget(
            cx=center_x,
            cy=center_y,
            area=area,
            bbox=(x_min, y_min, roi_width, roi_height),
            active_ratio=active_ratio,
        )

    def _find_roi_boundaries(self, sums: np.ndarray) -> Tuple[int, int]:
        """Find ROI boundaries using Algorithm 3.

        Find first and last positions where at least 'consecutive' values
        are above the average.

        Args:
            sums: Array of column or row intensity sums

        Returns:
            (min_idx, max_idx) boundary indices
        """
        avg = np.mean(sums)
        above_avg = sums > avg

        min_idx = 0
        max_idx = len(sums) - 1

        # Find first position with 'consecutive' values above average
        for i in range(len(sums) - self.consecutive + 1):
            if all(above_avg[i : i + self.consecutive]):
                min_idx = i
                break

        # Find last position with 'consecutive' values above average
        for i in range(len(sums) - 1, self.consecutive - 2, -1):
            if all(above_avg[i - self.consecutive + 1 : i + 1]):
                max_idx = i
                break

        return min_idx, max_idx

    def _make_square_roi(
        self,
        event_frame: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
    ) -> Tuple[int, int, int, int]:
        """Make ROI square by selecting the most active region.

        Based on Algorithm 3: if x_range > y_range, slide square window
        horizontally to find maximum intensity sum.

        Args:
            event_frame: Grayscale event frame
            x_min, x_max, y_min, y_max: Initial ROI boundaries

        Returns:
            (x_min, x_max, y_min, y_max) adjusted to form a square
        """
        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range == y_range:
            return x_min, x_max, y_min, y_max

        if x_range > y_range:
            # Slide square window horizontally to find max intensity
            best_x_min = x_min
            best_sum = 0
            for x in range(x_min, x_max - y_range + 1):
                window_sum = np.sum(event_frame[y_min:y_max, x : x + y_range])
                if window_sum > best_sum:
                    best_sum = window_sum
                    best_x_min = x
            return best_x_min, best_x_min + y_range, y_min, y_max
        else:
            # Slide square window vertically to find max intensity
            best_y_min = y_min
            best_sum = 0
            for y in range(y_min, y_max - x_range + 1):
                window_sum = np.sum(event_frame[y : y + x_range, x_min:x_max])
                if window_sum > best_sum:
                    best_sum = window_sum
                    best_y_min = y
            return x_min, x_max, best_y_min, best_y_min + x_range

    def __repr__(self) -> str:
        return (
            f"DVSTracker({self.width}x{self.height}, "
            f"consecutive={self.consecutive}, "
            f"min_active={self.min_active_ratio})"
        )
