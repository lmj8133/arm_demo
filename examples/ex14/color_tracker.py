"""HSV color-based object tracking module.

Provides ColorTracker class for detecting and tracking objects by color
using HSV color space. No AI model required.

Example:
    tracker = ColorTracker(color="red", min_area=500)
    target = tracker.detect(frame)
    if target:
        print(f"Target at {target.center} with area {target.area}")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ColorTarget:
    """Detected color target.

    Attributes:
        cx: Center X coordinate (pixels)
        cy: Center Y coordinate (pixels)
        area: Contour area (pixels^2)
        bbox: Bounding box as (x, y, width, height)
    """

    cx: float
    cy: float
    area: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (cx, cy) in pixels."""
        return (self.cx, self.cy)

    def normalized_center(self, img_width: int, img_height: int) -> Tuple[float, float]:
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
            f"ColorTarget(center=({self.cx:.0f}, {self.cy:.0f}), area={self.area:.0f})"
        )


# HSV color range type: list of (lower, upper) tuples
# Each tuple defines a range in HSV space
HSVRange = List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]


class ColorTracker:
    """HSV color-based object tracker.

    Detects objects by color in HSV color space. Supports preset colors
    (red, green, blue, yellow) or custom HSV ranges.

    Example:
        # Using preset color
        tracker = ColorTracker(color="red")
        target = tracker.detect(frame)

        # Using custom HSV range
        tracker = ColorTracker(hsv_ranges=[
            ((30, 100, 100), (50, 255, 255)),  # Yellow-ish
        ])
        target = tracker.detect(frame)
    """

    # Preset HSV ranges for common colors
    # Red wraps around hue 0/180, so we need two ranges
    PRESETS: Dict[str, HSVRange] = {
        "red": [
            ((0, 100, 100), (10, 255, 255)),  # Low red
            ((160, 100, 100), (180, 255, 255)),  # High red
        ],
        "green": [
            ((35, 100, 100), (85, 255, 255)),
        ],
        "blue": [
            ((100, 100, 100), (130, 255, 255)),
        ],
        "yellow": [
            ((20, 100, 100), (35, 255, 255)),
        ],
    }

    def __init__(
        self,
        color: Optional[str] = None,
        hsv_ranges: Optional[HSVRange] = None,
        min_area: int = 500,
    ):
        """Initialize ColorTracker.

        Args:
            color: Preset color name ("red", "green", "blue", "yellow")
            hsv_ranges: Custom HSV ranges as list of (lower, upper) tuples
                        Each tuple contains HSV values (H: 0-180, S: 0-255, V: 0-255)
            min_area: Minimum contour area to consider (default: 500 pixels^2)

        Raises:
            ValueError: If neither color nor hsv_ranges is specified,
                       or if color is not a valid preset
        """
        if color is None and hsv_ranges is None:
            raise ValueError("Must specify either 'color' or 'hsv_ranges'")

        if color is not None:
            color_lower = color.lower()
            if color_lower not in self.PRESETS:
                valid = ", ".join(self.PRESETS.keys())
                raise ValueError(f"Unknown color '{color}'. Valid options: {valid}")
            self.hsv_ranges: HSVRange = self.PRESETS[color_lower]
            self.color_name = color_lower
        else:
            # hsv_ranges is guaranteed not None here (checked above)
            assert hsv_ranges is not None
            self.hsv_ranges = hsv_ranges
            self.color_name = "custom"

        self.min_area = min_area

    def set_color(self, color: str) -> None:
        """Change to a preset color.

        Args:
            color: Preset color name ("red", "green", "blue", "yellow")

        Raises:
            ValueError: If color is not a valid preset
        """
        color_lower = color.lower()
        if color_lower not in self.PRESETS:
            valid = ", ".join(self.PRESETS.keys())
            raise ValueError(f"Unknown color '{color}'. Valid options: {valid}")
        self.hsv_ranges = self.PRESETS[color_lower]
        self.color_name = color_lower

    def set_hsv_ranges(self, hsv_ranges: HSVRange) -> None:
        """Set custom HSV ranges.

        Args:
            hsv_ranges: List of (lower, upper) HSV tuples
        """
        self.hsv_ranges = hsv_ranges
        self.color_name = "custom"

    def create_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create binary mask for the target color.

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            Binary mask (H, W) where 255 = color detected
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create combined mask for all HSV ranges
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for lower, upper in self.hsv_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

    def detect(self, frame: np.ndarray) -> Optional[ColorTarget]:
        """Detect largest color blob in frame.

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            ColorTarget for the largest detected blob, or None if no detection
        """
        mask = self.create_mask(frame)
        return self._find_largest_contour(mask)

    def detect_all(self, frame: np.ndarray, max_count: int = 10) -> List[ColorTarget]:
        """Detect all color blobs in frame.

        Args:
            frame: BGR image as numpy array (H, W, C)
            max_count: Maximum number of targets to return (default: 10)

        Returns:
            List of ColorTarget objects sorted by area (descending)
        """
        mask = self.create_mask(frame)
        return self._find_all_contours(mask, max_count)

    def _find_largest_contour(self, mask: np.ndarray) -> Optional[ColorTarget]:
        """Find the largest contour in the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter by minimum area and find largest
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        return self._contour_to_target(largest)

    def _find_all_contours(self, mask: np.ndarray, max_count: int) -> List[ColorTarget]:
        """Find all contours in the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Filter by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        if not valid_contours:
            return []

        # Sort by area descending and limit count
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[
            :max_count
        ]

        return [self._contour_to_target(c) for c in sorted_contours]

    def _contour_to_target(self, contour: np.ndarray) -> ColorTarget:
        """Convert a contour to ColorTarget."""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Compute center using moments for better accuracy
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            # Fallback to bbox center
            cx = x + w / 2
            cy = y + h / 2

        return ColorTarget(cx=cx, cy=cy, area=area, bbox=(x, y, w, h))

    @classmethod
    def available_colors(cls) -> List[str]:
        """Get list of available preset color names."""
        return list(cls.PRESETS.keys())

    def __repr__(self) -> str:
        return f"ColorTracker(color={self.color_name}, min_area={self.min_area})"
