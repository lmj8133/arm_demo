"""Laser dot tracking module.

Detects and tracks a laser pointer dot on a surface using
brightness-peak detection. Unlike HSV color tracking, this works
even when the laser dot saturates the camera sensor and appears
near-white rather than red.

Detection strategy:
  1. Convert to grayscale and apply Gaussian blur
  2. Find the brightness peak (local maximum)
  3. Threshold to isolate the bright spot
  4. Find small contours and pick the brightest one
  5. Validate: must be small, bright, and low saturation

Calibrated from recording_20260209_112349.avi:
  - Center pixel: V=245-255, S=3-19 (nearly white due to sensor saturation)
  - Dot size: 1-10 pixels (very small)
  - Background (white paper): ~200 gray level
  - Laser dot: ~215-255 gray level

Example:
    tracker = LaserTracker()
    target = tracker.detect(frame)
    if target:
        print(f"Laser at {target.center} with area {target.area}")
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaserTarget:
    """Detected laser dot target.

    Attributes:
        cx: Center X coordinate (pixels)
        cy: Center Y coordinate (pixels)
        area: Contour area (pixels^2)
        brightness: Peak brightness value (0-255)
        bbox: Bounding box as (x, y, width, height)
    """

    cx: float
    cy: float
    area: float
    brightness: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (cx, cy) in pixels."""
        return (self.cx, self.cy)

    def normalized_center(
        self, img_width: int, img_height: int
    ) -> Tuple[float, float]:
        """Get center normalized to 0-1 range."""
        return (self.cx / img_width, self.cy / img_height)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (x1, y1, x2, y2)."""
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

    def __repr__(self) -> str:
        return (
            f"LaserTarget(center=({self.cx:.0f}, {self.cy:.0f}), "
            f"area={self.area:.0f}, brightness={self.brightness:.0f})"
        )


class LaserTracker:
    """Laser dot tracker using brightness-peak detection.

    Designed for red laser pointers that saturate camera sensors,
    appearing as bright near-white spots rather than red spots.

    Example:
        # Basic usage
        tracker = LaserTracker()
        target = tracker.detect(frame)

        # With ROI (region of interest)
        tracker = LaserTracker(roi=(170, 180, 380, 390))
        target = tracker.detect(frame)

        # Tuned for specific conditions
        tracker = LaserTracker(
            brightness_threshold=235,
            max_dot_area=50,
            blur_kernel=9,
        )
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        brightness_threshold: int = 240,
        max_dot_area: int = 100,
        min_dot_area: int = 1,
        max_saturation: int = 40,
        blur_kernel: int = 5,
    ):
        """Initialize LaserTracker.

        Args:
            roi: Region of interest as (x1, y1, x2, y2), or None for full frame
            brightness_threshold: Minimum V (brightness) for laser center (0-255).
                                  Calibrated value: 240 (from recording analysis)
            max_dot_area: Maximum contour area for a laser dot (pixels^2)
            min_dot_area: Minimum contour area for a laser dot (pixels^2)
            max_saturation: Maximum S (saturation) for laser center.
                            Laser dots appear near-white, so S should be low.
                            Calibrated value: 40 (from recording analysis)
            blur_kernel: Gaussian blur kernel size for noise reduction
        """
        self.roi = roi
        self.brightness_threshold = brightness_threshold
        self.max_dot_area = max_dot_area
        self.min_dot_area = min_dot_area
        self.max_saturation = max_saturation
        self.blur_kernel = blur_kernel

    def detect(self, frame: np.ndarray) -> Optional[LaserTarget]:
        """Detect laser dot in frame.

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            LaserTarget for the detected laser dot, or None if not found
        """
        targets = self._detect_candidates(frame)
        if not targets:
            return None
        # Return the brightest candidate
        return max(targets, key=lambda t: t.brightness)

    def detect_all(
        self, frame: np.ndarray, max_count: int = 5
    ) -> List[LaserTarget]:
        """Detect all laser dot candidates in frame.

        Args:
            frame: BGR image as numpy array (H, W, C)
            max_count: Maximum number of targets to return

        Returns:
            List of LaserTarget objects sorted by brightness (descending)
        """
        targets = self._detect_candidates(frame)
        targets.sort(key=lambda t: t.brightness, reverse=True)
        return targets[:max_count]

    def create_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create binary mask for laser dot detection.

        Useful for debugging and visualization.

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            Binary mask (H, W) where 255 = potential laser dot
        """
        work, offset_x, offset_y = self._get_work_region(frame)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        # Bright pixels with low saturation (overexposed laser characteristic)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.brightness_threshold]),
            np.array([180, self.max_saturation, 255]),
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # If ROI is used, place mask back into full-frame coordinates
        if self.roi is not None:
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[offset_y:offset_y + work.shape[0],
                      offset_x:offset_x + work.shape[1]] = mask
            return full_mask

        return mask

    def _get_work_region(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, int, int]:
        """Extract the working region (ROI or full frame).

        Returns:
            (region, offset_x, offset_y) where offsets map back to full frame
        """
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            return frame[y1:y2, x1:x2], x1, y1
        return frame, 0, 0

    def _detect_candidates(self, frame: np.ndarray) -> List[LaserTarget]:
        """Internal detection pipeline."""
        work, offset_x, offset_y = self._get_work_region(frame)
        if work.size == 0:
            return []

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        ks = self.blur_kernel
        if ks % 2 == 0:
            ks += 1
        blurred = cv2.GaussianBlur(gray, (ks, ks), 0)

        # Create mask: high brightness + low saturation
        # This captures the overexposed laser center
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        bright_mask = v_channel >= self.brightness_threshold
        low_sat_mask = s_channel <= self.max_saturation
        combined = (bright_mask & low_sat_mask).astype(np.uint8) * 255

        # Morphological cleanup to remove isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        targets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_dot_area or area > self.max_dot_area:
                continue

            # Compute center
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w / 2
                cy = y + h / 2

            # Get peak brightness at the center
            icx, icy = int(cx), int(cy)
            if 0 <= icy < blurred.shape[0] and 0 <= icx < blurred.shape[1]:
                brightness = float(blurred[icy, icx])
            else:
                brightness = 0.0

            x, y, w, h = cv2.boundingRect(c)

            targets.append(LaserTarget(
                cx=cx + offset_x,
                cy=cy + offset_y,
                area=area,
                brightness=brightness,
                bbox=(x + offset_x, y + offset_y, w, h),
            ))

        # Fallback: if no contours found, try peak brightness method
        if not targets:
            target = self._detect_by_peak(blurred, hsv, offset_x, offset_y)
            if target is not None:
                targets.append(target)

        return targets

    def _detect_by_peak(
        self,
        blurred: np.ndarray,
        hsv: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> Optional[LaserTarget]:
        """Fallback: detect laser by finding the absolute brightness peak.

        This handles cases where the laser dot is too small to form a contour
        but still creates a visible brightness peak.
        """
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

        if max_val < self.brightness_threshold:
            return None

        cx, cy = max_loc
        # Verify the peak has low saturation (characteristic of laser)
        if cy < hsv.shape[0] and cx < hsv.shape[1]:
            s_val = hsv[cy, cx, 1]
            if s_val > self.max_saturation:
                return None

        return LaserTarget(
            cx=float(cx + offset_x),
            cy=float(cy + offset_y),
            area=1.0,
            brightness=float(max_val),
            bbox=(max(0, cx + offset_x - 2), max(0, cy + offset_y - 2), 5, 5),
        )

    @staticmethod
    def calibrate_from_video(
        video_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        sample_count: int = 20,
    ) -> dict:
        """Analyze a video to extract laser dot HSV parameters.

        Args:
            video_path: Path to the calibration video file
            roi: Region of interest (x1, y1, x2, y2), or None for full frame
            sample_count: Number of frames to sample

        Returns:
            Dict with calibration stats (brightness, saturation ranges)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, min(sample_count, total), dtype=int)

        v_peaks = []
        s_at_peaks = []
        positions = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            if roi:
                x1, y1, x2, y2 = roi
                work = frame[y1:y2, x1:x2]
                ox, oy = x1, y1
            else:
                work = frame
                ox, oy = 0, 0

            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)

            _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
            cx, cy = max_loc

            v_peaks.append(max_val)
            s_at_peaks.append(int(hsv[cy, cx, 1]))
            positions.append((cx + ox, cy + oy))

        cap.release()

        if not v_peaks:
            raise ValueError(
                f"No frames could be read from video: {video_path}"
            )

        return {
            "brightness_min": float(np.min(v_peaks)),
            "brightness_max": float(np.max(v_peaks)),
            "brightness_mean": float(np.mean(v_peaks)),
            "saturation_min": int(np.min(s_at_peaks)),
            "saturation_max": int(np.max(s_at_peaks)),
            "saturation_mean": float(np.mean(s_at_peaks)),
            "position_x_mean": float(np.mean([p[0] for p in positions])),
            "position_y_mean": float(np.mean([p[1] for p in positions])),
            "position_x_std": float(np.std([p[0] for p in positions])),
            "position_y_std": float(np.std([p[1] for p in positions])),
            "suggested_brightness_threshold": int(np.min(v_peaks) * 0.95),
            "suggested_max_saturation": int(np.max(s_at_peaks) * 1.5),
            "num_samples": len(v_peaks),
        }

    def __repr__(self) -> str:
        roi_str = f", roi={self.roi}" if self.roi else ""
        return (
            f"LaserTracker(brightness>={self.brightness_threshold}, "
            f"saturation<={self.max_saturation}, "
            f"area=[{self.min_dot_area},{self.max_dot_area}]{roi_str})"
        )
