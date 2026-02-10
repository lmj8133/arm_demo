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

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

# --- Shared constants / utilities (imported by calibrate_laser, main_laser_drawing) ---

DEFAULT_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "laser_profile.json")

ROTATE_FLAGS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def parse_roi(roi_str: str) -> Tuple[int, int, int, int]:
    """Parse ROI string 'x1,y1,x2,y2' into a 4-int tuple."""
    parts = [int(x.strip()) for x in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 4 comma-separated integers: x1,y1,x2,y2")
    return (parts[0], parts[1], parts[2], parts[3])


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


@dataclass
class LaserProfile:
    """Persisted calibration profile for LaserTracker."""

    brightness_threshold: int = 240
    max_saturation: int = 40
    min_dot_area: int = 1
    max_dot_area: int = 100
    blur_kernel: int = 5
    use_hue_filter: bool = False
    hue_red_low_upper: int = 10       # H range: [0, this]
    hue_red_high_lower: int = 170     # H range: [this, 180]
    hue_min_saturation: int = 30      # Min S for hue ring validation

    def save(self, path: str) -> None:
        """Save profile to JSON file."""
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "LaserProfile":
        """Load profile from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return LaserProfile.from_dict(data)

    @staticmethod
    def from_dict(d: dict) -> "LaserProfile":
        """Create profile from dict, ignoring unknown keys."""
        valid = {f.name for f in dataclasses.fields(LaserProfile)}
        return LaserProfile(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return dataclasses.asdict(self)


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
        use_hue_filter: bool = False,
        hue_red_low_upper: int = 10,
        hue_red_high_lower: int = 170,
        hue_min_saturation: int = 30,
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
            use_hue_filter: Enable hue-ring validation for red laser
            hue_red_low_upper: Upper bound of low red hue range [0, this]
            hue_red_high_lower: Lower bound of high red hue range [this, 180]
            hue_min_saturation: Minimum saturation for hue ring validation
        """
        self.roi = roi
        self.brightness_threshold = brightness_threshold
        self.max_dot_area = max_dot_area
        self.min_dot_area = min_dot_area
        self.max_saturation = max_saturation
        self.blur_kernel = blur_kernel
        self.use_hue_filter = use_hue_filter
        self.hue_red_low_upper = hue_red_low_upper
        self.hue_red_high_lower = hue_red_high_lower
        self.hue_min_saturation = hue_min_saturation

    @classmethod
    def from_profile(
        cls,
        profile: "LaserProfile",
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> "LaserTracker":
        """Create tracker from a saved profile."""
        return cls(
            roi=roi,
            brightness_threshold=profile.brightness_threshold,
            max_saturation=profile.max_saturation,
            min_dot_area=profile.min_dot_area,
            max_dot_area=profile.max_dot_area,
            blur_kernel=profile.blur_kernel,
            use_hue_filter=profile.use_hue_filter,
            hue_red_low_upper=profile.hue_red_low_upper,
            hue_red_high_lower=profile.hue_red_high_lower,
            hue_min_saturation=profile.hue_min_saturation,
        )

    def to_profile(self) -> "LaserProfile":
        """Export current parameters as a LaserProfile."""
        return LaserProfile(
            brightness_threshold=self.brightness_threshold,
            max_saturation=self.max_saturation,
            min_dot_area=self.min_dot_area,
            max_dot_area=self.max_dot_area,
            blur_kernel=self.blur_kernel,
            use_hue_filter=self.use_hue_filter,
            hue_red_low_upper=self.hue_red_low_upper,
            hue_red_high_lower=self.hue_red_high_lower,
            hue_min_saturation=self.hue_min_saturation,
        )

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

        # When hue filter is active, also include red-hue pixels around
        # the overexposed center so the mask visualization shows what
        # the hue ring validates against.
        if self.use_hue_filter:
            hue_low = cv2.inRange(
                hsv,
                np.array([0, self.hue_min_saturation, self.brightness_threshold]),
                np.array([self.hue_red_low_upper, 255, 255]),
            )
            hue_high = cv2.inRange(
                hsv,
                np.array([self.hue_red_high_lower, self.hue_min_saturation,
                           self.brightness_threshold]),
                np.array([180, 255, 255]),
            )
            hue_mask = cv2.bitwise_or(hue_low, hue_high)
            mask = cv2.bitwise_or(mask, hue_mask)

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

    def _validate_hue_ring(
        self,
        hsv: np.ndarray,
        cx: int,
        cy: int,
        ring_inner: int = 2,
        ring_outer: int = 8,
        min_red_ratio: float = 0.3,
    ) -> bool:
        """Check if annular ring around (cx, cy) contains enough red hue pixels.

        The laser center is overexposed (low S, unstable H), but the
        surrounding glow retains red hue information.  We sample an
        annular ring and require >= *min_red_ratio* of its pixels to
        be red (H in [0, hue_red_low_upper] or [hue_red_high_lower, 180])
        with S >= hue_min_saturation.
        """
        h_img, w_img = hsv.shape[:2]

        # Build coordinate grid for ring
        y_lo = max(0, cy - ring_outer)
        y_hi = min(h_img, cy + ring_outer + 1)
        x_lo = max(0, cx - ring_outer)
        x_hi = min(w_img, cx + ring_outer + 1)

        patch = hsv[y_lo:y_hi, x_lo:x_hi]
        if patch.size == 0:
            return False

        # Create distance mask for annular ring
        ys = np.arange(y_lo, y_hi) - cy
        xs = np.arange(x_lo, x_hi) - cx
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        dist_sq = xx * xx + yy * yy
        ring_mask = (dist_sq >= ring_inner * ring_inner) & (
            dist_sq <= ring_outer * ring_outer
        )

        ring_pixels = patch[ring_mask]
        if len(ring_pixels) == 0:
            return False

        h_vals = ring_pixels[:, 0]
        s_vals = ring_pixels[:, 1]

        is_red = (
            ((h_vals <= self.hue_red_low_upper) | (h_vals >= self.hue_red_high_lower))
            & (s_vals >= self.hue_min_saturation)
        )
        ratio = np.count_nonzero(is_red) / len(ring_pixels)
        return ratio >= min_red_ratio

    @staticmethod
    def _collect_ring_stats(
        hsv: np.ndarray, cx: int, cy: int,
        ring_inner: int = 2, ring_outer: int = 8,
    ) -> dict:
        """Collect raw HSV statistics from an annular ring around (cx, cy).

        Returns:
            Dict with h_vals, s_vals (numpy arrays) and pixel_count.
            Empty arrays if ring has no pixels.
        """
        h_img, w_img = hsv.shape[:2]

        y_lo = max(0, cy - ring_outer)
        y_hi = min(h_img, cy + ring_outer + 1)
        x_lo = max(0, cx - ring_outer)
        x_hi = min(w_img, cx + ring_outer + 1)

        patch = hsv[y_lo:y_hi, x_lo:x_hi]
        if patch.size == 0:
            return {"h_vals": np.array([], dtype=np.uint8),
                    "s_vals": np.array([], dtype=np.uint8),
                    "pixel_count": 0}

        ys = np.arange(y_lo, y_hi) - cy
        xs = np.arange(x_lo, x_hi) - cx
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        dist_sq = xx * xx + yy * yy
        ring_mask = (dist_sq >= ring_inner * ring_inner) & (
            dist_sq <= ring_outer * ring_outer
        )

        ring_pixels = patch[ring_mask]
        if len(ring_pixels) == 0:
            return {"h_vals": np.array([], dtype=np.uint8),
                    "s_vals": np.array([], dtype=np.uint8),
                    "pixel_count": 0}

        return {
            "h_vals": ring_pixels[:, 0],
            "s_vals": ring_pixels[:, 1],
            "pixel_count": len(ring_pixels),
        }

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

            # Hue ring validation: reject candidates without red glow
            if self.use_hue_filter and not self._validate_hue_ring(
                hsv, icx, icy
            ):
                continue

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

        # Hue ring validation for peak-based fallback
        if self.use_hue_filter and not self._validate_hue_ring(hsv, cx, cy):
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
        h_at_peaks = []
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
            h_at_peaks.append(int(hsv[cy, cx, 0]))
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
            "hue_min": int(np.min(h_at_peaks)),
            "hue_max": int(np.max(h_at_peaks)),
            "hue_mean": float(np.mean(h_at_peaks)),
            "position_x_mean": float(np.mean([p[0] for p in positions])),
            "position_y_mean": float(np.mean([p[1] for p in positions])),
            "position_x_std": float(np.std([p[0] for p in positions])),
            "position_y_std": float(np.std([p[1] for p in positions])),
            "suggested_brightness_threshold": int(np.min(v_peaks) * 0.95),
            "suggested_max_saturation": int(np.max(s_at_peaks) * 1.5),
            "num_samples": len(v_peaks),
        }

    @staticmethod
    def calibrate_profile_from_video(
        video_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        auto_quad: bool = True,
        skip_frames: int = 0,
        rotate: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple["LaserProfile", dict]:
        """Analyze a calibration video and derive a LaserProfile automatically.

        Uses a loose bootstrap tracker to detect laser candidates, then
        computes thresholds from collected statistics.

        Args:
            video_path: Path to the calibration video file
            roi: Manual ROI as (x1, y1, x2, y2); skips quad detection if given
            auto_quad: Auto-detect black quad to derive ROI (when roi is None)
            skip_frames: Process every N-th frame (0 = all frames)
            rotate: Rotate frames CW by degrees (0, 90, 180, 270)
            progress_callback: Called with (current_frame, total_frames)

        Returns:
            (LaserProfile, stats_dict) where stats_dict contains raw statistics
            and quad information.

        Raises:
            FileNotFoundError: If video cannot be opened
            ValueError: If no laser candidates detected in any frame
        """
        rotate_flag = ROTATE_FLAGS.get(rotate)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Read first frame for ROI detection
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise FileNotFoundError(f"Cannot read first frame: {video_path}")
        if rotate_flag is not None:
            first_frame = cv2.rotate(first_frame, rotate_flag)

        frame_h, frame_w = first_frame.shape[:2]
        stats: dict = {
            "video_path": video_path,
            "total_frames": total,
            "fps": fps,
            "resolution": (frame_w, frame_h),
            "rotate": rotate,
            "quad_corners": None,
        }

        # --- Step 0: ROI detection ---
        detected_roi = roi
        if detected_roi is not None:
            stats["roi_source"] = "manual"
        elif auto_quad:
            try:
                from quad_detector import QuadDetector
            except ImportError:
                import os
                import sys
                sys.path.insert(0, os.path.dirname(__file__))
                from quad_detector import QuadDetector

            qd = QuadDetector()
            quad = qd.detect(first_frame)
            if quad is not None:
                detected_roi = quad.as_xyxy()
                stats["quad_corners"] = quad.corners.tolist()
                stats["roi_source"] = "quad"
                print(f"[CALIBRATE] Quad detected, ROI={detected_roi}")
            else:
                print("[WARNING] Quad not detected, using full frame")
                stats["roi_source"] = "full_frame"
        else:
            stats["roi_source"] = "full_frame"

        # --- Step 1: Bootstrap detection (loose thresholds) ---
        bootstrap = LaserTracker(
            roi=detected_roi,
            brightness_threshold=200,
            max_saturation=80,
            min_dot_area=0,
            max_dot_area=500,
            blur_kernel=5,
            use_hue_filter=False,
        )

        v_peaks: list = []
        s_peaks: list = []
        areas: list = []
        ring_h_all: list = []
        ring_s_all: list = []
        red_ratios: list = []
        frames_processed = 0
        frames_detected = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames if requested
            if skip_frames > 0 and (frame_idx - 1) % (skip_frames + 1) != 0:
                continue

            if rotate_flag is not None:
                frame = cv2.rotate(frame, rotate_flag)

            frames_processed += 1
            if progress_callback:
                progress_callback(frame_idx, total)

            candidates = bootstrap._detect_candidates(frame)
            if not candidates:
                continue

            # Pick brightest candidate
            best = max(candidates, key=lambda t: t.brightness)
            frames_detected += 1

            v_peaks.append(best.brightness)
            # Read S at center from the work region
            work, ox, oy = bootstrap._get_work_region(frame)
            hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
            lcx = int(best.cx - ox)
            lcy = int(best.cy - oy)
            if 0 <= lcy < hsv.shape[0] and 0 <= lcx < hsv.shape[1]:
                s_peaks.append(int(hsv[lcy, lcx, 1]))
            else:
                s_peaks.append(0)
            areas.append(best.area)

            # Collect ring stats
            ring = LaserTracker._collect_ring_stats(hsv, lcx, lcy)
            if ring["pixel_count"] > 0:
                h_vals = ring["h_vals"]
                s_vals = ring["s_vals"]
                ring_h_all.extend(h_vals.tolist())
                ring_s_all.extend(s_vals.tolist())

                # Red ratio for this sample
                is_red = (h_vals <= 10) | (h_vals >= 170)
                red_ratio = np.count_nonzero(is_red) / len(h_vals)
                red_ratios.append(red_ratio)

        cap.release()

        if not v_peaks:
            raise ValueError(
                f"No laser candidates detected in {frames_processed} frames "
                f"from {video_path}"
            )

        detection_rate = frames_detected / max(1, frames_processed)
        if detection_rate < 0.5:
            print(
                f"[WARNING] Low detection rate: {frames_detected}/{frames_processed} "
                f"({detection_rate:.0%})"
            )

        # --- Step 2: Derive thresholds from statistics ---
        v_arr = np.array(v_peaks)
        s_arr = np.array(s_peaks)
        a_arr = np.array(areas)

        brightness_threshold = max(200, int(np.percentile(v_arr, 5) * 0.92))
        max_saturation = min(80, int(np.percentile(s_arr, 95) * 1.3))
        min_dot_area = max(0, int(np.percentile(a_arr, 2) * 0.5))
        max_dot_area = min(500, int(np.percentile(a_arr, 98) * 2.0))

        # Hue analysis
        use_hue_filter = False
        hue_red_low_upper = 10
        hue_red_high_lower = 170
        hue_min_saturation = 30

        ring_h_arr = np.array(ring_h_all, dtype=np.uint8)
        ring_s_arr = np.array(ring_s_all, dtype=np.uint8)

        if len(ring_h_arr) > 0:
            h_low = ring_h_arr[ring_h_arr <= 30]
            h_high = ring_h_arr[ring_h_arr >= 150]

            if len(h_low) > 0:
                hue_red_low_upper = int(np.clip(
                    np.percentile(h_low, 95) + 3, 5, 30
                ))
            if len(h_high) > 0:
                hue_red_high_lower = int(np.clip(
                    np.percentile(h_high, 5) - 3, 150, 179
                ))

            # Decide whether to enable hue filter
            red_sample_count = len(h_low) + len(h_high)
            median_red_ratio = float(np.median(red_ratios)) if red_ratios else 0.0
            use_hue_filter = median_red_ratio >= 0.25 and red_sample_count > 20

            # Min saturation for hue ring
            if len(ring_s_arr) > 0:
                # Filter to only red-hue ring pixels for S threshold
                is_red_mask = (ring_h_arr <= hue_red_low_upper) | (
                    ring_h_arr >= hue_red_high_lower
                )
                ring_s_red = ring_s_arr[is_red_mask]
                if len(ring_s_red) > 0:
                    hue_min_saturation = max(
                        20, int(np.percentile(ring_s_red, 5) * 0.8)
                    )

        profile = LaserProfile(
            brightness_threshold=brightness_threshold,
            max_saturation=max_saturation,
            min_dot_area=min_dot_area,
            max_dot_area=max_dot_area,
            blur_kernel=5,
            use_hue_filter=use_hue_filter,
            hue_red_low_upper=hue_red_low_upper,
            hue_red_high_lower=hue_red_high_lower,
            hue_min_saturation=hue_min_saturation,
        )

        stats.update({
            "frames_processed": frames_processed,
            "frames_detected": frames_detected,
            "detection_rate": detection_rate,
            "v_peaks": {"min": float(v_arr.min()), "max": float(v_arr.max()),
                        "mean": float(v_arr.mean()), "p5": float(np.percentile(v_arr, 5))},
            "s_peaks": {"min": float(s_arr.min()), "max": float(s_arr.max()),
                        "mean": float(s_arr.mean()), "p95": float(np.percentile(s_arr, 95))},
            "areas": {"min": float(a_arr.min()), "max": float(a_arr.max()),
                      "mean": float(a_arr.mean())},
            "hue_red_samples": int(len(ring_h_arr)),
            "median_red_ratio": float(np.median(red_ratios)) if red_ratios else 0.0,
            "roi": detected_roi,
        })

        return profile, stats

    def __repr__(self) -> str:
        roi_str = f", roi={self.roi}" if self.roi else ""
        hue_str = (
            f", hue=[0-{self.hue_red_low_upper}|{self.hue_red_high_lower}-180]"
            if self.use_hue_filter else ""
        )
        return (
            f"LaserTracker(brightness>={self.brightness_threshold}, "
            f"saturation<={self.max_saturation}, "
            f"area=[{self.min_dot_area},{self.max_dot_area}]"
            f"{hue_str}{roi_str})"
        )
