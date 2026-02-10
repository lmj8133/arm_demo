"""Black quadrilateral detection on white paper.

Detects hand-drawn black quadrilaterals (rectangles/trapezoids) on white
paper using Canny edge detection + contour approximation. Designed for
static scenes where a camera views a white surface with a drawn black border.

Detection pipeline:
  1. Optional ROI crop
  2. Grayscale + Gaussian blur
  3. Canny edge detection
  4. Dilate edges to close small gaps
  5. Find contours + polygon approximation (approxPolyDP)
  6. Filter: 4 vertices, area range, convexity, aspect ratio, angles
  7. Interior brightness check — white paper inside the quad
  8. Adaptive epsilon search if initial attempt fails
  9. Sub-pixel corner refinement (cornerSubPix)
  10. Corner ordering: TL, TR, BR, BL (clockwise)

Calibrated from recording_20260209_112349.avi:
  - Black line gray: ~100-130 (lighter due to camera angle), white paper: ~195-200
  - Line width: ~8-15 px
  - Detected inner quad area: ~11000 px^2 in 640x480 frame

Example:
    detector = QuadDetector()
    target = detector.detect(frame)
    if target:
        print(f"Quad corners: {target.corners}")
        print(f"Area: {target.area:.0f}, Center: {target.center}")
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class QuadTarget:
    """Detected quadrilateral target.

    Attributes:
        corners: 4 corner points as (4, 2) array, clockwise [TL, TR, BR, BL]
        area: Polygon area (pixels^2)
        perimeter: Polygon perimeter (pixels)
        contour: Original OpenCV contour
    """

    corners: np.ndarray  # shape (4, 2), dtype float32
    area: float
    perimeter: float
    contour: np.ndarray

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (cx, cy) in pixels."""
        cx = float(self.corners[:, 0].mean())
        cy = float(self.corners[:, 1].mean())
        return (cx, cy)

    def normalized_center(
        self, img_width: int, img_height: int
    ) -> Tuple[float, float]:
        """Get center normalized to 0-1 range."""
        cx, cy = self.center
        return (cx / img_width, cy / img_height)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Get axis-aligned bounding box as (x1, y1, x2, y2)."""
        x1 = int(self.corners[:, 0].min())
        y1 = int(self.corners[:, 1].min())
        x2 = int(self.corners[:, 0].max())
        y2 = int(self.corners[:, 1].max())
        return (x1, y1, x2, y2)

    def corner_labels(self) -> List[Tuple[str, Tuple[int, int]]]:
        """Get labeled corners as [(label, (x, y)), ...]."""
        labels = ["TL", "TR", "BR", "BL"]
        return [
            (label, (int(self.corners[i, 0]), int(self.corners[i, 1])))
            for i, label in enumerate(labels)
        ]

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"QuadTarget(center=({cx:.0f}, {cy:.0f}), "
            f"area={self.area:.0f})"
        )


class QuadDetector:
    """Black quadrilateral detector on white paper.

    Uses Canny edge detection to find the inner boundary of drawn black
    lines. The detected contour represents the white interior enclosed
    by the drawn quadrilateral.

    Example:
        # Basic usage
        detector = QuadDetector()
        target = detector.detect(frame)

        # With ROI
        detector = QuadDetector(roi=(170, 180, 400, 390))
        target = detector.detect(frame)
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area: int = 2000,
        max_area: int = 80000,
        blur_kernel: int = 5,
        min_angle: float = 30.0,
        min_interior_brightness: int = 150,
        max_interior_std: float = 40.0,
    ):
        """Initialize QuadDetector.

        Args:
            roi: Region of interest as (x1, y1, x2, y2), or None for full frame
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            min_area: Minimum polygon area (pixels^2)
            max_area: Maximum polygon area (pixels^2)
            blur_kernel: Gaussian blur kernel size
            min_angle: Minimum interior angle (degrees) to reject degenerate quads
            min_interior_brightness: Minimum mean gray inside the quad (white paper check)
            max_interior_std: Maximum gray std inside the quad (uniform surface check)
        """
        self.roi = roi
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel
        self.min_angle = min_angle
        self.min_interior_brightness = min_interior_brightness
        self.max_interior_std = max_interior_std

    def detect(self, frame: np.ndarray) -> Optional[QuadTarget]:
        """Detect the most prominent black quadrilateral.

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            QuadTarget or None if no quadrilateral found
        """
        targets = self.detect_all(frame)
        if not targets:
            return None
        # Return the largest by area
        return max(targets, key=lambda t: t.area)

    def detect_all(
        self, frame: np.ndarray, max_count: int = 5
    ) -> List[QuadTarget]:
        """Detect all black quadrilaterals in frame.

        Args:
            frame: BGR image as numpy array (H, W, C)
            max_count: Maximum number of results

        Returns:
            List of QuadTarget sorted by area (descending)
        """
        work, offset_x, offset_y = self._get_work_region(frame)
        if work.size == 0:
            return []

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        ks = self.blur_kernel | 1  # ensure odd
        blurred = cv2.GaussianBlur(gray, (ks, ks), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Dilate to close small edge gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find all contours (RETR_LIST to get both inner and outer)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        targets = []
        for contour in contours:
            target = self._try_extract_quad(contour, gray, offset_x, offset_y)
            if target is not None:
                targets.append(target)

        targets.sort(key=lambda t: t.area, reverse=True)
        return targets[:max_count]

    def create_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create Canny edge mask (for debugging).

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            Edge mask (H, W) where 255 = edge detected
        """
        work, offset_x, offset_y = self._get_work_region(frame)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        ks = self.blur_kernel | 1
        blurred = cv2.GaussianBlur(gray, (ks, ks), 0)

        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        if self.roi is not None:
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[
                offset_y : offset_y + work.shape[0],
                offset_x : offset_x + work.shape[1],
            ] = edges
            return full_mask

        return edges

    def _get_work_region(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, int, int]:
        """Extract working region (ROI or full frame).

        Returns:
            (region, offset_x, offset_y) for mapping back to full frame
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

    def _try_extract_quad(
        self,
        contour: np.ndarray,
        gray: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> Optional[QuadTarget]:
        """Try to extract a valid quadrilateral from a contour.

        Uses adaptive epsilon search: tries multiple epsilon values
        to find a 4-vertex polygon approximation.
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 50:
            return None

        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return None

        # Try multiple epsilon values to find 4-vertex approximation
        epsilons = [0.02, 0.015, 0.025, 0.03, 0.04]
        for eps_factor in epsilons:
            approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
            if len(approx) == 4:
                target = self._validate_quad(
                    approx, contour, gray, offset_x, offset_y
                )
                if target is not None:
                    return target

        return None

    def _validate_quad(
        self,
        approx: np.ndarray,
        contour: np.ndarray,
        gray: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> Optional[QuadTarget]:
        """Validate a 4-vertex polygon and build QuadTarget if valid."""
        area = cv2.contourArea(approx)
        if area < self.min_area or area > self.max_area:
            return None

        # Must be convex
        if not cv2.isContourConvex(approx):
            return None

        # Check aspect ratio of bounding rect
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            return None
        aspect = w / h
        if aspect < 0.3 or aspect > 3.0:
            return None

        # Check interior angles
        pts = approx.reshape(4, 2).astype(np.float64)
        if not self._check_angles(pts):
            return None

        # Check interior brightness — must be bright white paper
        if not self._check_interior(approx, gray):
            return None

        # Sub-pixel corner refinement
        corners = self._refine_corners(pts, gray)

        # Map corners back to full frame coordinates
        corners[:, 0] += offset_x
        corners[:, 1] += offset_y

        # Order corners clockwise: TL, TR, BR, BL
        ordered = self._order_corners(corners)

        perimeter = cv2.arcLength(approx, True)

        return QuadTarget(
            corners=ordered.astype(np.float32),
            area=float(area),
            perimeter=float(perimeter),
            contour=contour,
        )

    def _check_angles(self, pts: np.ndarray) -> bool:
        """Check that all interior angles are above min_angle."""
        for i in range(4):
            p1 = pts[(i - 1) % 4]
            p2 = pts[i]
            p3 = pts[(i + 1) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return False

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle < self.min_angle:
                return False

        return True

    def _check_interior(self, approx: np.ndarray, gray: np.ndarray) -> bool:
        """Verify that the interior is bright white paper.

        Creates a filled mask of the polygon, erodes slightly to avoid
        edge pixels, then checks mean brightness and uniformity.
        """
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [approx], 255)

        # Erode to exclude edge pixels from the measurement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=5)

        if not np.any(mask > 0):
            return False

        interior_mean = cv2.mean(gray, mask=mask)[0]
        interior_std = float(np.std(gray[mask > 0]))

        return (
            interior_mean >= self.min_interior_brightness
            and interior_std <= self.max_interior_std
        )

    def _refine_corners(self, pts: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Refine corner positions to sub-pixel accuracy."""
        corners = pts.astype(np.float32).reshape(-1, 1, 2)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.01,
        )
        try:
            refined = cv2.cornerSubPix(
                gray, corners, winSize=(5, 5), zeroZone=(-1, -1),
                criteria=criteria,
            )
            return refined.reshape(4, 2)
        except cv2.error:
            # Fallback if cornerSubPix fails (e.g., corners near border)
            return pts.astype(np.float32)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Order corners clockwise: TL, TR, BR, BL.

        Strategy: sort by Y to get top/bottom pairs, then sort each by X.
        """
        # Sort by Y coordinate
        y_sorted = pts[pts[:, 1].argsort()]

        # Top two points (smallest Y)
        top = y_sorted[:2]
        top = top[top[:, 0].argsort()]  # sort by X: left, right
        tl, tr = top[0], top[1]

        # Bottom two points (largest Y)
        bottom = y_sorted[2:]
        bottom = bottom[bottom[:, 0].argsort()]  # sort by X: left, right
        bl, br = bottom[0], bottom[1]

        return np.array([tl, tr, br, bl], dtype=pts.dtype)

    def __repr__(self) -> str:
        roi_str = f", roi={self.roi}" if self.roi else ""
        return (
            f"QuadDetector(canny=[{self.canny_low},{self.canny_high}], "
            f"area=[{self.min_area},{self.max_area}]{roi_str})"
        )
