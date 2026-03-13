"""Interactive quadrilateral calibration for DVS coordinate mapping.

Provides a drag-based four-corner calibration UI using a hybrid RGB preview
from the XenReal camera, plus homography utilities for mapping DVS pixel
coordinates to a normalised unit square.

Public API:
    run_quad_calibration(xe_cam, scale, window_name, initial_corners) -> Optional[np.ndarray]
    compute_homography(corners)          -> np.ndarray
    warp_point(matrix, x, y)             -> Tuple[float, float]
    save_calibration(corners, path)      -> None
    load_calibration(path)               -> np.ndarray
    DEFAULT_CALIBRATION_PATH             -> str
    default_corners(width, height, margin)   -> np.ndarray
    grab_gray_frame(xe_cam)                  -> Optional[np.ndarray]
    draw_overlay(display, corners, scale, active_idx) -> None

Usage:
    import example_open_xe_001d_laser as xe_cam
    corners = run_quad_calibration(xe_cam, scale=3)
    if corners is not None:
        H = compute_homography(corners)
        nx, ny = warp_point(H, target.cx, target.cy)
"""

import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np

# DVS sensor resolution (ESC001D fixed)
DVS_WIDTH = 164
DVS_HEIGHT = 160

DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dvs_calibration.json"
)

# Corner label colours (BGR)
_CORNER_COLORS = {
    "TL": (0, 255, 0),    # green
    "TR": (255, 0, 0),    # blue
    "BR": (0, 0, 255),    # red
    "BL": (0, 255, 255),  # yellow
}
_CORNER_ORDER = ["TL", "TR", "BR", "BL"]


# ---------------------------------------------------------------------------
# Homography helpers (self-contained, mirrors ex16 logic)
# ---------------------------------------------------------------------------

def compute_homography(corners: np.ndarray) -> np.ndarray:
    """Compute perspective transform from quad corners to unit square.

    Args:
        corners: (4, 2) array ordered [TL, TR, BR, BL] in *rotated* pixel coords.

    Returns:
        3x3 homography matrix mapping pixel coords to unit square.
        Origin at bottom-left (BL=0,0), x-right, y-up — same as RGB.
    """
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    # TL(0,1) TR(1,1) BR(1,0) BL(0,0) — x-right, y-up, same as RGB calibration_store
    dst = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def warp_point(matrix: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Apply perspective transform to a single point."""
    pt = np.array([[[x, y]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, matrix)
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


def save_calibration(corners: np.ndarray, path: str) -> bool:
    """Save quad corners to JSON file. Returns True on success."""
    data = {
        "corners": corners.tolist(),
        "dvs_width": DVS_WIDTH,
        "dvs_height": DVS_HEIGHT,
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError as e:
        print(f"[CAL] Failed to save calibration: {e}")
        return False


def load_calibration(path: str) -> np.ndarray:
    """Load quad corners from JSON file.

    Returns (4, 2) float32 array [TL, TR, BR, BL].
    Raises FileNotFoundError / ValueError on failure.
    """
    with open(path) as f:
        data = json.load(f)
    corners = np.array(data["corners"], dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"Invalid corners shape: {corners.shape}")
    return corners


# ---------------------------------------------------------------------------
# Reusable helpers (also used by dual calibration in ex17)
# ---------------------------------------------------------------------------

def default_corners(
    width: int = DVS_HEIGHT,
    height: int = DVS_WIDTH,
    margin: float = 0.20,
) -> np.ndarray:
    """Return default quad corners inset by *margin* fraction (rotated space: w=160, h=164)."""
    mx = width * margin
    my = height * margin
    return np.array([
        [mx, my],                       # TL
        [width - 1 - mx, my],           # TR
        [width - 1 - mx, height - 1 - my],  # BR
        [mx, height - 1 - my],          # BL
    ], dtype=np.float32)


def grab_gray_frame(xe_cam) -> Optional[np.ndarray]:
    """Capture one hybrid RGB frame (gray channel) from xe_cam, rotated.

    Returns (DVS_WIDTH, DVS_HEIGHT) uint8 grayscale (rotated space), or None.
    """
    try:
        _dvs, gray = xe_cam.g_cap.XeGetFrame(
            xe_cam.g_xereal_mode, xe_cam.g_xereal_bit_depth
        )
        if gray is None:
            return None
        # gray comes as flat or (H, W); reshape if needed
        if gray.ndim == 1:
            gray = gray.reshape((DVS_HEIGHT, DVS_WIDTH))
        gray = gray.astype(np.uint8)
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.flip(gray, 1)
        return gray
    except Exception:
        return None


def draw_overlay(
    display: np.ndarray,
    corners: np.ndarray,
    scale: int,
    active_idx: Optional[int] = None,
) -> None:
    """Draw quad edges, corner circles, and labels on the display image."""
    pts = (corners * scale).astype(int)

    # Edges
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(display, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

    # Corners
    for i, label in enumerate(_CORNER_ORDER):
        color = _CORNER_COLORS[label]
        cx, cy = int(pts[i][0]), int(pts[i][1])
        radius = 8 if i != active_idx else 12
        cv2.circle(display, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(display, (cx, cy), radius + 2, color, 2, cv2.LINE_AA)
        cv2.putText(
            display, label, (cx + 12, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )


# ---------------------------------------------------------------------------
# Main calibration UI
# ---------------------------------------------------------------------------

def run_quad_calibration(
    xe_cam,
    scale: int = 3,
    window_name: str = "Quad Calibration",
    initial_corners: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Run interactive quad calibration with drag-to-move corners.

    Displays the hybrid RGB preview (grayscale) upscaled by *scale*,
    overlaid with the quad.  Drag corners to reposition; accept or cancel
    via keyboard.

    Args:
        xe_cam: The ``example_open_xe_001d_laser`` module (already started
                in hybrid config mode).
        scale: Display upscale factor (default 3).
        window_name: OpenCV window title.

    Returns:
        (4, 2) float32 array of corners [TL, TR, BR, BL] in DVS pixel
        coordinates, or *None* if the user cancelled.

    Controls:
        Enter       — confirm calibration
        R           — reset corners to default
        Q / Esc     — cancel
    """
    corners = initial_corners.copy() if initial_corners is not None else default_corners()
    dragging_idx: Optional[int] = None
    hit_radius = 15  # px in display space

    # ----- mouse callback -----
    def _mouse_cb(event, mx, my, flags, param):
        nonlocal corners, dragging_idx

        if event == cv2.EVENT_LBUTTONDOWN:
            # Find nearest corner in display space
            pts_disp = corners * scale
            dists = np.sqrt(((pts_disp - [mx, my]) ** 2).sum(axis=1))
            idx = int(np.argmin(dists))
            if dists[idx] < hit_radius:
                dragging_idx = idx

        elif event == cv2.EVENT_MOUSEMOVE and dragging_idx is not None:
            # Clamp to rotated DVS pixel range (w=DVS_HEIGHT, h=DVS_WIDTH)
            nx = np.clip(mx / scale, 0, DVS_HEIGHT - 1)
            ny = np.clip(my / scale, 0, DVS_WIDTH - 1)
            corners[dragging_idx] = [nx, ny]

        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = None

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_cb)

    print("[CAL] Quad calibration started")
    print("[CAL] Drag corners to adjust.  [Enter] confirm | [R] reset | [Q/Esc] cancel")

    # Rotated dimensions: width=DVS_HEIGHT, height=DVS_WIDTH
    rot_w, rot_h = DVS_HEIGHT, DVS_WIDTH

    while True:
        gray = grab_gray_frame(xe_cam)
        if gray is None:
            # Fallback: black frame (rotated space)
            gray = np.zeros((rot_h, rot_w), dtype=np.uint8)

        # Upscale to display
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        display = cv2.resize(
            bgr, (rot_w * scale, rot_h * scale),
            interpolation=cv2.INTER_NEAREST,
        )

        draw_overlay(display, corners, scale, dragging_idx)

        # Help text
        h_disp = display.shape[0]
        cv2.putText(
            display,
            "[Enter] confirm  [R] reset  [Q/Esc] cancel",
            (10, h_disp - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
        )
        cv2.putText(
            display, "QUAD CALIBRATION", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # Enter
            cv2.destroyWindow(window_name)
            print(f"[CAL] Confirmed corners: {corners.tolist()}")
            return corners.copy()

        elif key in (ord("r"), ord("R")):
            corners = default_corners()
            print("[CAL] Corners reset to default")

        elif key in (ord("q"), ord("Q"), 27):  # q / Esc
            cv2.destroyWindow(window_name)
            print("[CAL] Calibration cancelled")
            return None
