#!/usr/bin/env python3
"""DVS vs RGB dual-stream laser tracking comparison tool.

Displays DVS (event camera) and RGB (webcam) laser tracking side-by-side,
each with its own trajectory canvas, for visual comparison of tracking
performance between the two modalities.

Architecture:
    DVSReaderThread (background, ~200fps)     Main Thread (~30fps, webcam-driven)
      - Reads DVS events continuously           - Reads webcam frames
      - Runs detect_from_events()               - Gets latest DVS result
      - Updates DVS canvas at ~200fps            - Runs RGB laser detect
      - Stores result under Lock                 - Updates RGB canvas at ~30fps
                                                 - Renders both canvases for display
                                                 - Composes display + keyboard

Usage:
    # Default (DVS camera=2, RGB camera=0):
    python3 examples/ex17/dual_tracker_compare.py

    # Custom devices:
    python3 examples/ex17/dual_tracker_compare.py --dvs-camera 2 --rgb-camera 0

    # Skip DVS calibration (use simple normalization):
    python3 examples/ex17/dual_tracker_compare.py --no-dvs-cal

    # With RGB laser profile + noise mask:
    python3 examples/ex17/dual_tracker_compare.py \\
        --noise-mask recordings/no_signal.npy \\
        --load-profile examples/ex16/laser_profile.json

Controls:
    q     - Quit
    space - Toggle tracking on/off
    c     - Clear both trajectory canvases
    d     - Re-detect RGB quadrilateral
    v     - Cycle display layout (FULL -> TRAJECTORY -> PIP -> ...)
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# ex15/ — DVS tracker, quad calibrator
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "ex15"))
# ex16/ — RGB tracker, quad detector, trajectory canvas
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "ex16"))
# src/ — shared utilities
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
# XenReal SDK
sys.path.insert(0, "/workspace/xenreal_001d/src")
sys.path.insert(0, "/workspace/xenreal_001d")

# --- DVS imports (ex15) ---
from quad_calibrator import (
    run_quad_calibration,
    compute_homography as dvs_compute_homography,
    warp_point as dvs_warp_point,
    save_calibration, load_calibration, DEFAULT_CALIBRATION_PATH,
)
from dvs_laser_tracker import DVSLaserTracker

# Display helpers reused from ex15/main_dvs_tracking.py
from main_dvs_tracking import dvs_frame_to_bgr, draw_dvs_target_scaled

# --- RGB imports (ex16) ---
from laser_tracker import (
    DEFAULT_PROFILE_PATH,
    ROTATE_FLAGS,
    LaserProfile,
    LaserTracker,
)
from quad_detector import QuadDetector, QuadTarget
from trajectory_canvas import TrajectoryCanvas
from main_laser_drawing import (
    draw_quad,
    draw_target,
    compute_homography as rgb_compute_homography,
    warp_point as rgb_warp_point,
    detect_quad_roi,
    run_calibration,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DVS_WIDTH = 164
DVS_HEIGHT = 160
DVS_ONLY_CONFIG = "/workspace/xenreal_001d/ESC001D_DV_RAW4_200FPS_20260204_modify.cfg"
HYBRID_CONFIG = "/workspace/xenreal_001d/ESC001D_2D_RAW8_DV_RAW2.cfg"

LAYOUT_FULL = 0
LAYOUT_TRAJECTORY = 1
LAYOUT_PIP = 2
LAYOUT_NAMES = ["FULL", "TRAJECTORY", "PIP"]

WINDOW_NAME = "DVS vs RGB Compare"


# ---------------------------------------------------------------------------
# DVSReaderThread — background DVS reader + tracker
# ---------------------------------------------------------------------------

class DVSReaderThread:
    """Background thread that reads DVS frames at native rate (~200fps).

    Owns a TrajectoryCanvas internally so that canvas.update() runs at the
    full DVS frame rate (~200fps) instead of the main-loop rate (~30fps).
    The main thread calls render_canvas() / clear_canvas() for display.
    """

    def __init__(
        self,
        xe_cam,
        tracker: DVSLaserTracker,
        homography: Optional[np.ndarray],
        scale: int = 3,
        canvas_size: int = 400,
        idle_clear: float = 0,
    ):
        self._xe_cam = xe_cam
        self._tracker = tracker
        self._homography = homography
        self._scale = scale

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_target = None  # DVSTarget or None
        self._latest_warped: Optional[Tuple[float, float]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0

        # Canvas owned by this thread, updated at ~200fps
        self._canvas = TrajectoryCanvas(size=canvas_size, idle_clear=idle_clear)
        self._canvas_lock = threading.Lock()
        # Main thread can toggle tracking on/off
        self._tracking_enabled = True

    def start(self) -> None:
        """Start the background reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background reader thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self):
        """Return (frame, target, warped_xy, fps) — non-blocking snapshot."""
        with self._lock:
            return (
                self._latest_frame,
                self._latest_target,
                self._latest_warped,
                self._fps,
            )

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._homography

    def render_canvas(self) -> np.ndarray:
        """Thread-safe canvas render for main thread display."""
        with self._canvas_lock:
            return self._canvas.render()

    def clear_canvas(self) -> None:
        """Thread-safe canvas clear."""
        with self._canvas_lock:
            self._canvas.clear()

    @property
    def tracking_enabled(self) -> bool:
        return self._tracking_enabled

    @tracking_enabled.setter
    def tracking_enabled(self, value: bool) -> None:
        self._tracking_enabled = value

    def _run(self) -> None:
        """Reader loop: capture frames, run tracker, store results."""
        fps_frames = 0
        fps_timer = time.time()

        while self._running:
            event_frame = self._xe_cam.get_frame_laser_nparray()
            if event_frame is None:
                continue

            # Track laser spot
            target = self._tracker.detect_from_events(event_frame)

            # Compute warped coordinate
            warped = None
            if target and self._homography is not None:
                wx, wy = dvs_warp_point(self._homography, target.cx, target.cy)
                if 0.0 <= wx <= 1.0 and 0.0 <= wy <= 1.0:
                    warped = (wx, wy)

            # Update canvas at native DVS rate (~200fps)
            with self._canvas_lock:
                if not self._tracking_enabled:
                    self._canvas.update(False, 0.0, 0.0)
                elif target is not None and self._homography is not None:
                    # Has calibration — only draw points inside trapezoid
                    if warped is not None:
                        self._canvas.update(True, warped[0], warped[1])
                    else:
                        self._canvas.update(False, 0.0, 0.0)  # outside bounds → pen up
                elif target is not None:
                    # No calibration — fallback to simple normalization
                    nx = target.cx / DVS_WIDTH
                    ny = 1.0 - (target.cy / DVS_HEIGHT)
                    self._canvas.update(True, nx, ny)
                else:
                    self._canvas.update(False, 0.0, 0.0)

            # FPS counter
            fps_frames += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_val = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer = now
            else:
                fps_val = self._fps

            with self._lock:
                self._latest_frame = event_frame
                self._latest_target = target
                self._latest_warped = warped
                self._fps = fps_val


# ---------------------------------------------------------------------------
# Display composition helpers
# ---------------------------------------------------------------------------

def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize image to target height while preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def _pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    """Pad image on the right with black to reach target_w."""
    h, w = img.shape[:2]
    if w >= target_w:
        return img[:, :target_w]
    pad = np.zeros((h, target_w - w, 3), dtype=np.uint8)
    return np.hstack([img, pad])


def _make_label_bar(text: str, width: int, height: int = 28,
                    bg_color=(40, 40, 40), fg_color=(220, 220, 220)) -> np.ndarray:
    """Create a thin label bar image."""
    bar = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg_color, 1, cv2.LINE_AA)
    return bar


def _draw_status_on(panel: np.ndarray, label: str, tracking: bool,
                    fps: float, coord: Optional[Tuple[float, float]]) -> None:
    """Draw a simplified status line on top of a panel."""
    status = "ON" if tracking else "OFF"
    color = (0, 255, 0) if tracking else (0, 0, 255)
    text = f"{label} | Track: {status} | FPS: {fps:.0f}"
    if coord:
        text += f" | ({coord[0]:.2f}, {coord[1]:.2f})"
    cv2.putText(panel, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _compose_full(
    dvs_display: np.ndarray,
    rgb_display: np.ndarray,
    dvs_canvas: np.ndarray,
    rgb_canvas: np.ndarray,
) -> np.ndarray:
    """Layout 0 — FULL: 4-panel grid.

    [ DVS camera ] [ RGB camera ]
    [ DVS canvas ] [ RGB canvas ]
    """
    # Unify camera row height
    cam_h = max(dvs_display.shape[0], rgb_display.shape[0])
    dvs_cam = _resize_to_height(dvs_display, cam_h)
    rgb_cam = _resize_to_height(rgb_display, cam_h)

    # Unify camera row width
    cam_row_w = dvs_cam.shape[1] + rgb_cam.shape[1]
    cam_row = np.hstack([dvs_cam, rgb_cam])

    # Canvas row — resize to match camera row width
    canvas_h = dvs_canvas.shape[0]
    canvas_total_w = dvs_canvas.shape[1] + rgb_canvas.shape[1]

    canvas_row = np.hstack([dvs_canvas, rgb_canvas])
    if canvas_total_w != cam_row_w:
        canvas_row = cv2.resize(
            canvas_row,
            (cam_row_w, int(canvas_h * cam_row_w / canvas_total_w)),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.vstack([cam_row, canvas_row])


def _compose_trajectory(
    dvs_canvas: np.ndarray,
    rgb_canvas: np.ndarray,
) -> np.ndarray:
    """Layout 1 — TRAJECTORY: side-by-side canvases only."""
    return np.hstack([dvs_canvas, rgb_canvas])


def _compose_pip(
    dvs_display: np.ndarray,
    rgb_display: np.ndarray,
    dvs_canvas: np.ndarray,
    rgb_canvas: np.ndarray,
    pip_h: int = 120,
) -> np.ndarray:
    """Layout 2 — PIP: trajectory canvases with small camera preview inset."""
    # Resize camera previews to small PIP size
    dvs_pip = _resize_to_height(dvs_display, pip_h)
    rgb_pip = _resize_to_height(rgb_display, pip_h)

    # Place PIP in top-left corner of each canvas
    dvs_out = dvs_canvas.copy()
    rgb_out = rgb_canvas.copy()

    pip_margin = 6
    # DVS PIP
    dph, dpw = dvs_pip.shape[:2]
    if dpw < dvs_out.shape[1] - pip_margin and dph < dvs_out.shape[0] - pip_margin:
        dvs_out[pip_margin:pip_margin + dph, pip_margin:pip_margin + dpw] = dvs_pip
        cv2.rectangle(dvs_out, (pip_margin, pip_margin),
                      (pip_margin + dpw, pip_margin + dph), (100, 100, 100), 1)

    # RGB PIP
    rph, rpw = rgb_pip.shape[:2]
    if rpw < rgb_out.shape[1] - pip_margin and rph < rgb_out.shape[0] - pip_margin:
        rgb_out[pip_margin:pip_margin + rph, pip_margin:pip_margin + rpw] = rgb_pip
        cv2.rectangle(rgb_out, (pip_margin, pip_margin),
                      (pip_margin + rpw, pip_margin + rph), (100, 100, 100), 1)

    return np.hstack([dvs_out, rgb_out])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DVS vs RGB dual-stream laser tracking comparison"
    )
    parser.add_argument(
        "--dvs-camera", type=int, default=2,
        help="DVS device index (default: 2)",
    )
    parser.add_argument(
        "--rgb-camera", default="0",
        help="USB webcam device index or path (default: 0)",
    )
    parser.add_argument(
        "--rgb-rotate", type=int, choices=[0, 90, 180, 270], default=90,
        help="RGB frame rotation in degrees (default: 90)",
    )
    parser.add_argument(
        "--scale", type=int, default=3,
        help="DVS display scale factor (default: 3)",
    )
    parser.add_argument(
        "--idle-clear", type=float, default=1.0,
        help="Trajectory auto-clear idle timeout in seconds (0=disable, default: 1)",
    )
    parser.add_argument(
        "--noise-mask", metavar="PATH",
        help="DVS hot-pixel noise mask .npy path",
    )
    parser.add_argument(
        "--load-profile", default=DEFAULT_PROFILE_PATH,
        help="RGB laser profile JSON path",
    )
    parser.add_argument(
        "--no-dvs-cal", action="store_true",
        help="Skip DVS quad calibration (use simple normalization)",
    )
    parser.add_argument(
        "--dvs-cal", type=str, default=DEFAULT_CALIBRATION_PATH, metavar="PATH",
        help=f"DVS calibration file path (default: {DEFAULT_CALIBRATION_PATH})",
    )
    parser.add_argument(
        "--no-rgb-quad", action="store_true",
        help="Skip RGB quadrilateral detection",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Enter RGB HSV interactive calibration before tracking",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DVS vs RGB Dual-Stream Laser Tracking Comparison (ex17)")
    print("=" * 60)

    # ===================================================================
    # Phase 1: DVS Calibration
    # ===================================================================

    device = f"/dev/video{args.dvs_camera}"
    print(f"\n[DVS] Device: {device}")

    import example_open_xe_001d_laser as xe_cam
    xe_cam.DEVICE = device

    dvs_homography: Optional[np.ndarray] = None

    if not args.no_dvs_cal:
        # Try loading saved corners as initial positions
        saved_corners = None
        if os.path.isfile(args.dvs_cal):
            try:
                saved_corners = load_calibration(args.dvs_cal)
                print(f"[DVS] Loaded saved corners from {args.dvs_cal}")
            except (ValueError, KeyError) as e:
                print(f"[DVS WARNING] Invalid calibration file: {e}")

        # Open in hybrid mode for calibration
        print("[DVS] Starting hybrid camera for quad calibration...")
        xe_cam.CONFIG_ABS_PATH = HYBRID_CONFIG
        xe_cam.start_camera_laser()

        corners = run_quad_calibration(
            xe_cam, scale=args.scale, initial_corners=saved_corners,
        )
        if corners is not None:
            dvs_homography = dvs_compute_homography(corners)
            save_calibration(corners, args.dvs_cal)
            print(f"[DVS] Homography computed, saved to {args.dvs_cal}")
        else:
            print("[DVS] Calibration skipped, using simple normalization")

        # Switch to DVS-only mode
        print("[DVS] Switching to DVS-only mode...")
        xe_cam.close_camera(xe_cam.g_cap)

    xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
    xe_cam.start_camera_laser()
    print(f"[DVS] Camera ready ({DVS_WIDTH}x{DVS_HEIGHT})")

    # Create DVS tracker
    dvs_tracker = DVSLaserTracker(
        width=DVS_WIDTH,
        height=DVS_HEIGHT,
        noise_mask_path=args.noise_mask,
    )
    print(f"[DVS] Tracker: {dvs_tracker}")

    # ===================================================================
    # Phase 2: RGB Webcam Calibration
    # ===================================================================

    print()
    try:
        rgb_dev = int(args.rgb_camera)
    except ValueError:
        rgb_dev = args.rgb_camera
    print(f"[RGB] Device: {rgb_dev}")

    cap = cv2.VideoCapture(rgb_dev)
    if isinstance(rgb_dev, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open RGB camera: {rgb_dev}")
        xe_cam.close_camera(xe_cam.g_cap)
        return 1

    rotate_flag = ROTATE_FLAGS.get(args.rgb_rotate)

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first RGB frame")
        cap.release()
        xe_cam.close_camera(xe_cam.g_cap)
        return 1
    if rotate_flag is not None:
        first_frame = cv2.rotate(first_frame, rotate_flag)

    actual_h, actual_w = first_frame.shape[:2]
    print(f"[RGB] Frame size: {actual_w}x{actual_h}"
          f"{f' (rotated {args.rgb_rotate}°)' if rotate_flag else ''}")

    # Create RGB laser tracker
    if os.path.isfile(args.load_profile):
        profile = LaserProfile.load(args.load_profile)
        rgb_tracker = LaserTracker.from_profile(profile)
        print(f"[RGB] Loaded profile: {args.load_profile}")
    else:
        if args.load_profile != DEFAULT_PROFILE_PATH:
            print(f"[WARNING] Profile not found: {args.load_profile}")
        print("[RGB] Using default tracker parameters")
        rgb_tracker = LaserTracker()

    # Quad detection
    quad: Optional[QuadTarget] = None
    rgb_homography: Optional[np.ndarray] = None
    quad_detector = QuadDetector()

    if not args.no_rgb_quad:
        quad = detect_quad_roi(first_frame, quad_detector)
        if quad:
            rgb_tracker.roi = quad.as_xyxy()
            rgb_homography = rgb_compute_homography(quad)
    else:
        print("[RGB] Quad detection disabled")

    print(f"[RGB] Tracker: {rgb_tracker}")

    # Optional HSV calibration
    if args.calibrate:
        save_path = os.path.join(os.path.dirname(__file__), "..", "ex16",
                                 "laser_profile.json")
        run_calibration(cap, rgb_tracker, save_path, rotate_flag)
        print(f"[RGB] Post-calibration tracker: {rgb_tracker}")

    # ===================================================================
    # Phase 3: Start DVS background reader + main loop
    # ===================================================================

    dvs_reader = DVSReaderThread(
        xe_cam, dvs_tracker, dvs_homography, scale=args.scale,
        canvas_size=400, idle_clear=args.idle_clear,
    )
    dvs_reader.start()

    rgb_canvas = TrajectoryCanvas(size=400, idle_clear=args.idle_clear)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    tracking_enabled = True
    layout_mode = LAYOUT_FULL
    rgb_fps = 0.0
    rgb_fps_frames = 0
    rgb_fps_timer = time.time()

    print()
    print("[INFO] Controls: [q]uit [space]toggle [c]lear [d]re-detect [v]layout")
    print(f"[INFO] Layout: {LAYOUT_NAMES[layout_mode]}")
    print()

    try:
        while True:
            # --- 1. Read RGB frame (blocking, ~30fps) ---
            ret, rgb_frame = cap.read()
            if not ret:
                print("[WARNING] RGB frame capture failed")
                break
            if rotate_flag is not None:
                rgb_frame = cv2.rotate(rgb_frame, rotate_flag)

            rgb_h, rgb_w = rgb_frame.shape[:2]

            # RGB FPS counter
            rgb_fps_frames += 1
            now = time.time()
            if now - rgb_fps_timer >= 1.0:
                rgb_fps = rgb_fps_frames / (now - rgb_fps_timer)
                rgb_fps_frames = 0
                rgb_fps_timer = now

            # --- 2. Get latest DVS result (non-blocking) ---
            dvs_event_frame, dvs_target, dvs_warped, dvs_fps = dvs_reader.get_latest()

            # --- 3. RGB tracking ---
            rgb_target = rgb_tracker.detect(rgb_frame) if tracking_enabled else None

            # --- 4. DVS coord for status display (canvas updated in bg thread) ---
            dvs_coord: Optional[Tuple[float, float]] = dvs_warped

            # --- 5. Update RGB canvas ---
            rgb_coord: Optional[Tuple[float, float]] = None
            if rgb_target and rgb_homography is not None:
                nx, ny = rgb_warp_point(rgb_homography, rgb_target.cx, rgb_target.cy)
                if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                    rgb_canvas.update(True, nx, ny)
                    rgb_coord = (nx, ny)
                else:
                    rgb_canvas.update(False, 0.0, 0.0)
            elif rgb_target:
                nx, ny = rgb_target.normalized_center(rgb_w, rgb_h)
                rgb_canvas.update(True, nx, ny)
                rgb_coord = (nx, ny)
            else:
                rgb_canvas.update(False, 0.0, 0.0)

            # --- 6. Build display panels ---

            # DVS camera display
            if dvs_event_frame is not None:
                dvs_display = dvs_frame_to_bgr(dvs_event_frame, scale=args.scale)
                draw_dvs_target_scaled(dvs_display, dvs_target, scale=args.scale)
            else:
                # Placeholder while waiting for first DVS frame
                dvs_display = np.zeros(
                    (DVS_HEIGHT * args.scale, DVS_WIDTH * args.scale, 3),
                    dtype=np.uint8,
                )
                cv2.putText(dvs_display, "DVS: waiting...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # Draw status on DVS display
            _draw_status_on(dvs_display, "DVS", tracking_enabled, dvs_fps, dvs_coord)

            # RGB camera display
            rgb_display = rgb_frame.copy()
            if quad:
                draw_quad(rgb_display, quad)
            if rgb_target:
                draw_target(rgb_display, rgb_target)
            _draw_status_on(rgb_display, "RGB", tracking_enabled, rgb_fps, rgb_coord)

            # Render canvases (DVS canvas updated at ~200fps in bg thread)
            dvs_canvas_img = dvs_reader.render_canvas()
            rgb_canvas_img = rgb_canvas.render()

            # --- 7. Compose final frame based on layout ---
            if layout_mode == LAYOUT_FULL:
                composed = _compose_full(
                    dvs_display, rgb_display, dvs_canvas_img, rgb_canvas_img,
                )
            elif layout_mode == LAYOUT_TRAJECTORY:
                composed = _compose_trajectory(dvs_canvas_img, rgb_canvas_img)
            else:  # LAYOUT_PIP
                composed = _compose_pip(
                    dvs_display, rgb_display, dvs_canvas_img, rgb_canvas_img,
                )

            # Layout label in bottom-right
            lbl = f"[v] Layout: {LAYOUT_NAMES[layout_mode]}"
            ch, cw = composed.shape[:2]
            cv2.putText(composed, lbl, (cw - 220, ch - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
                        cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, composed)

            # --- 8. Keyboard handling ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] Quit")
                break

            elif key == ord(" "):
                tracking_enabled = not tracking_enabled
                dvs_reader.tracking_enabled = tracking_enabled
                state = "enabled" if tracking_enabled else "paused"
                print(f"[INFO] Tracking {state}")

            elif key == ord("c"):
                dvs_reader.clear_canvas()
                rgb_canvas.clear()
                print("[INFO] Both trajectories cleared")

            elif key == ord("d"):
                # Re-detect RGB quad on current frame
                print("[INFO] Re-detecting RGB quad...")
                new_quad = detect_quad_roi(rgb_frame, quad_detector)
                if new_quad:
                    quad = new_quad
                    rgb_tracker.roi = quad.as_xyxy()
                    rgb_homography = rgb_compute_homography(quad)
                    print("[OK] RGB quad updated")

            elif key == ord("v"):
                layout_mode = (layout_mode + 1) % 3
                print(f"[INFO] Layout: {LAYOUT_NAMES[layout_mode]}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    # ===================================================================
    # Phase 4: Cleanup
    # ===================================================================

    print("[INFO] Shutting down...")
    dvs_reader.stop()

    try:
        xe_cam.close_camera(xe_cam.g_cap)
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
