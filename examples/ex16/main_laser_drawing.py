#!/usr/bin/env python3
"""Real-time laser-guided drawing with Piper robotic arm.

Integrates camera-based laser tracking with robotic arm drawing using
a FIFO producer-consumer architecture. The camera thread detects laser
positions and enqueues commands; the arm thread dequeues and executes
them faithfully via DrawingController.move().

Architecture:
    Main Thread (Camera + GUI)  -->  CommandBridge (FIFO queue)  -->  Arm Thread (Consumer)

    - Laser detected: (True, nx, ny) enqueued
    - Laser lost:     (False, 0, 0)  enqueued (pen up, idempotent)
    - Arm processes commands in strict FIFO order

Usage:
    # Camera-only test (no hardware):
    uv run python examples/ex16/main_laser_drawing.py --camera 0 --no-arm

    # Full integration (on Orin):
    bash scripts/can_activate.sh can0 1000000
    uv run python examples/ex16/main_laser_drawing.py --camera 0 --can can0 --speed 0.2

Controls:
    q     - Quit (safe shutdown)
    space - Toggle tracking on/off
    d     - Re-detect quad, recompute homography
    r     - Pause tracking (arm finishes queued commands)
    m     - Toggle mask overlay
    +/-   - Adjust brightness threshold
    t     - Enter/exit interactive HSV calibration mode
    s     - Quick-save tracker parameters to profile JSON
"""

import argparse
import os
import queue
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# Add paths for local imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ex15"))

from laser_tracker import (
    DEFAULT_PROFILE_PATH,
    ROTATE_FLAGS,
    LaserProfile,
    LaserTarget,
    LaserTracker,
    parse_roi,
)
from quad_detector import QuadDetector, QuadTarget
from trajectory_canvas import TrajectoryCanvas

# Colors for quad corner labels (BGR)
CORNER_COLORS = {
    "TL": (0, 255, 0),    # green
    "TR": (255, 0, 0),    # blue
    "BR": (0, 0, 255),    # red
    "BL": (0, 255, 255),  # yellow
}


# ---------------------------------------------------------------------------
# CommandBridge — Thread-safe FIFO command queue
# ---------------------------------------------------------------------------

class CommandBridge:
    """Thread-safe FIFO command bridge. Every command is preserved."""

    def __init__(self, maxsize: int = 5000):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, write: bool, x: float, y: float) -> None:
        """Producer: enqueue one command per frame."""
        try:
            self._queue.put_nowait((write, x, y))
        except queue.Full:
            pass  # safety valve — drop if queue impossibly full

    def get(self, timeout: float = 0.1) -> Optional[Tuple[bool, float, float]]:
        """Consumer: dequeue next command."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# ArmThread — Background arm consumer
# ---------------------------------------------------------------------------

class ArmThread:
    """Background thread that consumes commands from CommandBridge."""

    def __init__(self, bridge: CommandBridge, can_name: str, speed: float):
        self._bridge = bridge
        self._can_name = can_name
        self._speed = speed
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Public state (readable from main thread for GUI)
        self.is_ready = threading.Event()
        self.is_running = False
        self.error: Optional[str] = None
        self.move_count = 0
        self.fail_count = 0

        # Internal references (set during _init_arm)
        self._conn = None
        self._drawer = None

    def start(self) -> None:
        """Launch daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal stop."""
        self._stop_event.set()

    def join(self, timeout: float = 20.0) -> None:
        """Wait for completion."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        """Thread entry point."""
        try:
            self._init_arm()
            self.is_ready.set()
            self.is_running = True
            self._consume_loop()
        except Exception as e:
            self.error = str(e)
            print(f"[ARM ERROR] {e}")
        finally:
            self._cleanup()
            self.is_running = False

    def _init_arm(self) -> None:
        """Initialize arm hardware."""
        from piper_demo import PiperConnection, MotionController, JointReader
        from drawing import DrawingController, DrawingConfig

        print(f"[ARM] Connecting to {self._can_name}...")
        self._conn = PiperConnection(can_name=self._can_name)
        self._conn.connect()

        print("[ARM] Enabling arm...")
        self._conn.enable(go_home=False)
        time.sleep(1)

        motion = MotionController(self._conn.piper)
        reader = JointReader(self._conn.piper)

        config = DrawingConfig(
            draw_speed=self._speed,
            move_speed=self._speed,
        )
        self._drawer = DrawingController(motion, reader, config)

        # Move to center position (pen up)
        print("[ARM] Moving to center (0.5, 0.5)...")
        ok = self._drawer.move(False, 0.5, 0.5)
        if not ok:
            raise RuntimeError("Cannot reach center position (0.5, 0.5)")
        print("[ARM] Ready!")

    def _consume_loop(self) -> None:
        """Main consumer loop — processes commands from bridge."""
        while not self._stop_event.is_set():
            cmd = self._bridge.get(timeout=0.1)
            if cmd is None:
                continue

            write, x, y = cmd
            ok = self._drawer.move(write, x, y)
            self.move_count += 1
            if not ok:
                self.fail_count += 1

    def _cleanup(self) -> None:
        """Safe shutdown of arm hardware."""
        if self._drawer is not None:
            try:
                self._drawer.safe_disable()
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.safe_disable(return_home=False)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GUI drawing helpers (from main_laser_quad.py)
# ---------------------------------------------------------------------------


def draw_quad(frame: np.ndarray, target: QuadTarget) -> None:
    """Draw quadrilateral annotation on frame."""
    corners = target.corners.astype(int)

    # Draw edges in green
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    # Draw corners with labels
    for label, (x, y) in target.corner_labels():
        color = CORNER_COLORS[label]
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 8, color, 2)
        cv2.putText(
            frame, label, (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    # Center cross
    cx, cy = target.center
    cv2.drawMarker(
        frame, (int(cx), int(cy)), (255, 255, 255),
        cv2.MARKER_CROSS, 15, 1,
    )


def draw_target(
    frame: np.ndarray,
    target: LaserTarget,
    color: tuple = (0, 255, 0),
) -> None:
    """Draw laser target annotation on frame."""
    cx, cy = int(target.cx), int(target.cy)

    # Crosshair
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

    # Circle
    radius = max(12, int(target.area ** 0.5 * 3))
    cv2.circle(frame, (cx, cy), radius, color, 1)

    # Label
    label = f"br={target.brightness:.0f} a={target.area:.0f}"
    cv2.putText(frame, label, (cx + 15, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def compute_homography(quad: QuadTarget) -> np.ndarray:
    """Compute perspective transform from quad corners to unit square.

    Maps quad corners [TL, TR, BR, BL] to [(0,1), (1,1), (1,0), (0,0)].
    Origin at bottom-left (BL), x-right, y-up.
    """
    src = quad.corners.astype(np.float32)  # shape (4, 2), order: TL TR BR BL
    dst = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def compensate_for_arm(nx: float, ny: float, rotate_deg: int) -> Tuple[float, float]:
    """Undo frame rotation in normalized space so arm coords stay correct."""
    if rotate_deg == 90:
        return 1 - ny, nx
    elif rotate_deg == 180:
        return 1 - nx, 1 - ny
    elif rotate_deg == 270:
        return ny, 1 - nx
    return nx, ny


def warp_point(matrix: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Apply perspective transform to a single point."""
    pt = np.array([[[x, y]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, matrix)
    return float(warped[0, 0, 0]), float(warped[0, 0, 1])


def draw_status(
    frame: np.ndarray,
    tracker: LaserTracker,
    tracking_enabled: bool,
    target: Optional[LaserTarget],
    homography: Optional[np.ndarray],
    roi_source: str,
    show_mask: bool,
    fps: float,
    arm: Optional[ArmThread],
    bridge: Optional[CommandBridge],
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Tracking status
    status = "TRACKING" if tracking_enabled else "PAUSED"
    status_color = (0, 255, 0) if tracking_enabled else (0, 165, 255)
    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # ROI source label
    roi_color = {
        "QUAD ROI": (0, 255, 0),
        "MANUAL ROI": (0, 165, 255),
        "NO ROI": (0, 0, 255),
    }.get(roi_source, (200, 200, 200))
    cv2.putText(frame, roi_source, (150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

    # FPS
    cv2.putText(frame, f"{fps:.0f} fps", (w - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Target info — normalized via homography if available
    if target:
        if homography is not None:
            norm_x, norm_y = warp_point(homography, target.cx, target.cy)
        else:
            norm_x, norm_y = target.normalized_center(w, h)
        cv2.putText(frame,
                    f"Laser: ({target.cx:.0f},{target.cy:.0f}) "
                    f"norm=({norm_x:.2f},{norm_y:.2f})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Laser: NOT DETECTED", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Arm status line
    if arm is not None:
        if arm.error:
            arm_text = f"ARM: ERROR - {arm.error[:40]}"
            arm_color = (0, 0, 255)
        elif not arm.is_ready.is_set():
            arm_text = "ARM: INITIALIZING..."
            arm_color = (0, 165, 255)
        else:
            q_size = bridge.pending if bridge else 0
            arm_text = f"ARM: moves={arm.move_count} fail={arm.fail_count} queue={q_size}"
            arm_color = (255, 255, 0)
        cv2.putText(frame, arm_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_color, 1)
    elif bridge is not None:
        # --no-arm mode: show queue stats
        cv2.putText(frame, f"QUEUE: {bridge.pending} (no-arm mode)", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # ROI rectangle
    if tracker.roi:
        x1, y1, x2, y2 = tracker.roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)

    # Help
    help_text = "[q]uit [space]toggle [d]etect quad [r]eset [m]ask [c]lear [+/-]threshold [t]une [s]ave"
    cv2.putText(frame, help_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)


def detect_quad_roi(
    frame: np.ndarray,
    detector: QuadDetector,
) -> Optional[QuadTarget]:
    """Run quad detection on a frame and print results."""
    target = detector.detect(frame)
    if target is None:
        print("[WARNING] No quadrilateral detected")
        return None

    corners_str = " ".join(
        f"{label}({x},{y})" for label, (x, y) in target.corner_labels()
    )
    print(f"[OK] Quad detected: {corners_str}")
    print(f"[OK] Auto ROI set: {target.as_xyxy()}")
    return target


# ---------------------------------------------------------------------------
# Interactive HSV Calibration
# ---------------------------------------------------------------------------

def run_calibration(
    cap: cv2.VideoCapture,
    tracker: LaserTracker,
    save_path: str,
    rotate_flag=None,
) -> None:
    """Interactive HSV calibration with trackbars.

    Adjusts tracker parameters in real-time and optionally saves to JSON.
    Press 's' to save, 'q' to exit calibration.
    """
    win = "HSV Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # Trackbar callbacks (no-op, we poll values)
    def _noop(_):
        pass

    cv2.createTrackbar("V min (brightness)", win, tracker.brightness_threshold, 255, _noop)
    cv2.createTrackbar("S max (saturation)", win, tracker.max_saturation, 255, _noop)
    cv2.createTrackbar("Area min", win, tracker.min_dot_area, 50, _noop)
    cv2.createTrackbar("Area max", win, tracker.max_dot_area, 500, _noop)
    cv2.createTrackbar("Blur kernel", win, tracker.blur_kernel, 21, _noop)
    cv2.createTrackbar("Hue filter", win, int(tracker.use_hue_filter), 1, _noop)
    cv2.createTrackbar("H red low", win, tracker.hue_red_low_upper, 30, _noop)
    cv2.createTrackbar("H red high", win, tracker.hue_red_high_lower, 180, _noop)

    print("[CALIBRATE] Trackbar window opened. [s]ave  [q]uit calibration")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotate_flag is not None:
            frame = cv2.rotate(frame, rotate_flag)

        # Read trackbar values and apply to tracker
        tracker.brightness_threshold = max(100, cv2.getTrackbarPos("V min (brightness)", win))
        tracker.max_saturation = cv2.getTrackbarPos("S max (saturation)", win)
        tracker.min_dot_area = cv2.getTrackbarPos("Area min", win)
        tracker.max_dot_area = max(tracker.min_dot_area + 1, cv2.getTrackbarPos("Area max", win))
        bk = cv2.getTrackbarPos("Blur kernel", win)
        tracker.blur_kernel = max(1, bk if bk % 2 == 1 else bk + 1)
        tracker.use_hue_filter = bool(cv2.getTrackbarPos("Hue filter", win))
        tracker.hue_red_low_upper = cv2.getTrackbarPos("H red low", win)
        tracker.hue_red_high_lower = max(150, cv2.getTrackbarPos("H red high", win))

        # Detect and visualize
        target = tracker.detect(frame)
        mask = tracker.create_mask(frame)

        # Top half: frame with mask overlay + detection marker
        display = frame.copy()
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[:, :, 0] = 0
        mask_bgr[:, :, 2] = 0
        display = cv2.addWeighted(display, 1.0, mask_bgr, 0.5, 0)
        if target:
            draw_target(display, target)

        # Bottom half: pure mask (scaled to 3-channel)
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Stack vertically
        combined = np.vstack([display, mask_vis])

        # Status text on combined image
        hue_label = "ON" if tracker.use_hue_filter else "OFF"
        status = (
            f"V>={tracker.brightness_threshold} S<={tracker.max_saturation} "
            f"area=[{tracker.min_dot_area},{tracker.max_dot_area}] "
            f"blur={tracker.blur_kernel} hue={hue_label}"
        )
        cv2.putText(combined, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        det_text = f"Detected: {target}" if target else "Detected: NONE"
        cv2.putText(combined, det_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, "[s]ave  [q]uit", (10, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        cv2.imshow(win, combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            profile = tracker.to_profile()
            profile.save(save_path)
            print(f"[CALIBRATE] Profile saved to {save_path}")

        elif key == ord("q"):
            print("[CALIBRATE] Exiting calibration mode")
            break

    cv2.destroyWindow(win)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time laser-guided drawing with Piper arm"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--video", metavar="PATH",
        help="Video file path (for testing without live camera)",
    )
    source.add_argument(
        "--camera", default="0",
        help="Camera device index or path (default: 0)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera frame width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera frame height (default: 480)",
    )
    parser.add_argument(
        "--roi", type=str, default=None,
        help="Fallback ROI as x1,y1,x2,y2 (used if quad not detected)",
    )
    parser.add_argument(
        "--brightness", type=int, default=240,
        help="Brightness threshold for laser detection (default: 240)",
    )
    parser.add_argument(
        "--saturation", type=int, default=40,
        help="Max saturation for laser detection (default: 40)",
    )
    parser.add_argument(
        "--no-quad", action="store_true",
        help="Disable automatic quad detection",
    )
    parser.add_argument(
        "--record", metavar="PATH",
        help="Record output to video file",
    )
    # Arm arguments
    parser.add_argument(
        "--can", default="can0",
        help="CAN interface name (default: can0)",
    )
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Arm speed factor 0.1-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--no-arm", action="store_true",
        help="Camera-only mode (no arm control)",
    )
    parser.add_argument(
        "--idle-clear", type=float, default=1, metavar="SEC",
        help="Auto-clear canvas after SEC seconds of pen-up idle (0=disabled, default: 1)",
    )
    parser.add_argument(
        "--rotate", type=int, choices=[0, 90, 180, 270], default=90,
        help="Rotate camera frame CW by degrees (default: 90 for side-mounted camera)",
    )
    # HSV calibration / profile arguments
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Enter interactive HSV calibration mode before tracking",
    )
    parser.add_argument(
        "--load-profile", metavar="PATH", default=DEFAULT_PROFILE_PATH,
        help=f"Calibration profile path (default: {DEFAULT_PROFILE_PATH})",
    )
    parser.add_argument(
        "--save-profile", metavar="PATH", default=DEFAULT_PROFILE_PATH,
        help=f"Path to save calibration profile (default: {DEFAULT_PROFILE_PATH})",
    )
    parser.add_argument(
        "--hue-filter", action="store_true",
        help="Enable red hue ring validation (rejects non-red bright spots)",
    )
    parser.add_argument(
        "--calibrate-video", metavar="PATH",
        help="Run offline calibration on a video, save profile, and exit",
    )
    args = parser.parse_args()

    # --- Offline calibration shortcut ---
    if args.calibrate_video:
        print("[INFO] Running offline calibration...")
        calib_roi = None
        if args.roi:
            calib_roi = parse_roi(args.roi)
        try:
            profile, stats = LaserTracker.calibrate_profile_from_video(
                video_path=args.calibrate_video,
                roi=calib_roi,
                auto_quad=not args.no_quad,
                rotate=args.rotate,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] Calibration failed: {e}")
            return 1
        save_path = args.save_profile
        profile.save(save_path)
        det = stats["frames_detected"]
        proc = stats["frames_processed"]
        rate = stats["detection_rate"]
        print(f"[OK] Detected {det}/{proc} ({rate:.1%})")
        print(f"[OK] Profile saved to {save_path}")
        return 0

    # Frame rotation setup
    rotate_flag = ROTATE_FLAGS.get(args.rotate)  # None if 0

    # Parse fallback ROI
    fallback_roi = None
    if args.roi:
        try:
            fallback_roi = parse_roi(args.roi)
        except ValueError as e:
            print(f"[ERROR] Invalid ROI: {e}")
            return 1

    speed = max(0.1, min(1.0, args.speed))

    print("=" * 60)
    print("Real-Time Laser-Guided Drawing (ex16)")
    print("=" * 60)

    # --- Phase 1: Camera calibration ---

    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        try:
            dev = int(args.camera)
        except ValueError:
            dev = args.camera
        cap = cv2.VideoCapture(dev)
        if isinstance(dev, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source_name = f"camera {dev}"

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source_name}")
        return 1

    print(f"[OK] Opened {source_name}")

    # Create laser tracker (from profile or CLI args)
    if os.path.isfile(args.load_profile):
        try:
            profile = LaserProfile.load(args.load_profile)
            tracker = LaserTracker.from_profile(profile, roi=fallback_roi)
            print(f"[OK] Loaded profile: {args.load_profile}")
        except (ValueError, KeyError) as e:
            print(f"[ERROR] Invalid profile {args.load_profile}: {e}")
            return 1
    else:
        if args.load_profile != DEFAULT_PROFILE_PATH:
            # User explicitly specified a path that doesn't exist
            print(f"[ERROR] Profile not found: {args.load_profile}")
            return 1
        print("[INFO] No profile found, using CLI defaults")
        tracker = LaserTracker(
            roi=fallback_roi,
            brightness_threshold=args.brightness,
            max_saturation=args.saturation,
            use_hue_filter=args.hue_filter,
        )

    # Create quad detector
    quad_detector = QuadDetector()

    # Read first frame for quad detection
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        cap.release()
        return 1
    if rotate_flag is not None:
        first_frame = cv2.rotate(first_frame, rotate_flag)

    actual_h, actual_w = first_frame.shape[:2]
    print(f"[OK] Frame size: {actual_w}x{actual_h}"
          f"{f' (rotated {args.rotate}°)' if rotate_flag is not None else ''}")

    # Quad detection on first frame
    quad: Optional[QuadTarget] = None
    homography: Optional[np.ndarray] = None
    roi_source = "NO ROI"

    if not args.no_quad:
        quad = detect_quad_roi(first_frame, quad_detector)
        if quad:
            tracker.roi = quad.as_xyxy()
            roi_source = "QUAD ROI"
            homography = compute_homography(quad)
        elif fallback_roi:
            print(f"[INFO] Falling back to manual ROI: {fallback_roi}")
            roi_source = "MANUAL ROI"
    else:
        if fallback_roi:
            roi_source = "MANUAL ROI"
        print("[INFO] Quad detection disabled")

    print(f"[OK] Tracker: {tracker}")

    # --- Optional: interactive calibration before tracking ---
    if args.calibrate:
        run_calibration(cap, tracker, args.save_profile, rotate_flag)
        print(f"[OK] Post-calibration tracker: {tracker}")

    # Rewind video to process first frame in the main loop
    if args.video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Optional recorder
    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(args.record, fourcc, video_fps, (actual_w, actual_h))
        print(f"[OK] Recording to: {args.record}")

    # --- Phase 2: Arm initialization (background thread) ---

    bridge = CommandBridge()
    canvas = TrajectoryCanvas(idle_clear=args.idle_clear)
    arm: Optional[ArmThread] = None

    if not args.no_arm:
        arm = ArmThread(bridge, can_name=args.can, speed=speed)
        arm.start()
        print("[INFO] Arm initialization started in background...")
    else:
        print("[INFO] Arm control disabled (--no-arm)")

    # --- Phase 2.5: Calibration confirmation ---

    cv2.namedWindow("Laser Drawing", cv2.WINDOW_AUTOSIZE)
    print("[INFO] Waiting for calibration confirmation... Press [ENTER] to start tracking.")

    while True:
        ret, frame = cap.read()
        if ret and rotate_flag is not None:
            frame = cv2.rotate(frame, rotate_flag)
        if not ret:
            break

        display = frame.copy()
        if quad:
            draw_quad(display, quad)

        # Build prompt based on readiness
        arm_ready = (arm is None) or arm.is_ready.is_set()
        has_cal = homography is not None

        if not arm_ready:
            prompt = "Arm initializing..."
        elif not has_cal:
            prompt = "No quad detected. [d] detect | [q] quit"
        else:
            prompt = "Calibration OK! [ENTER] start | [d] re-detect | [q] quit"

        # Draw prompt banner
        h_frame = display.shape[0]
        cv2.rectangle(display, (0, h_frame - 40), (display.shape[1], h_frame), (0, 0, 0), -1)
        cv2.putText(display, prompt, (10, h_frame - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Laser Drawing", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and has_cal and arm_ready:  # ENTER
            bridge.put(True, 0.5, 0.5)
            print("[INFO] Pen down at center. Prepare, then press [ENTER] to start tracking.")
            # --- Stage 2: wait for tracking start ---
            while True:
                ret2, frame2 = cap.read()
                if ret2 and rotate_flag is not None:
                    frame2 = cv2.rotate(frame2, rotate_flag)
                if not ret2:
                    break
                disp2 = frame2.copy()
                if quad:
                    draw_quad(disp2, quad)
                h2 = disp2.shape[0]
                cv2.rectangle(disp2, (0, h2 - 40), (disp2.shape[1], h2), (0, 0, 0), -1)
                cv2.putText(disp2, "Pen down. Prepare, then [ENTER] to track | [q] quit",
                            (10, h2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Laser Drawing", disp2)
                k2 = cv2.waitKey(1) & 0xFF
                if k2 == 13:  # ENTER
                    bridge.put(False, 0.0, 0.0)
                    print("[INFO] Pen up. Tracking started.")
                    break
                elif k2 == ord("d"):
                    print("[INFO] Re-detecting quad...")
                    quad = detect_quad_roi(frame2, quad_detector)
                    if quad:
                        tracker.roi = quad.as_xyxy()
                        roi_source = "QUAD ROI"
                        homography = compute_homography(quad)
                    elif fallback_roi:
                        tracker.roi = fallback_roi
                        roi_source = "MANUAL ROI"
                        homography = None
                elif k2 == ord("q"):
                    print("[INFO] Quit")
                    cap.release()
                    cv2.destroyAllWindows()
                    if arm:
                        arm.stop()
                        arm.join()
                    return 0
            break
        elif key == ord("d"):
            print("[INFO] Re-detecting quad...")
            quad = detect_quad_roi(frame, quad_detector)
            if quad:
                tracker.roi = quad.as_xyxy()
                roi_source = "QUAD ROI"
                homography = compute_homography(quad)
            elif fallback_roi:
                tracker.roi = fallback_roi
                roi_source = "MANUAL ROI"
                homography = None
        elif key == ord("q"):
            print("[INFO] Quit")
            cap.release()
            cv2.destroyAllWindows()
            if arm:
                arm.stop()
                arm.join()
            return 0

    # --- Phase 3: Main loop ---

    cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)
    print()
    print("[INFO] Controls: [q]uit [space]toggle [d]etect quad [r]eset [m]ask [c]lear [+/-]threshold [t]une [s]ave")

    tracking_enabled = True
    show_mask = False
    fps = 0.0
    fps_timer = time.time()
    fps_frames = 0

    try:
        while True:
            # Check for arm errors
            if arm and arm.error:
                print(f"[ERROR] Arm thread failed: {arm.error}")
                break

            ret, frame = cap.read()
            if ret and rotate_flag is not None:
                frame = cv2.rotate(frame, rotate_flag)
            if not ret:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if ret and rotate_flag is not None:
                        frame = cv2.rotate(frame, rotate_flag)
                    if not ret:
                        print("[ERROR] Cannot read video after rewind")
                        break
                else:
                    print("[WARNING] Failed to capture frame")
                    break

            fps_frames += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer = now

            h, w = frame.shape[:2]

            # Detect laser
            target = tracker.detect(frame) if tracking_enabled else None

            # Every frame → one command into queue + trajectory canvas
            if target and homography is not None:
                nx, ny = warp_point(homography, target.cx, target.cy)
                if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                    arm_nx, arm_ny = compensate_for_arm(nx, ny, args.rotate)
                    bridge.put(True, arm_nx, arm_ny)
                    canvas.update(True, nx, ny)
                else:
                    bridge.put(False, 0.0, 0.0)
                    canvas.update(False, 0.0, 0.0)
            elif target:
                nx, ny = target.normalized_center(w, h)
                arm_nx, arm_ny = compensate_for_arm(nx, ny, args.rotate)
                bridge.put(True, arm_nx, arm_ny)
                canvas.update(True, nx, ny)
            else:
                bridge.put(False, 0.0, 0.0)  # pen up (idle)
                canvas.update(False, 0.0, 0.0)

            # Draw display
            display = frame.copy()

            if show_mask:
                mask = tracker.create_mask(frame)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_bgr[:, :, 0] = 0
                mask_bgr[:, :, 2] = 0
                display = cv2.addWeighted(display, 1.0, mask_bgr, 0.5, 0)

            # Draw quad overlay
            if quad:
                draw_quad(display, quad)

            if target:
                draw_target(display, target)

            draw_status(display, tracker, tracking_enabled, target,
                        homography, roi_source, show_mask, fps, arm, bridge)

            if writer:
                writer.write(display)

            cv2.imshow("Laser Drawing", display)
            cv2.imshow("Trajectory", canvas.render())

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] Quit")
                break

            elif key == ord(" "):
                tracking_enabled = not tracking_enabled
                print(f"[INFO] Tracking {'enabled' if tracking_enabled else 'paused'}")

            elif key == ord("d"):
                # Re-detect quad from current frame
                print("[INFO] Re-detecting quad...")
                quad = detect_quad_roi(frame, quad_detector)
                if quad:
                    tracker.roi = quad.as_xyxy()
                    roi_source = "QUAD ROI"
                    homography = compute_homography(quad)
                elif fallback_roi:
                    tracker.roi = fallback_roi
                    roi_source = "MANUAL ROI"
                    homography = None
                    print(f"[INFO] Falling back to manual ROI: {fallback_roi}")
                else:
                    tracker.roi = None
                    roi_source = "NO ROI"
                    homography = None

            elif key == ord("r"):
                tracking_enabled = False
                print("[INFO] Tracking paused")

            elif key == ord("m"):
                show_mask = not show_mask
                print(f"[INFO] Mask overlay {'on' if show_mask else 'off'}")

            elif key in (ord("+"), ord("=")):
                tracker.brightness_threshold = min(255, tracker.brightness_threshold + 5)
                print(f"[INFO] Brightness threshold: {tracker.brightness_threshold}")

            elif key in (ord("-"), ord("_")):
                tracker.brightness_threshold = max(100, tracker.brightness_threshold - 5)
                print(f"[INFO] Brightness threshold: {tracker.brightness_threshold}")

            elif key == ord("c"):
                canvas.clear()
                print("[INFO] Trajectory cleared")

            elif key == ord("t"):
                # Enter interactive calibration mode
                print("[INFO] Entering calibration mode...")
                run_calibration(cap, tracker, args.save_profile, rotate_flag)
                print(f"[INFO] Resumed tracking: {tracker}")

            elif key == ord("s"):
                # Quick-save current tracker parameters
                profile = tracker.to_profile()
                profile.save(args.save_profile)
                print(f"[INFO] Profile saved to {args.save_profile}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    # --- Phase 4: Cleanup ---
    print("[INFO] Cleaning up...")

    if arm:
        arm.stop()
        arm.join(timeout=20)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if arm:
        print(f"[OK] Arm stats: moves={arm.move_count}, failures={arm.fail_count}")

    print("[OK] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
