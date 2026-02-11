#!/usr/bin/env python3
"""DVS camera tracking with trajectory preview and optional arm control.

Uses XenReal ESC001D DVS (Dynamic Vision Sensor) for real-time tracking
and displays an accumulated trajectory on a separate canvas window.
Supports two tracker modes: laser spot (default) and hand tracking (--hand).

Architecture:
    Main Thread (DVS Camera + GUI)  -->  CommandBridge (FIFO)  -->  Arm Thread (optional)

Usage:
    # Laser tracking (default, camera-only):
    python3 examples/ex15/main_dvs_tracking.py --camera 2

    # Laser tracking with pre-recorded noise mask:
    python3 examples/ex15/main_dvs_tracking.py --camera 2 --noise-mask recordings/no_signal.npy

    # Hand tracking mode:
    python3 examples/ex15/main_dvs_tracking.py --camera 2 --hand

    # Full integration with arm (on Orin with CAN bus):
    bash scripts/can_activate.sh can0 1000000
    python3 examples/ex15/main_dvs_tracking.py --camera 2 --arm --can can0 --speed 0.2

Controls:
    q     - Quit (safe shutdown)
    space - Toggle tracking on/off
    c     - Clear trajectory canvas
    r     - Pause tracking
    s     - Toggle DVS frame recording (saves .npy on stop)
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# ex15/ — dvs_tracker, drawing
sys.path.insert(0, _SCRIPT_DIR)
# ex16/ — trajectory_canvas
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "ex16"))
# src/ — piper_demo
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
# XenReal SDK
sys.path.insert(0, "/workspace/xenreal_001d/src")
sys.path.insert(0, "/workspace/xenreal_001d")

from dvs_tracker import DVSTracker
from trajectory_canvas import TrajectoryCanvas
from quad_calibrator import (
    run_quad_calibration, compute_homography, warp_point,
    save_calibration, load_calibration, DEFAULT_CALIBRATION_PATH,
)

# DVS camera dimensions (ESC001D fixed resolution)
DVS_WIDTH = 164
DVS_HEIGHT = 160

# Camera config paths
DVS_ONLY_CONFIG = "/workspace/xenreal_001d/ESC001D_DV_RAW4_200FPS_20260204_modify.cfg"
HYBRID_CONFIG = "/workspace/xenreal_001d/ESC001D_2D_RAW8_DV_RAW2.cfg"


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
# Display helpers
# ---------------------------------------------------------------------------

def dvs_frame_to_bgr(event_frame: np.ndarray, scale: int = 3) -> np.ndarray:
    """Convert grayscale DVS event frame to BGR, upscaled for display.

    Args:
        event_frame: Grayscale frame (H, W), dtype uint8.
        scale: Integer upscale factor (default 3: 164x160 -> 492x480).

    Returns:
        BGR image suitable for cv2.imshow().
    """
    bgr = cv2.cvtColor(event_frame, cv2.COLOR_GRAY2BGR)
    if scale > 1:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (w * scale, h * scale),
                          interpolation=cv2.INTER_NEAREST)
    return bgr


def draw_dvs_target_scaled(
    frame: np.ndarray,
    target,
    scale: int = 3,
) -> None:
    """Draw bounding box and crosshair for DVS target on scaled frame.

    Args:
        frame: BGR display frame (already upscaled).
        target: DVSTarget instance, or None (no-op).
        scale: Scale factor used for the frame.
    """
    if target is None:
        return

    x, y, w, h = target.bbox
    x1 = x * scale
    y1 = y * scale
    x2 = (x + w) * scale
    y2 = (y + h) * scale

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crosshair at center
    cx = int(target.cx * scale)
    cy = int(target.cy * scale)
    arm = 12
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (0, 255, 0), 1, cv2.LINE_AA)

    # Label
    label = f"({target.cx:.0f},{target.cy:.0f}) a={target.area:.0f}"
    cv2.putText(frame, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def draw_status(
    frame: np.ndarray,
    tracking_enabled: bool,
    target,
    fps: float,
    arm: Optional[ArmThread],
    bridge: Optional[CommandBridge],
    recording: bool = False,
    record_count: int = 0,
    homography: Optional[np.ndarray] = None,
) -> None:
    """Draw status overlay on the DVS display frame."""
    h, w = frame.shape[:2]

    # Tracking status
    status = "TRACKING" if tracking_enabled else "PAUSED"
    status_color = (0, 255, 0) if tracking_enabled else (0, 165, 255)
    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Calibration indicator
    if homography is not None:
        cv2.putText(frame, "CALIBRATED", (180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # FPS
    cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Target info
    if target and homography is not None:
        wx, wy = warp_point(homography, target.cx, target.cy)
        in_bounds = 0.0 <= wx <= 1.0 and 0.0 <= wy <= 1.0
        tag = "" if in_bounds else " [OUT]"
        cv2.putText(frame,
                    f"Target: ({target.cx:.0f},{target.cy:.0f}) "
                    f"warp=({wx:.2f},{wy:.2f}){tag}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if in_bounds else (0, 0, 255), 1)
    elif target:
        nx, ny = target.normalized_center(DVS_WIDTH, DVS_HEIGHT)
        cv2.putText(frame,
                    f"Target: ({target.cx:.0f},{target.cy:.0f}) "
                    f"norm=({nx:.2f},{ny:.2f})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Target: NOT DETECTED", (10, 50),
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
        cv2.putText(frame, f"QUEUE: {bridge.pending} (no-arm mode)", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Recording indicator
    if recording:
        cv2.putText(frame, "REC", (w - 70, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"{record_count} frames", (w - 120, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Help
    help_text = "[q]uit [space]toggle [c]lear [r]eset [s]ave"
    cv2.putText(frame, help_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DVS camera hand tracking with trajectory preview"
    )
    parser.add_argument(
        "--camera", type=int, default=2,
        help="DVS camera device index (default: 2) -> /dev/videoN",
    )
    parser.add_argument(
        "--consecutive", type=int, default=3,
        help="ROI detection threshold (default: 3)",
    )
    parser.add_argument(
        "--min-active", type=float, default=0.01,
        help="Min active pixel ratio for detection (default: 0.01)",
    )
    parser.add_argument(
        "--min-roi-size", type=int, default=10,
        help="Min ROI size in pixels (default: 10)",
    )
    parser.add_argument(
        "--scale", type=int, default=3,
        help="Display scale factor (default: 3, 164x160 -> 492x480)",
    )
    parser.add_argument(
        "--idle-clear", type=float, default=1, metavar="SEC",
        help="Auto-clear trajectory after SEC seconds idle (0=disabled, default: 1)",
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
        "--arm", action="store_true",
        help="Enable arm control (requires CAN bus)",
    )
    parser.add_argument(
        "--no-arm", action="store_true", default=True,
        help="Camera-only mode (default)",
    )
    parser.add_argument(
        "--record-dir", default="recordings",
        help="Recording output directory (default: recordings/)",
    )
    # Tracker mode (laser is default, --hand to switch)
    parser.add_argument(
        "--hand", action="store_true",
        help="Use DVS hand tracker instead of laser tracker",
    )
    parser.add_argument(
        "--noise-mask", type=str, default=None,
        help="Path to .npy noise recording for hot pixel calibration (laser mode)",
    )
    parser.add_argument(
        "--no-cal", action="store_true",
        help="Skip quad calibration (use simple normalization)",
    )
    parser.add_argument(
        "--dvs-cal", type=str, default=DEFAULT_CALIBRATION_PATH, metavar="PATH",
        help=f"DVS calibration file path (default: {DEFAULT_CALIBRATION_PATH})",
    )
    args = parser.parse_args()

    # --arm overrides --no-arm
    use_arm = args.arm
    speed = max(0.1, min(1.0, args.speed))

    print("=" * 60)
    print("DVS Camera Hand Tracking with Trajectory Preview (ex15)")
    print("=" * 60)

    # --- Phase 1: DVS Camera Init + Calibration ---

    device = f"/dev/video{args.camera}"
    print(f"[INFO] DVS device: {device}")

    import example_open_xe_001d_laser as xe_cam
    xe_cam.DEVICE = device

    # Determine whether to run quad calibration
    need_calibration = (not args.hand) and (not args.no_cal)
    homography: Optional[np.ndarray] = None

    if need_calibration:
        # Try loading saved corners as initial positions
        saved_corners = None
        if os.path.isfile(args.dvs_cal):
            try:
                saved_corners = load_calibration(args.dvs_cal)
                print(f"[OK] Loaded saved corners from {args.dvs_cal}")
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Invalid calibration file: {e}")

        # Phase 1a: open camera in HYBRID config for RGB preview
        print("[INFO] Starting hybrid camera for quad calibration...")
        xe_cam.CONFIG_ABS_PATH = HYBRID_CONFIG
        xe_cam.start_camera_laser()

        # Phase 1b: run interactive calibration (with pre-filled corners if available)
        corners = run_quad_calibration(
            xe_cam, scale=args.scale, initial_corners=saved_corners,
        )
        if corners is not None:
            homography = compute_homography(corners)
            save_calibration(corners, args.dvs_cal)
            print(f"[OK] Homography computed, saved to {args.dvs_cal}")
        else:
            print("[INFO] Calibration skipped, using simple normalization")

        # Phase 1c: close hybrid camera, reopen in DVS-only mode
        print("[INFO] Switching to DVS-only mode...")
        xe_cam.close_camera(xe_cam.g_cap)
        xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
        xe_cam.start_camera_laser()
        print(f"[OK] DVS camera ready ({DVS_WIDTH}x{DVS_HEIGHT})")
    else:
        # Direct DVS-only startup (hand mode or --no-cal)
        xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
        print("[INFO] Starting XenReal DVS camera...")
        xe_cam.start_camera_laser()
        print(f"[OK] DVS camera ready ({DVS_WIDTH}x{DVS_HEIGHT})")

    # Create DVS tracker
    if not args.hand:
        from dvs_laser_tracker import DVSLaserTracker
        tracker = DVSLaserTracker(
            width=DVS_WIDTH,
            height=DVS_HEIGHT,
            noise_mask_path=args.noise_mask,
        )
    else:
        tracker = DVSTracker(
            width=DVS_WIDTH,
            height=DVS_HEIGHT,
            consecutive=args.consecutive,
            min_active_ratio=args.min_active,
            min_roi_size=args.min_roi_size,
        )
    print(f"[OK] Tracker: {tracker}")

    # --- Phase 2: Arm Init (optional) ---

    bridge = CommandBridge()
    canvas = TrajectoryCanvas(idle_clear=args.idle_clear)
    arm: Optional[ArmThread] = None

    if use_arm:
        arm = ArmThread(bridge, can_name=args.can, speed=speed)
        arm.start()
        print("[INFO] Arm initialization started in background...")
    else:
        print("[INFO] Arm control disabled (camera-only mode)")

    # --- Phase 3: Main Loop ---

    cv2.namedWindow("DVS Tracking", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)
    print()
    print("[INFO] Controls: [q]uit [space]toggle [c]lear [r]eset [s]ave")

    tracking_enabled = True
    recording = False
    recorded_frames: list = []
    record_dir = args.record_dir
    os.makedirs(record_dir, exist_ok=True)
    fps = 0.0
    fps_timer = time.time()
    fps_frames = 0

    try:
        while True:
            # Check for arm errors
            if arm and arm.error:
                print(f"[ERROR] Arm thread failed: {arm.error}")
                break

            # Capture DVS event frame
            event_frame = xe_cam.get_frame_laser_nparray()
            if event_frame is None:
                continue

            # FPS counter
            fps_frames += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer = now

            # Record raw event frame if recording
            if recording:
                recorded_frames.append(event_frame.copy())

            # Detect hand from events
            target = tracker.detect_from_events(event_frame) if tracking_enabled else None

            # Feed command bridge + trajectory canvas
            if target and homography is not None:
                wx, wy = warp_point(homography, target.cx, target.cy)
                if 0.0 <= wx <= 1.0 and 0.0 <= wy <= 1.0:
                    canvas.update(True, wx, wy)    # canvas mirrors camera: nx=L→R(wx), ny=B→T(wy)
                    bridge.put(True, wx, wy)       # arm: user convention (vertical, horizontal)
                else:
                    canvas.update(False, 0.0, 0.0)
                    bridge.put(False, 0.0, 0.0)
            elif target:
                nx, ny = target.normalized_center(DVS_WIDTH, DVS_HEIGHT)
                # Y-flip: DVS origin is top-left, canvas origin is bottom-left
                canvas.update(True, nx, 1.0 - ny)
                bridge.put(True, nx, ny)
            else:
                canvas.update(False, 0.0, 0.0)
                bridge.put(False, 0.0, 0.0)

            # Render DVS display
            display = dvs_frame_to_bgr(event_frame, scale=args.scale)
            draw_dvs_target_scaled(display, target, scale=args.scale)
            draw_status(display, tracking_enabled, target, fps, arm, bridge,
                        recording=recording, record_count=len(recorded_frames),
                        homography=homography)

            cv2.imshow("DVS Tracking", display)
            cv2.imshow("Trajectory", canvas.render())

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] Quit")
                break

            elif key == ord(" "):
                tracking_enabled = not tracking_enabled
                print(f"[INFO] Tracking {'enabled' if tracking_enabled else 'paused'}")

            elif key == ord("c"):
                canvas.clear()
                print("[INFO] Trajectory cleared")

            elif key == ord("r"):
                tracking_enabled = False
                print("[INFO] Tracking paused")

            elif key == ord("s"):
                if not recording:
                    recording = True
                    recorded_frames = []
                    print("[REC] Recording started...")
                else:
                    recording = False
                    if recorded_frames:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        path = os.path.join(record_dir, f"dvs_{ts}.npy")
                        np.save(path, np.array(recorded_frames, dtype=np.uint8))
                        print(f"[REC] Saved {len(recorded_frames)} frames -> {path}")
                    else:
                        print("[REC] No frames recorded")
                    recorded_frames = []

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    # --- Phase 4: Cleanup ---

    # Auto-save unfinished recording
    if recording and recorded_frames:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(record_dir, f"dvs_{ts}.npy")
        np.save(path, np.array(recorded_frames, dtype=np.uint8))
        print(f"[REC] Auto-saved {len(recorded_frames)} frames -> {path}")

    print("[INFO] Cleaning up...")

    if arm:
        arm.stop()
        arm.join(timeout=20)

    xe_cam.close_camera(xe_cam.g_cap)
    cv2.destroyAllWindows()

    if arm:
        print(f"[OK] Arm stats: moves={arm.move_count}, failures={arm.fail_count}")

    print("[OK] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
