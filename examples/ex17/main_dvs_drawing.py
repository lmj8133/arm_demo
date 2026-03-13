#!/usr/bin/env python3
"""DVS laser-guided drawing with Piper robotic arm.

Uses XenReal ESC001D DVS (~200fps) for real-time laser tracking and drives
the Piper robotic arm for drawing via a FIFO producer-consumer architecture.

Architecture:
    DVSDrawingThread (bg, ~200fps)    Main Thread (~30fps)         ArmThread (bg)
      - xe_cam.get_frame_laser()       - dvs_drawing.get_latest()   - bridge.get()
      - tracker.detect_from_events()   - cv2.imshow() + keyboard    - drawer.move(w, x, y)
      - warp -> normalised (0-1)^2     - display DVS + canvas       - (blocks until done)
      - canvas.update() @ 200fps
      - bridge.put() every frame       (no bridge.put in main loop)

Usage:
    # Camera-only test (no hardware, no calibration):
    python3 examples/ex17/main_dvs_drawing.py --dvs-camera 2 --no-arm --no-dvs-cal

    # With quad calibration (no arm):
    python3 examples/ex17/main_dvs_drawing.py --dvs-camera 2 --no-arm

    # Full integration (on Orin):
    bash scripts/can_activate.sh can0 1000000
    python3 examples/ex17/main_dvs_drawing.py --dvs-camera 2 --can can0 --speed 0.2

Controls:
    q     - Quit (safe shutdown)
    space - Toggle tracking on/off
    c     - Clear trajectory canvas
    r     - Pause tracking
    d     - Re-run DVS quad calibration (camera mode switch)
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

sys.path.insert(0, _SCRIPT_DIR)                                     # ex17/
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "ex15"))          # ex15/
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "ex16"))          # ex16/
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))               # src/
sys.path.insert(0, "/workspace/xenreal_001d/src")                    # XenReal SDK
sys.path.insert(0, "/workspace/xenreal_001d")                        # XenReal SDK root

# --- DVS imports (ex15) ---
from quad_calibrator import (
    run_quad_calibration,
    compute_homography as dvs_compute_homography,
    warp_point as dvs_warp_point,
    save_calibration, load_calibration, DEFAULT_CALIBRATION_PATH,
)
from dvs_laser_tracker import DVSLaserTracker

# Display helpers from ex15/main_dvs_tracking.py
from main_dvs_tracking import dvs_frame_to_bgr, draw_dvs_target_scaled

# --- ex16 imports ---
from trajectory_canvas import TrajectoryCanvas
from gripper_tune import run_gripper_tune

# --- ex17 imports (DVSDrawingThread defined below) ---

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DVS_WIDTH = 164
DVS_HEIGHT = 160
DVS_ONLY_CONFIG = "/workspace/xenreal_001d/ESC001D_DV_RAW4_200FPS_20260204_modify.cfg"
HYBRID_CONFIG = "/workspace/xenreal_001d/ESC001D_2D_RAW8_DV_RAW2.cfg"


# ---------------------------------------------------------------------------
# DVSDrawingThread — Background DVS reader that pushes every frame to bridge
# ---------------------------------------------------------------------------

class DVSDrawingThread:
    """Background thread that reads DVS frames at native rate (~200fps).

    Like DVSReaderThread but additionally calls bridge.put() on every frame
    so that no high-frequency detail is lost.  Uses a TrajectoryCanvas
    (injected or self-created).  Pen-up flood prevention: only one pen-up
    command is sent on the writing→idle transition.
    """

    def __init__(
        self,
        xe_cam,
        tracker: DVSLaserTracker,
        homography: Optional[np.ndarray],
        bridge: "CommandBridge",
        scale: int = 3,
        canvas_size: int = 400,
        idle_clear: float = 0,
        write_confirm: int = 1,
        canvas: Optional["TrajectoryCanvas"] = None,
        canvas_lock: Optional[threading.Lock] = None,
    ):
        self._xe_cam = xe_cam
        self._tracker = tracker
        self._homography = homography
        self._bridge = bridge
        self._scale = scale

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_target = None  # DVSTarget or None
        self._latest_warped: Optional[Tuple[float, float]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0

        # Canvas: use injected (persistent) or create own (standalone)
        if canvas is not None:
            self._canvas = canvas
            self._canvas_lock = canvas_lock or threading.Lock()
        else:
            self._canvas = TrajectoryCanvas(size=canvas_size, idle_clear=idle_clear,
                                            write_confirm=write_confirm)
            self._canvas_lock = threading.Lock()
        # Main thread can toggle tracking on/off
        self._tracking_enabled = True
        # Pen-up flood prevention
        self._last_was_writing = False

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
        """Reader loop: capture frames, run tracker, push every frame to bridge."""
        fps_frames = 0
        fps_timer = time.time()

        while self._running:
            event_frame = self._xe_cam.get_frame_laser_nparray()
            if event_frame is None:
                continue

            # Rotate to match display convention (CCW 90° + flip H)
            event_frame = cv2.rotate(event_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            event_frame = cv2.flip(event_frame, 1)

            # Track laser spot (in rotated space)
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
                    if warped is not None:
                        self._canvas.update(True, warped[0], warped[1])
                    else:
                        self._canvas.update(False, 0.0, 0.0)
                elif target is not None:
                    # No calibration — fallback (rotated: w=DVS_HEIGHT, h=DVS_WIDTH)
                    nx = target.cx / DVS_HEIGHT
                    ny = 1.0 - (target.cy / DVS_WIDTH)
                    self._canvas.update(True, nx, ny)
                else:
                    self._canvas.update(False, 0.0, 0.0)

            # Push every frame to bridge (pen-up flood prevention)
            if self._tracking_enabled and warped is not None:
                # Swap axes: _normalized_to_meters maps cx→arm_y, cy→arm_x
                self._bridge.put(True, 1.0 - warped[1], warped[0])
                self._last_was_writing = True
            elif self._last_was_writing:
                self._bridge.put(False, 0.0, 0.0)
                self._last_was_writing = False
            # else: idle — don't flood queue with pen-up commands

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

    @property
    def piper(self):
        """Access C_PiperInterface_V2 (available after is_ready)."""
        return self._conn.piper if self._conn is not None else None

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

def draw_dvs_drawing_status(
    frame: np.ndarray,
    tracking_enabled: bool,
    target,
    warped: Optional[Tuple[float, float]],
    fps: float,
    arm: Optional[ArmThread],
    bridge: Optional[CommandBridge],
    has_homography: bool,
) -> None:
    """Draw status overlay on the DVS display frame for drawing mode."""
    h, w = frame.shape[:2]

    # Tracking status
    status = "TRACKING" if tracking_enabled else "PAUSED"
    status_color = (0, 255, 0) if tracking_enabled else (0, 165, 255)
    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Calibration indicator
    if has_homography:
        cv2.putText(frame, "CALIBRATED", (180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # FPS
    cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Target info
    if target and warped is not None:
        cv2.putText(frame,
                    f"Target: ({target.cx:.0f},{target.cy:.0f}) "
                    f"warp=({warped[0]:.2f},{warped[1]:.2f})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    elif target:
        nx = target.cx / DVS_WIDTH
        ny = target.cy / DVS_HEIGHT
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

    # Help
    help_text = "[q]uit [space]toggle [c]lear [r]eset [d]calibrate"
    cv2.putText(frame, help_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DVS laser-guided drawing with Piper arm"
    )
    parser.add_argument(
        "--dvs-camera", type=int, default=2,
        help="DVS device index -> /dev/videoN (default: 2)",
    )
    parser.add_argument(
        "--scale", type=int, default=3,
        help="DVS display scale factor (default: 3)",
    )
    parser.add_argument(
        "--noise-mask", type=str, default=None, metavar="PATH",
        help="Hot-pixel noise mask .npy path (laser mode)",
    )
    parser.add_argument(
        "--no-dvs-cal", action="store_true",
        help="Skip DVS quad calibration (use simple normalization)",
    )
    parser.add_argument(
        "--dvs-cal", type=str, default=DEFAULT_CALIBRATION_PATH, metavar="PATH",
        help=f"DVS calibration file path (default: {DEFAULT_CALIBRATION_PATH})",
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
        "--no-gripper-tune", action="store_true",
        help="Skip gripper fine-tune TUI before tracking",
    )
    parser.add_argument(
        "--idle-clear", type=float, default=0, metavar="SEC",
        help="Auto-clear canvas after SEC seconds idle (0=disabled, default: 0)",
    )
    parser.add_argument(
        "--canvas-size", type=int, default=400,
        help="Trajectory canvas pixel size (default: 400)",
    )
    args = parser.parse_args()

    speed = max(0.1, min(1.0, args.speed))

    print("=" * 60)
    print("DVS Laser-Guided Drawing (ex17)")
    print("=" * 60)

    # ===================================================================
    # Phase 1: DVS Camera Init + Quad Calibration
    # ===================================================================

    device = f"/dev/video{args.dvs_camera}"
    print(f"[INFO] DVS device: {device}")

    import example_open_xe_001d_laser as xe_cam
    xe_cam.DEVICE = device

    need_calibration = not args.no_dvs_cal
    dvs_homography: Optional[np.ndarray] = None

    if need_calibration:
        # Try loading saved corners as initial positions
        saved_corners = None
        if os.path.isfile(args.dvs_cal):
            try:
                saved_corners = load_calibration(args.dvs_cal)
                print(f"[OK] Loaded saved corners from {args.dvs_cal}")
            except (ValueError, KeyError) as e:
                print(f"[WARNING] Invalid calibration file: {e}")

        # Open camera in HYBRID config for RGB preview
        print("[INFO] Starting hybrid camera for quad calibration...")
        xe_cam.CONFIG_ABS_PATH = HYBRID_CONFIG
        xe_cam.start_camera_laser()

        corners = run_quad_calibration(
            xe_cam, scale=args.scale, initial_corners=saved_corners,
        )
        if corners is not None:
            dvs_homography = dvs_compute_homography(corners)
            save_calibration(corners, args.dvs_cal)
            print(f"[OK] Homography computed, saved to {args.dvs_cal}")
        else:
            print("[INFO] Calibration skipped, using simple normalization")

        # Close hybrid camera, reopen in DVS-only mode
        print("[INFO] Switching to DVS-only mode...")
        xe_cam.close_camera(xe_cam.g_cap)
    else:
        print("[INFO] DVS calibration disabled (--no-dvs-cal)")

    # Open / reopen DVS-only mode for tracking
    xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
    xe_cam.start_camera_laser()
    print(f"[OK] DVS camera ready ({DVS_WIDTH}x{DVS_HEIGHT})")

    # Create DVS laser tracker
    dvs_tracker = DVSLaserTracker(
        width=DVS_HEIGHT,
        height=DVS_WIDTH,
        noise_mask_path=args.noise_mask,
    )
    print(f"[OK] Tracker: {dvs_tracker}")

    # ===================================================================
    # Phase 2: Arm Init (background thread)
    # ===================================================================

    bridge = CommandBridge()
    arm: Optional[ArmThread] = None

    if not args.no_arm:
        arm = ArmThread(bridge, can_name=args.can, speed=speed)
        arm.start()
        print("[INFO] Arm initialization started in background...")
    else:
        print("[INFO] Arm control disabled (--no-arm)")

    # ===================================================================
    # Phase 2.5: Calibration Confirmation
    # ===================================================================

    cv2.namedWindow("DVS Drawing", cv2.WINDOW_AUTOSIZE)
    has_homography = dvs_homography is not None
    print("[INFO] Waiting for calibration confirmation... Press [ENTER] to start tracking.")

    # --- Stage 1: Wait for arm ready + confirm calibration ---
    while True:
        event_frame = xe_cam.get_frame_laser_nparray()
        if event_frame is None:
            continue

        display = dvs_frame_to_bgr(event_frame, scale=args.scale)

        arm_ready = (arm is None) or arm.is_ready.is_set()

        if not arm_ready:
            prompt = "Arm initializing..."
        elif not has_homography:
            prompt = "No calibration. [ENTER] start | [d] calibrate | [q] quit"
        else:
            prompt = "Calibration OK! [ENTER] start | [d] re-calibrate | [q] quit"

        # Draw prompt banner
        h_frame = display.shape[0]
        cv2.rectangle(display, (0, h_frame - 40), (display.shape[1], h_frame), (0, 0, 0), -1)
        cv2.putText(display, prompt, (10, h_frame - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("DVS Drawing", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 13 and arm_ready:  # ENTER
            bridge.put(True, 0.5, 0.5)
            print("[INFO] Pen down at center. Prepare, then press [ENTER] to start tracking.")
            # --- Stage 2: pen down, wait for tracking start ---
            while True:
                ef2 = xe_cam.get_frame_laser_nparray()
                if ef2 is None:
                    continue
                disp2 = dvs_frame_to_bgr(ef2, scale=args.scale)
                h2 = disp2.shape[0]
                cv2.rectangle(disp2, (0, h2 - 40), (disp2.shape[1], h2), (0, 0, 0), -1)
                cv2.putText(disp2, "Pen down. Prepare, then [ENTER] to track | [q] quit",
                            (10, h2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.imshow("DVS Drawing", disp2)
                k2 = cv2.waitKey(30) & 0xFF
                if k2 == 13:  # ENTER
                    # Gripper fine-tune phase
                    if arm is not None and not args.no_gripper_tune:
                        print("[INFO] Entering gripper fine-tune TUI...")
                        run_gripper_tune(
                            piper=arm.piper,
                            can_name=args.can,
                        )
                        print("[INFO] Gripper tune complete.")
                    bridge.put(False, 0.0, 0.0)
                    print("[INFO] Pen up. Tracking started.")
                    break
                elif k2 == ord("q"):
                    print("[INFO] Quit")
                    xe_cam.close_camera(xe_cam.g_cap)
                    cv2.destroyAllWindows()
                    if arm:
                        arm.stop()
                        arm.join()
                    return 0
            break

        elif key == ord("d"):
            # Re-run DVS quad calibration
            print("[INFO] Re-running DVS quad calibration...")
            xe_cam.close_camera(xe_cam.g_cap)
            xe_cam.CONFIG_ABS_PATH = HYBRID_CONFIG
            xe_cam.start_camera_laser()

            saved = None
            if os.path.isfile(args.dvs_cal):
                try:
                    saved = load_calibration(args.dvs_cal)
                except (ValueError, KeyError):
                    pass

            new_corners = run_quad_calibration(
                xe_cam, scale=args.scale, initial_corners=saved,
            )
            if new_corners is not None:
                dvs_homography = dvs_compute_homography(new_corners)
                save_calibration(new_corners, args.dvs_cal)
                has_homography = True
                print(f"[OK] Homography updated, saved to {args.dvs_cal}")
            else:
                print("[INFO] Calibration cancelled")

            xe_cam.close_camera(xe_cam.g_cap)
            xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
            xe_cam.start_camera_laser()

        elif key == ord("q"):
            print("[INFO] Quit")
            xe_cam.close_camera(xe_cam.g_cap)
            cv2.destroyAllWindows()
            if arm:
                arm.stop()
                arm.join()
            return 0

    # ===================================================================
    # Phase 3: DVSDrawingThread + Main Loop
    # ===================================================================

    dvs_drawing = DVSDrawingThread(
        xe_cam, dvs_tracker, dvs_homography, bridge,
        scale=args.scale, canvas_size=args.canvas_size,
        idle_clear=args.idle_clear, write_confirm=3,
    )
    dvs_drawing.start()

    cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)
    print()
    print("[INFO] Controls: [q]uit [space]toggle [c]lear [r]eset [d]calibrate")

    tracking_enabled = True

    try:
        while True:
            # Check for arm errors
            if arm and arm.error:
                print(f"[ERROR] Arm thread failed: {arm.error}")
                break

            # 1. Get latest DVS result (non-blocking)
            dvs_frame, dvs_target, dvs_warped, dvs_fps = dvs_drawing.get_latest()

            if dvs_frame is None:
                # No DVS frame yet — wait a bit
                cv2.waitKey(30)
                continue

            # 2. Display (bridge.put is handled by DVSDrawingThread at ~200fps)
            display = dvs_frame_to_bgr(dvs_frame, scale=args.scale)
            draw_dvs_target_scaled(display, dvs_target, scale=args.scale)
            draw_dvs_drawing_status(
                display, tracking_enabled, dvs_target, dvs_warped,
                dvs_fps, arm, bridge, has_homography,
            )
            cv2.imshow("DVS Drawing", display)
            cv2.imshow("Trajectory", dvs_drawing.render_canvas())

            # 4. Keyboard
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                print("[INFO] Quit")
                break

            elif key == ord(" "):
                tracking_enabled = not tracking_enabled
                dvs_drawing.tracking_enabled = tracking_enabled
                print(f"[INFO] Tracking {'enabled' if tracking_enabled else 'paused'}")

            elif key == ord("c"):
                dvs_drawing.clear_canvas()
                print("[INFO] Trajectory cleared")

            elif key == ord("r"):
                tracking_enabled = False
                dvs_drawing.tracking_enabled = False
                print("[INFO] Tracking paused")

            elif key == ord("d"):
                # Re-run DVS quad calibration (camera mode switch)
                print("[INFO] Re-running DVS quad calibration...")
                dvs_drawing.stop()
                xe_cam.close_camera(xe_cam.g_cap)

                xe_cam.CONFIG_ABS_PATH = HYBRID_CONFIG
                xe_cam.start_camera_laser()

                saved = None
                if os.path.isfile(args.dvs_cal):
                    try:
                        saved = load_calibration(args.dvs_cal)
                    except (ValueError, KeyError):
                        pass

                new_corners = run_quad_calibration(
                    xe_cam, scale=args.scale, initial_corners=saved,
                )
                if new_corners is not None:
                    dvs_homography = dvs_compute_homography(new_corners)
                    save_calibration(new_corners, args.dvs_cal)
                    has_homography = True
                    print(f"[OK] Homography updated, saved to {args.dvs_cal}")

                # Switch back to DVS-only mode
                xe_cam.close_camera(xe_cam.g_cap)
                xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
                xe_cam.start_camera_laser()

                # Restart DVS drawing thread with new homography
                dvs_drawing = DVSDrawingThread(
                    xe_cam, dvs_tracker, dvs_homography, bridge,
                    scale=args.scale, canvas_size=args.canvas_size,
                    idle_clear=args.idle_clear, write_confirm=3,
                )
                dvs_drawing.tracking_enabled = tracking_enabled
                dvs_drawing.start()
                print("[OK] DVS drawing thread restarted")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    # ===================================================================
    # Phase 4: Cleanup
    # ===================================================================

    print("[INFO] Cleaning up...")
    dvs_drawing.stop()

    if arm:
        arm.stop()
        arm.join(timeout=20)

    try:
        xe_cam.close_camera(xe_cam.g_cap)
    except Exception:
        pass

    cv2.destroyAllWindows()

    if arm:
        print(f"[OK] Arm stats: moves={arm.move_count}, failures={arm.fail_count}")

    print("[OK] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
