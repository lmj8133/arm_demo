#!/usr/bin/env python3
"""Example 14c: DVS event-based tracking with automatic arm control.

Integrates USB camera + simulated DVS event detection + Piper arm control
for automatic tracking of hand motion. Uses frame differencing to simulate
DVS sensor behavior.

Usage:
    uv run python examples/ex14/main_dvs.py
    uv run python examples/ex14/main_dvs.py --threshold 15 --speed 0.3
    uv run python examples/ex14/main_dvs.py --no-arm --camera 0  # test without arm
    uv run python examples/ex14/main_dvs.py --no-display  # headless mode

Controls (GUI and console):
    q     - Quit and safely disable arm
    space - Toggle tracking on/off (GUI only)
    r     - Return to center position
    +/=   - Increase DVS threshold
    -     - Decrease DVS threshold

Console commands (type + Enter):
    Same as above, useful in headless mode (--no-display)

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
"""

import argparse
import os
import queue
import sys
import threading
import time
from typing import Optional

import cv2

# Add src to path for package import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from camera import CameraCapture, CameraCaptureError
from dvs_tracker import DVSTarget, DVSTracker

# Movement throttle settings
MIN_MOVE_INTERVAL_SEC = 0.3  # Minimum time between arm movements
MIN_POSITION_CHANGE = 0.03  # Minimum normalized position change to trigger move


def console_input_thread(cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Background thread to read console input (non-blocking for main loop)."""
    while not stop_event.is_set():
        try:
            cmd = input()
            cmd_queue.put(cmd.strip().lower())
        except EOFError:
            break


def draw_target(
    frame,
    target: DVSTarget,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw target bounding box and center on frame."""
    x1, y1, x2, y2 = target.as_xyxy()

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw center crosshair
    cx, cy = int(target.cx), int(target.cy)
    cv2.drawMarker(
        frame,
        (cx, cy),
        (0, 0, 255),  # Red center point
        cv2.MARKER_CROSS,
        20,
        2,
    )

    # Draw active ratio label
    label = f"active: {target.active_ratio:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def draw_status(
    frame,
    tracking_enabled: bool,
    threshold: int,
    last_target: Optional[DVSTarget],
    last_move_result,
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Status text
    status = "TRACKING" if tracking_enabled else "PAUSED"
    status_color = (0, 255, 0) if tracking_enabled else (0, 165, 255)
    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

    # DVS threshold
    cv2.putText(
        frame,
        f"DVS threshold: {threshold}",
        (150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    # Target info
    if last_target:
        norm_x, norm_y = last_target.normalized_center(w, h)
        target_text = f"Target: ({norm_x:.2f}, {norm_y:.2f})"
        cv2.putText(
            frame,
            target_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    # Arm status
    if last_move_result:
        arm_text = f"Arm: ({last_move_result.y_cam:.2f}, {last_move_result.z_cam:.2f})"
        cv2.putText(
            frame,
            arm_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )

    # Help text
    help_text = "[q]uit [space]toggle [r]eset [+/-]threshold"
    cv2.putText(
        frame,
        help_text,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (128, 128, 128),
        1,
    )


def should_move(
    current_y: float,
    current_z: float,
    target_y: float,
    target_z: float,
    last_move_time: float,
) -> bool:
    """Check if arm should move based on throttle settings."""
    # Check time interval
    if time.time() - last_move_time < MIN_MOVE_INTERVAL_SEC:
        return False

    # Check position change
    dy = abs(target_y - current_y)
    dz = abs(target_z - current_z)

    return dy > MIN_POSITION_CHANGE or dz > MIN_POSITION_CHANGE


def main():
    parser = argparse.ArgumentParser(
        description="DVS event-based tracking with arm control"
    )
    parser.add_argument(
        "--can",
        default="can0",
        help="CAN interface name (default: can0)",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera device index or path (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera frame height (default: 480)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=15,
        help="DVS threshold (0-255, default: 15)",
    )
    parser.add_argument(
        "--consecutive",
        "-c",
        type=int,
        default=3,
        help="ROI boundary consecutive pixels (default: 3)",
    )
    parser.add_argument(
        "--min-active",
        "-m",
        type=float,
        default=0.2,
        help="Minimum active pixel ratio (default: 0.2)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Arm movement speed factor 0.1-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI display (headless mode)",
    )
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="Disable arm control (test tracking only)",
    )
    parser.add_argument(
        "--override-z",
        type=float,
        default=None,
        metavar="METERS",
        help="Override Z height for larger XY reach (e.g., 0.07 for 70mm). "
        "Use scripts/find_optimal_z.py to find optimal value.",
    )
    parser.add_argument(
        "--workspace-range",
        type=float,
        default=0.15,
        metavar="METERS",
        help="Half-range for XZ workspace (default: 0.15 = ±150mm)",
    )
    args = parser.parse_args()

    # Parse camera device (int or string)
    try:
        camera_device = int(args.camera)
    except ValueError:
        camera_device = args.camera

    # Clamp speed
    speed = max(0.1, min(1.0, args.speed))

    print("=" * 60)
    print("DVS Event-Based Tracking with Arm Control (ex14 DVS)")
    print("=" * 60)

    # --- Initialize camera ---
    print(f"[INFO] Opening camera: {camera_device}")
    try:
        camera = CameraCapture(
            device=camera_device,
            width=args.width,
            height=args.height,
        )
        camera.open()
        print(f"[OK] Camera opened: {camera.frame_size}")
    except CameraCaptureError as e:
        print(f"[ERROR] {e}")
        return 1

    # --- Initialize DVS tracker ---
    print(f"[INFO] Initializing DVS tracker: threshold={args.threshold}")
    tracker = DVSTracker(
        threshold=args.threshold,
        consecutive=args.consecutive,
        min_active_ratio=args.min_active,
    )
    print(f"[OK] Tracker ready: {tracker}")

    # --- Initialize arm (optional) ---
    controller = None
    conn = None
    if not args.no_arm:
        try:
            from piper_demo import JointReader, MotionController, PiperConnection
            from piper_demo.vision_arm_controller import VisionArmController

            print(f"[INFO] Connecting to arm on {args.can}...")
            conn = PiperConnection(can_name=args.can)
            conn.connect()

            reader = JointReader(conn.piper)
            motion = MotionController(conn.piper)

            # Enable arm and move to home
            print("[INFO] Enabling arm and moving to home...")
            conn.enable(go_home=True, home_speed=20, home_settle_sec=1.0)

            # Skip initial readings for stability
            for _ in range(5):
                reader.read_joints()
                time.sleep(0.05)

            # Create vision controller
            print("[INFO] Computing workspace from HOME_POSITION...")
            controller, home_fk = VisionArmController.from_home_position(
                PiperConnection.HOME_POSITION,
                reader,
                motion,
                workspace_range_xz=args.workspace_range,
                invert_cam_y=True,
                invert_cam_z=True,
                override_z=args.override_z,
            )

            # Show workspace info
            ws = controller.config.workspace
            x_mm, y_mm, z_mm = home_fk.position_mm()
            print(f"[INFO] HOME FK: X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm")
            if args.override_z is not None:
                print(
                    f"[INFO] Override Z: {args.override_z*1000:.1f} mm "
                    f"(delta: {(args.override_z - home_fk.z)*1000:+.1f} mm)"
                )
            print(
                f"[INFO] Workspace X (front/back): "
                f"{ws.x_arm.min*1000:.0f} ~ {ws.x_arm.max*1000:.0f} mm"
            )
            print(
                f"[INFO] Workspace Y (left/right): "
                f"{ws.y_arm.min*1000:.0f} ~ {ws.y_arm.max*1000:.0f} mm"
            )

            print("[OK] Arm ready!")

        except Exception as e:
            print(f"[ERROR] Arm initialization failed: {e}")
            camera.release()
            return 1
    else:
        print("[INFO] Arm control disabled (--no-arm)")

    # --- Main loop ---
    print()
    print("[INFO] Starting tracking loop...")
    print("[INFO] Console commands: q=quit, r=reset, +/-=threshold")
    if not args.no_display:
        print("[INFO] Press 'q' to quit, 'space' to toggle tracking")
        print("[INFO] Press '+/-' to adjust threshold, 'r' to reset")
        # Create window before loop
        cv2.namedWindow("DVS Tracking", cv2.WINDOW_AUTOSIZE)

    # Start console input thread
    cmd_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=console_input_thread, args=(cmd_queue, stop_event), daemon=True
    )
    input_thread.start()

    tracking_enabled = False
    last_move_time = 0.0
    last_target_y = 0.5
    last_target_z = 0.5
    last_target: Optional[DVSTarget] = None
    last_move_result = None

    running = True
    frame_count = 0

    try:
        while running:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("[WARNING] Failed to capture frame")
                continue

            frame_count += 1
            h, w = frame.shape[:2]

            # Run DVS detection
            target = tracker.detect(frame)
            last_target = target

            # Process detection
            if target and tracking_enabled and controller:
                # Get normalized center
                norm_x, norm_y = target.normalized_center(w, h)

                target_y = norm_x  # Camera horizontal -> Arm Y
                target_z = norm_y  # Camera vertical -> Arm Z

                # Check if we should move
                if should_move(
                    last_target_y,
                    last_target_z,
                    target_y,
                    target_z,
                    last_move_time,
                ):
                    result = controller.move_to_normalized(
                        target_y,
                        target_z,
                        speed_factor=speed,
                        wait=False,
                    )
                    last_move_result = result
                    last_move_time = time.time()
                    last_target_y = target_y
                    last_target_z = target_z

                    if result.ik_converged:
                        status = "OK" if not result.near_singularity else "SING"
                        if frame_count % 10 == 0:
                            print(
                                f"[{status}] cam=({target_y:.2f}, {target_z:.2f}) "
                                f"-> arm=({result.x_arm*1000:.0f}, {result.z_arm*1000:.0f})mm"
                            )
                    else:
                        print(f"[IK FAIL] pos_err={result.position_error*1000:.1f}mm")

            # Display
            if not args.no_display:
                # Get DVS event frame for display
                event_frame = tracker.last_event_frame
                if event_frame is not None:
                    # Convert grayscale to BGR for colored overlays
                    display_frame = cv2.cvtColor(event_frame, cv2.COLOR_GRAY2BGR)
                else:
                    # First frame: show original as grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                # Draw target ROI
                if target:
                    draw_target(display_frame, target)

                # Draw status overlay
                draw_status(
                    display_frame,
                    tracking_enabled,
                    tracker.threshold,
                    last_target,
                    last_move_result,
                )

                cv2.imshow("DVS Tracking", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[INFO] Quit requested")
                    running = False

                elif key == ord(" "):
                    tracking_enabled = not tracking_enabled
                    status = "enabled" if tracking_enabled else "disabled"
                    print(f"[INFO] Tracking {status}")

                elif key == ord("r"):
                    tracking_enabled = False
                    print("[INFO] Returning to center...")
                    if controller:
                        result = controller.move_to_normalized(
                            0.5,
                            0.5,
                            speed_factor=speed,
                            wait=True,
                        )
                        last_move_result = result
                    last_target_y = 0.5
                    last_target_z = 0.5

                elif key in (ord("+"), ord("=")):
                    # Increase threshold
                    tracker.threshold = min(255, tracker.threshold + 5)
                    print(f"[INFO] DVS threshold: {tracker.threshold}")

                elif key == ord("-"):
                    # Decrease threshold
                    tracker.threshold = max(1, tracker.threshold - 5)
                    print(f"[INFO] DVS threshold: {tracker.threshold}")

            # Handle console input
            while not cmd_queue.empty():
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break

                if cmd == "q":
                    print("[INFO] Quit requested (console)")
                    running = False

                elif cmd == "r":
                    tracking_enabled = False
                    print("[INFO] Returning to center...")
                    if controller:
                        result = controller.move_to_normalized(
                            0.5,
                            0.5,
                            speed_factor=speed,
                            wait=True,
                        )
                        last_move_result = result
                    last_target_y = 0.5
                    last_target_z = 0.5

                elif cmd == "+":
                    tracker.threshold = min(255, tracker.threshold + 5)
                    print(f"[INFO] DVS threshold: {tracker.threshold}")

                elif cmd == "-":
                    tracker.threshold = max(1, tracker.threshold - 5)
                    print(f"[INFO] DVS threshold: {tracker.threshold}")

            if args.no_display:
                # Headless mode: small delay to prevent CPU spinning
                time.sleep(0.01)

    except KeyboardInterrupt:
        print()
        print("[INFO] Interrupted by user")

    # --- Cleanup ---
    print("[INFO] Cleaning up...")
    stop_event.set()

    if not args.no_display:
        cv2.destroyAllWindows()

    camera.release()
    print("[OK] Camera released")

    if conn and not args.no_arm:
        print("[INFO] Safely disabling arm...")
        conn.safe_disable(return_home=True, home_speed=20)
        print("[OK] Arm disabled")

    print("[OK] Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
