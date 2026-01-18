#!/usr/bin/env python3
"""Example 14b: HSV color tracking with automatic arm control.

Integrates USB camera + HSV color detection + Piper arm control
for automatic tracking of colored objects. No AI model required.

Usage:
    uv run python examples/ex14/main_color.py
    uv run python examples/ex14/main_color.py --color red --speed 0.3
    uv run python examples/ex14/main_color.py --color green --min-area 1000
    uv run python examples/ex14/main_color.py --no-display  # headless mode

Controls (when display enabled):
    q     - Quit and safely disable arm
    space - Toggle tracking on/off
    r     - Return to center position
    1     - Track red
    2     - Track green
    3     - Track blue
    4     - Track yellow

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
"""

import argparse
import os
import sys
import time
from typing import Optional

import cv2

# Add src to path for package import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from piper_demo import JointReader, MotionController, PiperConnection
from piper_demo.vision_arm_controller import VisionArmController

from camera import CameraCapture, CameraCaptureError
from color_tracker import ColorTarget, ColorTracker


# Movement throttle settings
MIN_MOVE_INTERVAL_SEC = 0.3  # Minimum time between arm movements
MIN_POSITION_CHANGE = 0.03  # Minimum normalized position change to trigger move

# Color key mappings
COLOR_KEYS = {
    ord("1"): "red",
    ord("2"): "green",
    ord("3"): "blue",
    ord("4"): "yellow",
}


def draw_target(
    frame,
    target: ColorTarget,
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
        color,
        cv2.MARKER_CROSS,
        20,
        2,
    )

    # Draw area label
    label = f"area: {target.area:.0f}"
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
    color_name: str,
    last_target: Optional[ColorTarget],
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

    # Color name
    cv2.putText(
        frame,
        f"Color: {color_name.upper()}",
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
    help_text = "[q]uit [space]toggle [r]eset [1-4]color"
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
        description="HSV color tracking with arm control"
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
        "--color",
        default="red",
        choices=ColorTracker.available_colors(),
        help="Color to track (default: red)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum detection area in pixels (default: 500)",
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
    args = parser.parse_args()

    # Parse camera device (int or string)
    try:
        camera_device = int(args.camera)
    except ValueError:
        camera_device = args.camera

    # Clamp speed
    speed = max(0.1, min(1.0, args.speed))

    print("=" * 60)
    print("HSV Color Tracking with Arm Control (ex14 color)")
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

    # --- Initialize color tracker ---
    print(f"[INFO] Initializing color tracker: {args.color}")
    try:
        tracker = ColorTracker(color=args.color, min_area=args.min_area)
        print(f"[OK] Tracker ready: {tracker}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        camera.release()
        return 1

    # --- Initialize arm ---
    print(f"[INFO] Connecting to arm on {args.can}...")
    try:
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
            PiperConnection.HOME_POSITION, reader, motion,
            invert_cam_y=True,   # Flip: was False (default)
            invert_cam_z=False,  # Flip: was True (default)
        )

        # Show workspace info
        ws = controller.config.workspace
        x_mm, y_mm, z_mm = home_fk.position_mm()
        print(f"[INFO] HOME FK: X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm")
        print(
            f"[INFO] Workspace X: {ws.x_arm.min*1000:.0f} ~ {ws.x_arm.max*1000:.0f} mm"
        )
        print(
            f"[INFO] Workspace Z: {ws.z_arm.min*1000:.0f} ~ {ws.z_arm.max*1000:.0f} mm"
        )

        print("[OK] Arm ready!")

    except Exception as e:
        print(f"[ERROR] Arm initialization failed: {e}")
        camera.release()
        return 1

    # --- Main loop ---
    print()
    print("[INFO] Starting tracking loop...")
    if not args.no_display:
        print("[INFO] Press 'q' to quit, 'space' to toggle tracking, 'r' to reset")
        print("[INFO] Press 1-4 to switch color (red/green/blue/yellow)")
        # Create window before loop
        cv2.namedWindow("Color Tracking", cv2.WINDOW_AUTOSIZE)

    tracking_enabled = True
    last_move_time = 0.0
    last_target_y = 0.5
    last_target_z = 0.5
    last_target: Optional[ColorTarget] = None
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

            # Run detection
            target = tracker.detect(frame)
            last_target = target

            # Process detection
            if target and tracking_enabled:
                # Get normalized center
                # Image X (horizontal) -> Arm X (front-back)
                # Image Y (vertical) -> Arm Z (left-right)
                norm_x, norm_y = target.normalized_center(w, h)
                target_y = norm_y  # Camera vertical -> cam Y -> Arm Z (left-right)
                target_z = norm_x  # Camera horizontal -> cam Z -> Arm X (front-back)

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
                        wait=False,  # Non-blocking for smooth tracking
                    )
                    last_move_result = result
                    last_move_time = time.time()
                    last_target_y = target_y
                    last_target_z = target_z

                    if result.ik_converged:
                        status = "OK" if not result.near_singularity else "SING"
                        if frame_count % 10 == 0:  # Reduce log spam
                            print(
                                f"[{status}] cam=({target_y:.2f}, {target_z:.2f}) "
                                f"-> arm=({result.x_arm*1000:.0f}, {result.z_arm*1000:.0f})mm"
                            )
                    else:
                        print(f"[IK FAIL] pos_err={result.position_error*1000:.1f}mm")

            # Display
            if not args.no_display:
                # Draw color mask overlay (optional, for debugging)
                # mask = tracker.create_mask(frame)
                # mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                # frame = cv2.addWeighted(frame, 1.0, mask_colored, 0.3, 0)

                # Draw target
                if target:
                    draw_target(frame, target)

                # Draw status overlay
                draw_status(
                    frame,
                    tracking_enabled,
                    tracker.color_name,
                    last_target,
                    last_move_result,
                )

                cv2.imshow("Color Tracking", frame)

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
                    print("[INFO] Returning to center...")
                    result = controller.move_to_normalized(
                        0.5,
                        0.5,
                        speed_factor=speed,
                        wait=True,
                    )
                    last_move_result = result
                    last_target_y = 0.5
                    last_target_z = 0.5

                elif key in COLOR_KEYS:
                    new_color = COLOR_KEYS[key]
                    tracker.set_color(new_color)
                    print(f"[INFO] Switched to color: {new_color}")

            else:
                # Headless mode: small delay to prevent CPU spinning
                time.sleep(0.01)

    except KeyboardInterrupt:
        print()
        print("[INFO] Interrupted by user")

    # --- Cleanup ---
    print("[INFO] Cleaning up...")

    if not args.no_display:
        cv2.destroyAllWindows()

    camera.release()
    print("[OK] Camera released")

    print("[INFO] Safely disabling arm...")
    conn.safe_disable(return_home=True, home_speed=20)
    print("[OK] Arm disabled")

    print("[OK] Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
