#!/usr/bin/env python3
"""Example 08: Linear interpolation movement (MoveL).

This example demonstrates how to move the end-effector in a straight line
using MoveL mode. Unlike move_cartesian (point-to-point), MoveL ensures
the end-effector follows a linear path in Cartesian space.

Key difference:
- move_cartesian (coord_type=0x00): Point-to-point, path may be curved
- move_linear (coord_type=0x02): Straight-line path guaranteed

Usage:
    # First, activate CAN bus
    bash scripts/can_activate.sh can0 1000000

    # Move to position using linear interpolation
    uv run python examples/08_move_linear.py --x 0.25 --y 0.0 --z 0.2

    # Use continuous mode with feedback
    uv run python examples/08_move_linear.py --x 0.25 --y 0.0 --z 0.2 --continuous -v

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
         Not all positions may be reachable.
"""

import sys
import os
import argparse
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader, MotionController
from piper_demo.utils import deg_to_rad


def main():
    parser = argparse.ArgumentParser(
        description="Move Piper arm using linear interpolation (MoveL)"
    )
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument("--x", type=float, required=True, help="X position in meters")
    parser.add_argument("--y", type=float, required=True, help="Y position in meters")
    parser.add_argument("--z", type=float, required=True, help="Z position in meters")
    parser.add_argument(
        "--roll", type=float, default=0.0, help="Roll angle in degrees (default: 0)"
    )
    parser.add_argument(
        "--pitch", type=float, default=0.0, help="Pitch angle in degrees (default: 0)"
    )
    parser.add_argument(
        "--yaw", type=float, default=0.0, help="Yaw angle in degrees (default: 0)"
    )
    parser.add_argument(
        "--speed", type=float, default=0.2, help="Speed factor 0.0-1.0 (default: 0.2)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Use continuous control mode with pose feedback",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for continuous mode (default: 10)",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="Settle time in seconds after reaching target (default: 0.5)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print pose updates during motion"
    )
    args = parser.parse_args()

    print("[INFO] MoveL (Linear Interpolation) mode")
    print("       Target pose:")
    print(f"       Position: ({args.x:.3f}, {args.y:.3f}, {args.z:.3f}) m")
    print(
        f"       Orientation: (R:{args.roll:.1f}, P:{args.pitch:.1f}, Y:{args.yaw:.1f}) deg"
    )
    print(f"       Speed factor: {args.speed}")
    mode_str = (
        f"Continuous (settle: {args.settle}s)" if args.continuous else "Single command"
    )
    print(f"       Mode: {mode_str}")
    print("[WARN] The arm will move in a STRAIGHT LINE! Ensure workspace is clear.")
    print()

    try:
        with PiperConnection(can_name=args.can) as conn:
            reader = JointReader(conn.piper)
            motion = MotionController(conn.piper, speed_factor=args.speed)

            # Read current end-effector pose
            pose = reader.read_end_pose()
            print("[INFO] Current end-effector pose:")
            print(pose)
            print()

            # Enable arm (auto moves to home position)
            print("[INFO] Enabling arm (moving to home position)...")
            conn.enable()
            print("[OK] Arm enabled and at home position")

            # Convert target orientation to radians
            roll_rad = deg_to_rad(args.roll)
            pitch_rad = deg_to_rad(args.pitch)
            yaw_rad = deg_to_rad(args.yaw)

            if args.continuous:
                # Use continuous control mode with feedback
                print("[INFO] Moving to target pose (MoveL continuous mode)...")

                last_print_time = [0.0]

                def pose_callback(current_pose):
                    """Print pose updates at 2Hz max."""
                    now = time.time()
                    if args.verbose and now - last_print_time[0] >= 0.5:
                        last_print_time[0] = now
                        x_mm, y_mm, z_mm = current_pose.position_mm()
                        print(
                            f"       Position: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm"
                        )

                reached = motion.move_linear_continuous(
                    x=args.x,
                    y=args.y,
                    z=args.z,
                    roll=roll_rad,
                    pitch=pitch_rad,
                    yaw=yaw_rad,
                    speed_factor=args.speed,
                    timeout_sec=args.timeout,
                    settle_sec=args.settle,
                    pose_callback=pose_callback if args.verbose else None,
                )

                if reached:
                    print("[INFO] Target pose reached!")
                else:
                    print("[WARN] Timeout reached before arriving at target pose")

            else:
                # Use single command mode
                print("[INFO] Moving to target pose (MoveL single command)...")
                motion.move_linear(
                    x=args.x,
                    y=args.y,
                    z=args.z,
                    roll=roll_rad,
                    pitch=pitch_rad,
                    yaw=yaw_rad,
                    speed_factor=args.speed,
                )

                # Wait for motion using pose feedback
                print("[INFO] Waiting for motion to complete...")
                reached = reader.wait_for_pose(
                    target_x=args.x,
                    target_y=args.y,
                    target_z=args.z,
                    target_roll=roll_rad,
                    target_pitch=pitch_rad,
                    target_yaw=yaw_rad,
                    timeout_sec=args.timeout,
                )

                if reached:
                    print("[INFO] Target pose reached!")
                else:
                    print("[WARN] Timeout reached before arriving at target pose")

            # Read final pose
            final_pose = reader.read_end_pose()
            print("\n[INFO] Final end-effector pose:")
            print(final_pose)

            # Safely disable arm (returns to home first)
            print("\n[INFO] Safely disabling arm (returning to home)...")
            conn.safe_disable()

            return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
