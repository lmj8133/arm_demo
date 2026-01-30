#!/usr/bin/env python3
"""Example 04: Cartesian pose movement.

This example demonstrates how to move the end-effector
to a Cartesian position with pose feedback.

Usage:
    # First, activate CAN bus
    bash scripts/can_activate.sh can0 1000000

    # Move to a Cartesian position (uses continuous mode by default)
    uv run python examples/04_move_pose.py --x 0.3 --y 0.0 --z 0.2

    # Use single command mode (without continuous feedback)
    uv run python examples/04_move_pose.py --x 0.3 --y 0.0 --z 0.2 --no-continuous

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
        description="Move Piper arm end-effector to Cartesian position"
    )
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument("--x", type=float, required=True, help="X position in meters")
    parser.add_argument("--y", type=float, required=True, help="Y position in meters")
    parser.add_argument("--z", type=float, required=True, help="Z position in meters")
    parser.add_argument(
        "--roll",
        type=float,
        default=None,
        help="Roll angle in degrees (default: current)",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=None,
        help="Pitch angle in degrees (default: current)",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=None,
        help="Yaw angle in degrees (default: current)",
    )
    parser.add_argument(
        "--speed", type=float, default=0.2, help="Speed factor 0.0-1.0 (default: 0.2)"
    )
    parser.add_argument(
        "--no-continuous",
        action="store_true",
        help="Use single command mode instead of continuous control",
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

    print(f"[INFO] Target position: ({args.x:.3f}, {args.y:.3f}, {args.z:.3f}) m")
    print(f"       Speed factor: {args.speed}")
    mode_str = (
        "Single command"
        if args.no_continuous
        else f"Continuous (settle: {args.settle}s)"
    )
    print(f"       Mode: {mode_str}")
    print("[WARN] The arm will move! Ensure workspace is clear.")
    print()

    try:
        with PiperConnection(can_name=args.can) as conn:
            reader = JointReader(conn.piper)
            motion = MotionController(conn.piper, speed_factor=args.speed)

            # Read current joint state
            state = reader.read_joints()
            print("[INFO] Current joint state:")
            print(state)
            print()

            # Read current end-effector pose
            pose = reader.read_end_pose()
            print("[INFO] Current end-effector pose:")
            print(pose)
            print()

            # Use current orientation if not specified
            current_roll_deg, current_pitch_deg, current_yaw_deg = (
                pose.orientation_deg()
            )
            roll_deg = args.roll if args.roll is not None else current_roll_deg
            pitch_deg = args.pitch if args.pitch is not None else current_pitch_deg
            yaw_deg = args.yaw if args.yaw is not None else current_yaw_deg

            print(
                f"[INFO] Target orientation: (R:{roll_deg:.1f}°, P:{pitch_deg:.1f}°, Y:{yaw_deg:.1f}°)"
            )
            if args.roll is None or args.pitch is None or args.yaw is None:
                print("       (Using current values for unspecified orientation)")
            print()

            # Enable arm (auto moves to home position)
            print("[INFO] Enabling arm (moving to home position)...")
            conn.enable()
            print("[OK] Arm enabled and at home position")

            # Convert target orientation to radians
            roll_rad = deg_to_rad(roll_deg)
            pitch_rad = deg_to_rad(pitch_deg)
            yaw_rad = deg_to_rad(yaw_deg)

            if not args.no_continuous:
                # Use continuous control mode with feedback
                print("[INFO] Moving to target pose (continuous mode)...")

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

                reached = motion.move_cartesian_continuous(
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
                # Use single command mode (original behavior)
                print("[INFO] Moving to target pose (single command)...")
                motion.move_cartesian(
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
