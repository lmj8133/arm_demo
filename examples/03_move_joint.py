#!/usr/bin/env python3
"""Example 03: Single joint movement.

This example demonstrates how to move individual joints
of the Piper arm.

Usage:
    # First, activate CAN bus
    bash scripts/can_activate.sh can0 1000000

    # Move joint 1 to 30 degrees
    uv run python examples/03_move_joint.py --joint 1 --angle 30

    # Move joint 2 to 45 degrees with slower speed
    uv run python examples/03_move_joint.py --joint 2 --angle 45 --speed 0.2

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
"""

import sys
import os
import argparse
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader, MotionController
from piper_demo.utils import deg_to_rad, rad_to_deg


def main():
    parser = argparse.ArgumentParser(description="Move a single joint of the Piper arm")
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--joint", type=int, required=True, choices=range(1, 7),
        help="Joint number (1-6)"
    )
    parser.add_argument(
        "--angle", type=float, required=True, help="Target angle in degrees"
    )
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Speed factor 0.0-1.0 (default: 0.3)"
    )
    args = parser.parse_args()

    joint_idx = args.joint - 1  # Convert to 0-indexed
    target_rad = deg_to_rad(args.angle)

    print(f"[INFO] Target: Joint {args.joint} -> {args.angle:.1f}° at speed {args.speed}")
    print(f"[WARN] The arm will move! Ensure workspace is clear.")
    print()

    try:
        with PiperConnection(can_name=args.can) as conn:
            reader = JointReader(conn.piper)
            motion = MotionController(conn.piper, speed_factor=args.speed)

            # Read current position
            state = reader.read_joints()
            current_deg = state.positions_deg()[joint_idx]
            print(f"[INFO] Current position: Joint {args.joint} = {current_deg:.2f}°")

            # Enable arm (auto moves to home position)
            print("[INFO] Enabling arm (moving to home position)...")
            conn.enable()
            print("[OK] Arm enabled and at home position")

            # Build target positions (home + modified joint)
            home_positions = [0.0] * 6
            target_positions = home_positions.copy()
            target_positions[joint_idx] = target_rad

            # Execute motion
            print(f"[INFO] Moving Joint {args.joint} to {args.angle:.1f}°...")
            motion.move_joint(target_positions, args.speed)

            # Wait for motion to complete
            reached = reader.wait_for_position(
                target_positions,
                tolerance_rad=deg_to_rad(2.0),  # 2 degree tolerance
                timeout_sec=10.0,
            )

            # Read final position
            state = reader.read_joints()
            final_deg = state.positions_deg()[joint_idx]

            if reached:
                print(f"[OK] Motion complete: Joint {args.joint} = {final_deg:.2f}°")
            else:
                print(f"[WARN] Timeout reached: Joint {args.joint} = {final_deg:.2f}°")

            # Hold position for user to see the result
            print("[INFO] Holding position for 3 seconds...")
            time.sleep(3.0)

            # Safely disable arm
            print("[INFO] Safely disabling arm (returning to home)...")
            conn.safe_disable()

            return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
