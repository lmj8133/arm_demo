#!/usr/bin/env python3
"""Example 05: Gripper control.

This example demonstrates how to control the Piper arm gripper.

Usage:
    # First, activate CAN bus
    bash scripts/can_activate.sh can0 1000000

    # Open gripper
    uv run python examples/05_gripper.py --action open

    # Close gripper
    uv run python examples/05_gripper.py --action close

    # Set specific opening (in mm)
    uv run python examples/05_gripper.py --action set --position 40
"""

import sys
import os
import argparse
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, GripperController


def main():
    parser = argparse.ArgumentParser(description="Control Piper arm gripper")
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--action",
        choices=["open", "close", "set", "read"],
        required=True,
        help="Gripper action",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=40.0,
        help="Gripper opening in mm (for 'set' action, default: 40)",
    )
    parser.add_argument(
        "--speed", type=int, default=500, help="Gripper speed 0-1000 (default: 500)"
    )
    args = parser.parse_args()

    print(f"[INFO] Gripper action: {args.action}")

    try:
        with PiperConnection(can_name=args.can) as conn:
            gripper = GripperController(
                conn.piper,
                speed=args.speed,
            )

            # Read current position
            current_mm = gripper.read_position_mm()
            print(f"[INFO] Current gripper position: {current_mm:.1f} mm")

            if args.action == "read":
                return 0

            # Enable arm for gripper control (skip home for gripper-only control)
            print("[INFO] Enabling arm (skipping home for gripper-only control)...")
            conn.enable(go_home=False)
            time.sleep(0.3)

            if args.action == "open":
                print("[INFO] Opening gripper...")
                gripper.open(speed=args.speed)

            elif args.action == "close":
                print("[INFO] Closing gripper...")
                gripper.close(speed=args.speed)

            elif args.action == "set":
                print(f"[INFO] Setting gripper to {args.position:.1f} mm...")
                gripper.set_position_mm(args.position, speed=args.speed)

            # Wait for gripper motion
            time.sleep(1.0)

            # Read final position
            final_mm = gripper.read_position_mm()
            print(f"[OK] Final gripper position: {final_mm:.1f} mm")

            # Safely disable arm (returns to home first)
            conn.safe_disable()
            return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
