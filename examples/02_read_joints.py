#!/usr/bin/env python3
"""Example 02: Continuous joint reading with optional pose display.

This example demonstrates how to continuously read and display
joint positions from the Piper arm, with optional end-effector pose.

Usage:
    # First, activate CAN bus
    bash scripts/can_activate.sh can0 1000000

    # Run this example (press Ctrl+C to stop)
    uv run python examples/02_read_joints.py

    # Show SDK-reported end-effector pose (recommended for move_cartesian)
    uv run python examples/02_read_joints.py --pose

    # Show calculated FK (for comparison/debugging)
    uv run python examples/02_read_joints.py --fk

    # Show both SDK pose and calculated FK
    uv run python examples/02_read_joints.py --pose --fk
"""

import sys
import os
import argparse
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "piper_demo"))

from piper_demo import PiperConnection, JointReader
from kinematics import forward_kinematics


def make_callback(reader: JointReader, show_pose: bool, show_fk: bool):
    """Create callback function with pose/FK options."""

    def print_state(state):
        """Callback to print joint state and optional pose/FK."""
        positions_deg = state.positions_deg()
        joints_str = " | ".join(
            [f"J{i + 1}:{p:6.1f}°" for i, p in enumerate(positions_deg)]
        )
        gripper_mm = state.gripper * 1000

        extra = ""

        if show_pose:
            # Read SDK-reported end-effector pose
            pose = reader.read_end_pose()
            x_mm, y_mm, z_mm = pose.position_mm()
            extra += f" | SDK: X:{x_mm:6.1f} Y:{y_mm:6.1f} Z:{z_mm:6.1f}"

        if show_fk:
            # Compute forward kinematics
            fk = forward_kinematics(state.positions)
            x_mm, y_mm, z_mm = fk.position_mm()
            extra += f" | FK: X:{x_mm:6.1f} Y:{y_mm:6.1f} Z:{z_mm:6.1f}"

        print(f"\r{joints_str} | Grip:{gripper_mm:4.0f}mm{extra}", end="", flush=True)

    return print_state


def main():
    parser = argparse.ArgumentParser(description="Read Piper arm joint positions")
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--rate", type=float, default=10.0, help="Reading rate in Hz (default: 10)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (default: infinite)",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Show SDK-reported end-effector pose (use this for move_cartesian)",
    )
    parser.add_argument(
        "--fk", action="store_true", help="Show calculated FK position (for debugging)"
    )
    args = parser.parse_args()

    print(f"[INFO] Connecting to Piper arm on {args.can}...")
    print(f"[INFO] Reading at {args.rate} Hz", end="")
    if args.duration:
        print(f" for {args.duration} seconds")
    else:
        print(" (press Ctrl+C to stop)")
    if args.pose:
        print("[INFO] SDK pose display enabled (use these values for move_cartesian)")
    if args.fk:
        print("[INFO] Calculated FK display enabled")

    print("-" * 100)

    try:
        with PiperConnection(can_name=args.can) as conn:
            reader = JointReader(conn.piper)

            # Skip first few readings (may contain default values)
            for _ in range(3):
                reader.read_joints()
                time.sleep(0.05)

            # Start monitoring
            header = "Joint Positions (degrees)"
            if args.pose:
                header += " + SDK Pose (mm)"
            if args.fk:
                header += " + FK (mm)"
            print(header + ":")

            reader.monitor(
                callback=make_callback(reader, args.pose, args.fk),
                rate_hz=args.rate,
                duration_sec=args.duration,
            )

            print("\n" + "-" * 100)
            print("[INFO] Monitoring stopped")
            return 0

    except KeyboardInterrupt:
        print("\n" + "-" * 100)
        print("[INFO] Stopped by user")
        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
