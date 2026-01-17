#!/usr/bin/env python3
"""Read current joint angles for SAFE_HOME_POSITION.

Manually move the arm to the desired safe home position, then run this script
to read the joint angles.

Usage:
    bash scripts/can_activate.sh can0 1000000
    # Manually move arm to safe home position
    uv run python examples/measure_safe_home.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader


def main():
    print("=" * 60)
    print("Read Current Position for SAFE_HOME_POSITION")
    print("=" * 60)

    with PiperConnection(can_name="can0", auto_slave_mode=True) as conn:
        reader = JointReader(conn.piper)

        # Read current pose
        pose = reader.read_end_pose()
        print(f"\nCurrent Cartesian pose:")
        print(f"  X = {pose.x * 1000:.2f} mm")
        print(f"  Y = {pose.y * 1000:.2f} mm")
        print(f"  Z = {pose.z * 1000:.2f} mm")
        roll_deg = pose.roll * 180 / 3.14159
        pitch_deg = pose.pitch * 180 / 3.14159
        yaw_deg = pose.yaw * 180 / 3.14159
        print(f"  Roll  = {roll_deg:.1f}°")
        print(f"  Pitch = {pitch_deg:.1f}°")
        print(f"  Yaw   = {yaw_deg:.1f}°")

        # Read joint angles
        state = reader.read_joints()
        print(f"\nJoint angles (radians):")
        for i, p in enumerate(state.positions):
            deg = p * 180 / 3.14159
            print(f"  Joint {i+1}: {p:.5f} rad ({deg:.2f}°)")

        # Output ready-to-copy format
        rounded = [round(p, 5) for p in state.positions]
        print("\n" + "=" * 60)
        print("Copy this line to connection.py:")
        print("=" * 60)
        print(f"SAFE_HOME_POSITION = {rounded}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
