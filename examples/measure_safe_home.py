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

from piper_demo import PiperConnection, JointReader, forward_kinematics


def main():
    print("=" * 60)
    print("Read Current Position for SAFE_HOME_POSITION")
    print("=" * 60)

    with PiperConnection(can_name="can0", auto_slave_mode=True) as conn:
        reader = JointReader(conn.piper)

        # Read current pose
        pose = reader.read_end_pose()
        print("\nCurrent Cartesian pose:")
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
        print("\nJoint angles (radians):")
        for i, p in enumerate(state.positions):
            deg = p * 180 / 3.14159
            print(f"  Joint {i + 1}: {p:.5f} rad ({deg:.2f}°)")

        # Output ready-to-copy format
        rounded = [round(p, 5) for p in state.positions]
        print("\n" + "=" * 60)
        print("Copy this line to connection.py:")
        print("=" * 60)
        print(f"SAFE_HOME_POSITION = {rounded}")
        print("=" * 60)

        # Calculate XYZ offset from HOME_POSITION using forward kinematics
        home_fk = forward_kinematics(PiperConnection.HOME_POSITION)
        home_xyz = home_fk.position_mm()  # (x, y, z) in mm
        current_xyz = (pose.x * 1000, pose.y * 1000, pose.z * 1000)

        offset_x = current_xyz[0] - home_xyz[0]
        offset_y = current_xyz[1] - home_xyz[1]
        offset_z = current_xyz[2] - home_xyz[2]

        print("\n" + "=" * 60)
        print("XYZ Offset from HOME_POSITION (current - home):")
        print("=" * 60)
        sign_x = "+" if offset_x >= 0 else ""
        sign_y = "+" if offset_y >= 0 else ""
        sign_z = "+" if offset_z >= 0 else ""
        print(f"  dX = {sign_x}{offset_x:.2f} mm")
        print(f"  dY = {sign_y}{offset_y:.2f} mm")
        print(f"  dZ = {sign_z}{offset_z:.2f} mm")

        print("\nCopy-paste ready (mm):")
        print(f"XYZ_OFFSET = ({offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f})")

        print(
            f"\nReference HOME XYZ (from FK): ({home_xyz[0]:.2f}, {home_xyz[1]:.2f}, {home_xyz[2]:.2f}) mm"
        )
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
