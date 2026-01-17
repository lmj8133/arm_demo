#!/usr/bin/env python3
"""Example 07: Forward kinematics test.

Compute end-effector pose from joint angles using Modified DH parameters.
Compare calculated result with SDK feedback to verify kinematics.

Usage:
    # Test with real arm (requires CAN connection)
    uv run python examples/07_forward_kinematics.py

    # Test with simulated joint angles (no hardware needed)
    uv run python examples/07_forward_kinematics.py --simulate

    # Test specific joint angles (degrees)
    uv run python examples/07_forward_kinematics.py --simulate --joints 0 45 -30 0 0 0
"""

import sys
import os
import argparse
import math

# Add src/piper_demo to path for direct module import (avoid __init__.py which requires piper_sdk)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "piper_demo"))

from kinematics import forward_kinematics
from utils import deg_to_rad, rad_to_deg


def print_pose(label: str, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
    """Print pose in a formatted way."""
    print(f"\n{label}:")
    print(f"  Position:    X={x*1000:8.2f} mm, Y={y*1000:8.2f} mm, Z={z*1000:8.2f} mm")
    print(f"  Orientation: R={rad_to_deg(roll):8.2f}°, P={rad_to_deg(pitch):8.2f}°, Y={rad_to_deg(yaw):8.2f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Forward kinematics test for Piper arm"
    )
    parser.add_argument(
        "--can", default="can0",
        help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run without hardware (use default or specified joint angles)"
    )
    parser.add_argument(
        "--joints", nargs=6, type=float, default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Joint angles in degrees (default: all zeros)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Piper Arm Forward Kinematics Test")
    print("=" * 60)

    # Determine joint angles to use
    if args.joints:
        joint_angles_deg = args.joints
        joint_angles_rad = [deg_to_rad(j) for j in joint_angles_deg]
        print(f"\nUsing specified joint angles (degrees):")
    elif args.simulate:
        joint_angles_deg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_angles_rad = [deg_to_rad(j) for j in joint_angles_deg]
        print(f"\nUsing simulated joint angles (degrees):")
    else:
        joint_angles_rad = None
        joint_angles_deg = None
        print(f"\nReading joint angles from arm on {args.can}...")

    # If using real arm, read joint angles and SDK pose
    sdk_pose = None
    if not args.simulate:
        try:
            from piper_demo import PiperConnection, JointReader
            import time

            with PiperConnection(can_name=args.can) as conn:
                reader = JointReader(conn.piper)

                # Skip initial readings
                for _ in range(5):
                    reader.read_joints()
                    time.sleep(0.05)

                # Read current state
                joint_state = reader.read_joints()
                joint_angles_rad = joint_state.positions
                joint_angles_deg = [rad_to_deg(j) for j in joint_angles_rad]

                # Read SDK end pose for comparison
                sdk_pose = reader.read_end_pose()

                print("Read from arm successfully.")

        except Exception as e:
            print(f"[ERROR] Failed to connect to arm: {e}")
            print("[INFO] Use --simulate to run without hardware")
            return 1

    # Display joint angles
    print("-" * 60)
    print("Input Joint Angles:")
    for i, (deg, rad) in enumerate(zip(joint_angles_deg, joint_angles_rad)):
        print(f"  J{i+1}: {deg:8.2f}° ({rad:8.4f} rad)")

    # Compute forward kinematics
    print("-" * 60)
    print("Computing forward kinematics...")

    fk = forward_kinematics(joint_angles_rad)
    print_pose("Calculated (FK)", fk.x, fk.y, fk.z, fk.roll, fk.pitch, fk.yaw)

    # Compare with SDK if available
    if sdk_pose:
        print_pose(
            "SDK Feedback",
            sdk_pose.x, sdk_pose.y, sdk_pose.z,
            sdk_pose.roll, sdk_pose.pitch, sdk_pose.yaw
        )

        # Compute errors
        pos_error = math.sqrt(
            (fk.x - sdk_pose.x)**2 +
            (fk.y - sdk_pose.y)**2 +
            (fk.z - sdk_pose.z)**2
        ) * 1000  # mm

        print("-" * 60)
        print("Comparison:")
        print(f"  Position error: {pos_error:.2f} mm")
        print(f"  Roll error:     {rad_to_deg(abs(fk.roll - sdk_pose.roll)):.2f}°")
        print(f"  Pitch error:    {rad_to_deg(abs(fk.pitch - sdk_pose.pitch)):.2f}°")
        print(f"  Yaw error:      {rad_to_deg(abs(fk.yaw - sdk_pose.yaw)):.2f}°")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
