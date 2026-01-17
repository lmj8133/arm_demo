#!/usr/bin/env python3
"""Example 10: Move arm to target pose using inverse kinematics.

Compute IK for target pose and move the arm to that configuration.

Usage:
    # Move to specific pose (x, y, z in mm; roll, pitch, yaw in degrees)
    uv run python examples/10_move_to_pose.py --target 300 0 200 0 90 0

    # Move with slower speed
    uv run python examples/10_move_to_pose.py --target 300 0 200 0 90 0 --speed 0.1

    # Dry run (compute IK only, don't move)
    uv run python examples/10_move_to_pose.py --target 300 0 200 0 90 0 --dry-run

    # Use custom IK parameters
    uv run python examples/10_move_to_pose.py --target 300 0 200 0 90 0 --damping 0.01 --max-iter 200

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
"""

import sys
import os
import argparse
import math
import time

# Add src to path for package import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "piper_demo"))

from kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics, IKConfig
from utils import deg_to_rad, rad_to_deg


def print_pose(label: str, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
    """Print pose in a formatted way."""
    print(f"{label}:")
    print(f"  Position:    X={x*1000:8.2f} mm, Y={y*1000:8.2f} mm, Z={z*1000:8.2f} mm")
    print(f"  Orientation: R={rad_to_deg(roll):8.2f}°, P={rad_to_deg(pitch):8.2f}°, Y={rad_to_deg(yaw):8.2f}°")


def print_joints(label: str, angles_rad: list):
    """Print joint angles."""
    print(f"{label}:")
    angles_deg = [rad_to_deg(a) for a in angles_rad]
    print(f"  [{angles_deg[0]:7.2f}°, {angles_deg[1]:7.2f}°, {angles_deg[2]:7.2f}°, "
          f"{angles_deg[3]:7.2f}°, {angles_deg[4]:7.2f}°, {angles_deg[5]:7.2f}°]")


def main():
    parser = argparse.ArgumentParser(
        description="Move Piper arm to target pose using inverse kinematics"
    )
    parser.add_argument(
        "--can", default="can0",
        help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--target", nargs=6, type=float, required=True,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Target pose: x,y,z in mm; roll,pitch,yaw in degrees"
    )
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Speed factor 0.0-1.0 (default: 0.3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute IK only, don't move the arm"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100,
        help="Maximum IK iterations (default: 100)"
    )
    parser.add_argument(
        "--damping", type=float, default=0.05,
        help="IK damping factor (default: 0.05)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Move to Pose (IK + Motion Control)")
    print("=" * 60)

    # Parse target pose
    target_x = args.target[0] / 1000.0  # mm to m
    target_y = args.target[1] / 1000.0
    target_z = args.target[2] / 1000.0
    target_roll = deg_to_rad(args.target[3])
    target_pitch = deg_to_rad(args.target[4])
    target_yaw = deg_to_rad(args.target[5])

    print()
    print_pose("Target Pose", target_x, target_y, target_z, target_roll, target_pitch, target_yaw)

    # Connect to arm and read current state
    print()
    print(f"Connecting to arm on {args.can}...")

    try:
        from piper_demo import PiperConnection, JointReader, MotionController

        with PiperConnection(can_name=args.can) as conn:
            reader = JointReader(conn.piper)

            # Skip initial readings
            for _ in range(5):
                reader.read_joints()
                time.sleep(0.05)

            # Read current joint angles for initial guess
            joint_state = reader.read_joints()
            current_joints = joint_state.positions

            # Read current pose for display
            current_pose = reader.read_end_pose()

            print()
            print_pose("Current Pose", current_pose.x, current_pose.y, current_pose.z,
                      current_pose.roll, current_pose.pitch, current_pose.yaw)
            print()
            print_joints("Current Joints", current_joints)

            # Compute IK
            print()
            print("-" * 60)
            print("Computing Inverse Kinematics...")

            config = IKConfig(
                max_iterations=args.max_iter,
                damping_factor=args.damping,
                position_tolerance=1e-4,
                orientation_tolerance=1e-3,
            )

            result = inverse_kinematics(
                target_x, target_y, target_z,
                target_roll, target_pitch, target_yaw,
                initial_guess=current_joints,
                config=config
            )

            if result.converged:
                print(f"[OK] IK converged in {result.iterations} iterations")
            else:
                print(f"[WARNING] IK did not converge after {result.iterations} iterations")

            print(f"  Position error:    {result.position_error*1000:.4f} mm")
            print(f"  Orientation error: {rad_to_deg(result.orientation_error):.4f}°")
            print()
            print_joints("IK Solution", result.joint_angles)

            # Verify with FK
            fk = forward_kinematics(result.joint_angles)
            print()
            print_pose("Expected Pose (FK)", fk.x, fk.y, fk.z, fk.roll, fk.pitch, fk.yaw)

            # Check if IK solution is acceptable
            if result.position_error > 0.001:  # > 1mm
                print()
                print(f"[WARNING] Position error {result.position_error*1000:.2f}mm exceeds 1mm threshold")
                if not args.dry_run:
                    response = input("Continue anyway? [y/N]: ")
                    if response.lower() != 'y':
                        print("Aborted.")
                        return 1

            # Move arm
            print()
            print("-" * 60)

            if args.dry_run:
                print("[DRY RUN] Skipping arm movement")
                print()
                print("To execute this motion, run without --dry-run flag")
            else:
                # Enable arm first (required for motion)
                print("[INFO] Enabling arm...")
                conn.enable()
                print("[OK] Arm enabled")
                print()

                print(f"[INFO] Moving arm to target pose (speed={args.speed})...")
                motion = MotionController(conn.piper)
                motion.move_joint(result.joint_angles, speed_factor=args.speed)

                # Wait for motion to complete
                print("[INFO] Waiting for motion to complete...")
                reached = reader.wait_for_position(
                    result.joint_angles,
                    tolerance_rad=deg_to_rad(2.0),  # 2 degree tolerance
                    timeout_sec=15.0,
                )

                if reached:
                    print("[OK] Motion complete")
                else:
                    print("[WARNING] Motion timeout - may not have reached target")

                # Read final pose
                time.sleep(0.3)
                for _ in range(5):
                    reader.read_joints()
                    time.sleep(0.05)

                final_pose = reader.read_end_pose()
                final_joints = reader.read_joints()

                print()
                print_pose("Final Pose", final_pose.x, final_pose.y, final_pose.z,
                          final_pose.roll, final_pose.pitch, final_pose.yaw)
                print()
                print_joints("Final Joints", final_joints.positions)

                # Calculate final error
                final_pos_err = math.sqrt(
                    (final_pose.x - target_x)**2 +
                    (final_pose.y - target_y)**2 +
                    (final_pose.z - target_z)**2
                )
                print()
                print(f"Final position error: {final_pos_err*1000:.2f} mm")

                # Hold position briefly then safe disable
                print()
                print("[INFO] Holding position for 2 seconds...")
                time.sleep(2.0)

                print("[INFO] Safely disabling arm...")
                conn.safe_disable()
                print("[OK] Arm disabled")

            print()
            print("=" * 60)
            return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
