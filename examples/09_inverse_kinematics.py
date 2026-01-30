#!/usr/bin/env python3
"""Example 09: Inverse kinematics test.

Compute joint angles from target end-effector pose using Damped Least Squares (DLS).
Verify solution by computing forward kinematics and comparing.

Usage:
    # Test with default target pose (no hardware needed)
    uv run python examples/09_inverse_kinematics.py

    # Test with specific target pose (x, y, z in mm; roll, pitch, yaw in degrees)
    uv run python examples/09_inverse_kinematics.py --target 300 0 200 0 90 0

    # Use current arm pose as target (requires CAN connection)
    uv run python examples/09_inverse_kinematics.py --from-arm

    # Use custom initial guess (degrees)
    uv run python examples/09_inverse_kinematics.py --initial 0 30 -30 0 0 0
"""

import sys
import os
import argparse
import math

# Add src to path for package import (piper_demo) and direct module import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "piper_demo"))

from kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics, IKConfig, verify_ik_solution
from jacobian import compute_jacobian, is_near_singularity, jacobian_determinant
from utils import deg_to_rad, rad_to_deg


def print_pose(
    label: str, x: float, y: float, z: float, roll: float, pitch: float, yaw: float
):
    """Print pose in a formatted way."""
    print(f"\n{label}:")
    print(
        f"  Position:    X={x * 1000:8.2f} mm, Y={y * 1000:8.2f} mm, Z={z * 1000:8.2f} mm"
    )
    print(
        f"  Orientation: R={rad_to_deg(roll):8.2f}°, P={rad_to_deg(pitch):8.2f}°, Y={rad_to_deg(yaw):8.2f}°"
    )


def print_joints(label: str, angles_rad: list):
    """Print joint angles."""
    print(f"\n{label}:")
    for i, rad in enumerate(angles_rad):
        print(f"  J{i + 1}: {rad_to_deg(rad):8.2f}° ({rad:8.4f} rad)")


def main():
    parser = argparse.ArgumentParser(
        description="Inverse kinematics test for Piper arm"
    )
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--target",
        nargs=6,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Target pose: x,y,z in mm; roll,pitch,yaw in degrees",
    )
    parser.add_argument(
        "--from-arm", action="store_true", help="Read current arm pose as target"
    )
    parser.add_argument(
        "--initial",
        nargs=6,
        type=float,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Initial guess in degrees (default: [0, 30, -30, 0, 0, 0])",
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Maximum iterations (default: 100)"
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.05,
        help="Damping factor for DLS (default: 0.05)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Piper Arm Inverse Kinematics Test")
    print("=" * 60)

    # Determine target pose
    current_joints = None  # Will be set if reading from arm
    if args.from_arm:
        print(f"\nReading current pose from arm on {args.can}...")
        try:
            from piper_demo import PiperConnection, JointReader
            import time

            with PiperConnection(can_name=args.can) as conn:
                reader = JointReader(conn.piper)

                # Skip initial readings
                for _ in range(5):
                    reader.read_joints()
                    time.sleep(0.05)

                # Read current joint angles (for initial guess)
                joint_state = reader.read_joints()
                current_joints = joint_state.positions

                sdk_pose = reader.read_end_pose()
                target_x = sdk_pose.x
                target_y = sdk_pose.y
                target_z = sdk_pose.z
                target_roll = sdk_pose.roll
                target_pitch = sdk_pose.pitch
                target_yaw = sdk_pose.yaw
                print("Read from arm successfully.")

        except Exception as e:
            print(f"[ERROR] Failed to connect to arm: {e}")
            return 1
    elif args.target:
        # Convert from mm and degrees to meters and radians
        target_x = args.target[0] / 1000.0
        target_y = args.target[1] / 1000.0
        target_z = args.target[2] / 1000.0
        target_roll = deg_to_rad(args.target[3])
        target_pitch = deg_to_rad(args.target[4])
        target_yaw = deg_to_rad(args.target[5])
    else:
        # Default reachable target pose
        target_x = 0.35
        target_y = 0.0
        target_z = 0.25
        target_roll = 0.0
        target_pitch = math.pi / 2  # 90 degrees
        target_yaw = 0.0
        print("\nUsing default target pose")

    print_pose(
        "Target Pose",
        target_x,
        target_y,
        target_z,
        target_roll,
        target_pitch,
        target_yaw,
    )

    # Initial guess
    if args.initial:
        initial_guess = [deg_to_rad(j) for j in args.initial]
    elif current_joints is not None:
        # Use current joint angles when reading from arm
        initial_guess = current_joints
        print("\n[INFO] Using current joint angles as initial guess")
    else:
        initial_guess = [0.0, deg_to_rad(30), deg_to_rad(-30), 0.0, 0.0, 0.0]

    print_joints("Initial Guess", initial_guess)

    # Configure IK solver
    config = IKConfig(
        max_iterations=args.max_iter,
        damping_factor=args.damping,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
    )

    print("-" * 60)
    print("Running Inverse Kinematics (Damped Least Squares)...")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Damping factor: {config.damping_factor}")
    print(f"  Position tolerance: {config.position_tolerance * 1000:.3f} mm")
    print(f"  Orientation tolerance: {rad_to_deg(config.orientation_tolerance):.3f}°")

    # Compute IK
    result = inverse_kinematics(
        target_x,
        target_y,
        target_z,
        target_roll,
        target_pitch,
        target_yaw,
        initial_guess=initial_guess,
        config=config,
    )

    print("-" * 60)
    if result.converged:
        print(f"[SUCCESS] Converged in {result.iterations} iterations")
    else:
        print(f"[WARNING] Did not converge after {result.iterations} iterations")

    print(f"  Position error:    {result.position_error * 1000:.4f} mm")
    print(f"  Orientation error: {rad_to_deg(result.orientation_error):.4f}°")

    print_joints("IK Solution", result.joint_angles)

    # Verify by computing FK
    print("-" * 60)
    print("Verification (Forward Kinematics):")
    fk = forward_kinematics(result.joint_angles)
    print_pose("Computed Pose (FK)", fk.x, fk.y, fk.z, fk.roll, fk.pitch, fk.yaw)

    # Verify errors
    pos_err, ori_err = verify_ik_solution(
        target_x,
        target_y,
        target_z,
        target_roll,
        target_pitch,
        target_yaw,
        result.joint_angles,
    )
    print(f"\n  Verification position error:    {pos_err * 1000:.4f} mm")
    print(f"  Verification orientation error: {rad_to_deg(ori_err):.4f}°")

    # Check singularity at solution
    print("-" * 60)
    J = compute_jacobian(result.joint_angles)
    det = jacobian_determinant(J)
    singular = is_near_singularity(result.joint_angles)
    print("Jacobian Analysis:")
    print(f"  Determinant: {det:.6f}")
    print(f"  Near singularity: {'Yes' if singular else 'No'}")

    print("\n" + "=" * 60)
    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(main())
