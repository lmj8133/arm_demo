#!/usr/bin/env python3
"""Test drawing workspace using IK + joint control (Method 2).

This script ensures the drawing pose (pen pointing down) is maintained
by calculating joint angles with IK, then moving with joint control.

## Logic:
1. Connect to arm, enable (without go_home)
2. Move to drawing start pose using joint control
3. For each target point (X, Y, Z):
   a. Calculate joint angles using IK (fixed orientation)
   b. If IK converged → move with joint control
   c. If IK failed → skip the point
4. Safely disable arm

Usage:
    uv run python examples/ex15/test_drawing_ik.py
    uv run python examples/ex15/test_drawing_ik.py --speed 0.2
"""

import argparse
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from piper_demo import PiperConnection, MotionController, JointReader
from piper_demo.inverse_kinematics import inverse_kinematics, IKConfig
from piper_demo.kinematics import forward_kinematics


# =============================================================================
# DRAWING POSE CONFIGURATION
# =============================================================================

# Drawing pose joint angles (radians)
# J1=1.19°, J2=116.22°, J3=-38.69°, J4=-4.54°, J5=-74.60°, J6=0.93°
DRAWING_JOINTS = [0.02070, 2.02836, -0.67533, -0.07916, -1.30194, 0.01630]

# Drawing pose end-effector orientation (radians)
# RX=-11.9°, RY=87.9° (pen pointing down), RZ=-6.4°
DRAW_ROLL = math.radians(-11.9)
DRAW_PITCH = math.radians(90)
DRAW_YAW = math.radians(-6.4)

# Optimal Z height (from find_optimal_z.py)
OPTIMAL_Z = 0.2667  # 266.7mm

# Reachable workspace at optimal Z
WORKSPACE_X_MIN = 0.196   # 196mm
WORKSPACE_X_MAX = 0.496   # 496mm
WORKSPACE_Y_MIN = -0.136  # -136mm
WORKSPACE_Y_MAX = 0.164   # 164mm


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_pose_from_joints(joints: list, label: str = "FK"):
    """Calculate and print end-effector pose from joint angles."""
    fk = forward_kinematics(joints)
    print(f"[{label}] Position: X={fk.x*1000:.1f}, Y={fk.y*1000:.1f}, Z={fk.z*1000:.1f} mm")
    print(f"[{label}] Orientation: RX={math.degrees(fk.roll):.1f}°, RY={math.degrees(fk.pitch):.1f}°, RZ={math.degrees(fk.yaw):.1f}°")
    return fk


def move_with_ik(
    motion: MotionController,
    target_x: float,
    target_y: float,
    target_z: float,
    current_joints: list,
    speed: float = 0.2,
):
    """
    Move to target position using IK + joint control.

    Args:
        motion: MotionController instance
        target_x, target_y, target_z: Target position in meters
        current_joints: Current joint angles (used as IK initial guess)
        speed: Movement speed factor

    Returns:
        (success, new_joints): success flag and resulting joint angles
    """
    # IK configuration
    ik_config = IKConfig(
        max_iterations=100,
        damping_factor=0.05,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
    )

    # Calculate joint angles using IK
    result = inverse_kinematics(
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        target_roll=DRAW_ROLL,
        target_pitch=DRAW_PITCH,
        target_yaw=DRAW_YAW,
        initial_guess=current_joints,
        config=ik_config,
    )

    if not result.converged:
        print(f"    [IK FAIL] Position error: {result.position_error*1000:.2f}mm")
        return False, current_joints

    print(f"    [IK OK] Iterations: {result.iterations}, Error: {result.position_error*1000:.3f}mm")

    # Move using joint control
    motion.move_joint(result.joint_angles, speed_factor=speed)

    # Wait for motion to complete
    time.sleep(2.0)

    return True, result.joint_angles


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test drawing workspace with IK")
    parser.add_argument("--can", default="can0", help="CAN interface")
    parser.add_argument("--speed", type=float, default=0.2, help="Movement speed (0.1-1.0)")
    parser.add_argument("--z", type=float, default=OPTIMAL_Z, help="Z height in meters")
    args = parser.parse_args()

    print_section("Drawing Workspace Test (IK + Joint Control)")

    print()
    print("Configuration:")
    print(f"  Z height: {args.z * 1000:.1f} mm")
    print(f"  X range: {WORKSPACE_X_MIN * 1000:.0f} ~ {WORKSPACE_X_MAX * 1000:.0f} mm")
    print(f"  Y range: {WORKSPACE_Y_MIN * 1000:.0f} ~ {WORKSPACE_Y_MAX * 1000:.0f} mm")
    print(f"  Speed: {args.speed}")
    print()
    print("Drawing orientation (fixed):")
    print(f"  RX={math.degrees(DRAW_ROLL):.1f}°, RY={math.degrees(DRAW_PITCH):.1f}°, RZ={math.degrees(DRAW_YAW):.1f}°")

    # Calculate workspace center
    x_center = (WORKSPACE_X_MIN + WORKSPACE_X_MAX) / 2
    y_center = (WORKSPACE_Y_MIN + WORKSPACE_Y_MAX) / 2

    # Define test points (use 80% of range for safety margin)
    margin = 0.8
    x_min_safe = WORKSPACE_X_MIN + (x_center - WORKSPACE_X_MIN) * (1 - margin)
    x_max_safe = WORKSPACE_X_MAX - (WORKSPACE_X_MAX - x_center) * (1 - margin)
    y_min_safe = WORKSPACE_Y_MIN + (y_center - WORKSPACE_Y_MIN) * (1 - margin)
    y_max_safe = WORKSPACE_Y_MAX - (WORKSPACE_Y_MAX - y_center) * (1 - margin)

    test_points = [
        ("Center", x_center, y_center),
        ("Front", x_max_safe, y_center),
        ("Back", x_min_safe, y_center),
        ("Left", x_center, y_max_safe),
        ("Right", x_center, y_min_safe),
        ("Front-Left", x_max_safe, y_max_safe),
        ("Front-Right", x_max_safe, y_min_safe),
        ("Back-Left", x_min_safe, y_max_safe),
        ("Back-Right", x_min_safe, y_min_safe),
        ("Center", x_center, y_center),
    ]

    print()
    print(f"Test points ({len(test_points)}):")
    for name, x, y in test_points:
        print(f"  {name}: X={x*1000:.0f}mm, Y={y*1000:.0f}mm, Z={args.z*1000:.0f}mm")

    # -------------------------------------------------------------------------
    # Step 1: Connect (using context manager for safe exit)
    # -------------------------------------------------------------------------
    print_section("Step 1: Connect to Arm")

    with PiperConnection(can_name=args.can) as conn:
        motion = MotionController(conn.piper)
        reader = JointReader(conn.piper)
        print("[OK] Connected")

        # ---------------------------------------------------------------------
        # Step 2: Enable (no home)
        # ---------------------------------------------------------------------
        print_section("Step 2: Enable Arm")

        print("[INFO] Enabling arm (without go_home)...")
        conn.enable(go_home=False)
        time.sleep(1)

        # Read current position
        joint_state = reader.read_joints()
        joints = joint_state.positions
        print(f"[INFO] Current joints: {[f'{math.degrees(j):.1f}°' for j in joints]}")
        print_pose_from_joints(joints, "Current")

        # ---------------------------------------------------------------------
        # Step 3: Move to workspace center
        # ---------------------------------------------------------------------
        print_section("Step 3: Move to Workspace Center")

        print(f"[INFO] Target: Center of workspace")
        print(f"       X={x_center*1000:.0f}mm, Y={y_center*1000:.0f}mm, Z={args.z*1000:.0f}mm")

        input("\nPress Enter to move to workspace center (Ctrl+C to abort)...")

        print("[INFO] Moving with IK + joint control...")
        success, current_joints = move_with_ik(
            motion, x_center, y_center, args.z,
            DRAWING_JOINTS, speed=args.speed
        )

        if not success:
            print("[ERROR] Failed to move to center, aborting")
            conn.safe_disable(return_home=True)
            return

        print_pose_from_joints(current_joints, "Actual")

        # ---------------------------------------------------------------------
        # Step 4: Test XY movement with IK
        # ---------------------------------------------------------------------
        print_section("Step 4: Test XY Movement")

        input("\nPress Enter to start XY movement test...")

        success_count = 0
        fail_count = 0

        for i, (name, x, y) in enumerate(test_points):
            print()
            print(f"[{i+1}/{len(test_points)}] Moving to {name}: X={x*1000:.0f}mm, Y={y*1000:.0f}mm, Z={args.z*1000:.0f}mm")

            success, current_joints = move_with_ik(
                motion, x, y, args.z,
                current_joints, speed=args.speed
            )

            if success:
                # Verify with FK
                fk = print_pose_from_joints(current_joints, "Actual")
                pos_error = math.sqrt((fk.x - x)**2 + (fk.y - y)**2 + (fk.z - args.z)**2)
                print(f"    Position error: {pos_error*1000:.2f}mm")
                success_count += 1
            else:
                fail_count += 1

            time.sleep(0.5)

        # ---------------------------------------------------------------------
        # Step 5: Summary
        # ---------------------------------------------------------------------
        print_section("Summary")

        print(f"Success: {success_count}/{len(test_points)}")
        print(f"Failed: {fail_count}/{len(test_points)}")

        # Safely disable arm before exiting context
        print()
        print("[INFO] Returning to safe home and disabling arm...")
        conn.safe_disable(return_home=True)
        print("[OK] Arm disabled")

    print("[OK] Done!")


if __name__ == "__main__":
    main()
