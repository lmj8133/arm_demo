#!/usr/bin/env python3
"""Test drawing figure-8 pattern using IK + joint control.

This script draws a figure-8 (lemniscate) pattern to verify that
all points in the drawing workspace are reachable.

Usage:
    uv run python examples/ex15/test_figure8.py
    uv run python examples/ex15/test_figure8.py --scale 0.03
    uv run python examples/ex15/test_figure8.py --points 100
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
# DRAWING CONFIGURATION
# =============================================================================

# Drawing pose orientation (pen pointing down)
DRAW_ROLL = math.radians(-11.9)
DRAW_PITCH = math.radians(90)
DRAW_YAW = math.radians(-6.4)

# Initial joint angles for IK seed
DRAWING_JOINTS = [0.02070, 2.02836, -0.67533, -0.07916, -1.30194, 0.01630]

# Z heights
DRAW_Z = 0.2667   # Drawing height (pen on paper)
SAFE_Z = 0.30     # Safe height (pen lifted)

# Workspace center
X_CENTER = 0.346
Y_CENTER = 0.014

# Default figure-8 scale
DEFAULT_SCALE = 0.05  # 50mm


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def generate_figure8_points(x_center, y_center, scale, num_points=50):
    """Generate figure-8 (lemniscate) path points.

    Uses parametric equations:
        x(t) = scale * sin(t) + x_center
        y(t) = scale * sin(t) * cos(t) + y_center

    Args:
        x_center, y_center: Center of the figure-8
        scale: Size of the figure-8 (half-width)
        num_points: Number of points to generate

    Returns:
        List of (x, y) tuples
    """
    points = []
    for i in range(num_points + 1):
        t = 2 * math.pi * i / num_points
        x = x_center + scale * math.sin(t)
        y = y_center + scale * math.sin(t) * math.cos(t)
        points.append((x, y))
    return points


def move_with_ik(motion, target_x, target_y, target_z, current_joints, speed=0.2):
    """Move to target position using IK + joint control.

    Args:
        motion: MotionController instance
        target_x, target_y, target_z: Target position in meters
        current_joints: Current joint angles (used as IK initial guess)
        speed: Movement speed factor

    Returns:
        (success, new_joints): success flag and resulting joint angles
    """
    ik_config = IKConfig(
        max_iterations=100,
        damping_factor=0.05,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
    )

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
        return False, current_joints

    motion.move_joint(result.joint_angles, speed_factor=speed)
    return True, result.joint_angles


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test drawing figure-8 pattern")
    parser.add_argument("--can", default="can0", help="CAN interface")
    parser.add_argument("--speed", type=float, default=0.3, help="Movement speed (0.1-1.0)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Figure-8 size in meters")
    parser.add_argument("--points", type=int, default=50, help="Number of path points")
    parser.add_argument("--draw-z", type=float, default=DRAW_Z, help="Drawing Z height")
    parser.add_argument("--safe-z", type=float, default=SAFE_Z, help="Safe Z height")
    args = parser.parse_args()

    print_section("Figure-8 Drawing Test")

    print()
    print("Configuration:")
    print(f"  Center: X={X_CENTER*1000:.0f}mm, Y={Y_CENTER*1000:.0f}mm")
    print(f"  Scale: {args.scale*1000:.0f}mm")
    print(f"  Points: {args.points}")
    print(f"  Draw Z: {args.draw_z*1000:.1f}mm")
    print(f"  Safe Z: {args.safe_z*1000:.1f}mm")
    print(f"  Speed: {args.speed}")

    # Generate figure-8 path
    path_points = generate_figure8_points(X_CENTER, Y_CENTER, args.scale, args.points)

    print()
    print(f"Path points: {len(path_points)}")
    xs = [p[0] for p in path_points]
    ys = [p[1] for p in path_points]
    print(f"  X range: [{min(xs)*1000:.1f}, {max(xs)*1000:.1f}] mm")
    print(f"  Y range: [{min(ys)*1000:.1f}, {max(ys)*1000:.1f}] mm")

    # -------------------------------------------------------------------------
    # Connect and run
    # -------------------------------------------------------------------------
    print_section("Step 1: Connect to Arm")

    with PiperConnection(can_name=args.can) as conn:
        motion = MotionController(conn.piper)
        reader = JointReader(conn.piper)
        print("[OK] Connected")

        # ---------------------------------------------------------------------
        # Step 2: Enable
        # ---------------------------------------------------------------------
        print_section("Step 2: Enable Arm")

        print("[INFO] Enabling arm (without go_home)...")
        conn.enable(go_home=False)
        time.sleep(1)

        joint_state = reader.read_joints()
        current_joints = list(joint_state.positions)
        print(f"[INFO] Current joints: {[f'{math.degrees(j):.1f}' for j in current_joints]}")

        # ---------------------------------------------------------------------
        # Step 3: Move to start point (pen up)
        # ---------------------------------------------------------------------
        print_section("Step 3: Move to Start Point (Pen Up)")

        start_x, start_y = path_points[0]
        print(f"[INFO] Start point: X={start_x*1000:.1f}mm, Y={start_y*1000:.1f}mm, Z={args.safe_z*1000:.1f}mm")

        input("\nPress Enter to move to start point (Ctrl+C to abort)...")

        print("[INFO] Moving to start point (pen up)...")
        success, current_joints = move_with_ik(
            motion, start_x, start_y, args.safe_z,
            DRAWING_JOINTS, speed=args.speed
        )

        if not success:
            print("[ERROR] Failed to move to start point")
            conn.safe_disable(return_home=True)
            return

        time.sleep(1)
        print("[OK] At start point")

        # ---------------------------------------------------------------------
        # Step 4: Pen down
        # ---------------------------------------------------------------------
        print_section("Step 4: Pen Down")

        input("\nPress Enter to lower pen (Ctrl+C to abort)...")

        print("[INFO] Lowering pen...")
        success, current_joints = move_with_ik(
            motion, start_x, start_y, args.draw_z,
            current_joints, speed=args.speed
        )

        if not success:
            print("[ERROR] Failed to lower pen")
            conn.safe_disable(return_home=True)
            return

        time.sleep(0.5)
        print("[OK] Pen down")

        # ---------------------------------------------------------------------
        # Step 5: Draw figure-8
        # ---------------------------------------------------------------------
        print_section("Step 5: Drawing Figure-8")

        input("\nPress Enter to start drawing (Ctrl+C to abort)...")

        success_count = 0
        fail_count = 0
        failed_points = []

        print()
        for i, (x, y) in enumerate(path_points[1:], start=1):
            # Progress indicator
            progress = i / len(path_points) * 100
            print(f"\r[{progress:5.1f}%] Point {i}/{len(path_points)-1}: X={x*1000:.1f}mm, Y={y*1000:.1f}mm", end="", flush=True)

            success, current_joints = move_with_ik(
                motion, x, y, args.draw_z,
                current_joints, speed=args.speed
            )

            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_points.append((i, x, y))
                print(f" [FAIL]")

            # Small delay between points for smooth motion
            time.sleep(0.05)

        print()
        print("[OK] Drawing complete")

        # ---------------------------------------------------------------------
        # Step 6: Pen up
        # ---------------------------------------------------------------------
        print_section("Step 6: Pen Up")

        print("[INFO] Lifting pen...")
        end_x, end_y = path_points[-1]
        success, current_joints = move_with_ik(
            motion, end_x, end_y, args.safe_z,
            current_joints, speed=args.speed
        )

        if not success:
            print("[WARN] Failed to lift pen normally")

        time.sleep(0.5)
        print("[OK] Pen up")

        # ---------------------------------------------------------------------
        # Step 7: Summary
        # ---------------------------------------------------------------------
        print_section("Summary")

        total_points = len(path_points) - 1
        print(f"Total points: {total_points}")
        print(f"Success: {success_count} ({success_count/total_points*100:.1f}%)")
        print(f"Failed: {fail_count} ({fail_count/total_points*100:.1f}%)")

        if failed_points:
            print()
            print("Failed points:")
            for idx, x, y in failed_points:
                print(f"  Point {idx}: X={x*1000:.1f}mm, Y={y*1000:.1f}mm")

        # ---------------------------------------------------------------------
        # Step 8: Safe disable
        # ---------------------------------------------------------------------
        print()
        print("[INFO] Returning to safe home and disabling arm...")
        conn.safe_disable(return_home=True)
        print("[OK] Arm disabled")

    print("[OK] Done!")


if __name__ == "__main__":
    main()
