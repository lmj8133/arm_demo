#!/usr/bin/env python3
"""Draw workspace boundary rectangle using inline IK + continuous motion.

Uses figure8-style fire-and-forget movement: IK solve inline with chained
initial guesses, move_joint without waiting for arrival, fixed sleep interval.

Flow:
1. Connect and enable arm
2. Move to start corner at pen-up height (wait for position)
3. Pen down: lower to drawing Z (wait for position)
4. Trace rectangle boundary (inline IK, no wait)
5. Pen up: raise back to safe Z (wait for position)
6. Disable arm

Usage:
    python3 test/test_draw_boundary.py --can can1 --speed 0.3
    python3 test/test_draw_boundary.py --can can1 --steps 60 --interval 0.05
"""

import argparse
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, MotionController, JointReader
from piper_demo.inverse_kinematics import inverse_kinematics, IKConfig
from piper_demo.kinematics import forward_kinematics


# =============================================================================
# CONFIGURATION
# =============================================================================

# Drawing pose joint angles (IK initial guess)
DRAWING_JOINTS = [0, 2.02786, -0.57318, 0.13404, -1.19086, 0.01389]

# Derive fixed orientation from drawing pose
_home_fk = forward_kinematics(DRAWING_JOINTS)
DRAW_ROLL = _home_fk.roll
DRAW_PITCH = _home_fk.pitch
DRAW_YAW = _home_fk.yaw

# Heights
OPTIMAL_Z = 0.18       # pen down (drawing plane)
PEN_UP_OFFSET = 0.03   # pen up = OPTIMAL_Z + offset

# Workspace boundaries
WORKSPACE_X_MIN = 0.170
WORKSPACE_X_MAX = 0.470
WORKSPACE_Y_MIN = -0.161
WORKSPACE_Y_MAX = 0.139

# IK solver config
IK_CFG = IKConfig(
    max_iterations=100,
    damping_factor=0.05,
    position_tolerance=1e-4,
    orientation_tolerance=1e-3,
)


# =============================================================================
# HELPERS
# =============================================================================

def move_ik(motion, reader, x, y, z, speed, timeout_sec=10.0):
    """Move to (x, y, z) via IK + joint control, wait until arrived.

    Used for pen-up/pen-down vertical moves where precision matters.

    Returns:
        success: True if IK converged and arm reached target
    """
    current_joints = list(reader.read_joints().positions)

    result = inverse_kinematics(
        target_x=x, target_y=y, target_z=z,
        target_roll=DRAW_ROLL,
        target_pitch=DRAW_PITCH,
        target_yaw=DRAW_YAW,
        initial_guess=current_joints,
        config=IK_CFG,
    )
    if not result.converged:
        print(f"  [IK FAIL] ({x*1000:.0f}, {y*1000:.0f}, {z*1000:.0f}) "
              f"err={result.position_error*1000:.2f}mm")
        return False

    motion.move_joint(result.joint_angles, speed_factor=speed)
    reader.wait_for_position(
        result.joint_angles,
        tolerance_rad=0.035,
        timeout_sec=timeout_sec,
    )
    return True


def move_with_ik(motion, target_x, target_y, target_z, current_joints, speed=0.2):
    """Move to target position using IK, fire-and-forget (no wait).

    Args:
        motion: MotionController instance
        target_x, target_y, target_z: Target position in meters
        current_joints: Current joint angles (used as IK initial guess)
        speed: Movement speed factor

    Returns:
        (success, new_joints): success flag and resulting joint angles
    """
    result = inverse_kinematics(
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        target_roll=DRAW_ROLL,
        target_pitch=DRAW_PITCH,
        target_yaw=DRAW_YAW,
        initial_guess=current_joints,
        config=IK_CFG,
    )

    if not result.converged:
        return False, current_joints

    motion.move_joint(result.joint_angles, speed_factor=speed)
    return True, result.joint_angles


def generate_rectangle_path(x_min, x_max, y_min, y_max, steps):
    """Generate rectangle boundary points (CCW from bottom-left).

    Returns list of (x, y) tuples.
    """
    path = []
    # Bottom edge: (x_min,y_min) -> (x_max,y_min)
    for i in range(steps):
        t = i / steps
        path.append((x_min + (x_max - x_min) * t, y_min))
    # Right edge: (x_max,y_min) -> (x_max,y_max)
    for i in range(steps):
        t = i / steps
        path.append((x_max, y_min + (y_max - y_min) * t))
    # Top edge: (x_max,y_max) -> (x_min,y_max)
    for i in range(steps):
        t = i / steps
        path.append((x_max - (x_max - x_min) * t, y_max))
    # Left edge: (x_min,y_max) -> (x_min,y_min)
    for i in range(steps):
        t = i / steps
        path.append((x_min, y_max - (y_max - y_min) * t))
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Draw workspace boundary rectangle")
    parser.add_argument("--can", default="can0", help="CAN interface")
    parser.add_argument("--speed", type=float, default=0.3, help="Speed factor (0.1-1.0)")
    parser.add_argument("--z", type=float, default=OPTIMAL_Z, help="Drawing Z height (m)")
    parser.add_argument("--pen-up-offset", type=float, default=PEN_UP_OFFSET,
                        help="Pen-up height above drawing Z (m)")
    parser.add_argument("--steps", type=int, default=60, help="Points per edge")
    parser.add_argument("--interval", type=float, default=0.05,
                        help="Sleep between move commands (seconds)")
    args = parser.parse_args()

    z_draw = args.z
    z_up = args.z + args.pen_up_offset

    # Generate path (close the rectangle by appending start point)
    path = generate_rectangle_path(
        WORKSPACE_X_MIN, WORKSPACE_X_MAX,
        WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
        args.steps,
    )
    path.append(path[0])
    total = len(path)

    # Step distance
    edge_len_x = (WORKSPACE_X_MAX - WORKSPACE_X_MIN) * 1000
    edge_len_y = (WORKSPACE_Y_MAX - WORKSPACE_Y_MIN) * 1000
    step_dist_x = edge_len_x / args.steps
    step_dist_y = edge_len_y / args.steps

    print("=" * 60)
    print("Draw Workspace Boundary")
    print("=" * 60)
    print(f"  Drawing Z : {z_draw*1000:.0f} mm")
    print(f"  Pen-up Z  : {z_up*1000:.0f} mm")
    print(f"  X range   : {WORKSPACE_X_MIN*1000:.0f} ~ {WORKSPACE_X_MAX*1000:.0f} mm")
    print(f"  Y range   : {WORKSPACE_Y_MIN*1000:.0f} ~ {WORKSPACE_Y_MAX*1000:.0f} mm")
    print(f"  Steps/edge: {args.steps}  (total {total} points)")
    print(f"  Step dist : ~{step_dist_x:.1f}mm (X), ~{step_dist_y:.1f}mm (Y)")
    print(f"  Speed     : {args.speed}")
    print(f"  Interval  : {args.interval}s  (est. {total * args.interval:.1f}s to draw)")
    print(f"  Orientation: roll={math.degrees(DRAW_ROLL):.1f}, "
          f"pitch={math.degrees(DRAW_PITCH):.1f}, yaw={math.degrees(DRAW_YAW):.1f}")
    print()

    start_x, start_y = path[0]

    with PiperConnection(can_name=args.can) as conn:
        motion = MotionController(conn.piper)
        reader = JointReader(conn.piper)

        # --- Enable ---
        print("[1] Enabling arm...")
        conn.enable(go_home=False)
        time.sleep(1)

        joints = reader.read_joints().positions
        print(f"    Current joints: {[f'{math.degrees(j):.1f}' for j in joints]}")

        # --- Move to start at pen-up height ---
        print(f"\n[2] Moving to start ({start_x*1000:.0f}, {start_y*1000:.0f}) "
              f"at pen-up Z={z_up*1000:.0f}mm...")
        ok = move_ik(motion, reader, start_x, start_y, z_up, args.speed)
        if not ok:
            print("[ERROR] Cannot reach start point, aborting")
            conn.safe_disable(return_home=True)
            return

        pose = reader.read_end_pose()
        print(f"    Reached: X={pose.x*1000:.1f}, Y={pose.y*1000:.1f}, Z={pose.z*1000:.1f} mm")

        input("\nPress Enter to pen-down and start drawing (Ctrl+C to abort)...")

        # --- Pen down ---
        print(f"\n[3] Pen down -> Z={z_draw*1000:.0f}mm")
        ok = move_ik(motion, reader, start_x, start_y, z_draw, args.speed)
        if not ok:
            print("[ERROR] Pen down failed, aborting")
            conn.safe_disable(return_home=True)
            return

        # --- Draw boundary (inline IK, fire-and-forget) ---
        print(f"\n[4] Drawing boundary ({total} points, interval={args.interval}s)...")
        current_joints = list(reader.read_joints().positions)
        success_count = 0
        fail_count = 0

        t0 = time.monotonic()
        for i, (x, y) in enumerate(path):
            success, current_joints = move_with_ik(
                motion, x, y, z_draw, current_joints, speed=args.speed
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

            if (i + 1) % 20 == 0 or i == total - 1:
                progress = (i + 1) / total * 100
                print(f"    [{progress:5.1f}%] {i+1}/{total}  ok={success_count} fail={fail_count}")

            time.sleep(args.interval)

        elapsed = time.monotonic() - t0
        print(f"    Done in {elapsed:.1f}s")

        # --- Pen up ---
        print(f"\n[5] Pen up -> Z={z_up*1000:.0f}mm")
        move_ik(motion, reader, start_x, start_y, z_up, args.speed)

        # --- Summary ---
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Points     : {total}")
        print(f"  Success    : {success_count}")
        print(f"  IK failures: {fail_count}")
        print(f"  Draw time  : {elapsed:.1f}s")

        # --- Disable ---
        print("\n[6] Disabling arm...")
        conn.safe_disable(return_home=True)
        print("[OK] Done!")


if __name__ == "__main__":
    main()
