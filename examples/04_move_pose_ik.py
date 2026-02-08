#!/usr/bin/env python3
"""Move arm to Cartesian position using IK + joint control.

Flow:
1. Move to initial joint angles (joint control)
2. Compute IK for target position
3. Move to IK result (joint control)

Usage:
    # Move to position (uses default initial joints and orientation)
    uv run python examples/04_move_pose_ik.py --x 0.2 --y 0 --z 0.2

    # Specify orientation
    uv run python examples/04_move_pose_ik.py --x 0.2 --y 0 --z 0.2 --roll -180 --pitch 0 --yaw 90

    # Specify initial joint angles (degrees)
    uv run python examples/04_move_pose_ik.py --x 0.2 --y 0 --z 0.2 --init-joints 0 75 -42 -3 62 1
"""

import sys
import os
import argparse
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader, MotionController
from piper_demo.inverse_kinematics import inverse_kinematics, IKConfig
from piper_demo.kinematics import forward_kinematics


# Default initial joints (HOME_POSITION from find_optimal_z.py)
#DEFAULT_INIT_JOINTS_DEG = [0, 75.229, -41.940, -3.245, 61.653, 1.401]
DEFAULT_INIT_JOINTS_DEG = [0, 117.419, -32.810, 5.728, -70, 1.368]

def main():
    parser = argparse.ArgumentParser(
        description="Move arm to position using IK + joint control"
    )
    parser.add_argument("--can", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--x", type=float, required=True, help="X position in meters")
    parser.add_argument("--y", type=float, required=True, help="Y position in meters")
    parser.add_argument("--z", type=float, required=True, help="Z position in meters")
    parser.add_argument("--roll", type=float, default=None,
                        help="Roll (RX) in degrees (default: from init-joints FK)")
    parser.add_argument("--pitch", type=float, default=None,
                        help="Pitch (RY) in degrees (default: from init-joints FK)")
    parser.add_argument("--yaw", type=float, default=None,
                        help="Yaw (RZ) in degrees (default: from init-joints FK)")
    parser.add_argument("--init-joints", type=float, nargs=6, default=None,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="Initial joint angles in degrees (default: HOME_POSITION)")
    parser.add_argument("--speed", type=float, default=0.3,
                        help="Speed factor 0.0-1.0 (default: 0.3)")
    parser.add_argument("--no-home", action="store_true",
                        help="Don't go to home position on enable")
    args = parser.parse_args()

    # Initial joints
    if args.init_joints:
        init_joints_deg = args.init_joints
    else:
        init_joints_deg = DEFAULT_INIT_JOINTS_DEG
    init_joints_rad = [math.radians(j) for j in init_joints_deg]

    # Compute FK for initial joints (for default orientation)
    init_fk = forward_kinematics(init_joints_rad)

    # Target orientation (default: from init-joints FK)
    if args.roll is not None:
        roll_rad = math.radians(args.roll)
        roll_deg = args.roll
    else:
        roll_rad = init_fk.roll
        roll_deg = math.degrees(roll_rad)

    if args.pitch is not None:
        pitch_rad = math.radians(args.pitch)
        pitch_deg = args.pitch
    else:
        pitch_rad = init_fk.pitch
        pitch_deg = math.degrees(pitch_rad)

    if args.yaw is not None:
        yaw_rad = math.radians(args.yaw)
        yaw_deg = args.yaw
    else:
        yaw_rad = init_fk.yaw
        yaw_deg = math.degrees(yaw_rad)

    print("=" * 60)
    print("Move to Position (IK + Joint Control)")
    print("=" * 60)
    print()
    print("Step 1: Move to initial joints")
    print(f"  Joints: {[f'{j:.1f}°' for j in init_joints_deg]}")
    print(f"  FK Position: X={init_fk.x*1000:.1f}mm, Y={init_fk.y*1000:.1f}mm, Z={init_fk.z*1000:.1f}mm")
    print(f"  FK Orientation: RX={math.degrees(init_fk.roll):.1f}°, RY={math.degrees(init_fk.pitch):.1f}°, RZ={math.degrees(init_fk.yaw):.1f}°")
    print()
    print("Step 2: IK to target position")
    print(f"  Position: X={args.x*1000:.1f}mm, Y={args.y*1000:.1f}mm, Z={args.z*1000:.1f}mm")
    print(f"  Orientation: RX={roll_deg:.1f}°, RY={pitch_deg:.1f}°, RZ={yaw_deg:.1f}°")
    if args.roll is None or args.pitch is None or args.yaw is None:
        print("  (Orientation from init-joints FK)")
    print()
    print(f"Speed: {args.speed}")
    print()

    # IK configuration
    ik_config = IKConfig(
        max_iterations=200,
        damping_factor=0.05,
        position_tolerance=2e-3,    # 2mm
        orientation_tolerance=2e-2, # ~1.15°
    )

    try:
        with PiperConnection(can_name=args.can) as conn:
            reader = JointReader(conn.piper)
            motion = MotionController(conn.piper, speed_factor=args.speed)

            # Enable arm
            print("[INFO] Enabling arm...")
            conn.enable(go_home=not args.no_home)
            time.sleep(1)
            print("[OK] Arm enabled")
            print()

            # Step 1: Move to initial joints
            print("[Step 1] Moving to initial joints...")
            input("Press Enter to move (Ctrl+C to abort)...")

            motion.move_joint(init_joints_rad, speed_factor=args.speed)
            time.sleep(2)

            # Verify position
            joint_state = reader.read_joints()
            current_joints = list(joint_state.positions)
            current_fk = forward_kinematics(current_joints)
            print(f"[OK] At initial position:")
            print(f"     X={current_fk.x*1000:.1f}mm, Y={current_fk.y*1000:.1f}mm, Z={current_fk.z*1000:.1f}mm")
            print(f"     RX={math.degrees(current_fk.roll):.1f}°, RY={math.degrees(current_fk.pitch):.1f}°, RZ={math.degrees(current_fk.yaw):.1f}°")
            print()

            # Step 2: Compute IK
            print("[Step 2] Computing IK...")
            result = inverse_kinematics(
                target_x=args.x,
                target_y=args.y,
                target_z=args.z,
                target_roll=roll_rad,
                target_pitch=pitch_rad,
                target_yaw=yaw_rad,
                initial_guess=current_joints,
                config=ik_config,
            )

            if not result.converged:
                print(f"[ERROR] IK failed!")
                print(f"        Position error: {result.position_error*1000:.2f}mm")
                print(f"        Orientation error: {math.degrees(result.orientation_error):.2f}°")
                conn.safe_disable(return_home=True)
                return 1

            print(f"[OK] IK converged (iter={result.iterations}, err={result.position_error*1000:.2f}mm)")
            print()

            # Step 3: Move to IK result
            print("[Step 3] Moving to target position...")
            input("Press Enter to move (Ctrl+C to abort)...")

            motion.move_joint(result.joint_angles, speed_factor=args.speed)
            time.sleep(2)

            # Read final position
            joint_state = reader.read_joints()
            final_joints = list(joint_state.positions)
            final_fk = forward_kinematics(final_joints)

            print()
            print("[OK] Final position:")
            print(f"     X={final_fk.x*1000:.1f}mm, Y={final_fk.y*1000:.1f}mm, Z={final_fk.z*1000:.1f}mm")
            print(f"     RX={math.degrees(final_fk.roll):.1f}°, RY={math.degrees(final_fk.pitch):.1f}°, RZ={math.degrees(final_fk.yaw):.1f}°")
            print()

            # Print joint angles for code
            print("[INFO] Joint angles for code:")
            angles_deg = [math.degrees(j) for j in final_joints]
            print(f"  Degrees: {[f'{a:.3f}' for a in angles_deg]}")
            print(f"  Radians: {[f'{r:.5f}' for r in final_joints]}")
            print()

            # Safe disable
            print("[INFO] Safely disabling arm...")
            conn.safe_disable(return_home=True)
            print("[OK] Done")

            return 0

    except KeyboardInterrupt:
        print("\n[INFO] Aborted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
