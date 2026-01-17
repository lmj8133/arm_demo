#!/usr/bin/env python3
"""Debug script for Cartesian control diagnostic.

This script tests each axis individually and prints detailed debug info
to help diagnose the Y-axis issue.

Usage:
    bash scripts/can_activate.sh can0 1000000
    uv run python examples/debug_cartesian.py --axis y --offset 0.05
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader


def main():
    parser = argparse.ArgumentParser(description="Debug Cartesian control")
    parser.add_argument("--can", default="can0", help="CAN interface")
    parser.add_argument(
        "--axis", choices=["x", "y", "z"], default="y", help="Axis to test"
    )
    parser.add_argument(
        "--offset", type=float, default=0.05, help="Offset in meters (default: 0.05)"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Test duration in seconds"
    )
    parser.add_argument(
        "--speed", type=int, default=20, help="Speed value 0-100 (default: 20)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Cartesian Control Debug Tool")
    print("=" * 60)

    with PiperConnection(can_name=args.can) as conn:
        piper = conn.piper
        reader = JointReader(piper)

        # Read initial pose
        print("\n[1] Reading initial pose...")
        pose = reader.read_end_pose()
        print(f"    X = {pose.x:.6f} m ({pose.x * 1000:.3f} mm)")
        print(f"    Y = {pose.y:.6f} m ({pose.y * 1000:.3f} mm)")
        print(f"    Z = {pose.z:.6f} m ({pose.z * 1000:.3f} mm)")
        print(f"    Roll  = {pose.roll:.4f} rad")
        print(f"    Pitch = {pose.pitch:.4f} rad")
        print(f"    Yaw   = {pose.yaw:.4f} rad")

        # Compute target
        target_x = pose.x
        target_y = pose.y
        target_z = pose.z
        target_roll = pose.roll
        target_pitch = pose.pitch
        target_yaw = pose.yaw

        if args.axis == "x":
            target_x += args.offset
        elif args.axis == "y":
            target_y += args.offset
        elif args.axis == "z":
            target_z += args.offset

        print(f"\n[2] Target pose (offset {args.axis.upper()} by {args.offset:+.3f} m):")
        print(f"    X = {target_x:.6f} m ({target_x * 1000:.3f} mm)")
        print(f"    Y = {target_y:.6f} m ({target_y * 1000:.3f} mm)")
        print(f"    Z = {target_z:.6f} m ({target_z * 1000:.3f} mm)")

        # Convert to SDK units (0.001mm for position, 0.001deg for orientation)
        x_001mm = int(target_x * 1_000_000)
        y_001mm = int(target_y * 1_000_000)
        z_001mm = int(target_z * 1_000_000)
        roll_001deg = int(target_roll * 180 / 3.14159265 * 1000)
        pitch_001deg = int(target_pitch * 180 / 3.14159265 * 1000)
        yaw_001deg = int(target_yaw * 180 / 3.14159265 * 1000)

        print(f"\n[3] SDK units (0.001mm / 0.001deg):")
        print(f"    X = {x_001mm}")
        print(f"    Y = {y_001mm}")
        print(f"    Z = {z_001mm}")
        print(f"    Roll  = {roll_001deg}")
        print(f"    Pitch = {pitch_001deg}")
        print(f"    Yaw   = {yaw_001deg}")

        # Enable arm
        print("\n[4] Enabling arm...")
        conn.enable()
        print("    Arm enabled")

        # Read pose after enable (arm returns to home)
        time.sleep(1.0)
        home_pose = reader.read_end_pose()
        print(f"\n[5] Pose after enable (home position):")
        print(f"    X = {home_pose.x:.6f} m ({home_pose.x * 1000:.3f} mm)")
        print(f"    Y = {home_pose.y:.6f} m ({home_pose.y * 1000:.3f} mm)")
        print(f"    Z = {home_pose.z:.6f} m ({home_pose.z * 1000:.3f} mm)")

        # Recalculate target based on home position
        if args.axis == "x":
            target_x = home_pose.x + args.offset
            target_y = home_pose.y
            target_z = home_pose.z
        elif args.axis == "y":
            target_x = home_pose.x
            target_y = home_pose.y + args.offset
            target_z = home_pose.z
        elif args.axis == "z":
            target_x = home_pose.x
            target_y = home_pose.y
            target_z = home_pose.z + args.offset

        target_roll = home_pose.roll
        target_pitch = home_pose.pitch
        target_yaw = home_pose.yaw

        x_001mm = int(target_x * 1_000_000)
        y_001mm = int(target_y * 1_000_000)
        z_001mm = int(target_z * 1_000_000)
        roll_001deg = int(target_roll * 180 / 3.14159265 * 1000)
        pitch_001deg = int(target_pitch * 180 / 3.14159265 * 1000)
        yaw_001deg = int(target_yaw * 180 / 3.14159265 * 1000)

        print(f"\n[6] Recalculated target from home (SDK units):")
        print(f"    X = {x_001mm} (target: {target_x * 1000:.3f} mm)")
        print(f"    Y = {y_001mm} (target: {target_y * 1000:.3f} mm)")
        print(f"    Z = {z_001mm} (target: {target_z * 1000:.3f} mm)")

        # Send motion commands
        print(f"\n[7] Sending motion commands for {args.duration}s...")
        print(f"    MotionCtrl_2(0x01, 0x00, {args.speed}, 0x00)")
        print(f"    EndPoseCtrl({x_001mm}, {y_001mm}, {z_001mm}, "
              f"{roll_001deg}, {pitch_001deg}, {yaw_001deg})")

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < args.duration:
            # Send motion command
            piper.MotionCtrl_2(0x01, 0x00, args.speed, 0x00)
            piper.EndPoseCtrl(
                x_001mm, y_001mm, z_001mm,
                roll_001deg, pitch_001deg, yaw_001deg
            )

            # Print progress every 0.5s
            if sample_count % 25 == 0:
                current = reader.read_end_pose()
                elapsed = time.time() - start_time
                dx = (current.x - home_pose.x) * 1000
                dy = (current.y - home_pose.y) * 1000
                dz = (current.z - home_pose.z) * 1000
                print(
                    f"    [{elapsed:.1f}s] "
                    f"dX={dx:+.1f}mm dY={dy:+.1f}mm dZ={dz:+.1f}mm"
                )

            sample_count += 1
            time.sleep(0.02)  # 50 Hz

        # Final pose
        final_pose = reader.read_end_pose()
        print(f"\n[8] Final pose:")
        print(f"    X = {final_pose.x:.6f} m ({final_pose.x * 1000:.3f} mm)")
        print(f"    Y = {final_pose.y:.6f} m ({final_pose.y * 1000:.3f} mm)")
        print(f"    Z = {final_pose.z:.6f} m ({final_pose.z * 1000:.3f} mm)")

        print(f"\n[9] Movement summary:")
        dx = (final_pose.x - home_pose.x) * 1000
        dy = (final_pose.y - home_pose.y) * 1000
        dz = (final_pose.z - home_pose.z) * 1000
        print(f"    Delta X = {dx:+.3f} mm")
        print(f"    Delta Y = {dy:+.3f} mm")
        print(f"    Delta Z = {dz:+.3f} mm")

        expected = args.offset * 1000
        if args.axis == "x":
            actual = dx
        elif args.axis == "y":
            actual = dy
        else:
            actual = dz

        print(f"\n    Expected {args.axis.upper()} change: {expected:+.1f} mm")
        print(f"    Actual {args.axis.upper()} change:   {actual:+.1f} mm")

        if abs(actual) < 1.0:
            print(f"\n    [FAIL] {args.axis.upper()} axis did NOT move!")
        elif abs(actual - expected) > 10:
            print(f"\n    [WARN] Movement differs from expected by {abs(actual - expected):.1f} mm")
        else:
            print(f"\n    [OK] {args.axis.upper()} axis moved as expected")

        print("\n[10] Disabling arm...")
        conn.safe_disable()
        print("     Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
