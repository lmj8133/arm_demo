#!/usr/bin/env python3
"""Debug script to test MOVEL mode with official example orientation.

This script tests X/Y movement using the same orientation as the official
piper_ctrl_moveL.py example to avoid singularity issues.

Usage:
    bash scripts/can_activate.sh can0 1000000
    uv run python examples/debug_moveL.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader


def main():
    print("=" * 60)
    print("MOVEL Mode Test (Official Example Style)")
    print("=" * 60)

    # Official example uses these orientation values (in 0.001 degrees)
    # RX = -179.9°, RY = 0°, RZ = -179.9°
    OFFICIAL_RX = -179900
    OFFICIAL_RY = 0
    OFFICIAL_RZ = -179900

    # Position values (in 0.001 mm)
    # Start position: (150mm, -50mm, 150mm)
    START_X = 150000
    START_Y = -50000
    START_Z = 150000

    # Target position: (150mm, +50mm, 150mm) - Y changes by 100mm
    TARGET_X = 150000
    TARGET_Y = 50000
    TARGET_Z = 150000

    with PiperConnection(can_name="can0") as conn:
        piper = conn.piper
        reader = JointReader(piper)

        # Enable arm
        print("\n[1] Enabling arm...")
        conn.enable()
        print("    Arm enabled")

        # Read current pose
        time.sleep(0.5)
        pose = reader.read_end_pose()
        print(f"\n[2] Current pose after enable:")
        print(f"    X = {pose.x * 1000:.1f} mm")
        print(f"    Y = {pose.y * 1000:.1f} mm")
        print(f"    Z = {pose.z * 1000:.1f} mm")
        roll_deg = pose.roll * 180 / 3.14159
        pitch_deg = pose.pitch * 180 / 3.14159
        yaw_deg = pose.yaw * 180 / 3.14159
        print(f"    Roll  = {roll_deg:.1f}°")
        print(f"    Pitch = {pitch_deg:.1f}°")
        print(f"    Yaw   = {yaw_deg:.1f}°")

        # Step 1: Use MOVEP to go to start position (official example style)
        print(f"\n[3] Moving to start position with MOVEP (0x00)...")
        print(f"    Target: ({START_X/1000:.0f}, {START_Y/1000:.0f}, {START_Z/1000:.0f}) mm")
        print(f"    Orientation: (RX={OFFICIAL_RX/1000:.1f}°, RY={OFFICIAL_RY/1000:.1f}°, RZ={OFFICIAL_RZ/1000:.1f}°)")

        # Send MOVEP command continuously for 3 seconds
        start_time = time.time()
        while time.time() - start_time < 3.0:
            piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)  # MOVEP mode
            piper.EndPoseCtrl(START_X, START_Y, START_Z, OFFICIAL_RX, OFFICIAL_RY, OFFICIAL_RZ)
            time.sleep(0.02)

        # Check position
        pose = reader.read_end_pose()
        print(f"\n[4] Pose after MOVEP:")
        print(f"    X = {pose.x * 1000:.1f} mm")
        print(f"    Y = {pose.y * 1000:.1f} mm")
        print(f"    Z = {pose.z * 1000:.1f} mm")

        # Step 2: Use MOVEL to move Y axis
        print(f"\n[5] Moving with MOVEL (0x02) - Y from -50 to +50 mm...")
        print(f"    Target: ({TARGET_X/1000:.0f}, {TARGET_Y/1000:.0f}, {TARGET_Z/1000:.0f}) mm")

        start_time = time.time()
        sample_count = 0
        initial_y = pose.y * 1000

        while time.time() - start_time < 5.0:
            piper.MotionCtrl_2(0x01, 0x02, 50, 0x00)  # MOVEL mode
            piper.EndPoseCtrl(TARGET_X, TARGET_Y, TARGET_Z, OFFICIAL_RX, OFFICIAL_RY, OFFICIAL_RZ)

            if sample_count % 25 == 0:
                current = reader.read_end_pose()
                elapsed = time.time() - start_time
                dy = current.y * 1000 - initial_y
                print(f"    [{elapsed:.1f}s] Y = {current.y * 1000:.1f} mm (dY = {dy:+.1f} mm)")

            sample_count += 1
            time.sleep(0.02)

        # Final check
        final_pose = reader.read_end_pose()
        print(f"\n[6] Final pose:")
        print(f"    X = {final_pose.x * 1000:.1f} mm")
        print(f"    Y = {final_pose.y * 1000:.1f} mm")
        print(f"    Z = {final_pose.z * 1000:.1f} mm")

        dy_total = final_pose.y * 1000 - initial_y
        print(f"\n[7] Y axis movement: {dy_total:+.1f} mm")

        if abs(dy_total) > 50:
            print("    [OK] Y axis moved significantly!")
        elif abs(dy_total) > 5:
            print("    [PARTIAL] Y axis moved but less than expected")
        else:
            print("    [FAIL] Y axis did not move")

        # Disable
        print("\n[8] Disabling arm...")
        conn.safe_disable()
        print("    Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
