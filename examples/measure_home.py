#!/usr/bin/env python3
"""Measure joint angles at the official safe pose for HOME_POSITION.

This script moves the arm to the official example position:
- Position: (150, -50, 150) mm
- Orientation: roll=-179.9°, pitch=0°, yaw=-179.9°

This pose has pitch=0° which avoids singularity issues and allows
reliable Y-axis Cartesian control.

Usage:
    bash scripts/can_activate.sh can0 1000000
    uv run python examples/measure_home.py

After running, copy the printed HOME_POSITION values to:
- src/piper_demo/connection.py:45
- src/piper_demo/motion.py:35
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader


def main():
    print("=" * 60)
    print("Measure HOME_POSITION at Official Safe Pose")
    print("=" * 60)

    # Official example orientation (in 0.001 degrees)
    # RX = -179.9°, RY = 0°, RZ = -179.9°
    OFFICIAL_RX = -179900
    OFFICIAL_RY = 0
    OFFICIAL_RZ = -179900

    # Position values (in 0.001 mm)
    # Official safe position: (150mm, -50mm, 150mm)
    TARGET_X = 150000
    TARGET_Y = -50000
    TARGET_Z = 150000

    print(f"\nTarget Cartesian pose:")
    print(f"  Position: ({TARGET_X/1000:.0f}, {TARGET_Y/1000:.0f}, {TARGET_Z/1000:.0f}) mm")
    print(f"  Orientation: (roll={OFFICIAL_RX/1000:.1f}°, pitch={OFFICIAL_RY/1000:.1f}°, yaw={OFFICIAL_RZ/1000:.1f}°)")

    with PiperConnection(can_name="can0") as conn:
        piper = conn.piper
        reader = JointReader(piper)

        # Enable arm (go_home=False to avoid moving to old home first)
        print("\n[1] Enabling arm (skipping old home position)...")
        conn.enable(go_home=False)
        print("    Arm enabled")

        # Read current pose
        time.sleep(0.3)
        pose = reader.read_end_pose()
        print(f"\n[2] Current pose:")
        print(f"    X = {pose.x * 1000:.1f} mm")
        print(f"    Y = {pose.y * 1000:.1f} mm")
        print(f"    Z = {pose.z * 1000:.1f} mm")
        roll_deg = pose.roll * 180 / 3.14159
        pitch_deg = pose.pitch * 180 / 3.14159
        yaw_deg = pose.yaw * 180 / 3.14159
        print(f"    Roll  = {roll_deg:.1f}°")
        print(f"    Pitch = {pitch_deg:.1f}°")
        print(f"    Yaw   = {yaw_deg:.1f}°")

        # Move to official safe pose using MOVEP
        print(f"\n[3] Moving to official safe pose with MOVEP...")
        duration_sec = 4.0
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)  # MOVEP mode
            piper.EndPoseCtrl(TARGET_X, TARGET_Y, TARGET_Z, OFFICIAL_RX, OFFICIAL_RY, OFFICIAL_RZ)
            time.sleep(0.02)

        # Wait for settle
        print("    Settling...")
        time.sleep(0.5)

        # Read final Cartesian pose
        pose = reader.read_end_pose()
        print(f"\n[4] Final Cartesian pose:")
        print(f"    X = {pose.x * 1000:.1f} mm")
        print(f"    Y = {pose.y * 1000:.1f} mm")
        print(f"    Z = {pose.z * 1000:.1f} mm")
        roll_deg = pose.roll * 180 / 3.14159
        pitch_deg = pose.pitch * 180 / 3.14159
        yaw_deg = pose.yaw * 180 / 3.14159
        print(f"    Roll  = {roll_deg:.1f}°")
        print(f"    Pitch = {pitch_deg:.1f}°")
        print(f"    Yaw   = {yaw_deg:.1f}°")

        # Read joint angles
        state = reader.read_joints()
        print(f"\n[5] Joint angles (radians):")
        for i, p in enumerate(state.positions):
            deg = p * 180 / 3.14159
            print(f"    Joint {i+1}: {p:.5f} rad ({deg:.2f}°)")

        # Output ready-to-copy format
        rounded = [round(p, 5) for p in state.positions]
        print("\n" + "=" * 60)
        print("Copy this line to connection.py and motion.py:")
        print("=" * 60)
        print(f"HOME_POSITION = {rounded}")
        print("=" * 60)

        # Return to safe position and disable
        print("\n[6] Disabling arm...")
        conn.safe_disable()
        print("    Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
