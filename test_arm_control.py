#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper Arm Control Test Script

This script tests basic arm control functionality:
1. Connect and enable the arm
2. Switch to CAN control mode
3. Read current joint positions
4. Move joints by small amounts
5. Return to original position
"""

import time
import sys
from piper_sdk import *


def wait_for_enable(piper, timeout=5.0):
    """Wait for arm to be enabled with timeout."""
    start = time.time()
    while time.time() - start < timeout:
        if piper.EnablePiper():
            return True
        time.sleep(0.01)
    return False


def print_status(piper, label=""):
    """Print current arm status."""
    status = piper.GetArmStatus()
    if label:
        print(f"\n=== {label} ===")
    print(status)
    return status


def print_joints(piper, label=""):
    """Print current joint positions."""
    joints = piper.GetArmJointMsgs()
    if label:
        print(f"\n=== {label} ===")
    print(joints)
    return joints


def main():
    can_port = "can0"
    if len(sys.argv) > 1:
        can_port = sys.argv[1]

    print(f"Connecting to {can_port}...")
    piper = C_PiperInterface_V2(can_port)
    piper.ConnectPort()
    time.sleep(0.2)

    # Step 1: Check initial status
    print_status(piper, "Initial Status")

    # Step 2: Enable arm
    print("\n[Step 1] Enabling arm...")
    if not wait_for_enable(piper, timeout=5.0):
        print("ERROR: Failed to enable arm within timeout!")
        return 1
    print("Arm enabled successfully!")

    # Step 3: Switch to CAN control mode + MOVE J
    print("\n[Step 2] Switching to CAN control mode (MOVE J)...")
    piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)  # ctrl_mode=CAN, move_mode=MOVEJ, speed=30%
    time.sleep(0.2)

    # Step 4: Verify status
    status = print_status(piper, "Status After Mode Switch")

    # Check if in CAN control mode
    if "CAN_CTRL" not in str(status):
        print("\nWARNING: Not in CAN_CTRL mode! Current mode shown above.")
        print("Try running piper_ctrl_reset.py first, then run this script again.")
        return 1

    # Step 5: Read current joint positions
    print_joints(piper, "Current Joint Positions")

    # Step 6: Ask user before moving
    print("\n" + "=" * 50)
    print("Ready to test joint movement.")
    print("The arm will move Joint 1 by ~5 degrees, then return.")
    print("Make sure the workspace is clear!")
    print("=" * 50)

    response = input("\nProceed with movement test? [y/N]: ").strip().lower()
    if response != 'y':
        print("Test cancelled by user.")
        return 0

    # Step 7: Small movement test
    # factor: 1000 * 180 / pi = 57295.7795 (converts radians to 0.001 degrees)
    factor = 57295.7795

    # Move joint 1 by ~5 degrees (0.087 rad)
    test_positions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Position 1: zero
        [0.75, 0.0, 0.0, 0.0, 0.0, 0.0],    # Position 2: J1 +5 deg
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Position 3: back to zero
    ]

    for i, pos in enumerate(test_positions):
        print(f"\n[Step 3.{i+1}] Moving to position {i+1}...")
        print(f"  Target (rad): {pos}")

        # Convert to 0.001 degree units
        joints = [round(p * factor) for p in pos]

        # Send control commands (need to send repeatedly for a short duration)
        for _ in range(100):  # Send for ~0.5 seconds
            piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            piper.JointCtrl(joints[0], joints[1], joints[2],
                           joints[3], joints[4], joints[5])
            time.sleep(0.005)

        # Wait for movement to complete
        time.sleep(1.0)
        print_status(piper, f"Status after position {i+1}")

    # Step 8: Final status
    print_joints(piper, "Final Joint Positions")
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
