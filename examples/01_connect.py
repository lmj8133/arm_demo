#!/usr/bin/env python3
"""Example 01: Basic connection test.

This example demonstrates how to connect to the Piper arm
and verify communication is working.

Usage:
    # First, activate CAN bus
    bash scripts/can_activate.sh can0 1000000

    # Run this example
    uv run python examples/01_connect.py
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, check_can_interface


def main():
    can_name = "can0"

    # Check CAN interface first
    print(f"[1/3] Checking CAN interface '{can_name}'...")
    if not check_can_interface(can_name):
        print(f"[ERROR] CAN interface '{can_name}' is not active.")
        print(f"        Run: bash scripts/can_activate.sh {can_name} 1000000")
        return 1

    print(f"[OK] CAN interface '{can_name}' is active")

    # Connect to arm
    print(f"\n[2/3] Connecting to Piper arm...")
    try:
        with PiperConnection(can_name=can_name) as conn:
            print(f"[OK] Connected: {conn}")

            # Read basic status
            print(f"\n[3/3] Reading arm status...")
            joint_msg = conn.piper.GetArmJointMsgs()
            print(f"[OK] Joint message received")
            print(f"     Joint 1: {joint_msg.joint_state.joint_1:.4f} rad")
            print(f"     Joint 2: {joint_msg.joint_state.joint_2:.4f} rad")
            print(f"     Joint 3: {joint_msg.joint_state.joint_3:.4f} rad")

            print("\n[SUCCESS] Connection test passed!")
            return 0

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
