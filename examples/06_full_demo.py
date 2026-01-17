#!/usr/bin/env python3
"""Example 06: Full demo - complete workflow demonstration.

This example demonstrates a complete workflow:
1. Connect to the arm
2. Read current joint positions
3. Enable the arm
4. Move to home position
5. Execute a simple pick-and-place sequence
6. Return to home and disable

Usage:
    # First, activate CAN bus
    bash scripts/can_activate.sh can0 1000000

    # Run the full demo
    uv run python examples/06_full_demo.py

    # Dry run (no actual movement)
    uv run python examples/06_full_demo.py --dry-run

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
"""

import sys
import os
import argparse
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import (
    PiperConnection,
    JointReader,
    MotionController,
    GripperController,
)
from piper_demo.utils import deg_to_rad


# Demo waypoints (joint positions in degrees)
WAYPOINTS = {
    "home": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ready": [0.0, 30.0, -30.0, 0.0, 0.0, 0.0],
    "pick_approach": [30.0, 45.0, -45.0, 0.0, -20.0, 0.0],
    "place_approach": [-30.0, 45.0, -45.0, 0.0, -20.0, 0.0],
}


def wait_with_progress(seconds: float, message: str = "Waiting"):
    """Wait with progress indicator."""
    print(f"[INFO] {message}", end="", flush=True)
    for _ in range(int(seconds * 2)):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print(" done")


def run_demo(conn, reader, motion, gripper, dry_run: bool = False):
    """Execute the demo sequence."""

    def move_to(name: str, positions_deg: list):
        """Helper to move to a waypoint."""
        print(f"\n[STEP] Moving to '{name}'...")
        if dry_run:
            print(f"       Target: {positions_deg}")
            time.sleep(0.5)
            return

        positions_rad = [deg_to_rad(p) for p in positions_deg]
        motion.move_joint(positions_rad)
        reader.wait_for_position(positions_rad, tolerance_rad=deg_to_rad(3.0))
        state = reader.read_joints()
        print(f"       Reached: {[f'{p:.1f}°' for p in state.positions_deg()]}")

    # Step 1: Read current state
    print("\n" + "=" * 60)
    print("STEP 1: Reading current state")
    print("=" * 60)
    state = reader.read_joints()
    print(state)
    print(f"Gripper: {gripper.read_position_mm():.1f} mm")

    # Step 2: Enable arm (auto moves to home position)
    print("\n" + "=" * 60)
    print("STEP 2: Enabling arm (moving to home position)")
    print("=" * 60)
    if not dry_run:
        conn.enable()
        print("[OK] Arm enabled and at home position")
    else:
        print("[DRY-RUN] Would enable arm and move to home")

    # Step 3: Move to home
    print("\n" + "=" * 60)
    print("STEP 3: Moving to home position")
    print("=" * 60)
    move_to("home", WAYPOINTS["home"])

    # Step 4: Open gripper
    print("\n" + "=" * 60)
    print("STEP 4: Opening gripper")
    print("=" * 60)
    if not dry_run:
        gripper.open()
        wait_with_progress(1.0, "Gripper opening")
    else:
        print("[DRY-RUN] Would open gripper")

    # Step 5: Move to ready position
    print("\n" + "=" * 60)
    print("STEP 5: Moving to ready position")
    print("=" * 60)
    move_to("ready", WAYPOINTS["ready"])

    # Step 6: Simulate pick
    print("\n" + "=" * 60)
    print("STEP 6: Simulating pick operation")
    print("=" * 60)
    move_to("pick_approach", WAYPOINTS["pick_approach"])
    if not dry_run:
        gripper.close()
        wait_with_progress(1.0, "Gripper closing (picking)")
    else:
        print("[DRY-RUN] Would close gripper")

    # Step 7: Simulate place
    print("\n" + "=" * 60)
    print("STEP 7: Simulating place operation")
    print("=" * 60)
    move_to("ready", WAYPOINTS["ready"])
    move_to("place_approach", WAYPOINTS["place_approach"])
    if not dry_run:
        gripper.open()
        wait_with_progress(1.0, "Gripper opening (placing)")
    else:
        print("[DRY-RUN] Would open gripper")

    # Step 8: Return home
    print("\n" + "=" * 60)
    print("STEP 8: Returning to home")
    print("=" * 60)
    move_to("ready", WAYPOINTS["ready"])
    move_to("home", WAYPOINTS["home"])

    # Step 9: Disable arm
    print("\n" + "=" * 60)
    print("STEP 9: Disabling arm")
    print("=" * 60)
    if not dry_run:
        conn.safe_disable()
        print("[OK] Arm safely disabled")
    else:
        print("[DRY-RUN] Would disable arm")


def main():
    parser = argparse.ArgumentParser(
        description="Full demo of Piper arm capabilities"
    )
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Speed factor 0.0-1.0 (default: 0.3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without moving the arm"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPER ARM FULL DEMO")
    print("=" * 60)
    print(f"CAN interface: {args.can}")
    print(f"Speed factor: {args.speed}")
    print(f"Dry run: {args.dry_run}")

    if not args.dry_run:
        print("\n[WARNING] This will physically move the robot arm!")
        print("          Press Ctrl+C within 3 seconds to cancel...")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n[INFO] Cancelled by user")
            return 0

    try:
        with PiperConnection(can_name=args.can, verify_can=not args.dry_run) as conn:
            reader = JointReader(conn.piper)
            motion = MotionController(conn.piper, speed_factor=args.speed)
            gripper = GripperController(conn.piper)

            run_demo(conn, reader, motion, gripper, dry_run=args.dry_run)

            print("\n" + "=" * 60)
            print("[SUCCESS] Demo completed!")
            print("=" * 60)
            return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
