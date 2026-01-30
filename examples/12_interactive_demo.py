#!/usr/bin/env python3
"""Example 12: Interactive demo with reference-point-based control.

Provides a while-loop interactive interface for controlling the Piper arm
using home-relative xyz coordinates.

Usage:
    uv run python examples/12_interactive_demo.py
    uv run python examples/12_interactive_demo.py --can can1

Commands:
    x y z     - Move to reference + (x, y, z) mm offset
    home      - Return to home position and reset reference point
    setref    - Set current position as new reference point
    status    - Show current position, reference point, and offset
    speed <v> - Set speed factor (0.1-1.0)
    disable   - Return to safe home, disable arm, and exit
    help      - Show command help

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass
from typing import Tuple

# Add src to path for package import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "piper_demo"))

from kinematics import forward_kinematics, FKResult
from inverse_kinematics import inverse_kinematics, IKConfig
from utils import deg_to_rad


@dataclass
class ReferencePoint:
    """Reference point for relative positioning."""

    x: float  # meters
    y: float  # meters
    z: float  # meters
    roll: float  # radians
    pitch: float  # radians
    yaw: float  # radians
    is_home: bool = True  # True if this is the home reference

    def position_mm(self) -> Tuple[float, float, float]:
        """Get position in millimeters."""
        return (self.x * 1000, self.y * 1000, self.z * 1000)


def print_banner():
    """Print the program banner."""
    print("=" * 60)
    print("Piper Interactive Demo")
    print("=" * 60)


def print_help():
    """Print command help."""
    print()
    print("Commands:")
    print("  x y z       Move to reference + (x, y, z) mm offset")
    print("  home        Return to home position and reset reference")
    print("  setref      Set current position as new reference point")
    print("  status      Show current position, reference, and offset")
    print("  speed <v>   Set speed factor (0.1-1.0)")
    print("  disable     Return to safe home, disable arm, and exit")
    print("  help        Show this help message")
    print()


def print_prompt(ref: ReferencePoint, current_pose: FKResult, speed: float):
    """Print the interactive prompt with status info."""
    print()
    ref_label = "home" if ref.is_home else "custom"
    ref_x, ref_y, ref_z = ref.position_mm()
    cur_x, cur_y, cur_z = current_pose.position_mm()

    print(f"Reference: {ref_label} (X={ref_x:.2f}, Y={ref_y:.2f}, Z={ref_z:.2f} mm)")
    print(f"Current:   X={cur_x:.2f}, Y={cur_y:.2f}, Z={cur_z:.2f} mm")
    print(f"Speed: {speed}")
    print()
    print("Commands: x y z | home | setref | status | speed <val> | disable | help")


def print_status(
    ref: ReferencePoint, current_pose: FKResult, joint_state, speed: float
):
    """Print detailed status information."""
    print()
    ref_label = "home" if ref.is_home else "custom"
    ref_x, ref_y, ref_z = ref.position_mm()
    cur_x, cur_y, cur_z = current_pose.position_mm()

    # Calculate offset from reference
    off_x = cur_x - ref_x
    off_y = cur_y - ref_y
    off_z = cur_z - ref_z

    print(f"Reference: {ref_label} (X={ref_x:.2f}, Y={ref_y:.2f}, Z={ref_z:.2f} mm)")
    print(f"Current:   X={cur_x:.2f}, Y={cur_y:.2f}, Z={cur_z:.2f} mm")
    print(f"Offset:    ({off_x:.2f}, {off_y:.2f}, {off_z:.2f}) mm")
    print(f"Speed: {speed}")
    print()
    print("Orientation:")
    r, p, y = current_pose.orientation_deg()
    print(f"  Roll={r:.2f}°, Pitch={p:.2f}°, Yaw={y:.2f}°")

    # Copy-paste ready (same format as measure_safe_home.py)
    joints = [round(p, 5) for p in joint_state.positions]
    print()
    print(f"joints = {joints}")


def parse_command(line: str) -> Tuple[str, list]:
    """Parse user input command.

    Returns:
        (command, args) tuple where command is one of:
        'move', 'home', 'setref', 'status', 'speed', 'disable', 'help', 'error'
    """
    line = line.strip()
    if not line:
        return ("empty", [])

    parts = line.split()
    cmd = parts[0].lower()

    # Named commands
    if cmd == "home":
        return ("home", [])
    elif cmd == "setref":
        return ("setref", [])
    elif cmd == "status":
        return ("status", [])
    elif cmd == "speed":
        if len(parts) < 2:
            return ("error", ["Speed command requires a value (0.1-1.0)"])
        try:
            val = float(parts[1])
            if val < 0.1 or val > 1.0:
                return ("error", ["Speed must be between 0.1 and 1.0"])
            return ("speed", [val])
        except ValueError:
            return ("error", [f"Invalid speed value: {parts[1]}"])
    elif cmd == "disable":
        return ("disable", [])
    elif cmd == "help":
        return ("help", [])

    # Try to parse as xyz coordinates
    if len(parts) == 3:
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            return ("move", [x, y, z])
        except ValueError:
            return ("error", [f"Invalid coordinates: {line}"])

    return ("error", [f"Unknown command: {line}"])


def execute_move(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    ref: ReferencePoint,
    speed: float,
    reader,
    motion,
    home_joints: list,
) -> bool:
    """Compute IK and move to target position.

    Args:
        x_mm, y_mm, z_mm: Offset from reference in millimeters
        ref: Reference point
        speed: Speed factor (0.0-1.0)
        reader: JointReader instance
        motion: MotionController instance
        home_joints: Home joint angles for IK initial guess

    Returns:
        True if move succeeded, False otherwise
    """
    # Calculate target position (reference + offset)
    target_x = ref.x + x_mm / 1000.0
    target_y = ref.y + y_mm / 1000.0
    target_z = ref.z + z_mm / 1000.0

    print(f"[INFO] Target: reference + ({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f}) mm")
    print("[INFO] Computing IK...")

    # Use reference orientation
    target_roll = ref.roll
    target_pitch = ref.pitch
    target_yaw = ref.yaw

    # Compute IK
    config = IKConfig(
        max_iterations=100,
        damping_factor=0.05,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
    )

    result = inverse_kinematics(
        target_x,
        target_y,
        target_z,
        target_roll,
        target_pitch,
        target_yaw,
        initial_guess=home_joints,
        config=config,
    )

    # Check convergence
    if not result.converged:
        print(f"[ERROR] IK did not converge (iterations={result.iterations})")
        return False

    # Check position error
    if result.position_error > 0.001:  # > 1mm
        print(
            f"[ERROR] Position error {result.position_error * 1000:.2f}mm exceeds 1mm threshold"
        )
        return False

    # Move arm
    print("[OK] Moving to target...")
    motion.move_joint(result.joint_angles, speed_factor=speed)

    # Wait for motion to complete
    reached = reader.wait_for_position(
        result.joint_angles,
        tolerance_rad=deg_to_rad(2.0),
        timeout_sec=15.0,
    )

    if reached:
        print("[OK] Done.")
    else:
        print("[WARNING] Motion timeout - may not have reached target")

    return reached


def main():
    parser = argparse.ArgumentParser(
        description="Interactive demo for Piper arm with reference-point-based control"
    )
    parser.add_argument(
        "--can", default="can0", help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Initial speed factor 0.1-1.0 (default: 0.3)",
    )
    args = parser.parse_args()

    print_banner()

    # Clamp initial speed
    speed = max(0.1, min(1.0, args.speed))

    print(f"[INFO] Connecting to arm on {args.can}...")

    try:
        from piper_demo import PiperConnection, JointReader, MotionController

        with PiperConnection(can_name=args.can) as conn:
            reader = JointReader(conn.piper)
            motion = MotionController(conn.piper)

            # Enable arm and move to home
            print("[INFO] Enabling arm and moving to home...")
            conn.enable(go_home=True, home_speed=20, home_settle_sec=1.0)

            # Skip initial readings for stability
            for _ in range(5):
                reader.read_joints()
                time.sleep(0.05)

            print("[OK] Ready!")

            # Calculate home position in Cartesian space
            home_joints = PiperConnection.HOME_POSITION
            home_fk = forward_kinematics(home_joints)

            # Initialize reference point to home
            ref = ReferencePoint(
                x=home_fk.x,
                y=home_fk.y,
                z=home_fk.z,
                roll=home_fk.roll,
                pitch=home_fk.pitch,
                yaw=home_fk.yaw,
                is_home=True,
            )

            # Main interaction loop
            running = True
            while running:
                # Read current pose
                current_pose = reader.read_end_pose()

                # Print prompt
                print_prompt(ref, current_pose, speed)

                # Get user input
                try:
                    line = input("> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    print("[INFO] Interrupted, exiting...")
                    break

                # Parse and execute command
                cmd, cmd_args = parse_command(line)

                if cmd == "empty":
                    continue

                elif cmd == "error":
                    print(f"[ERROR] {cmd_args[0]}")

                elif cmd == "help":
                    print_help()

                elif cmd == "status":
                    current_pose = reader.read_end_pose()
                    joint_state = reader.read_joints()
                    print_status(ref, current_pose, joint_state, speed)

                elif cmd == "speed":
                    speed = cmd_args[0]
                    print(f"[OK] Speed set to {speed}")

                elif cmd == "setref":
                    current_pose = reader.read_end_pose()
                    ref = ReferencePoint(
                        x=current_pose.x,
                        y=current_pose.y,
                        z=current_pose.z,
                        roll=current_pose.roll,
                        pitch=current_pose.pitch,
                        yaw=current_pose.yaw,
                        is_home=False,
                    )
                    print("[OK] Reference point updated to current position.")
                    rx, ry, rz = ref.position_mm()
                    print(f"Reference: custom (X={rx:.2f}, Y={ry:.2f}, Z={rz:.2f} mm)")

                elif cmd == "home":
                    print("[INFO] Returning to home position...")
                    motion.move_to_home(speed_factor=0.2)

                    # Wait for motion to complete
                    reached = reader.wait_for_position(
                        home_joints,
                        tolerance_rad=deg_to_rad(2.0),
                        timeout_sec=15.0,
                    )

                    if reached:
                        # Reset reference to home
                        ref = ReferencePoint(
                            x=home_fk.x,
                            y=home_fk.y,
                            z=home_fk.z,
                            roll=home_fk.roll,
                            pitch=home_fk.pitch,
                            yaw=home_fk.yaw,
                            is_home=True,
                        )
                        print("[OK] Reference point reset to home.")
                    else:
                        print("[WARNING] Motion timeout")

                elif cmd == "move":
                    x_mm, y_mm, z_mm = cmd_args
                    execute_move(
                        x_mm,
                        y_mm,
                        z_mm,
                        ref,
                        speed,
                        reader,
                        motion,
                        home_joints,
                    )

                elif cmd == "disable":
                    print("[INFO] Returning to safe home...")
                    print("[INFO] Disabling arm...")
                    conn.safe_disable(return_home=True, home_speed=20)
                    print("[OK] Goodbye!")
                    running = False

            # If loop exits without disable command, still safe disable
            if running:
                print("[INFO] Safely disabling arm...")
                conn.safe_disable(return_home=True, home_speed=20)
                print("[OK] Done.")

        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
