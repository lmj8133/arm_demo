#!/usr/bin/env python3
"""Example 13: Vision-guided arm control with normalized camera coordinates.

Demonstrates the VisionArmController for mapping camera coordinates (0-1)
to robot arm workspace positions.

Usage:
    uv run python examples/13_vision_control.py
    uv run python examples/13_vision_control.py --can can1
    uv run python examples/13_vision_control.py --config config/camera_workspace.yaml

Commands:
    y z       - Move to normalized camera position (0-1 range)
    center    - Move to workspace center (0.5, 0.5)
    corners   - Visit all four corners
    scan      - Scan workspace in grid pattern
    status    - Show current position in both coordinate systems
    speed <v> - Set speed factor (0.1-1.0)
    disable   - Return to safe home, disable arm, and exit
    help      - Show command help

WARNING: This will physically move the robot arm!
         Ensure the workspace is clear before running.
"""

import sys
import os
import argparse
import readline  # noqa: F401 - Enable command history
import time
from typing import Tuple

# Add src to path for package import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piper_demo import PiperConnection, JointReader, MotionController
from piper_demo.vision_arm_controller import VisionArmController, MoveResult


def print_banner():
    """Print the program banner."""
    print("=" * 60)
    print("Vision-Guided Arm Control Demo")
    print("Camera coordinates (0-1) → Arm workspace")
    print("=" * 60)


def print_help():
    """Print command help."""
    print()
    print("Commands:")
    print("  y z          Move to normalized camera position (0-1 range)")
    print("               Example: 0.5 0.5 moves to center")
    print("  center       Move to workspace center (0.5, 0.5)")
    print("  corners      Visit all four corners sequentially")
    print("  scan [n]     Scan workspace in n×n grid (default: 3)")
    print("  status       Show current position in both coordinate systems")
    print("  speed <v>    Set speed factor (0.1-1.0)")
    print("  disable      Return to safe home, disable arm, and exit")
    print("  help         Show this help message")
    print()


def print_result(result: MoveResult):
    """Print movement result details."""
    x_mm, y_mm, z_mm = result.position_mm()

    if result.ik_converged and result.motion_completed:
        print(f"[OK] Moved to camera ({result.y_cam:.2f}, {result.z_cam:.2f})")
        print(f"     Arm position: X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm")
    else:
        if not result.ik_converged:
            print(f"[ERROR] IK failed (iter={result.ik_iterations}, "
                  f"pos_err={result.position_error*1000:.2f}mm)")
        elif result.near_singularity:
            print(f"[ERROR] Near singularity at camera ({result.y_cam:.2f}, {result.z_cam:.2f})")
        elif not result.motion_completed:
            print(f"[WARNING] Motion timeout at camera ({result.y_cam:.2f}, {result.z_cam:.2f})")


def print_status(controller: VisionArmController, speed: float):
    """Print detailed status information."""
    # Get current position
    pose = controller.reader.read_end_pose()

    # Get camera coordinates
    y_cam, z_cam = controller.get_current_camera_coords()

    print()
    print(f"Arm Position:")
    print(f"  X={pose.x*1000:.1f}, Y={pose.y*1000:.1f}, Z={pose.z*1000:.1f} mm")
    print(f"Camera Coordinates (normalized):")
    print(f"  Y={y_cam:.3f}, Z={z_cam:.3f}")
    print(f"Speed: {speed}")
    print()

    # Print workspace bounds
    ws = controller.config.workspace
    print(f"Workspace bounds:")
    print(f"  X: {ws.x_arm.min*1000:.0f} to {ws.x_arm.max*1000:.0f} mm")
    print(f"  Y: {ws.y_arm.min*1000:.0f} to {ws.y_arm.max*1000:.0f} mm")
    print(f"  Z: {ws.z_arm.min*1000:.0f} to {ws.z_arm.max*1000:.0f} mm")


def parse_command(line: str) -> Tuple[str, list]:
    """Parse user input command."""
    line = line.strip()
    if not line:
        return ('empty', [])

    parts = line.split()
    cmd = parts[0].lower()

    if cmd == 'center':
        return ('move', [0.5, 0.5])
    elif cmd == 'corners':
        return ('corners', [])
    elif cmd == 'scan':
        grid_size = 3
        if len(parts) > 1:
            try:
                grid_size = int(parts[1])
                grid_size = max(2, min(5, grid_size))
            except ValueError:
                return ('error', ['Invalid grid size'])
        return ('scan', [grid_size])
    elif cmd == 'status':
        return ('status', [])
    elif cmd == 'speed':
        if len(parts) < 2:
            return ('error', ['Speed command requires a value (0.1-1.0)'])
        try:
            val = float(parts[1])
            if val < 0.1 or val > 1.0:
                return ('error', ['Speed must be between 0.1 and 1.0'])
            return ('speed', [val])
        except ValueError:
            return ('error', [f'Invalid speed value: {parts[1]}'])
    elif cmd == 'disable':
        return ('disable', [])
    elif cmd == 'help':
        return ('help', [])

    # Try to parse as y z coordinates
    if len(parts) == 2:
        try:
            y = float(parts[0])
            z = float(parts[1])
            if not (0 <= y <= 1 and 0 <= z <= 1):
                return ('error', ['Coordinates must be in range 0-1'])
            return ('move', [y, z])
        except ValueError:
            return ('error', [f'Invalid coordinates: {line}'])

    return ('error', [f'Unknown command: {line}'])


def main():
    parser = argparse.ArgumentParser(
        description="Vision-guided arm control demo"
    )
    parser.add_argument(
        "--can", default="can0",
        help="CAN interface name (default: can0)"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Initial speed factor 0.1-1.0 (default: 0.3)"
    )
    args = parser.parse_args()

    print_banner()

    # Clamp initial speed
    speed = max(0.1, min(1.0, args.speed))

    print(f"[INFO] Connecting to arm on {args.can}...")

    try:
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

            # Create controller
            if args.config:
                print(f"[INFO] Loading config from {args.config}")
                controller = VisionArmController.from_yaml(
                    args.config, reader, motion
                )
            else:
                # Auto-compute workspace from HOME_POSITION FK
                print("[INFO] Computing workspace from HOME_POSITION...")
                controller, home_fk = VisionArmController.from_home_position(
                    PiperConnection.HOME_POSITION, reader, motion
                )
                # Show computed workspace
                ws = controller.config.workspace
                x_mm, y_mm, z_mm = home_fk.position_mm()
                print(f"[INFO] HOME FK: X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm")
                print(f"[INFO] Workspace X: {ws.x_arm.min*1000:.0f} ~ {ws.x_arm.max*1000:.0f} mm")
                print(f"[INFO] Workspace Y: {ws.y_arm.min*1000:.0f} ~ {ws.y_arm.max*1000:.0f} mm")
                print(f"[INFO] Workspace Z: {ws.z_arm.min*1000:.0f} ~ {ws.z_arm.max*1000:.0f} mm")
                print(f"[INFO] Singularity threshold: {controller.config.motion.singularity_threshold:.2e}")

            print("[OK] Ready!")
            print_help()

            # Main interaction loop
            running = True
            while running:
                # Get current camera coords for prompt
                y_cam, z_cam = controller.get_current_camera_coords()

                print()
                print(f"Current: camera ({y_cam:.2f}, {z_cam:.2f})")

                try:
                    line = input("> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    print("[INFO] Interrupted, exiting...")
                    break

                cmd, cmd_args = parse_command(line)

                if cmd == 'empty':
                    continue

                elif cmd == 'error':
                    print(f"[ERROR] {cmd_args[0]}")

                elif cmd == 'help':
                    print_help()

                elif cmd == 'status':
                    print_status(controller, speed)

                elif cmd == 'speed':
                    speed = cmd_args[0]
                    print(f"[OK] Speed set to {speed}")

                elif cmd == 'move':
                    y_cam, z_cam = cmd_args
                    print(f"[INFO] Moving to camera ({y_cam:.2f}, {z_cam:.2f})...")
                    result = controller.move_to_normalized(
                        y_cam, z_cam,
                        speed_factor=speed,
                    )
                    print_result(result)

                elif cmd == 'corners':
                    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                    print("[INFO] Visiting corners...")
                    for i, (y, z) in enumerate(corners):
                        print(f"[{i+1}/4] Moving to ({y:.1f}, {z:.1f})...")
                        result = controller.move_to_normalized(
                            y, z,
                            speed_factor=speed,
                        )
                        print_result(result)
                        if result.motion_completed:
                            time.sleep(0.5)

                elif cmd == 'scan':
                    grid_size = cmd_args[0]
                    print(f"[INFO] Scanning {grid_size}×{grid_size} grid...")
                    results = controller.scan_workspace(
                        grid_size=grid_size,
                        speed_factor=speed,
                        dwell_sec=0.3,
                    )
                    success = sum(1 for r in results if r.motion_completed)
                    print(f"[OK] Scan complete: {success}/{len(results)} positions reached")

                elif cmd == 'disable':
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
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
