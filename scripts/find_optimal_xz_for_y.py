#!/usr/bin/env python3
"""Find optimal (X, Z) for maximizing Y-axis reachable range.

For YZ plane tracking (X fixed), this script finds the optimal fixed X value
and working height Z that maximizes the arm's reachable range along the Y-axis
(left/right direction).

Usage:
    uv run python scripts/find_optimal_xz_for_y.py
    uv run python scripts/find_optimal_xz_for_y.py --x-steps 15 --z-steps 15 --verbose
    uv run python scripts/find_optimal_xz_for_y.py --x-min 0.10 --x-max 0.25

Output:
    Prints Y range for each (X, Z) combination and the optimal values at the end.
"""

import argparse
import sys
import os

# Add src/piper_demo directly to path to avoid __init__.py triggering piper_sdk import
_piper_demo_path = os.path.join(os.path.dirname(__file__), "..", "src", "piper_demo")
sys.path.insert(0, _piper_demo_path)

# Import directly from modules (not package) to avoid piper_sdk dependency
from kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics, IKConfig

# HOME_POSITION from PiperConnection (copied to avoid piper_sdk dependency)
HOME_POSITION = [-0.05434, 0.13097, -0.89813, 0.01238, 0.74649, -0.73524]


def compute_y_range(
    x_fixed: float,
    z_height: float,
    y_range: tuple[float, float],
    roll: float,
    pitch: float,
    yaw: float,
    y_samples: int,
    initial_guess: list[float],
    ik_config: IKConfig,
) -> tuple[float, float, float, int, list[float] | None]:
    """Compute reachable Y range at given (X, Z) position.

    Args:
        x_fixed: Fixed X coordinate in meters
        z_height: Z coordinate (height) in meters
        y_range: (y_min, y_max) search range in meters
        roll, pitch, yaw: Target orientation in radians
        y_samples: Number of Y samples
        initial_guess: Initial joint angles for IK
        ik_config: IK solver configuration

    Returns:
        (y_min_reachable, y_max_reachable, span, count, center_joint_angles)
        center_joint_angles: IK solution closest to Y center, or None if unreachable
    """
    reachable = []  # [(y, joint_angles), ...]
    y_min, y_max = y_range

    for i in range(y_samples):
        y = y_min + (y_max - y_min) * i / (y_samples - 1)

        result = inverse_kinematics(
            x_fixed,
            y,
            z_height,
            roll,
            pitch,
            yaw,
            initial_guess=initial_guess,
            config=ik_config,
        )

        if result.converged:
            reachable.append((y, result.joint_angles))
            # Use solution as next initial guess for better convergence
            initial_guess = result.joint_angles

    if not reachable:
        return (0.0, 0.0, 0.0, 0, None)

    # Find solution closest to Y center
    y_center = (y_min + y_max) / 2
    center_solution = min(reachable, key=lambda r: abs(r[0] - y_center))

    y_vals = [r[0] for r in reachable]
    y_min_reach = min(y_vals)
    y_max_reach = max(y_vals)
    span = y_max_reach - y_min_reach

    return (y_min_reach, y_max_reach, span, len(reachable), center_solution[1])


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal (X, Z) for maximum Y-axis range (REP-103)"
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Minimum X in meters (default: HOME_X - 0.10)",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Maximum X in meters (default: HOME_X + 0.10)",
    )
    parser.add_argument(
        "--x-range-offset",
        type=float,
        default=0.10,
        help="X search half-range from HOME (default: 0.10 = ±100mm)",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=None,
        help="Minimum Z height in meters (default: HOME_Z - 0.05)",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="Maximum Z height in meters (default: HOME_Z + 0.10)",
    )
    parser.add_argument(
        "--z-range-below",
        type=float,
        default=0.05,
        help="Z search range below HOME (default: 0.05 = 50mm)",
    )
    parser.add_argument(
        "--z-range-above",
        type=float,
        default=0.10,
        help="Z search range above HOME (default: 0.10 = 100mm)",
    )
    parser.add_argument(
        "--x-steps",
        type=int,
        default=11,
        help="Number of X samples (default: 11)",
    )
    parser.add_argument(
        "--z-steps",
        type=int,
        default=11,
        help="Number of Z samples (default: 11)",
    )
    parser.add_argument(
        "--y-samples",
        type=int,
        default=40,
        help="Number of Y samples for range calculation (default: 40)",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y_MIN", "Y_MAX"),
        help="Y search range in meters (default: HOME_Y ± 0.20)",
    )
    parser.add_argument(
        "--y-range-offset",
        type=float,
        default=0.20,
        help="Y search half-range from HOME (default: 0.20 = ±200mm)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information for each combination",
    )
    args = parser.parse_args()

    # Compute HOME_POSITION FK for reference
    home_joints = HOME_POSITION
    home_fk = forward_kinematics(home_joints)

    print("=" * 60)
    print("Find Optimal (X, Z) for Maximum Y Range")
    print("(REP-103: X=front, Y=left, Z=up)")
    print("=" * 60)
    print()
    print("HOME_POSITION FK:")
    x_mm, y_mm, z_mm = home_fk.position_mm()
    print(f"  Position: X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm")
    r_deg, p_deg, yw_deg = home_fk.orientation_deg()
    print(f"  Orientation: roll={r_deg:.1f}°, pitch={p_deg:.1f}°, yaw={yw_deg:.1f}°")
    print()

    # Compute X range centered on HOME position
    x_min = args.x_min if args.x_min is not None else home_fk.x - args.x_range_offset
    x_max = args.x_max if args.x_max is not None else home_fk.x + args.x_range_offset

    # Compute Z range centered on HOME position
    z_min = args.z_min if args.z_min is not None else home_fk.z - args.z_range_below
    z_max = args.z_max if args.z_max is not None else home_fk.z + args.z_range_above

    # Compute Y search range
    if args.y_range is not None:
        y_search = tuple(args.y_range)
    else:
        y_search = (home_fk.y - args.y_range_offset, home_fk.y + args.y_range_offset)

    print("Search parameters:")
    print(
        f"  X (front/back) range: {x_min * 1000:.0f} ~ {x_max * 1000:.0f} mm ({args.x_steps} steps)"
    )
    print(
        f"  Z (height) range: {z_min * 1000:.0f} ~ {z_max * 1000:.0f} mm ({args.z_steps} steps)"
    )
    print(
        f"  Y (left/right) search: {y_search[0] * 1000:.0f} ~ {y_search[1] * 1000:.0f} mm ({args.y_samples} samples)"
    )
    print(f"  Total combinations: {args.x_steps * args.z_steps}")
    print()

    # IK configuration
    ik_config = IKConfig(
        max_iterations=100,
        damping_factor=0.05,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
    )

    # Use home orientation
    roll, pitch, yaw = home_fk.roll, home_fk.pitch, home_fk.yaw

    # Track best result
    best_x = None
    best_z = None
    best_span = 0.0
    best_y_min = 0.0
    best_y_max = 0.0
    best_count = 0
    best_center_joints = None  # IK solution at Y center for best (X, Z)

    print("Scanning (X, Z) combinations...")
    print("-" * 60)

    for i in range(args.x_steps):
        x = (
            x_min + (x_max - x_min) * i / (args.x_steps - 1)
            if args.x_steps > 1
            else x_min
        )

        for j in range(args.z_steps):
            z = (
                z_min + (z_max - z_min) * j / (args.z_steps - 1)
                if args.z_steps > 1
                else z_min
            )

            y_min_reach, y_max_reach, span, count, center_joints = compute_y_range(
                x_fixed=x,
                z_height=z,
                y_range=y_search,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                y_samples=args.y_samples,
                initial_guess=list(home_joints),
                ik_config=ik_config,
            )

            marker = ""
            if span > best_span:
                best_x = x
                best_z = z
                best_span = span
                best_y_min = y_min_reach
                best_y_max = y_max_reach
                best_count = count
                best_center_joints = center_joints  # Save the IK solution
                marker = " <- Best"

            if args.verbose or marker:
                print(
                    f"X={x * 1000:6.1f}mm, Z={z * 1000:6.1f}mm -> Y range: {span * 1000:6.1f} mm ({count:2d} pts){marker}"
                )
            elif count == 0:
                print(
                    f"X={x * 1000:6.1f}mm, Z={z * 1000:6.1f}mm -> Y range: 0 mm (unreachable)"
                )

    print("-" * 60)
    print()

    if best_x is not None and best_z is not None and best_span > 0:
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Optimal X (front/back): {best_x * 1000:.1f} mm ({best_x:.5f} m)")
        print(f"Optimal Z (height): {best_z * 1000:.1f} mm ({best_z:.5f} m)")
        print(
            f"Reachable Y range: {best_y_min * 1000:.1f} ~ {best_y_max * 1000:.1f} mm (span: {best_span * 1000:.1f} mm)"
        )
        print(f"IK-solvable points: {best_count}/{args.y_samples}")
        print()

        # Output suggested HOME_POSITION using saved IK solution from scan
        if best_center_joints is not None:
            angles_str = ", ".join(f"{a:.5f}" for a in best_center_joints)
            print("Suggested HOME_POSITION:")
            print(f"  HOME_POSITION = [{angles_str}]")
            print()
        else:
            print("[WARN] No IK solution found for center Y position")
            print()

        print("Usage in code:")
        print(f"  workspace_x_center = {best_x:.5f}  # meters")
        print(f"  override_z = {best_z:.5f}  # meters")
        print()

        # Compare with home position
        print(
            f"HOME X: {home_fk.x * 1000:.1f} mm, Delta: {(best_x - home_fk.x) * 1000:+.1f} mm"
        )
        print(
            f"HOME Z: {home_fk.z * 1000:.1f} mm, Delta: {(best_z - home_fk.z) * 1000:+.1f} mm"
        )
    else:
        print("[ERROR] No reachable points found in the search range!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
