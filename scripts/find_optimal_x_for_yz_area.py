#!/usr/bin/env python3
"""Find optimal X position for maximizing YZ plane reachable area.

For YZ plane tracking (X fixed), this script finds the optimal fixed X value
that maximizes the arm's reachable area in the YZ plane (left/right + up/down).

Usage:
    uv run python scripts/find_optimal_x_for_yz_area.py
    uv run python scripts/find_optimal_x_for_yz_area.py --x-steps 15 --yz-samples 25 --verbose
    uv run python scripts/find_optimal_x_for_yz_area.py --x-min 0.10 --x-max 0.25

Output:
    Prints YZ reachable area for each X value and the optimal X at the end.
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


def convex_hull_area(points: list[tuple[float, float]]) -> float:
    """Compute convex hull area using gift wrapping + shoelace formula.

    Args:
        points: List of (y, z) tuples

    Returns:
        Area of convex hull in square meters
    """
    if len(points) < 3:
        return 0.0

    # Remove duplicates and very close points
    unique_points = []
    for p in points:
        is_dup = False
        for q in unique_points:
            if abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)

    if len(unique_points) < 3:
        return 0.0

    # Find convex hull using gift wrapping (Jarvis march)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Start from leftmost point
    start = min(unique_points, key=lambda p: (p[0], p[1]))
    hull = []
    current = start

    while True:
        hull.append(current)
        candidate = unique_points[0]
        for p in unique_points[1:]:
            if candidate == current or cross(current, candidate, p) < 0:
                candidate = p
        current = candidate
        if current == start:
            break
        if len(hull) > len(unique_points):  # Safety break
            break

    if len(hull) < 3:
        return 0.0

    # Shoelace formula for polygon area
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]

    return abs(area) / 2.0


def compute_yz_reachable_area(
    x_fixed: float,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    roll: float,
    pitch: float,
    yaw: float,
    yz_samples: int,
    initial_guess: list[float],
    ik_config: IKConfig,
) -> tuple[float, list[tuple[float, float]], tuple[float, float], tuple[float, float]]:
    """Compute reachable area in YZ plane at given X position.

    Args:
        x_fixed: Fixed X coordinate (front/back) in meters
        y_range: (y_min, y_max) in meters (left/right)
        z_range: (z_min, z_max) in meters (height)
        roll, pitch, yaw: Target orientation in radians
        yz_samples: Number of samples per axis
        initial_guess: Initial joint angles for IK
        ik_config: IK solver configuration

    Returns:
        (area_m2, reachable_points, y_reachable_range, z_reachable_range)
        - area_m2: Convex hull area in square meters
        - reachable_points: List of (y, z) tuples that are IK-solvable
        - y_reachable_range: (y_min, y_max) of reachable points
        - z_reachable_range: (z_min, z_max) of reachable points
    """
    reachable = []  # [(y, z, joint_angles), ...]
    y_min, y_max = y_range
    z_min, z_max = z_range

    for i in range(yz_samples):
        for j in range(yz_samples):
            y = y_min + (y_max - y_min) * i / (yz_samples - 1)
            z = z_min + (z_max - z_min) * j / (yz_samples - 1)

            result = inverse_kinematics(
                x_fixed,
                y,
                z,
                roll,
                pitch,
                yaw,
                initial_guess=initial_guess,
                config=ik_config,
            )

            if result.converged:
                reachable.append((y, z, result.joint_angles))
                # Use solution as next initial guess for better convergence
                initial_guess = result.joint_angles

    if not reachable:
        return (0.0, [], (0.0, 0.0), (0.0, 0.0))

    # Extract (y, z) points for convex hull
    points = [(r[0], r[1]) for r in reachable]
    area = convex_hull_area(points)

    # Compute actual reachable ranges
    y_vals = [r[0] for r in reachable]
    z_vals = [r[1] for r in reachable]
    y_reachable_range = (min(y_vals), max(y_vals))
    z_reachable_range = (min(z_vals), max(z_vals))

    return (area, points, y_reachable_range, z_reachable_range)


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal X for maximum YZ plane reachable area (REP-103)"
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
        "--x-steps",
        type=int,
        default=11,
        help="Number of X samples (default: 11)",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y_MIN", "Y_MAX"),
        help="Y range in meters (default: HOME_Y ± 0.15)",
    )
    parser.add_argument(
        "--y-range-offset",
        type=float,
        default=0.15,
        help="Y search half-range from HOME (default: 0.15 = ±150mm)",
    )
    parser.add_argument(
        "--z-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Z_MIN", "Z_MAX"),
        help="Z range in meters (default: HOME_Z - 0.05 to HOME_Z + 0.10)",
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
        "--yz-samples",
        type=int,
        default=20,
        help="Samples per YZ axis (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information for each X value",
    )
    args = parser.parse_args()

    # Compute HOME_POSITION FK for reference
    home_joints = HOME_POSITION
    home_fk = forward_kinematics(home_joints)

    print("=" * 60)
    print("Find Optimal X for Maximum YZ Plane Reachable Area")
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

    # Compute Y range centered on HOME position
    if args.y_range is not None:
        y_range = tuple(args.y_range)
    else:
        y_range = (home_fk.y - args.y_range_offset, home_fk.y + args.y_range_offset)

    # Compute Z range centered on HOME position
    if args.z_range is not None:
        z_range = tuple(args.z_range)
    else:
        z_range = (home_fk.z - args.z_range_below, home_fk.z + args.z_range_above)

    print("Search parameters:")
    print(
        f"  X (front/back) range: {x_min * 1000:.0f} ~ {x_max * 1000:.0f} mm ({args.x_steps} steps)"
    )
    print(
        f"  Y (left/right) range: {y_range[0] * 1000:.0f} ~ {y_range[1] * 1000:.0f} mm"
    )
    print(f"  Z (height) range: {z_range[0] * 1000:.0f} ~ {z_range[1] * 1000:.0f} mm")
    print(
        f"  YZ grid: {args.yz_samples}x{args.yz_samples} = {args.yz_samples**2} samples per X"
    )
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
    best_area = 0.0
    best_count = 0
    best_y_reachable = (0.0, 0.0)
    best_z_reachable = (0.0, 0.0)

    print("Scanning X values...")
    print("-" * 60)

    for i in range(args.x_steps):
        x = (
            x_min + (x_max - x_min) * i / (args.x_steps - 1)
            if args.x_steps > 1
            else x_min
        )

        area, points, y_reach, z_reach = compute_yz_reachable_area(
            x_fixed=x,
            y_range=y_range,
            z_range=z_range,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            yz_samples=args.yz_samples,
            initial_guess=list(home_joints),
            ik_config=ik_config,
        )

        # Convert to cm² for readability
        area_cm2 = area * 10000

        marker = ""
        if area > best_area:
            best_x = x
            best_area = area
            best_count = len(points)
            best_y_reachable = y_reach
            best_z_reachable = z_reach
            marker = " <- Best"

        print(
            f"X={x * 1000:6.1f}mm -> Area={area_cm2:6.1f} cm² ({len(points):3d} pts){marker}"
        )

        if args.verbose and points:
            print(
                f"             Y: [{y_reach[0] * 1000:.0f}, {y_reach[1] * 1000:.0f}] mm"
            )
            print(
                f"             Z: [{z_reach[0] * 1000:.0f}, {z_reach[1] * 1000:.0f}] mm"
            )

    print("-" * 60)
    print()

    if best_x is not None and best_area > 0:
        # Compute center of reachable YZ range
        y_center = (best_y_reachable[0] + best_y_reachable[1]) / 2
        z_center = (best_z_reachable[0] + best_z_reachable[1]) / 2

        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Optimal X (front/back): {best_x * 1000:.1f} mm ({best_x:.5f} m)")
        print(f"Reachable YZ area: {best_area * 10000:.1f} cm²")
        print(f"IK-solvable points: {best_count}/{args.yz_samples**2}")
        print()
        print(
            f"Reachable Y range: [{best_y_reachable[0] * 1000:.1f}, {best_y_reachable[1] * 1000:.1f}] mm"
        )
        print(
            f"Reachable Z range: [{best_z_reachable[0] * 1000:.1f}, {best_z_reachable[1] * 1000:.1f}] mm"
        )
        print(f"Z center (reachable): {z_center * 1000:.1f} mm")
        print()

        # Solve IK at center of reachable YZ range for HOME_POSITION
        print("Computing HOME_POSITION at center of reachable YZ range...")
        center_result = inverse_kinematics(
            best_x,
            y_center,
            z_center,
            roll,
            pitch,
            yaw,
            initial_guess=list(home_joints),
            config=ik_config,
        )

        if center_result.converged:
            angles_str = ", ".join(f"{a:.5f}" for a in center_result.joint_angles)
            print()
            print("Suggested HOME_POSITION:")
            print(f"  HOME_POSITION = [{angles_str}]")
            print()

            # Verify with FK
            fk_result = forward_kinematics(center_result.joint_angles)
            print(
                f"FK verification: X={fk_result.x * 1000:.1f}, Y={fk_result.y * 1000:.1f}, Z={fk_result.z * 1000:.1f} mm"
            )
            print()
        else:
            print("[WARN] IK failed at center position, trying nearby points...")
            # Try a few points near the center
            found = False
            for dz in [0.01, -0.01, 0.02, -0.02]:
                retry_result = inverse_kinematics(
                    best_x,
                    y_center,
                    z_center + dz,
                    roll,
                    pitch,
                    yaw,
                    initial_guess=list(home_joints),
                    config=ik_config,
                )
                if retry_result.converged:
                    angles_str = ", ".join(
                        f"{a:.5f}" for a in retry_result.joint_angles
                    )
                    fk_result = forward_kinematics(retry_result.joint_angles)
                    print(f"Found solution at Z={fk_result.z * 1000:.1f} mm")
                    print()
                    print("Suggested HOME_POSITION:")
                    print(f"  HOME_POSITION = [{angles_str}]")
                    print()
                    found = True
                    break
            if not found:
                print("[ERROR] Could not find IK solution near center")
                print()

        print("Usage in code:")
        print(f"  workspace_x_center = {best_x:.5f}  # meters")
        print()

        # Compare with home position
        print(
            f"HOME X: {home_fk.x * 1000:.1f} mm, Delta: {(best_x - home_fk.x) * 1000:+.1f} mm"
        )
        print(
            f"HOME Z: {home_fk.z * 1000:.1f} mm, New Z center: {z_center * 1000:.1f} mm, Delta: {(z_center - home_fk.z) * 1000:+.1f} mm"
        )
    else:
        print("[ERROR] No reachable points found in the search range!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
