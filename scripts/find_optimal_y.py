#!/usr/bin/env python3
"""Find optimal Y height for maximizing XZ reachable area.

Samples Y values and calculates the convex hull area of IK-reachable points
in the XZ plane. Outputs the Y value with the largest reachable area.

Usage:
    uv run python scripts/find_optimal_y.py
    uv run python scripts/find_optimal_y.py --y-min 0.05 --y-max 0.15 --y-steps 20
    uv run python scripts/find_optimal_y.py --xz-samples 30 --verbose

Output:
    Prints reachable area for each Y value and the optimal Y at the end.
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
# Y=16mm, optimized for XZ reach
HOME_POSITION = [1.59868, 0.27609, -0.83856, 0.04926, 0.67869, -0.7526]


def convex_hull_area(points: list[tuple[float, float]]) -> float:
    """Compute convex hull area using gift wrapping + shoelace formula.

    Args:
        points: List of (x, z) tuples

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


def compute_reachable_area(
    y_height: float,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    roll: float,
    pitch: float,
    yaw: float,
    xz_samples: int,
    initial_guess: list[float],
    ik_config: IKConfig,
) -> tuple[float, list[tuple[float, float]]]:
    """Compute reachable area in XZ plane at given Y height.

    Args:
        y_height: Y coordinate in meters
        x_range: (x_min, x_max) in meters
        z_range: (z_min, z_max) in meters
        roll, pitch, yaw: Target orientation in radians
        xz_samples: Number of samples per axis
        initial_guess: Initial joint angles for IK
        ik_config: IK solver configuration

    Returns:
        (area_m2, list of reachable (x, z) points)
    """
    reachable_points = []
    x_min, x_max = x_range
    z_min, z_max = z_range

    for i in range(xz_samples):
        for j in range(xz_samples):
            x = x_min + (x_max - x_min) * i / (xz_samples - 1)
            z = z_min + (z_max - z_min) * j / (xz_samples - 1)

            result = inverse_kinematics(
                x, y_height, z,
                roll, pitch, yaw,
                initial_guess=initial_guess,
                config=ik_config,
            )

            if result.converged:
                reachable_points.append((x, z))
                # Use solution as next initial guess for better convergence
                initial_guess = result.joint_angles

    area = convex_hull_area(reachable_points)
    return area, reachable_points


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal Y height for maximum XZ reachable area"
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Minimum Y height in meters (default: HOME_Y - 0.05)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum Y height in meters (default: HOME_Y + 0.10)",
    )
    parser.add_argument(
        "--y-range-below",
        type=float,
        default=0.05,
        help="Y search range below HOME (default: 0.05 = 50mm)",
    )
    parser.add_argument(
        "--y-range-above",
        type=float,
        default=0.10,
        help="Y search range above HOME (default: 0.10 = 100mm)",
    )
    parser.add_argument(
        "--y-steps",
        type=int,
        default=11,
        help="Number of Y samples (default: 11)",
    )
    parser.add_argument(
        "--xz-samples",
        type=int,
        default=20,
        help="Samples per XZ axis (default: 20)",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("X_MIN", "X_MAX"),
        help="X range in meters (default: HOME_X ± 0.15)",
    )
    parser.add_argument(
        "--z-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Z_MIN", "Z_MAX"),
        help="Z range in meters (default: HOME_Z ± 0.15)",
    )
    parser.add_argument(
        "--xz-range",
        type=float,
        default=0.15,
        help="Half-range for X and Z centered on HOME (default: 0.15 = ±150mm)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information",
    )
    args = parser.parse_args()

    # Compute HOME_POSITION FK for reference orientation
    home_joints = HOME_POSITION
    home_fk = forward_kinematics(home_joints)

    print("=" * 60)
    print("Find Optimal Y Height for Maximum XZ Reachable Area")
    print("=" * 60)
    print()
    print(f"HOME_POSITION FK:")
    x_mm, y_mm, z_mm = home_fk.position_mm()
    print(f"  Position: X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm")
    r_deg, p_deg, yw_deg = home_fk.orientation_deg()
    print(f"  Orientation: roll={r_deg:.1f}°, pitch={p_deg:.1f}°, yaw={yw_deg:.1f}°")
    print()

    # Compute Y range centered on HOME position
    y_min = args.y_min if args.y_min is not None else home_fk.y - args.y_range_below
    y_max = args.y_max if args.y_max is not None else home_fk.y + args.y_range_above

    # Compute X/Z ranges centered on HOME position
    if args.x_range is not None:
        x_range = tuple(args.x_range)
    else:
        x_range = (home_fk.x - args.xz_range, home_fk.x + args.xz_range)

    if args.z_range is not None:
        z_range = tuple(args.z_range)
    else:
        z_range = (home_fk.z - args.xz_range, home_fk.z + args.xz_range)

    print(f"Search parameters:")
    print(f"  Y range: {y_min*1000:.0f} ~ {y_max*1000:.0f} mm ({args.y_steps} steps)")
    print(f"  X range: {x_range[0]*1000:.0f} ~ {x_range[1]*1000:.0f} mm")
    print(f"  Z range: {z_range[0]*1000:.0f} ~ {z_range[1]*1000:.0f} mm")
    print(f"  XZ grid: {args.xz_samples}x{args.xz_samples} = {args.xz_samples**2} samples per Y")
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

    results = []
    best_y = None
    best_area = 0.0
    best_count = 0

    print("Scanning Y values...")
    print("-" * 60)

    for i in range(args.y_steps):
        y = y_min + (y_max - y_min) * i / (args.y_steps - 1)

        area, points = compute_reachable_area(
            y_height=y,
            x_range=x_range,
            z_range=z_range,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            xz_samples=args.xz_samples,
            initial_guess=list(home_joints),
            ik_config=ik_config,
        )

        # Convert to cm² for readability
        area_cm2 = area * 10000

        results.append((y, area_cm2, len(points)))

        marker = ""
        if area > best_area:
            best_y = y
            best_area = area
            best_count = len(points)
            marker = " <- Best"

        print(f"Y={y*1000:6.1f}mm -> Area={area_cm2:6.1f} cm² ({len(points):3d} pts){marker}")

        if args.verbose and points:
            x_vals = [p[0] for p in points]
            z_vals = [p[1] for p in points]
            print(f"             X: [{min(x_vals)*1000:.0f}, {max(x_vals)*1000:.0f}] mm")
            print(f"             Z: [{min(z_vals)*1000:.0f}, {max(z_vals)*1000:.0f}] mm")

    print("-" * 60)
    print()

    if best_y is not None:
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Optimal Y: {best_y*1000:.1f} mm ({best_y:.5f} m)")
        print(f"Reachable area: {best_area*10000:.1f} cm²")
        print(f"IK-solvable points: {best_count}/{args.xz_samples**2}")
        print()
        print("Usage in code:")
        print(f"  override_y={best_y:.5f}  # meters")
        print()

        # Compare with home Y
        home_y_mm = home_fk.y * 1000
        delta = best_y * 1000 - home_y_mm
        print(f"HOME Y: {home_y_mm:.1f} mm")
        print(f"Delta: {delta:+.1f} mm")
    else:
        print("[ERROR] No reachable points found in the search range!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
