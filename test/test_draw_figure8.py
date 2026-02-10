#!/usr/bin/env python3
"""Draw a figure-8 pattern using DrawingController.

End-to-end integration test for the DrawingController module.
Uses the unified move(write, x, y) API with automatic pen transitions
and Cartesian interpolation.

Usage:
    uv run python test/test_draw_figure8.py
    uv run python test/test_draw_figure8.py --can can1 --scale 0.04
    uv run python test/test_draw_figure8.py --points 80 --speed 0.2
"""

import argparse
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "ex15"))

from piper_demo import PiperConnection, MotionController, JointReader
from drawing import DrawingController, DrawingConfig


def generate_figure8_points(x_center, y_center, scale, num_points=50):
    """Generate figure-8 (lemniscate) path points.

    Parametric equations:
        x(t) = scale * sin(t) + x_center
        y(t) = scale * sin(t) * cos(t) + y_center

    Returns:
        List of (x, y) tuples forming a closed loop.
    """
    points = []
    for i in range(num_points + 1):
        t = 2 * math.pi * i / num_points
        x = x_center + scale * math.sin(t)
        y = y_center + scale * math.sin(t) * math.cos(t)
        points.append((x, y))
    return points


def main():
    parser = argparse.ArgumentParser(description="Draw figure-8 using DrawingController")
    parser.add_argument("--can", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--speed", type=float, default=0.3, help="Speed factor 0.1-1.0 (default: 0.3)")
    parser.add_argument("--scale", type=float, default=0.03, help="Figure-8 size in meters (default: 0.03)")
    parser.add_argument("--points", type=int, default=50, help="Number of path points (default: 50)")
    args = parser.parse_args()

    # Build DrawingConfig with user speed
    config = DrawingConfig(draw_speed=args.speed, move_speed=args.speed)

    # Compute workspace center from config bounds
    x_center = (config.x_min + config.x_max) / 2
    y_center = (config.y_min + config.y_max) / 2

    # Generate path
    path = generate_figure8_points(x_center, y_center, args.scale, args.points)

    # Print configuration
    print("=" * 60)
    print("Figure-8 Drawing Test (DrawingController)")
    print("=" * 60)
    print(f"  Center    : X={x_center*1000:.0f}mm, Y={y_center*1000:.0f}mm")
    print(f"  Scale     : {args.scale*1000:.0f}mm")
    print(f"  Points    : {len(path)}")
    print(f"  Draw Z    : {config.draw_z*1000:.0f}mm")
    print(f"  Safe Z    : {config.safe_z*1000:.0f}mm")
    print(f"  Speed     : {args.speed}")
    print(f"  Workspace : X=[{config.x_min*1000:.0f}, {config.x_max*1000:.0f}] "
          f"Y=[{config.y_min*1000:.0f}, {config.y_max*1000:.0f}] mm")

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    print(f"  Path X    : [{min(xs)*1000:.1f}, {max(xs)*1000:.1f}] mm")
    print(f"  Path Y    : [{min(ys)*1000:.1f}, {max(ys)*1000:.1f}] mm")
    print()

    # -----------------------------------------------------------------
    # Connect and run
    # -----------------------------------------------------------------
    with PiperConnection(can_name=args.can) as conn:
        motion = MotionController(conn.piper)
        reader = JointReader(conn.piper)
        drawer = DrawingController(motion, reader, config)
        try:
            # --- Enable ---
            print("[1] Enabling arm...")
            conn.enable(go_home=False)
            time.sleep(1)

            joints = reader.read_joints().positions
            print(f"    Joints: {[f'{math.degrees(j):.1f}' for j in joints]}")

            # --- Travel to start ---
            start_x, start_y = path[0]
            print(f"\n[2] Traveling to start ({start_x*1000:.1f}, {start_y*1000:.1f})mm...")

            input("    Press Enter to move (Ctrl+C to abort)...")

            ok = drawer.move(False, start_x, start_y)
            if not ok:
                print("[ERROR] Cannot reach start, aborting")
                drawer.safe_disable()
                conn.safe_disable(return_home=False)
                return

            pose = reader.read_end_pose()
            print(f"    Reached: X={pose.x*1000:.1f}, Y={pose.y*1000:.1f}, Z={pose.z*1000:.1f}mm")

            # --- Draw figure-8 ---
            print(f"\n[3] Drawing figure-8 ({len(path) - 1} segments)...")
            input("    Press Enter to start drawing (Ctrl+C to abort)...")

            success_count = 0
            fail_count = 0
            t0 = time.monotonic()

            for i, (x, y) in enumerate(path[1:], start=1):
                ok = drawer.move(True, x, y)
                if ok:
                    success_count += 1
                else:
                    fail_count += 1

                if i % 10 == 0 or i == len(path) - 1:
                    progress = i / (len(path) - 1) * 100
                    print(f"    [{progress:5.1f}%] {i}/{len(path)-1}  ok={success_count} fail={fail_count}")

            elapsed = time.monotonic() - t0
            print(f"    Done in {elapsed:.1f}s")

            # --- Pen up and return ---
            print(f"\n[4] Pen up, returning to start...")
            drawer.move(False, start_x, start_y)

            # --- Summary ---
            print()
            print("=" * 60)
            print("Summary")
            print("=" * 60)
            total = len(path) - 1
            print(f"  Points      : {total}")
            print(f"  Success     : {success_count} ({success_count/total*100:.1f}%)")
            print(f"  IK failures : {fail_count}")
            print(f"  Draw time   : {elapsed:.1f}s")

            # --- Disable ---
            print("\n[5] Disabling arm...")
            drawer.safe_disable()
            conn.safe_disable(return_home=False)
            print("[OK] Done!")
        except KeyboardInterrupt:
            print("\n[!] Interrupted, safe-disabling arm...")
            drawer.safe_disable()
            conn.safe_disable(return_home=False)
            raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
