#!/usr/bin/env python3
"""Play back drawing commands from a CSV file using DrawingController.

Each row in the CSV represents a single move() call:
    write, x, y
where:
    write  — 1 = pen down (draw), 0 = pen up (travel)
    x, y   — normalized coordinates in [0, 1]

Example CSV content:
    0, 0.5, 0.5
    1, 0.3, 0.3
    1, 0.7, 0.3
    1, 0.7, 0.7
    0, 0.0, 0.0

Usage:
    uv run python test/test_csv_player.py path/to/commands.csv
    uv run python test/test_csv_player.py data.csv --can can1 --speed 0.2
    uv run python test/test_csv_player.py data.csv --dry-run
"""

import argparse
import csv
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "ex15"))

from piper_demo import PiperConnection, MotionController, JointReader
from drawing import DrawingController, DrawingConfig


def load_commands(csv_path):
    """Load move commands from a CSV file.

    Expected columns: write, x, y
    Lines starting with '#' and empty lines are skipped.

    Returns:
        List of (write: bool, x: float, y: float) tuples.
    """
    commands = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            if not row or row[0].strip().startswith("#"):
                continue
            if row[0].strip().lower() == "write":
                continue
            if len(row) < 3:
                print(f"[WARN] Line {lineno}: expected 3 columns, got {len(row)}, skipping")
                continue
            try:
                write = row[0].strip() in ("1", "true", "True", "TRUE")
                x = float(row[1].strip())
                y = float(row[2].strip())
            except ValueError as e:
                print(f"[WARN] Line {lineno}: parse error ({e}), skipping")
                continue
            commands.append((write, x, y))
    return commands


def validate_commands(commands):
    """Check that all coordinates are within [0, 1]."""
    issues = 0
    for i, (write, x, y) in enumerate(commands):
        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
            print(f"[WARN] Command {i}: coords ({x}, {y}) outside [0,1], will be clamped")
            issues += 1
    return issues


def main():
    parser = argparse.ArgumentParser(description="Play CSV drawing commands on the arm")
    parser.add_argument("csv_file", help="Path to CSV file with move commands")
    parser.add_argument("--can", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--speed", type=float, default=0.3, help="Speed factor 0.1-1.0 (default: 0.3)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate CSV only, do not move arm")
    args = parser.parse_args()

    # Load and validate
    commands = load_commands(args.csv_file)
    if not commands:
        print(f"[ERROR] No valid commands found in {args.csv_file}")
        sys.exit(1)

    warn_count = validate_commands(commands)

    draw_count = sum(1 for w, _, _ in commands if w)
    travel_count = len(commands) - draw_count
    print("=" * 60)
    print("CSV Drawing Player")
    print("=" * 60)
    print(f"  File      : {args.csv_file}")
    print(f"  Commands  : {len(commands)} ({draw_count} draw, {travel_count} travel)")
    print(f"  Warnings  : {warn_count}")
    print(f"  Speed     : {args.speed}")
    print()

    if args.dry_run:
        print("[DRY-RUN] Commands parsed successfully:")
        for i, (write, x, y) in enumerate(commands):
            action = "DRAW  " if write else "TRAVEL"
            print(f"  {i:4d}  {action}  ({x:.4f}, {y:.4f})")
        return

    config = DrawingConfig(draw_speed=args.speed, move_speed=args.speed)

    with PiperConnection(can_name=args.can) as conn:
        motion = MotionController(conn.piper)
        reader = JointReader(conn.piper)
        drawer = DrawingController(motion, reader, config)
        try:
            print("[1] Enabling arm...")
            conn.enable(go_home=False)
            time.sleep(1)

            joints = reader.read_joints().positions
            print(f"    Joints: {[f'{math.degrees(j):.1f}' for j in joints]}")

            total = len(commands)
            print(f"\n[2] Executing {total} commands...")
            input("    Press Enter to start (Ctrl+C to abort)...")

            success_count = 0
            fail_count = 0
            t0 = time.monotonic()

            for i, (write, x, y) in enumerate(commands):
                action = "DRAW" if write else "MOVE"
                ok = drawer.move(write, x, y)
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"    [{i}/{total}] {action} ({x:.3f}, {y:.3f}) -> FAIL")

                if (i + 1) % 10 == 0 or i == total - 1:
                    progress = (i + 1) / total * 100
                    print(f"    [{progress:5.1f}%] {i+1}/{total}  ok={success_count} fail={fail_count}")

            elapsed = time.monotonic() - t0

            print("\n[3] Pen up...")
            drawer.pen_up()

            print()
            print("=" * 60)
            print("Summary")
            print("=" * 60)
            print(f"  Commands    : {total}")
            print(f"  Success     : {success_count} ({success_count/total*100:.1f}%)")
            print(f"  IK failures : {fail_count}")
            print(f"  Elapsed     : {elapsed:.1f}s")

            print("\n[4] Disabling arm...")
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
