#!/usr/bin/env python3
"""Gripper Fine-Tune TUI for ex16 laser drawing workflow.

Interactive terminal UI for adjusting gripper position and effort before
tracking starts. Adapted from examples/14_tui_gripper.py for integration
with main_laser_drawing.py and main_laser_wizard.py.

Controls:
    Up/Down     Select parameter (position / effort)
    Left/Right  Adjust by +/-1
    [ / ]       Adjust by +/-10
    { / }       Adjust by +/-50
    Enter/Space Send current position
    O           Fully open (80 mm), send immediately
    C           Fully close (0 mm), send immediately
    R           Read hardware position & effort
    Q / Esc     Done (exit TUI)

Usage:
    # Dry-run (no hardware)
    python examples/ex16/gripper_tune.py --dry-run

    # Hardware (Orin target)
    bash scripts/can_activate.sh can0 1000000
    python examples/ex16/gripper_tune.py --can can0
"""

import sys
import os
import argparse
import curses
from dataclasses import dataclass
from typing import Optional

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from piper_demo import PiperConnection, GripperController

# Limits
POSITION_MIN = 0
POSITION_MAX = 80
EFFORT_MIN = 0
EFFORT_MAX = 5000

# Cursor indices
CURSOR_POSITION = 0
CURSOR_EFFORT = 1


@dataclass
class GripperTuiState:
    """Mutable state for the TUI loop."""

    position_mm: int = 40
    effort: int = 5000
    cursor: int = CURSOR_POSITION  # 0=position, 1=effort
    message: str = ""
    hw_position_mm: Optional[float] = None
    hw_effort_nm: Optional[float] = None


def _bar(value: float, min_val: float, max_val: float, width: int = 26) -> str:
    """Render a Unicode progress bar."""
    ratio = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _send_position(
    gripper: Optional[GripperController],
    state: GripperTuiState,
    dry_run: bool,
) -> None:
    """Send gripper position command (or log in dry-run)."""
    if dry_run:
        state.message = f"[DRY-RUN] pos={state.position_mm}mm effort={state.effort}"
        return
    if gripper is None:
        state.message = "No gripper connected"
        return
    gripper.set_position_mm(float(state.position_mm), effort=state.effort)
    state.message = f"Sent: {state.position_mm} mm (effort={state.effort})"


def _redraw(
    stdscr,
    state: GripperTuiState,
    can_name: str,
    dry_run: bool,
    title: str = "Gripper Fine-Tune",
) -> None:
    """Full-screen redraw."""
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    if height < 10 or width < 40:
        stdscr.addstr(0, 0, "Terminal too small (need 40x10)")
        stdscr.refresh()
        return

    col_w = min(width - 1, 70)  # clamp drawing width

    # --- Title bar (reversed) ---
    header = f" {title} | {can_name} "
    if dry_run:
        header += "[DRY-RUN] "
    header = header.ljust(col_w)
    stdscr.addstr(0, 0, header[:col_w], curses.A_REVERSE)

    # --- Help ---
    stdscr.addstr(1, 1, "[Up/Down]Select [Left/Right]+/-1 []/[]+/-10 {/}+/-50")
    stdscr.addstr(2, 1, "[Enter]Send [O]Open [C]Close [R]Read [Q/Esc]Done")

    # --- Parameter rows ---
    row = 4
    for idx, (label, value, unit, mn, mx) in enumerate(
        [
            ("Position", state.position_mm, "mm", POSITION_MIN, POSITION_MAX),
            ("Effort", state.effort, "mNm", EFFORT_MIN, EFFORT_MAX),
        ]
    ):
        prefix = "> " if state.cursor == idx else "  "
        attr = curses.A_BOLD if state.cursor == idx else curses.A_NORMAL
        bar = _bar(value, mn, mx)
        line = f"{prefix}{label:<12}{value:>5} {unit:<4} {bar}"
        stdscr.addstr(row + idx, 0, line[:col_w], attr)

    # --- HW readback ---
    row_hw = row + 3
    if state.hw_position_mm is not None:
        stdscr.addstr(row_hw, 2, f"HW Pos: {state.hw_position_mm:.1f} mm")
    if state.hw_effort_nm is not None:
        stdscr.addstr(row_hw + 1, 2, f"HW Effort: {state.hw_effort_nm:.3f} N\u00b7m")

    # --- Summary ---
    row_sum = row_hw + 3
    summary = f" pos={state.position_mm}mm effort={state.effort}mNm"
    stdscr.addstr(row_sum, 0, summary[:col_w])

    # --- Status message ---
    if state.message:
        stdscr.addstr(row_sum + 1, 1, f">> {state.message}"[:col_w], curses.A_DIM)

    stdscr.refresh()


def _adjust(state: GripperTuiState, delta: int) -> bool:
    """Adjust the value under cursor. Returns True if position changed."""
    if state.cursor == CURSOR_POSITION:
        state.position_mm = max(POSITION_MIN, min(POSITION_MAX, state.position_mm + delta))
        return True
    else:
        state.effort = max(EFFORT_MIN, min(EFFORT_MAX, state.effort + delta))
        return False


def tui_main_loop(
    stdscr,
    gripper: Optional[GripperController],
    conn: Optional[PiperConnection],
    state: GripperTuiState,
    can_name: str,
    dry_run: bool,
    title: str = "Gripper Fine-Tune",
) -> None:
    """Main curses event loop."""
    curses.curs_set(0)  # hide cursor
    stdscr.nodelay(False)
    stdscr.keypad(True)

    _redraw(stdscr, state, can_name, dry_run, title)

    while True:
        key = stdscr.getch()

        if key in (ord("q"), ord("Q"), 27):  # Q or Esc
            break

        elif key == curses.KEY_UP:
            state.cursor = CURSOR_POSITION

        elif key == curses.KEY_DOWN:
            state.cursor = CURSOR_EFFORT

        elif key == curses.KEY_LEFT:
            if _adjust(state, -1):
                _send_position(gripper, state, dry_run)

        elif key == curses.KEY_RIGHT:
            if _adjust(state, 1):
                _send_position(gripper, state, dry_run)

        elif key == ord("["):
            if _adjust(state, -10):
                _send_position(gripper, state, dry_run)

        elif key == ord("]"):
            if _adjust(state, 10):
                _send_position(gripper, state, dry_run)

        elif key == ord("{"):
            if _adjust(state, -50):
                _send_position(gripper, state, dry_run)

        elif key == ord("}"):
            if _adjust(state, 50):
                _send_position(gripper, state, dry_run)

        elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
            _send_position(gripper, state, dry_run)

        elif key in (ord("o"), ord("O")):
            state.position_mm = POSITION_MAX
            if not dry_run and gripper is not None:
                gripper.open(effort=state.effort)
                state.message = f"Opened (effort={state.effort})"
            else:
                state.message = f"[DRY-RUN] Open (effort={state.effort})"

        elif key in (ord("c"), ord("C")):
            state.position_mm = POSITION_MIN
            if not dry_run and gripper is not None:
                gripper.close(effort=state.effort)
                state.message = f"Closed (effort={state.effort})"
            else:
                state.message = f"[DRY-RUN] Close (effort={state.effort})"

        elif key in (ord("r"), ord("R")):
            if dry_run or gripper is None:
                state.message = "Not available in dry-run"
            else:
                try:
                    state.hw_position_mm = gripper.read_position_mm()
                    state.hw_effort_nm = gripper.read_effort()
                    state.message = (
                        f"Read: {state.hw_position_mm:.1f} mm, "
                        f"{state.hw_effort_nm:.3f} N\u00b7m"
                    )
                except Exception as e:
                    state.message = f"Read error: {e}"

        else:
            # Ignore unknown keys
            continue

        _redraw(stdscr, state, can_name, dry_run, title)


def run_gripper_tune(
    piper=None,
    effort: int = 5000,
    position_mm: int = 40,
    can_name: str = "can0",
    dry_run: bool = False,
    title: str = "Gripper Fine-Tune",
) -> GripperTuiState:
    """High-level entry point for gripper fine-tune TUI.

    Creates GripperController, initializes it, runs the curses TUI,
    and returns the final state.

    Args:
        piper: Connected C_PiperInterface_V2 instance (None for dry-run)
        effort: Initial effort / torque limit 0-5000
        position_mm: Initial position in mm (0-80)
        can_name: CAN interface name (for display only)
        dry_run: Run without hardware
        title: TUI title bar text

    Returns:
        Final GripperTuiState after user exits
    """
    state = GripperTuiState(
        position_mm=max(POSITION_MIN, min(POSITION_MAX, position_mm)),
        effort=max(EFFORT_MIN, min(EFFORT_MAX, effort)),
    )

    gripper: Optional[GripperController] = None

    if not dry_run and piper is not None:
        gripper = GripperController(piper, effort=state.effort)
        gripper.initialize()

    curses.wrapper(
        tui_main_loop, gripper, None, state, can_name, dry_run, title
    )

    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive TUI for Piper gripper fine-tuning"
    )
    parser.add_argument(
        "--can", default="can0", help="CAN interface (default: can0)"
    )
    parser.add_argument(
        "--effort", type=int, default=5000,
        help="Initial effort / torque limit 0-5000 in 0.001 N\u00b7m (default: 5000)"
    )
    parser.add_argument(
        "--position", type=int, default=40, help="Initial position mm (default: 40)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without hardware"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] Starting gripper fine-tune TUI without hardware...")
        run_gripper_tune(
            dry_run=True,
            effort=args.effort,
            position_mm=args.position,
            can_name=args.can,
        )
        return 0

    # Hardware path
    conn: Optional[PiperConnection] = None
    try:
        conn = PiperConnection(can_name=args.can)
        conn.connect()
        print(f"[INFO] Connected to {args.can}")

        conn.enable(go_home=False)
        print("[INFO] Arm enabled (no homing)")

        run_gripper_tune(
            piper=conn.piper,
            effort=args.effort,
            position_mm=args.position,
            can_name=args.can,
        )

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    finally:
        if conn is not None and conn.is_connected:
            print("[INFO] Disabling arm...")
            try:
                conn.safe_disable()
            except Exception:
                pass
            print("[INFO] Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
