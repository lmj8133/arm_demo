"""Real-time trajectory visualization canvas.

Collects normalized coordinates from laser tracking and renders
an accumulated path on a separate OpenCV window.
"""

import time
from typing import List, Tuple

import cv2
import numpy as np

# Type alias: a stroke is a list of (nx, ny) normalized points
Stroke = List[Tuple[float, float]]


class TrajectoryCanvas:
    """Accumulates laser pen strokes and renders them onto a canvas.

    Coordinates use normalized space where origin is bottom-left (Y-up),
    matching the homography output of main_laser_drawing.py.
    """

    def __init__(self, size: int = 600, margin: int = 40,
                 idle_clear: float = 0):
        self._size = size
        self._margin = margin
        self._draw_size = size - 2 * margin
        self._strokes: List[Stroke] = []
        self._current_stroke: Stroke = []
        self._pen_down = False
        # Last known cursor position (normalized)
        self._cursor: Tuple[float, float] = (0.0, 0.0)
        # Idle auto-clear (0 = disabled)
        self._idle_clear = idle_clear
        self._last_write_time: float = time.time()
        self._pending_clear = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, write: bool, nx: float, ny: float) -> None:
        """Feed one frame of data.

        Args:
            write: True if laser is detected (pen down), False otherwise.
            nx: Normalized x coordinate (0-1, left to right).
            ny: Normalized y coordinate (0-1, bottom to top).
        """
        if write:
            self._cursor = (nx, ny)
            self._last_write_time = time.time()
            if not self._pen_down:
                # Pen-down transition: clear old strokes if pending
                if self._pending_clear:
                    self._strokes.clear()
                    self._pending_clear = False
                self._current_stroke = [(nx, ny)]
                self._pen_down = True
            else:
                self._current_stroke.append((nx, ny))
        else:
            if self._pen_down:
                # Finish current stroke
                if len(self._current_stroke) >= 2:
                    self._strokes.append(self._current_stroke)
                self._current_stroke = []
                self._pen_down = False
            # Mark pending clear after idle timeout (do NOT clear yet)
            if (self._idle_clear > 0
                    and not self._pending_clear
                    and self._has_strokes()
                    and time.time() - self._last_write_time > self._idle_clear):
                self._pending_clear = True

    def render(self) -> np.ndarray:
        """Render the canvas and return a BGR image."""
        canvas = np.full((self._size, self._size, 3), 255, dtype=np.uint8)

        self._draw_grid(canvas)
        self._draw_strokes(canvas)
        self._draw_cursor(canvas)
        self._draw_info(canvas)

        return canvas

    def clear(self) -> None:
        """Clear all strokes."""
        self._strokes.clear()
        self._current_stroke.clear()
        self._pen_down = False
        self._pending_clear = False

    def _has_strokes(self) -> bool:
        """Return True if there are any completed or in-progress strokes."""
        return bool(self._strokes) or len(self._current_stroke) >= 2

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _to_pixel(self, nx: float, ny: float) -> Tuple[int, int]:
        """Convert normalized coords (origin bottom-left, Y-up) to pixel coords."""
        px = int(self._margin + nx * self._draw_size)
        py = int(self._margin + (1.0 - ny) * self._draw_size)
        return px, py

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self, canvas: np.ndarray) -> None:
        """Draw light grid lines at 0.25 intervals."""
        grid_color = (220, 220, 220)
        label_color = (180, 180, 180)
        m = self._margin
        d = self._draw_size

        # Draw border
        cv2.rectangle(canvas, (m, m), (m + d, m + d), (200, 200, 200), 1)

        for i in range(1, 4):
            frac = i * 0.25
            # Vertical line
            px = int(m + frac * d)
            cv2.line(canvas, (px, m), (px, m + d), grid_color, 1)
            cv2.putText(canvas, f"{frac:.2f}", (px - 12, m + d + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)
            # Horizontal line
            py = int(m + frac * d)
            cv2.line(canvas, (m, py), (m + d, py), grid_color, 1)
            # Y labels (inverted: top=1.0, bottom=0.0)
            cv2.putText(canvas, f"{1.0 - frac:.2f}", (m - 35, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)

        # Corner labels
        cv2.putText(canvas, "0.00", (m - 2, m + d + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)
        cv2.putText(canvas, "1.00", (m + d - 12, m + d + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)
        cv2.putText(canvas, "1.00", (m - 35, m + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)
        cv2.putText(canvas, "0.00", (m - 35, m + d + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)

    def _draw_polyline(self, canvas: np.ndarray, stroke: Stroke,
                       color: tuple, thickness: int = 2) -> None:
        """Draw a single stroke as a polyline."""
        if len(stroke) < 2:
            return
        pts = np.array([self._to_pixel(nx, ny) for nx, ny in stroke],
                       dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)

    def _draw_strokes(self, canvas: np.ndarray) -> None:
        """Draw all completed and in-progress strokes."""
        # Completed strokes in red
        for stroke in self._strokes:
            self._draw_polyline(canvas, stroke, (0, 0, 200))

        # Current (in-progress) stroke in bright red
        if self._pen_down and len(self._current_stroke) >= 2:
            self._draw_polyline(canvas, self._current_stroke, (0, 0, 255))

    def _draw_cursor(self, canvas: np.ndarray) -> None:
        """Draw crosshair cursor at current position."""
        if self._cursor == (0.0, 0.0) and not self._pen_down:
            return

        px, py = self._to_pixel(*self._cursor)
        arm = 8
        if self._pen_down:
            color = (0, 180, 0)  # green when drawing
        else:
            color = (160, 160, 160)  # gray when idle

        cv2.line(canvas, (px - arm, py), (px + arm, py), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (px, py - arm), (px, py + arm), color, 1, cv2.LINE_AA)

    def _draw_info(self, canvas: np.ndarray) -> None:
        """Draw status text."""
        total_strokes = len(self._strokes) + (1 if self._pen_down else 0)
        total_points = sum(len(s) for s in self._strokes) + len(self._current_stroke)

        # Pen state
        if self._pen_down:
            state_text = "PEN DOWN"
            state_color = (0, 160, 0)
        else:
            state_text = "PEN UP"
            state_color = (160, 160, 160)

        y_base = self._size - 12
        cv2.putText(canvas, state_text, (self._margin, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

        stats = f"strokes: {total_strokes}  pts: {total_points}"
        cv2.putText(canvas, stats, (self._margin + 130, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Cursor coordinates
        nx, ny = self._cursor
        coord_text = f"({nx:.2f}, {ny:.2f})"
        cv2.putText(canvas, coord_text, (self._size - 120, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Idle auto-clear countdown / pending indicator
        if (self._idle_clear > 0
                and not self._pen_down
                and self._has_strokes()):
            if self._pending_clear:
                cv2.putText(canvas, "pending clear",
                            (self._size - 150, y_base - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            else:
                remaining = self._idle_clear - (time.time() - self._last_write_time)
                if remaining > 0:
                    countdown = f"clear in {remaining:.0f}s"
                    cv2.putText(canvas, countdown,
                                (self._size - 130, y_base - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

        # Title
        cv2.putText(canvas, "Trajectory", (self._margin, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        cv2.putText(canvas, "[c] clear", (self._size - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
