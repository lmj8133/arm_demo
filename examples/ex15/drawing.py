"""
Drawing controller for Piper robotic arm on XY plane.

Provides a single `move(write, x, y)` API for table-surface drawing with pen control.
Uses IK + move_joint fire-and-forget pattern for smooth continuous motion.

Coordinate System (REP-103):
    +X -> Forward (arm front direction)
    +Y -> Left
    +Z -> Up

Example:
    >>> from piper_demo import PiperConnection, MotionController
    >>> from piper_demo.joint_reader import JointReader
    >>> from drawing import DrawingController, DrawingConfig
    >>>
    >>> with PiperConnection("can0") as conn:
    ...     conn.enable(go_home=True)
    ...     motion = MotionController(conn.piper)
    ...     reader = JointReader(conn.piper)
    ...     drawer = DrawingController(motion, reader)
    ...
    ...     drawer.move(False, 0.25, 0.0)   # travel to position
    ...     drawer.move(True, 0.30, 0.05)   # draw line to target
    ...     drawer.move(False, 0.25, 0.0)   # lift and travel back
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from piper_demo.motion import MotionController
from piper_demo.inverse_kinematics import inverse_kinematics, IKConfig
from piper_demo.kinematics import forward_kinematics
from piper_demo.joint_reader import JointReader

# Drawing pose joint angles (pen-down orientation)
DRAWING_JOINTS = [0, 2.02786, -0.57318, 0.13404, -1.19086, 0.01389]

# Derive fixed end-effector orientation from drawing pose via FK
_home_fk = forward_kinematics(DRAWING_JOINTS)
DRAW_ROLL = _home_fk.roll
DRAW_PITCH = _home_fk.pitch
DRAW_YAW = _home_fk.yaw

# IK solver configuration
IK_CFG = IKConfig(
    max_iterations=100,
    damping_factor=0.05,
    position_tolerance=1e-4,
    orientation_tolerance=1e-3,
)

# Home joint angles for safe shutdown (editable)
SAFE_HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass
class DrawingConfig:
    """Configuration for drawing operations."""

    # Drawing plane Z heights (meters)
    draw_z: float = 0.18  # Z height when pen touches surface
    safe_z: float = 0.21  # Z height when pen is lifted (draw_z + 0.03)

    # Motion parameters
    draw_speed: float = 0.3   # Speed factor for drawing (0-1)
    move_speed: float = 0.3   # Speed factor for travel moves
    interval: float = 0.01    # Fire-and-forget sleep interval (s)

    # Workspace limits (safety bounds in meters)
    x_min: float = 0.220
    x_max: float = 0.420
    y_min: float = -0.161
    y_max: float = 0.139

    # Interpolation
    max_step_length: float = 0.002  # Auto-interpolate steps longer than this (m)


class DrawingController:
    """
    High-level drawing controller for XY plane drawing.

    Uses a single `move(write, x, y)` method for all motion:
    - write=True:  pen at draw_z (drawing height)
    - write=False: pen at safe_z (travel height)
    """

    def __init__(
        self,
        motion: MotionController,
        reader: JointReader,
        config: Optional[DrawingConfig] = None,
    ):
        """
        Initialize the drawing controller.

        Args:
            motion: MotionController instance for arm movement.
            reader: JointReader for position feedback and IK chaining.
            config: Drawing configuration. Uses defaults if None.
        """
        self.motion = motion
        self.reader = reader
        self.config = config or DrawingConfig()

        # State (replaces DrawingState dataclass)
        self._x: float = 0.25
        self._y: float = 0.0
        self._writing: bool = False
        self._current_joints: Optional[List[float]] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def move(self, write: bool, x: float, y: float) -> bool:
        """
        Move to (x, y) with pen state controlled by `write`.

        Pure dispatcher — delegates to pen_up/pen_down/_travel_to/_draw_to.

        Args:
            write: True to draw (pen down), False to travel (pen up).
            x: Target X position (m).
            y: Target Y position (m).

        Returns:
            True if move completed successfully.
        """
        x, y = self._clamp_position(x, y)

        if self._writing and not write:
            # Draw -> travel: lift pen, then move at safe height
            if not self.pen_up():
                return False
            return self._travel_to(x, y)

        if not self._writing and write:
            # Travel -> draw: move to position, then lower pen
            if not self._travel_to(x, y):
                return False
            return self.pen_down()

        if self._writing:
            # Draw -> draw: interpolated line
            return self._draw_to(x, y)

        # Travel -> travel: direct move at safe height
        return self._travel_to(x, y)

    def pen_up(self) -> bool:
        """Lift pen at current position (idempotent)."""
        if not self._writing:
            return True
        success = self._move_to_xyz(
            self._x, self._y, self.config.safe_z,
            speed=self.config.move_speed,
        )
        if success:
            self._writing = False
        return success

    def pen_down(self) -> bool:
        """Lower pen at current position (idempotent)."""
        if self._writing:
            return True
        success = self._move_to_xyz(
            self._x, self._y, self.config.draw_z,
            speed=self.config.move_speed,
        )
        if success:
            self._writing = True
        return success

    def is_writing(self) -> bool:
        """Check if pen is currently down (drawing)."""
        return self._writing

    def get_position(self) -> Tuple[float, float]:
        """Get current XY position in meters."""
        return self._x, self._y

    def get_workspace_bounds(self) -> Tuple[float, float, float, float]:
        """Get workspace bounds (x_min, x_max, y_min, y_max) in meters."""
        return (
            self.config.x_min,
            self.config.x_max,
            self.config.y_min,
            self.config.y_max,
        )

    def get_workspace_size(self) -> Tuple[float, float]:
        """Get workspace size (width_x, height_y) in meters."""
        return (
            self.config.x_max - self.config.x_min,
            self.config.y_max - self.config.y_min,
        )

    def safe_disable(self, lift_height: float = 0.10) -> None:
        """Lift arm vertically and return to home for safe shutdown.

        Sequence:
            1. Pen up (if writing)
            2. Raise Z by lift_height (default 10cm)
            3. Move to SAFE_HOME_JOINTS zero position

        Args:
            lift_height: Vertical lift distance in meters (default: 0.10).
        """
        self.pen_up()
        lift_z = self.config.safe_z + lift_height
        self._move_to_xyz(
            self._x, self._y, lift_z,
            speed=self.config.move_speed,
            wait=True,
        )
        self.motion.move_joint(SAFE_HOME_JOINTS, speed_factor=self.config.move_speed)
        self.reader.wait_for_position(
            SAFE_HOME_JOINTS,
            tolerance_rad=0.035,
            timeout_sec=10.0,
        )

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _clamp_position(self, x: float, y: float) -> Tuple[float, float]:
        """Clamp position to workspace limits."""
        x = max(self.config.x_min, min(self.config.x_max, x))
        y = max(self.config.y_min, min(self.config.y_max, y))
        return x, y

    def _get_current_joints(self) -> List[float]:
        """Get current joint angles for IK initial guess."""
        if self._current_joints is not None:
            return self._current_joints
        return list(self.reader.read_joints().positions)

    def _move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        speed: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        Move to XYZ position via IK + move_joint.

        Args:
            x, y, z: Target position (m).
            speed: Speed factor (0-1).
            wait: If True, block until position reached.
                  If False, fire-and-forget with sleep interval.

        Returns:
            True if IK converged and move was issued.
        """
        if speed is None:
            speed = self.config.draw_speed if self._writing else self.config.move_speed

        current_joints = self._get_current_joints()

        result = inverse_kinematics(
            target_x=x,
            target_y=y,
            target_z=z,
            target_roll=DRAW_ROLL,
            target_pitch=DRAW_PITCH,
            target_yaw=DRAW_YAW,
            initial_guess=current_joints,
            config=IK_CFG,
        )

        if not result.converged:
            print(f"  [IK FAIL] target=({x:.3f}, {y:.3f}, {z:.3f}) "
                  f"err={result.position_error*1000:.2f}mm")
            return False

        self.motion.move_joint(result.joint_angles, speed_factor=speed)
        self._current_joints = result.joint_angles

        if wait:
            self.reader.wait_for_position(
                result.joint_angles,
                tolerance_rad=0.035,
                timeout_sec=10.0,
            )
        else:
            time.sleep(self.config.interval)

        self._x = x
        self._y = y
        return True

    def _travel_to(self, x: float, y: float) -> bool:
        """
        Travel to (x, y) at safe_z height, blocking until position reached.

        Args:
            x: Target X position (m), already clamped.
            y: Target Y position (m), already clamped.

        Returns:
            True if move completed successfully.
        """
        return self._move_to_xyz(
            x, y, self.config.safe_z,
            speed=self.config.move_speed,
            wait=True,
        )

    def _draw_to(self, x: float, y: float) -> bool:
        """
        Draw a straight line from current position to (x, y).

        Auto-interpolates when distance exceeds max_step_length to ensure
        the end-effector follows a straight Cartesian path.

        Args:
            x: Target X position (m), already clamped.
            y: Target Y position (m), already clamped.

        Returns:
            True if line completed successfully.
        """
        dx = x - self._x
        dy = y - self._y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= self.config.max_step_length:
            return self._move_to_xyz(x, y, self.config.draw_z, wait=False)

        # Auto-interpolate: split into ceil(dist / max_step) segments
        num_steps = math.ceil(dist / self.config.max_step_length)
        start_x, start_y = self._x, self._y
        for i in range(1, num_steps + 1):
            t = i / num_steps
            ix = start_x + dx * t
            iy = start_y + dy * t
            if not self._move_to_xyz(ix, iy, self.config.draw_z, wait=False):
                return False
        return True
