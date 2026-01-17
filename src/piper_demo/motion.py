"""Motion control for Piper arm.

Provides MotionController class for joint and Cartesian motion commands.
"""

import time
from typing import Callable, List, Optional, Tuple

from piper_sdk import C_PiperInterface_V2

from .utils import (
    clamp_joint_position,
    validate_joint_positions,
    deg_to_rad,
    rad_to_deg,
    JOINT_LIMITS_RAD,
)


class MotionError(Exception):
    """Exception raised for motion-related errors."""
    pass


class MotionController:
    """Control arm motion with safety limits.

    Example:
        motion = MotionController(piper)
        motion.move_to_home()
        motion.move_joint([0.1, 0.5, -0.3, 0.0, 0.0, 0.0])
    """

    # Home position (all joints at 0)
    HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Default speed factor (0.0 to 1.0)
    DEFAULT_SPEED_FACTOR = 0.3

    def __init__(
        self,
        piper: C_PiperInterface_V2,
        speed_factor: float = DEFAULT_SPEED_FACTOR,
        enforce_limits: bool = True,
    ):
        """Initialize MotionController.

        Args:
            piper: Connected C_PiperInterface_V2 instance
            speed_factor: Speed multiplier 0.0-1.0 (default: 0.3)
            enforce_limits: Clamp positions to joint limits (default: True)
        """
        self.piper = piper
        self.speed_factor = max(0.0, min(1.0, speed_factor))
        self.enforce_limits = enforce_limits
        self._is_moving = False

    # Conversion factor: radians to 0.001 degrees (milli-degrees)
    # factor = 1000 * 180 / pi ≈ 57295.7795
    RAD_TO_MILLIDEG = 57295.7795

    def move_joint(
        self,
        positions: List[float],
        speed_factor: Optional[float] = None,
        wait: bool = False,
    ) -> None:
        """Move arm to target joint positions.

        Args:
            positions: Target positions in radians (6 joints)
            speed_factor: Override default speed factor
            wait: Block until motion complete (not implemented yet)

        Raises:
            MotionError: If positions invalid and enforce_limits=False
        """
        if len(positions) != 6:
            raise MotionError(f"Expected 6 joint positions, got {len(positions)}")

        # Validate or clamp positions
        if self.enforce_limits:
            positions = [
                clamp_joint_position(i, pos)
                for i, pos in enumerate(positions)
            ]
        else:
            is_valid, error_msg = validate_joint_positions(positions)
            if not is_valid:
                raise MotionError(error_msg)

        # Calculate speed (SDK uses 0-100 range internally)
        speed = speed_factor if speed_factor is not None else self.speed_factor
        speed_value = int(speed * 100)

        # Convert radians to SDK units (0.001 degrees / milli-degrees)
        joints = [round(pos * self.RAD_TO_MILLIDEG) for pos in positions]

        # Send motion command
        self._is_moving = True
        # MotionCtrl_2(mode, coord_type, speed, reserved)
        # coord_type: 0x00 = Cartesian, 0x01 = Joint
        self.piper.MotionCtrl_2(0x01, 0x01, speed_value, 0x00)
        self.piper.JointCtrl(
            joints[0], joints[1], joints[2],
            joints[3], joints[4], joints[5],
        )

    def move_joint_deg(
        self,
        positions_deg: List[float],
        speed_factor: Optional[float] = None,
    ) -> None:
        """Move arm to target joint positions in degrees.

        Args:
            positions_deg: Target positions in degrees (6 joints)
            speed_factor: Override default speed factor
        """
        positions_rad = [deg_to_rad(p) for p in positions_deg]
        self.move_joint(positions_rad, speed_factor)

    def move_single_joint(
        self,
        joint_index: int,
        position: float,
        speed_factor: Optional[float] = None,
    ) -> None:
        """Move a single joint to target position.

        Args:
            joint_index: Joint index (0-5)
            position: Target position in radians
            speed_factor: Override default speed factor

        Note:
            Other joints will hold their current command positions.
        """
        if joint_index < 0 or joint_index >= 6:
            raise MotionError(f"Invalid joint index: {joint_index}")

        # Get current joint command (or use zeros as baseline)
        positions = self.HOME_POSITION.copy()
        positions[joint_index] = position
        self.move_joint(positions, speed_factor)

    def move_to_home(self, speed_factor: Optional[float] = None) -> None:
        """Move arm to home position (all joints at 0).

        Args:
            speed_factor: Override default speed factor
        """
        # Use slower speed for homing
        home_speed = speed_factor if speed_factor is not None else 0.2
        self.move_joint(self.HOME_POSITION, home_speed)

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        speed_factor: Optional[float] = None,
    ) -> None:
        """Move end-effector to Cartesian position.

        Args:
            x, y, z: Position in meters
            roll, pitch, yaw: Orientation in radians
            speed_factor: Override default speed factor

        Note:
            Coordinate frame is base-relative.
            Not all positions may be reachable.
            Uses SDK units: position in 0.001mm, angles in 0.001 degrees.
        """
        speed = speed_factor if speed_factor is not None else self.speed_factor
        speed_value = int(speed * 100)

        # Convert to SDK units (0.001mm for position, 0.001 degrees for angles)
        # meters -> mm -> 0.001mm: multiply by 1000 * 1000 = 1000000
        x_001mm = int(x * 1000000)
        y_001mm = int(y * 1000000)
        z_001mm = int(z * 1000000)
        # radians -> degrees -> 0.001 degrees: multiply by (180/pi) * 1000
        roll_001deg = int(rad_to_deg(roll) * 1000)
        pitch_001deg = int(rad_to_deg(pitch) * 1000)
        yaw_001deg = int(rad_to_deg(yaw) * 1000)

        self._is_moving = True
        # Must call MotionCtrl_2 before EndPoseCtrl
        # MotionCtrl_2(mode, coord_type, speed, reserved)
        # coord_type: 0x00 = Cartesian, 0x01 = Joint
        self.piper.MotionCtrl_2(0x01, 0x00, speed_value, 0x00)
        self.piper.EndPoseCtrl(
            x_001mm, y_001mm, z_001mm,
            roll_001deg, pitch_001deg, yaw_001deg,
        )

    def move_cartesian_continuous(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        tolerance_m: float = 0.005,
        tolerance_rad: float = 0.05,
        timeout_sec: float = 10.0,
        settle_sec: float = 0.5,
        rate_hz: float = 100.0,
        speed_factor: Optional[float] = None,
        pose_callback: Optional[Callable] = None,
    ) -> bool:
        """Move to Cartesian position with continuous control and feedback.

        Continuously sends motion commands until target is reached or timeout.
        After reaching target, continues sending commands for settle_sec to
        maintain position and let the arm stabilize.

        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            tolerance_m: Position tolerance in meters (default: 5mm)
            tolerance_rad: Orientation tolerance in radians (default: ~3°)
            timeout_sec: Timeout in seconds (default: 10s)
            settle_sec: Time to hold position after reaching target (default: 0.5s)
            rate_hz: Control loop frequency (default: 100Hz = 10ms)
            speed_factor: Override default speed factor
            pose_callback: Optional callback(pose) called each iteration

        Returns:
            True if target reached within timeout, False otherwise

        Example:
            def on_pose(pose):
                print(f"Current: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")

            reached = motion.move_cartesian_continuous(
                0.25, 0.0, 0.2,
                settle_sec=1.0,  # Hold for 1 second after reaching target
                pose_callback=on_pose
            )
        """
        import math
        from .joint_reader import JointReader, EndPoseState

        speed = speed_factor if speed_factor is not None else self.speed_factor
        speed_value = int(speed * 100)

        # Convert to SDK units
        x_001mm = int(x * 1000000)
        y_001mm = int(y * 1000000)
        z_001mm = int(z * 1000000)
        roll_001deg = int(rad_to_deg(roll) * 1000)
        pitch_001deg = int(rad_to_deg(pitch) * 1000)
        yaw_001deg = int(rad_to_deg(yaw) * 1000)

        reader = JointReader(self.piper)
        start_time = time.time()
        period = 1.0 / rate_hz
        self._is_moving = True
        reached_time = None  # Track when we first reached target

        try:
            while time.time() - start_time < timeout_sec:
                loop_start = time.time()

                # Send motion command continuously (SDK requirement)
                self.piper.MotionCtrl_2(0x01, 0x00, speed_value, 0x00)
                self.piper.EndPoseCtrl(
                    x_001mm, y_001mm, z_001mm,
                    roll_001deg, pitch_001deg, yaw_001deg,
                )

                # Read current pose
                pose = reader.read_end_pose()

                if pose_callback:
                    pose_callback(pose)

                # Check position error
                pos_error = math.sqrt(
                    (pose.x - x) ** 2 +
                    (pose.y - y) ** 2 +
                    (pose.z - z) ** 2
                )

                # Check orientation error
                orient_ok = (
                    abs(pose.roll - roll) <= tolerance_rad and
                    abs(pose.pitch - pitch) <= tolerance_rad and
                    abs(pose.yaw - yaw) <= tolerance_rad
                )

                if pos_error <= tolerance_m and orient_ok:
                    if reached_time is None:
                        reached_time = time.time()
                    # Continue sending commands for settle_sec to stabilize
                    elif time.time() - reached_time >= settle_sec:
                        self._is_moving = False
                        return True
                else:
                    # Reset if we drifted out of tolerance
                    reached_time = None

                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)

        except KeyboardInterrupt:
            pass

        self._is_moving = False
        return False

    def move_linear(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        speed_factor: Optional[float] = None,
    ) -> None:
        """Move end-effector in straight line (MoveL mode).

        Unlike move_cartesian (point-to-point), this ensures the
        end-effector follows a linear path in Cartesian space.

        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            speed_factor: Override default speed factor

        Note:
            Uses coord_type=0x02 for linear interpolation.
            The arm will move in a straight line from current to target.
        """
        speed = speed_factor if speed_factor is not None else self.speed_factor
        speed_value = int(speed * 100)

        # Convert to SDK units (0.001mm for position, 0.001 degrees for angles)
        x_001mm = int(x * 1000000)
        y_001mm = int(y * 1000000)
        z_001mm = int(z * 1000000)
        roll_001deg = int(rad_to_deg(roll) * 1000)
        pitch_001deg = int(rad_to_deg(pitch) * 1000)
        yaw_001deg = int(rad_to_deg(yaw) * 1000)

        self._is_moving = True
        # MotionCtrl_2(mode, coord_type, speed, reserved)
        # coord_type: 0x00 = Cartesian P2P, 0x01 = Joint, 0x02 = MoveL
        self.piper.MotionCtrl_2(0x01, 0x02, speed_value, 0x00)
        self.piper.EndPoseCtrl(
            x_001mm, y_001mm, z_001mm,
            roll_001deg, pitch_001deg, yaw_001deg,
        )

    def move_linear_continuous(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        tolerance_m: float = 0.005,
        tolerance_rad: float = 0.05,
        timeout_sec: float = 10.0,
        settle_sec: float = 0.5,
        rate_hz: float = 100.0,
        speed_factor: Optional[float] = None,
        pose_callback: Optional[Callable] = None,
    ) -> bool:
        """Move linear with continuous control until target reached.

        Similar to move_cartesian_continuous but uses MoveL mode (coord_type=0x02)
        to ensure the end-effector follows a straight-line path.

        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            tolerance_m: Position tolerance in meters (default: 5mm)
            tolerance_rad: Orientation tolerance in radians (default: ~3°)
            timeout_sec: Timeout in seconds (default: 10s)
            settle_sec: Time to hold position after reaching target (default: 0.5s)
            rate_hz: Control loop frequency (default: 100Hz)
            speed_factor: Override default speed factor
            pose_callback: Optional callback(pose) called each iteration

        Returns:
            True if target reached within timeout, False otherwise

        Example:
            reached = motion.move_linear_continuous(
                0.25, 0.0, 0.2,
                settle_sec=1.0,
            )
        """
        import math
        from .joint_reader import JointReader

        speed = speed_factor if speed_factor is not None else self.speed_factor
        speed_value = int(speed * 100)

        # Convert to SDK units
        x_001mm = int(x * 1000000)
        y_001mm = int(y * 1000000)
        z_001mm = int(z * 1000000)
        roll_001deg = int(rad_to_deg(roll) * 1000)
        pitch_001deg = int(rad_to_deg(pitch) * 1000)
        yaw_001deg = int(rad_to_deg(yaw) * 1000)

        reader = JointReader(self.piper)
        start_time = time.time()
        period = 1.0 / rate_hz
        self._is_moving = True
        reached_time = None

        try:
            while time.time() - start_time < timeout_sec:
                loop_start = time.time()

                # Send MoveL command continuously (coord_type=0x02)
                self.piper.MotionCtrl_2(0x01, 0x02, speed_value, 0x00)
                self.piper.EndPoseCtrl(
                    x_001mm, y_001mm, z_001mm,
                    roll_001deg, pitch_001deg, yaw_001deg,
                )

                # Read current pose
                pose = reader.read_end_pose()

                if pose_callback:
                    pose_callback(pose)

                # Check position error
                pos_error = math.sqrt(
                    (pose.x - x) ** 2 +
                    (pose.y - y) ** 2 +
                    (pose.z - z) ** 2
                )

                # Check orientation error
                orient_ok = (
                    abs(pose.roll - roll) <= tolerance_rad and
                    abs(pose.pitch - pitch) <= tolerance_rad and
                    abs(pose.yaw - yaw) <= tolerance_rad
                )

                if pos_error <= tolerance_m and orient_ok:
                    if reached_time is None:
                        reached_time = time.time()
                    elif time.time() - reached_time >= settle_sec:
                        self._is_moving = False
                        return True
                else:
                    reached_time = None

                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)

        except KeyboardInterrupt:
            pass

        self._is_moving = False
        return False

    def stop(self) -> None:
        """Emergency stop - halt all motion immediately."""
        self._is_moving = False
        # Send stop command with zero speed
        # MotionCtrl_2(mode, coord_type, speed, reserved)
        self.piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)

    def set_speed_factor(self, factor: float) -> None:
        """Set default speed factor.

        Args:
            factor: Speed multiplier 0.0-1.0
        """
        self.speed_factor = max(0.0, min(1.0, factor))

    @property
    def is_moving(self) -> bool:
        """Check if arm is currently executing motion."""
        return self._is_moving

    def get_joint_limits(self) -> List[Tuple[float, float]]:
        """Get joint limits in radians.

        Returns:
            List of (min, max) tuples for each joint
        """
        return JOINT_LIMITS_RAD.copy()

    def get_joint_limits_deg(self) -> List[Tuple[float, float]]:
        """Get joint limits in degrees.

        Returns:
            List of (min, max) tuples for each joint in degrees
        """
        return [
            (rad_to_deg(min_v), rad_to_deg(max_v))
            for min_v, max_v in JOINT_LIMITS_RAD
        ]
