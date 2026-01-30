"""Joint state reading and monitoring.

Provides JointReader class for reading current joint positions
and continuous monitoring with callbacks.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from piper_sdk import C_PiperInterface_V2

from .utils import rad_to_deg, deg_to_rad, format_joint_state, format_end_pose


@dataclass
class EndPoseState:
    """End-effector pose data container.

    Attributes:
        x, y, z: Position in meters
        roll, pitch, yaw: Orientation in radians
        timestamp: Reading timestamp
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    timestamp: float

    def __str__(self) -> str:
        return format_end_pose(self)

    def position_mm(self) -> Tuple[float, float, float]:
        """Get position in millimeters."""
        return (self.x * 1000, self.y * 1000, self.z * 1000)

    def orientation_deg(self) -> Tuple[float, float, float]:
        """Get orientation in degrees."""
        return (rad_to_deg(self.roll), rad_to_deg(self.pitch), rad_to_deg(self.yaw))


@dataclass
class JointState:
    """Joint state data container.

    Attributes:
        positions: Joint positions in radians (6 joints)
        gripper: Gripper opening in meters
        timestamp: Reading timestamp
    """

    positions: List[float]
    gripper: float
    timestamp: float

    def __str__(self) -> str:
        return format_joint_state(self.positions)

    def positions_deg(self) -> List[float]:
        """Get joint positions in degrees."""
        return [rad_to_deg(p) for p in self.positions]


class JointReader:
    """Read joint states from Piper arm.

    Example:
        reader = JointReader(piper)
        state = reader.read_joints()
        print(f"Joint 1: {state.positions_deg()[0]:.2f}°")

        # Continuous monitoring
        def callback(state):
            print(state)
        reader.monitor(callback, rate_hz=10, duration_sec=5.0)
    """

    def __init__(self, piper: C_PiperInterface_V2):
        """Initialize JointReader.

        Args:
            piper: Connected C_PiperInterface_V2 instance
        """
        self.piper = piper
        self._last_state: Optional[JointState] = None

    # SDK returns 0.001 degrees (milli-degrees), convert to radians
    MILLIDEG_TO_RAD = 3.14159265358979 / 180000

    def read_joints(self) -> JointState:
        """Read current joint state.

        Returns:
            JointState with current positions

        Note:
            First reading after connection may contain default values (0).
            Arm must be in slave mode (0xFC) to receive feedback.
        """
        joint_msg = self.piper.GetArmJointMsgs()

        # Extract positions from SDK message
        # SDK returns joint angles in 0.001 degrees (milli-degrees), convert to radians
        positions = [
            joint_msg.joint_state.joint_1 * self.MILLIDEG_TO_RAD,
            joint_msg.joint_state.joint_2 * self.MILLIDEG_TO_RAD,
            joint_msg.joint_state.joint_3 * self.MILLIDEG_TO_RAD,
            joint_msg.joint_state.joint_4 * self.MILLIDEG_TO_RAD,
            joint_msg.joint_state.joint_5 * self.MILLIDEG_TO_RAD,
            joint_msg.joint_state.joint_6 * self.MILLIDEG_TO_RAD,
        ]

        # Get gripper state
        # grippers_angle is in 0.001mm units, convert to meters
        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper = gripper_msg.gripper_state.grippers_angle / 1000000.0

        state = JointState(
            positions=positions,
            gripper=gripper,
            timestamp=time.time(),
        )
        self._last_state = state
        return state

    def monitor(
        self,
        callback: Callable[[JointState], None],
        rate_hz: float = 10.0,
        duration_sec: Optional[float] = None,
        stop_condition: Optional[Callable[[JointState], bool]] = None,
    ) -> None:
        """Continuously monitor joint states.

        Args:
            callback: Function called with each new JointState
            rate_hz: Monitoring frequency in Hz (default: 10)
            duration_sec: Stop after this duration (None = run until stop_condition)
            stop_condition: Function returning True to stop monitoring

        Example:
            def on_joint_update(state):
                print(f"J1: {state.positions_deg()[0]:.2f}°")

            # Monitor for 5 seconds
            reader.monitor(on_joint_update, rate_hz=10, duration_sec=5.0)

            # Monitor until joint 1 reaches 30°
            reader.monitor(
                on_joint_update,
                stop_condition=lambda s: s.positions_deg()[0] > 30
            )
        """
        period = 1.0 / rate_hz
        start_time = time.time()

        try:
            while True:
                loop_start = time.time()

                # Check duration limit
                if duration_sec and (loop_start - start_time) >= duration_sec:
                    break

                # Read and callback
                state = self.read_joints()
                callback(state)

                # Check stop condition
                if stop_condition and stop_condition(state):
                    break

                # Maintain loop rate
                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)

        except KeyboardInterrupt:
            pass  # Allow clean exit on Ctrl+C

    @property
    def last_state(self) -> Optional[JointState]:
        """Get the last read joint state."""
        return self._last_state

    def read_end_pose(self) -> EndPoseState:
        """Read current end-effector pose.

        Returns:
            EndPoseState with current position and orientation

        Note:
            First reading after connection may contain default values (0).
            Arm must be in slave mode (0xFC) to receive feedback.
        """
        pose_msg = self.piper.GetArmEndPoseMsgs()

        # SDK returns: X/Y/Z in 0.001mm, RX/RY/RZ in 0.001 degrees
        # Convert to meters and radians
        return EndPoseState(
            x=pose_msg.end_pose.X_axis / 1_000_000,  # 0.001mm -> m
            y=pose_msg.end_pose.Y_axis / 1_000_000,
            z=pose_msg.end_pose.Z_axis / 1_000_000,
            roll=deg_to_rad(pose_msg.end_pose.RX_axis / 1000),  # 0.001deg -> rad
            pitch=deg_to_rad(pose_msg.end_pose.RY_axis / 1000),
            yaw=deg_to_rad(pose_msg.end_pose.RZ_axis / 1000),
            timestamp=time.time(),
        )

    def wait_for_position(
        self,
        target_positions: List[float],
        tolerance_rad: float = 0.05,
        timeout_sec: float = 30.0,
        rate_hz: float = 10.0,
    ) -> bool:
        """Wait until arm reaches target position.

        Args:
            target_positions: Target positions in radians (6 joints)
            tolerance_rad: Position tolerance in radians (default: ~3°)
            timeout_sec: Timeout in seconds
            rate_hz: Check frequency

        Returns:
            True if position reached within timeout
        """
        start_time = time.time()
        period = 1.0 / rate_hz

        while time.time() - start_time < timeout_sec:
            state = self.read_joints()

            # Check if all joints are within tolerance
            all_reached = True
            for i, (current, target) in enumerate(
                zip(state.positions, target_positions)
            ):
                if abs(current - target) > tolerance_rad:
                    all_reached = False
                    break

            if all_reached:
                return True

            time.sleep(period)

        return False

    def wait_for_pose(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        target_roll: float = 0.0,
        target_pitch: float = 0.0,
        target_yaw: float = 0.0,
        tolerance_m: float = 0.005,
        tolerance_rad: float = 0.05,
        timeout_sec: float = 30.0,
        rate_hz: float = 10.0,
    ) -> bool:
        """Wait until end-effector reaches target pose.

        Args:
            target_x, target_y, target_z: Target position in meters
            target_roll, target_pitch, target_yaw: Target orientation in radians
            tolerance_m: Position tolerance in meters (default: 5mm)
            tolerance_rad: Orientation tolerance in radians (default: ~3°)
            timeout_sec: Timeout in seconds
            rate_hz: Check frequency

        Returns:
            True if pose reached within timeout
        """
        import math

        start_time = time.time()
        period = 1.0 / rate_hz

        while time.time() - start_time < timeout_sec:
            pose = self.read_end_pose()

            # Check position
            pos_error = math.sqrt(
                (pose.x - target_x) ** 2
                + (pose.y - target_y) ** 2
                + (pose.z - target_z) ** 2
            )
            if pos_error > tolerance_m:
                time.sleep(period)
                continue

            # Check orientation
            orient_ok = (
                abs(pose.roll - target_roll) <= tolerance_rad
                and abs(pose.pitch - target_pitch) <= tolerance_rad
                and abs(pose.yaw - target_yaw) <= tolerance_rad
            )
            if orient_ok:
                return True

            time.sleep(period)

        return False
