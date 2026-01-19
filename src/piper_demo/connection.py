"""Piper arm connection management.

Provides PiperConnection class for managing CAN bus connection,
slave mode configuration, and arm enable/disable.
"""

import time
from typing import Optional
from contextlib import contextmanager

from piper_sdk import C_PiperInterface_V2

from .utils import check_can_interface


class PiperConnectionError(Exception):
    """Exception raised for connection-related errors."""
    pass


class PiperConnection:
    """Context manager for Piper arm connection.

    Handles CAN connection, slave mode setup, and cleanup.
    Uses C_PiperInterface_V2 for gripper control compatibility.

    Example:
        with PiperConnection(can_name="can0") as conn:
            print(conn.piper.GetArmJointMsgs())

    Attributes:
        piper: The underlying C_PiperInterface_V2 instance
        can_name: CAN interface name
        is_connected: Whether currently connected
    """

    # Slave mode command for reading feedback
    SLAVE_MODE_CMD = 0xFC

    # Zero position: all joints at 0 radians (WARNING: near singularity, pitch~85°)
    ZERO_HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Home position: official safe pose with pitch=0° for reliable Cartesian control
    # Cartesian: (150, -50, 150) mm, orientation: roll=-179.9°, pitch=0°, yaw=-179.9°
    #HOME_POSITION = [-0.31936, 1.32373, -0.43202, 0.0, 0.76541, -0.31948]
    # for demo
    # HOME_POSITION = [1.62684, -0.00482, -0.00401, 0.02339, 0.12292, -0.73655]
    # for demo-center (Y=8.9mm, optimal height for maximizing XZ reach)
    #HOME_POSITION = [1.59689, 0.2561, -0.84699, 0.05036, 0.70714, -0.75259]
    #HOME_POSITION = [0.01726, -0.04299, 0.04384, 0.0403, 0.34798, -0.75709]
    #HOME_POSITION = [-0.63137, 1.39531, -0.01832, -0.73327, -1.09347, -0.52215]
    #HOME_POSITION = [-0.0278, 0.30056, -0.83139, -0.03681, 0.5104, -0.69689]
    HOME_POSITION = [-0.05434, 0.13097, -0.89813, 0.01238, 0.74649, -0.73524]


    # Safe home position: resting pose for disable
    # Cartesian: (56.13, 0, 213.27) mm, orientation: roll=0°, pitch=87°, yaw=0°
    SAFE_HOME_POSITION = [0.01726, -0.04299, 0.04384, 0.0403, 0.34798, -0.75709]

    # Conversion factor: radians to milli-degrees (1000 * 180 / pi)
    RAD_TO_MILLIDEG = 57295.7795

    def __init__(
        self,
        can_name: str = "can0",
        auto_slave_mode: bool = False,
        verify_can: bool = True,
    ):
        """Initialize PiperConnection.

        Args:
            can_name: CAN interface name (default: "can0")
            auto_slave_mode: Automatically set slave mode on connect (default: False).
                             Note: Slave mode may interfere with motion control commands.
                             Set to True only when reading feedback without motion control.
            verify_can: Verify CAN interface before connecting (default: True)
        """
        self.can_name = can_name
        self.auto_slave_mode = auto_slave_mode
        self.verify_can = verify_can

        self.piper: Optional[C_PiperInterface_V2] = None
        self.is_connected = False
        self._enabled = False

    def connect(self) -> "PiperConnection":
        """Establish connection to the Piper arm.

        Returns:
            self for method chaining

        Raises:
            PiperConnectionError: If CAN interface not available or connection fails
        """
        if self.is_connected:
            return self

        # Verify CAN interface is active
        if self.verify_can and not check_can_interface(self.can_name):
            raise PiperConnectionError(
                f"CAN interface '{self.can_name}' is not active. "
                f"Run: bash scripts/can_activate.sh {self.can_name} 1000000"
            )

        try:
            self.piper = C_PiperInterface_V2(self.can_name)
            self.piper.ConnectPort()
            self.is_connected = True

            # Set slave mode to enable feedback
            if self.auto_slave_mode:
                self.set_slave_mode()

            # Small delay for first CAN messages
            time.sleep(0.1)

        except Exception as e:
            self.is_connected = False
            raise PiperConnectionError(f"Failed to connect to Piper arm: {e}") from e

        return self

    def disconnect(self) -> None:
        """Disconnect from the Piper arm.

        Uses safe_disable() to return arm to home position before
        disabling motors, preventing gravity-induced falls.
        """
        if self._enabled:
            try:
                self.safe_disable(return_home=True)
            except Exception:
                # Fallback: direct disable if safe_disable fails
                try:
                    self.piper.DisableArm(7)
                except Exception:
                    pass

        self.is_connected = False
        self.piper = None

    def set_slave_mode(self) -> None:
        """Set arm to slave mode for reading feedback.

        Must be called before reading joint positions.
        """
        if not self.is_connected or self.piper is None:
            raise PiperConnectionError("Not connected to Piper arm")

        self.piper.MasterSlaveConfig(self.SLAVE_MODE_CMD, 0, 0, 0)

    def get_ctrl_mode(self) -> int:
        """Get current control mode.

        Returns:
            ctrl_mode: 0x00=Standby, 0x01=CAN_CTRL, 0x02=TEACHING_MODE
        """
        if not self.is_connected or self.piper is None:
            raise PiperConnectionError("Not connected to Piper arm")

        status = self.piper.GetArmStatus()
        ctrl_mode = status.arm_status.ctrl_mode
        # Handle enum type from SDK
        return ctrl_mode.value if hasattr(ctrl_mode, 'value') else int(ctrl_mode)

    def is_in_teaching_mode(self) -> bool:
        """Check if arm is currently in teaching mode."""
        try:
            return self.get_ctrl_mode() == 0x02
        except Exception:
            return False

    def _is_any_motor_enabled(self) -> bool:
        """Check if any motor is currently enabled."""
        try:
            status = self.piper.GetArmEnableStatus()
            return any(status)
        except Exception:
            return False

    def reset(self, max_retries: int = 3, delay_sec: float = 0.3) -> None:
        """Reset arm from teaching/MIT mode to position-velocity mode.

        Auto-detects current state:
        - If in teaching mode or motors enabled: DisableArm first
        - If fresh boot (standby, disabled): skip DisableArm

        Args:
            max_retries: Maximum number of reset attempts (default: 3)
            delay_sec: Delay between commands for state transition (default: 0.3)
        """
        if not self.is_connected or self.piper is None:
            raise PiperConnectionError("Not connected to Piper arm")

        for _attempt in range(max_retries):
            # Step 1: Only disable if in teaching mode or motors are enabled
            need_disable = self.is_in_teaching_mode() or self._is_any_motor_enabled()
            if need_disable:
                self.piper.DisableArm(7)
                time.sleep(delay_sec)

            # Step 2: Reset/Restore (use simple reset, not E-stop resume)
            self.piper.MotionCtrl_1(0x02, 0, 0)
            time.sleep(delay_sec)

            # Step 3: Switch to CAN control mode (MIT mode off)
            self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            time.sleep(delay_sec)

            # Verify we exited teaching mode
            if not self.is_in_teaching_mode():
                return  # Success

        # Final attempt without verification (best effort)
        if self.is_in_teaching_mode() or self._is_any_motor_enabled():
            self.piper.DisableArm(7)
            time.sleep(delay_sec)
        self.piper.MotionCtrl_1(0x02, 0, 0)
        time.sleep(delay_sec)
        self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        time.sleep(delay_sec)

    def set_control_mode(
        self,
        move_mode: int = 0x01,
        speed_percent: int = 30,
    ) -> None:
        """Switch arm to CAN control mode with specified move mode.

        Must be called after enable() to allow motion commands to work.

        Args:
            move_mode: Motion mode (0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L)
            speed_percent: Speed 0-100 (default: 30)
        """
        if not self.is_connected or self.piper is None:
            raise PiperConnectionError("Not connected to Piper arm")

        # ctrl_mode=0x01 (CAN_CTRL), move_mode, speed, mit_mode=0x00
        self.piper.MotionCtrl_2(0x01, move_mode, speed_percent, 0x00)
        time.sleep(0.1)

    def _move_to_home_and_wait(
        self,
        speed: int = 20,
        timeout_sec: float = 15.0,
        tolerance_millideg: int = 3000,
    ) -> bool:
        """Move to home position and wait for completion.

        Args:
            speed: Speed percentage (0-100)
            timeout_sec: Timeout for motion
            tolerance_millideg: Position tolerance in 0.001 degrees

        Returns:
            True if home position reached within timeout
        """
        # Send home command
        targets = [round(p * self.RAD_TO_MILLIDEG) for p in self.HOME_POSITION]
        self.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # CAN ctrl, MOVE_J
        self.piper.JointCtrl(*targets)

        # Wait for position with feedback
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            joint_msg = self.piper.GetArmJointMsgs()
            positions = [
                joint_msg.joint_state.joint_1,
                joint_msg.joint_state.joint_2,
                joint_msg.joint_state.joint_3,
                joint_msg.joint_state.joint_4,
                joint_msg.joint_state.joint_5,
                joint_msg.joint_state.joint_6,
            ]

            if all(abs(pos - tgt) <= tolerance_millideg for pos, tgt in zip(positions, targets)):
                return True

            time.sleep(0.1)

        return False

    def _move_to_safe_home_and_wait(
        self,
        speed: int = 20,
        timeout_sec: float = 15.0,
        tolerance_millideg: int = 3000,
    ) -> bool:
        """Move to safe home position and wait for completion.

        Args:
            speed: Speed percentage (0-100)
            timeout_sec: Timeout for motion
            tolerance_millideg: Position tolerance in 0.001 degrees

        Returns:
            True if safe home position reached within timeout
        """
        # Send safe home command
        joints = [round(p * self.RAD_TO_MILLIDEG) for p in self.SAFE_HOME_POSITION]
        self.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # CAN ctrl, MOVE_J
        self.piper.JointCtrl(*joints)

        # Wait for position with feedback
        targets = [round(p * self.RAD_TO_MILLIDEG) for p in self.SAFE_HOME_POSITION]
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            joint_msg = self.piper.GetArmJointMsgs()
            positions = [
                joint_msg.joint_state.joint_1,
                joint_msg.joint_state.joint_2,
                joint_msg.joint_state.joint_3,
                joint_msg.joint_state.joint_4,
                joint_msg.joint_state.joint_5,
                joint_msg.joint_state.joint_6,
            ]

            if all(abs(pos - tgt) <= tolerance_millideg for pos, tgt in zip(positions, targets)):
                return True

            time.sleep(0.1)

        return False

    def enable(
        self,
        timeout_sec: float = 5.0,
        force: bool = False,
        auto_reset: bool = True,
        auto_control_mode: bool = True,
        move_mode: int = 0x01,
        speed_percent: int = 30,
        go_home: bool = True,
        home_settle_sec: float = 1.0,
        home_speed: int = 20,
        home_timeout_sec: float = 15.0,
    ) -> bool:
        """Enable the arm motors using V2 EnablePiper.

        By default, this method will:
        1. Reset arm to ensure controllable state
        2. Call EnablePiper to enable motors
        3. Auto-switch to CAN control mode for motion commands
        4. Move to home position (all joints at 0)

        Args:
            timeout_sec: Timeout for enable operation
            force: Force enable even if already enabled (default: False)
            auto_reset: Reset arm before enabling (default: True)
            auto_control_mode: Auto-switch to CAN control mode (default: True)
            move_mode: Default move mode for auto_control_mode
                       (0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L, default: MOVE_J)
            speed_percent: Default speed for auto_control_mode (default: 30)
            go_home: Move to home position after enabling (default: True)
            home_settle_sec: Settle time after homing motion (default: 1.0)
            home_speed: Speed percentage for homing motion (default: 20)
            home_timeout_sec: Timeout for homing motion (default: 15.0)

        Returns:
            True if enabled successfully

        Raises:
            PiperConnectionError: If not connected or enable fails
        """
        if not self.is_connected or self.piper is None:
            raise PiperConnectionError("Not connected to Piper arm")

        # Auto-reset to ensure arm is in a controllable state
        if auto_reset:
            self.reset()

        # Check if already enabled to avoid unnecessary re-enable
        # SDK's EnablePiper() sends EnableArm(7) every call which may reset state
        if not force:
            enable_status = self.piper.GetArmEnableStatus()
            if all(enable_status):
                self._enabled = True
                # Still need to set control mode if requested
                if auto_control_mode:
                    self.set_control_mode(move_mode, speed_percent)
                # Move to home if requested (default: True)
                if go_home:
                    self._move_to_home_and_wait(home_speed, home_timeout_sec)
                    time.sleep(home_settle_sec)
                return True

        # V2 API: poll EnablePiper() until it returns True
        start_time = time.time()
        while not self.piper.EnablePiper():
            if time.time() - start_time > timeout_sec:
                raise PiperConnectionError(
                    f"Failed to enable arm within {timeout_sec}s"
                )
            time.sleep(0.01)

        # Switch to CAN control mode
        if auto_control_mode:
            self.set_control_mode(move_mode, speed_percent)

        # Move to home if requested (default: True)
        if go_home:
            self._move_to_home_and_wait(home_speed, home_timeout_sec)
            time.sleep(home_settle_sec)

        self._enabled = True
        return True

    def disable(self) -> None:
        """Disable the arm motors.

        Warning: This immediately disables motors, which may cause the arm
        to fall due to gravity. Use safe_disable() for controlled shutdown.
        """
        if not self.is_connected or self.piper is None:
            raise PiperConnectionError("Not connected to Piper arm")

        self.piper.DisableArm(7)  # Disable all joints
        self._enabled = False

    def safe_disable(
        self,
        return_home: bool = True,
        home_speed: int = 20,
        timeout_sec: float = 15.0,
        settle_sec: float = 1.0,
        pre_home_delay: float = 3.0,
    ) -> None:
        """Safely disable arm by returning to safe home position first.

        This prevents the arm from falling due to gravity when motors
        are disabled. The sequence is:
        1. Wait for pre_home_delay (let arm stabilize after motion)
        2. Move to safe home position (SAFE_HOME_POSITION) with position detection
        3. Wait for settle time after reaching position
        4. Switch to standby mode (holds position)
        5. Disable motors

        Args:
            return_home: Move to home position before disable (default: True)
            home_speed: Speed percentage for homing motion (default: 20)
            timeout_sec: Timeout for homing motion (default: 15.0)
            settle_sec: Settle time after reaching safe home (default: 1.0)
            pre_home_delay: Delay before homing to let arm stabilize (default: 3.0)

        Raises:
            PiperConnectionError: If not connected to the arm
        """
        if not self.is_connected or self.piper is None:
            raise PiperConnectionError("Not connected to Piper arm")

        if return_home and self._enabled:
            # Step 1: Wait for arm to stabilize after motion
            time.sleep(pre_home_delay)
            # Step 2: Move to safe home position with position detection
            self._move_to_safe_home_and_wait(home_speed, timeout_sec)

            # Step 3: Settle time after reaching position
            time.sleep(settle_sec)

        # Step 4: Switch to standby mode (holds position)
        self.piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
        time.sleep(0.3)

        # Step 5: Disable motors
        self.piper.DisableArm(7)
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if arm is currently enabled."""
        return self._enabled

    def __enter__(self) -> "PiperConnection":
        """Enter context manager - connect to arm."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - disconnect from arm."""
        self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"PiperConnection(can_name='{self.can_name}', status={status})"


@contextmanager
def piper_connection(can_name: str = "can0", **kwargs):
    """Convenience context manager for Piper connection.

    Example:
        with piper_connection("can0") as conn:
            print(conn.piper.GetArmJointMsgs())
    """
    conn = PiperConnection(can_name=can_name, **kwargs)
    try:
        yield conn.connect()
    finally:
        conn.disconnect()
