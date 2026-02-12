"""Gripper control for Piper arm.

Provides GripperController class for gripper open/close operations.
"""

import time
from typing import Optional

from piper_sdk import C_PiperInterface_V2

from .utils import clamp_gripper_position, GRIPPER_LIMIT_M


class GripperController:
    """Control Piper arm gripper.

    Example:
        gripper = GripperController(piper)
        gripper.initialize()  # Required before first control
        gripper.open()
        time.sleep(1)
        gripper.close()

        # Partial open (50mm)
        gripper.set_position(0.05)

    Note:
        GripperCtrl API: GripperCtrl(gripper_angle, gripper_effort, gripper_code, set_zero)
        - gripper_angle: 0.001mm units (0-80000 for 0-80mm)
        - gripper_effort: torque limit in 0.001 N·m (0-5000 for 0-5 N·m)
        - gripper_code: 0x00=disable, 0x01=enable, 0x02=disable+clear, 0x03=enable+clear
        - set_zero: 0x00=noop, 0xAE=set zero
    """

    # Default gripper parameters
    DEFAULT_EFFORT = 5000  # Effort value in 0.001 N·m (0-5000)
    EFFORT_MIN = 0
    EFFORT_MAX = 5000

    # Gripper control modes
    MODE_DISABLE_CLEAR = 0x02
    MODE_ENABLE = 0x01

    def __init__(
        self,
        piper: C_PiperInterface_V2,
        effort: int = DEFAULT_EFFORT,
    ):
        """Initialize GripperController.

        Args:
            piper: Connected C_PiperInterface_V2 instance
            effort: Default gripper effort / torque limit (0-5000, in 0.001 N·m)
        """
        self.piper = piper
        self.effort = max(self.EFFORT_MIN, min(self.EFFORT_MAX, effort))
        self._current_position: Optional[float] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize gripper before control.

        Must be called once before open/close/set_position.
        Sends disable+clear then enable to prepare the gripper.
        """
        # Send disable+clear command (mode=0x02)
        self.piper.GripperCtrl(0, self.effort, self.MODE_DISABLE_CLEAR, 0)
        time.sleep(0.5)  # Wait for gripper to process init
        # Then switch to enable mode (mode=0x01)
        self.piper.GripperCtrl(0, self.effort, self.MODE_ENABLE, 0)
        time.sleep(0.5)  # Wait for gripper to enter control mode
        self._initialized = True

    def open(self, effort: Optional[int] = None) -> None:
        """Fully open the gripper.

        Args:
            effort: Override default effort (torque limit)
        """
        self.set_position(GRIPPER_LIMIT_M[1], effort)

    def close(self, effort: Optional[int] = None) -> None:
        """Fully close the gripper.

        Args:
            effort: Override default effort (torque limit)
        """
        self.set_position(GRIPPER_LIMIT_M[0], effort)

    def set_position(
        self,
        position_m: float,
        effort: Optional[int] = None,
    ) -> None:
        """Set gripper to specific opening.

        Args:
            position_m: Target opening in meters (0.0 to 0.08)
            effort: Override default effort (torque limit)

        Note:
            Automatically calls initialize() on first control command.
        """
        # Auto-initialize on first control
        if not self._initialized:
            self.initialize()

        position_m = clamp_gripper_position(position_m)
        use_effort = effort if effort is not None else self.effort

        # Convert to SDK units (0.001mm)
        # meters * 1000000 = 0.001mm units
        position_001mm = int(position_m * 1000000)

        # Send gripper command
        # GripperCtrl(gripper_angle, gripper_effort, gripper_code, set_zero)
        self.piper.GripperCtrl(position_001mm, use_effort, self.MODE_ENABLE, 0)
        self._current_position = position_m

    def set_position_mm(
        self,
        position_mm: float,
        effort: Optional[int] = None,
    ) -> None:
        """Set gripper to specific opening in millimeters.

        Args:
            position_mm: Target opening in mm (0 to 80)
            effort: Override default effort (torque limit)
        """
        self.set_position(position_mm / 1000.0, effort)

    def read_position(self) -> float:
        """Read current gripper position.

        Returns:
            Current opening in meters
        """
        gripper_msg = self.piper.GetArmGripperMsgs()
        # grippers_angle is in 0.001mm units, convert to meters
        position = gripper_msg.gripper_state.grippers_angle / 1000000.0
        self._current_position = position
        return position

    def read_position_mm(self) -> float:
        """Read current gripper position in millimeters.

        Returns:
            Current opening in mm
        """
        return self.read_position() * 1000.0

    def read_effort_raw(self) -> int:
        """Read current gripper effort (raw SDK value).

        Returns:
            Effort in 0.001 N·m units (0-5000)
        """
        gripper_msg = self.piper.GetArmGripperMsgs()
        return gripper_msg.gripper_state.grippers_effort

    def read_effort(self) -> float:
        """Read current gripper effort in N·m.

        Returns:
            Effort in N·m (0.0-5.0)
        """
        return self.read_effort_raw() / 1000.0

    def set_effort(self, effort: int) -> None:
        """Set default gripper effort (torque limit).

        Args:
            effort: Effort value in 0.001 N·m (0-5000)
        """
        self.effort = max(self.EFFORT_MIN, min(self.EFFORT_MAX, effort))

    @property
    def current_position(self) -> Optional[float]:
        """Get last known gripper position in meters."""
        return self._current_position

    @property
    def limits(self) -> tuple:
        """Get gripper position limits (min, max) in meters."""
        return GRIPPER_LIMIT_M
