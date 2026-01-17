"""Gripper control for Piper arm.

Provides GripperController class for gripper open/close operations.
"""

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
        GripperCtrl API: GripperCtrl(position, speed, mode, control)
        - position: 0.001mm units (0-80000 for 0-80mm)
        - speed: 0-1000
        - mode: 0x01=control, 0x02=initialize
        - control: 0
    """

    # Default gripper parameters
    DEFAULT_SPEED = 1000  # Speed value (0-1000)

    # Gripper control modes
    MODE_INIT = 0x02
    MODE_CONTROL = 0x01

    def __init__(
        self,
        piper: C_PiperInterface_V2,
        speed: int = DEFAULT_SPEED,
    ):
        """Initialize GripperController.

        Args:
            piper: Connected C_PiperInterface_V2 instance
            speed: Default gripper speed (0-1000)
        """
        self.piper = piper
        self.speed = max(0, min(1000, speed))
        self._current_position: Optional[float] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize gripper before control.

        Must be called once before open/close/set_position.
        This sends mode=0x02 to prepare the gripper.
        """
        # Send initialization command (mode=0x02)
        self.piper.GripperCtrl(0, self.speed, self.MODE_INIT, 0)
        # Then switch to control mode (mode=0x01)
        self.piper.GripperCtrl(0, self.speed, self.MODE_CONTROL, 0)
        self._initialized = True

    def open(self, speed: Optional[int] = None) -> None:
        """Fully open the gripper.

        Args:
            speed: Override default speed
        """
        self.set_position(GRIPPER_LIMIT_M[1], speed)

    def close(self, speed: Optional[int] = None) -> None:
        """Fully close the gripper.

        Args:
            speed: Override default speed
        """
        self.set_position(GRIPPER_LIMIT_M[0], speed)

    def set_position(
        self,
        position_m: float,
        speed: Optional[int] = None,
    ) -> None:
        """Set gripper to specific opening.

        Args:
            position_m: Target opening in meters (0.0 to 0.08)
            speed: Override default speed

        Note:
            Automatically calls initialize() on first control command.
        """
        # Auto-initialize on first control
        if not self._initialized:
            self.initialize()

        position_m = clamp_gripper_position(position_m)
        use_speed = speed if speed is not None else self.speed

        # Convert to SDK units (0.001mm)
        # meters * 1000000 = 0.001mm units
        position_001mm = int(position_m * 1000000)

        # Send gripper command
        # GripperCtrl(position, speed, mode, control)
        self.piper.GripperCtrl(position_001mm, use_speed, self.MODE_CONTROL, 0)
        self._current_position = position_m

    def set_position_mm(
        self,
        position_mm: float,
        speed: Optional[int] = None,
    ) -> None:
        """Set gripper to specific opening in millimeters.

        Args:
            position_mm: Target opening in mm (0 to 80)
            speed: Override default speed
        """
        self.set_position(position_mm / 1000.0, speed)

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

    def set_speed(self, speed: int) -> None:
        """Set default gripper speed.

        Args:
            speed: Speed value (0-1000)
        """
        self.speed = max(0, min(1000, speed))

    @property
    def current_position(self) -> Optional[float]:
        """Get last known gripper position in meters."""
        return self._current_position

    @property
    def limits(self) -> tuple:
        """Get gripper position limits (min, max) in meters."""
        return GRIPPER_LIMIT_M
