"""Coordinate mapping for camera-to-arm projection.

Maps normalized camera coordinates (0-1) to robot arm workspace coordinates,
with configurable workspace bounds and safe Z-axis (height) projection strategies.

Coordinate system follows ROS REP-103 convention:
    - Arm X: forward/backward (前後)
    - Arm Y: left/right (左右)
    - Arm Z: up/down (高度)

Coordinate mapping:
    - Camera Y (horizontal, 0→1) → Arm Y (left-right)
    - Camera Z (vertical, 0→1)   → Arm X (front-back, inverted)
    - Arm Z (height)             → Dynamically computed for safety
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class AxisBounds:
    """Min/max bounds for a single axis in meters."""
    min: float
    max: float

    def clamp(self, value: float) -> float:
        """Clamp value to bounds."""
        return max(self.min, min(self.max, value))

    def range(self) -> float:
        """Get the range of this axis."""
        return self.max - self.min

    def center(self) -> float:
        """Get the center of this axis."""
        return (self.min + self.max) / 2.0


@dataclass
class WorkspaceBounds:
    """3D workspace boundary definition for the arm (REP-103 convention).

    Defines the reachable/safe workspace in arm coordinates:
        - x_arm: forward/backward (前後)
        - y_arm: left/right (左右)
        - z_arm: up/down (高度)
    """
    x_arm: AxisBounds = field(default_factory=lambda: AxisBounds(0.10, 0.35))
    y_arm: AxisBounds = field(default_factory=lambda: AxisBounds(-0.15, 0.15))
    z_arm: AxisBounds = field(default_factory=lambda: AxisBounds(0.05, 0.25))

    def clamp_position(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Clamp position to workspace bounds."""
        return (
            self.x_arm.clamp(x),
            self.y_arm.clamp(y),
            self.z_arm.clamp(z),
        )

    def is_within_bounds(
        self, x: float, y: float, z: float, margin: float = 0.0
    ) -> bool:
        """Check if position is within workspace bounds."""
        return (
            self.x_arm.min + margin <= x <= self.x_arm.max - margin
            and self.y_arm.min + margin <= y <= self.y_arm.max - margin
            and self.z_arm.min + margin <= z <= self.z_arm.max - margin
        )


@dataclass
class SafeHeightConfig:
    """Configuration for safe Z-axis (height) projection (REP-103 convention).

    Strategies:
        - "constant": Fixed Z value (base_z)
        - "adaptive": Z varies based on XY distance from center
    """
    strategy: str = "adaptive"
    # Constant strategy
    base_z: float = 0.10  # meters
    # Adaptive strategy parameters
    distance_factor: float = 0.05  # Z increase per meter from center
    z_min: float = 0.05  # meters
    z_max: float = 0.25  # meters

    def compute_z(
        self, x_arm: float, y_arm: float, workspace: WorkspaceBounds
    ) -> float:
        """Compute safe Z value based on strategy.

        Args:
            x_arm: Target X position in arm coordinates (meters)
            y_arm: Target Y position in arm coordinates (meters)
            workspace: Workspace bounds for reference

        Returns:
            Safe Z value in meters
        """
        if self.strategy == "constant":
            return self.base_z

        elif self.strategy == "adaptive":
            # Calculate distance from workspace center
            x_center = workspace.x_arm.center()
            y_center = workspace.y_arm.center()
            dx = x_arm - x_center
            dy = y_arm - y_center
            distance = math.sqrt(dx * dx + dy * dy)

            # Increase Z as we move further from center
            z = self.base_z + distance * self.distance_factor

            # Clamp to safe range
            return max(self.z_min, min(self.z_max, z))

        else:
            raise ValueError(f"Unknown safe height strategy: {self.strategy}")


@dataclass
class OrientationConfig:
    """End-effector orientation configuration (degrees)."""
    roll: float = 0.0
    pitch: float = 90.0  # Pointing downward
    yaw: float = 0.0

    def to_radians(self) -> Tuple[float, float, float]:
        """Convert to radians."""
        return (
            math.radians(self.roll),
            math.radians(self.pitch),
            math.radians(self.yaw),
        )


@dataclass
class MotionConfig:
    """Motion control configuration."""
    default_speed: float = 0.3
    singularity_check: bool = True
    singularity_threshold: float = 0.001


@dataclass
class CameraMappingConfig:
    """Complete configuration for camera-to-arm coordinate mapping.

    Example YAML (REP-103 convention):
        workspace:
          x_arm: { min: 0.10, max: 0.35 }  # forward/backward
          y_arm: { min: -0.15, max: 0.15 } # left/right
          z_arm: { min: 0.05, max: 0.25 }  # height
        safe_height:
          strategy: "adaptive"
          base_z: 0.10
          distance_factor: 0.05
          z_min: 0.05
          z_max: 0.25
        orientation:
          roll: 0.0
          pitch: 90.0
          yaw: 0.0
        motion:
          default_speed: 0.3
          singularity_check: true
    """
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    safe_height: SafeHeightConfig = field(default_factory=SafeHeightConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "CameraMappingConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            CameraMappingConfig instance

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If file doesn't exist
        """
        if yaml is None:
            raise ImportError("PyYAML is required: pip install pyyaml")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "CameraMappingConfig":
        """Create configuration from dictionary."""
        config = cls()

        if "workspace" in data:
            ws = data["workspace"]
            if "x_arm" in ws:
                config.workspace.x_arm = AxisBounds(**ws["x_arm"])
            if "z_arm" in ws:
                config.workspace.z_arm = AxisBounds(**ws["z_arm"])
            if "y_arm" in ws:
                config.workspace.y_arm = AxisBounds(**ws["y_arm"])

        if "safe_height" in data:
            sh = data["safe_height"]
            config.safe_height = SafeHeightConfig(
                strategy=sh.get("strategy", "adaptive"),
                base_z=sh.get("base_z", 0.10),
                distance_factor=sh.get("distance_factor", 0.05),
                z_min=sh.get("z_min", 0.05),
                z_max=sh.get("z_max", 0.25),
            )
            # Handle nested adaptive config
            if "adaptive" in sh:
                adaptive = sh["adaptive"]
                config.safe_height.base_z = adaptive.get("base_z", config.safe_height.base_z)
                config.safe_height.distance_factor = adaptive.get(
                    "distance_factor", config.safe_height.distance_factor
                )
                config.safe_height.z_min = adaptive.get("z_min", config.safe_height.z_min)
                config.safe_height.z_max = adaptive.get("z_max", config.safe_height.z_max)

        if "orientation" in data:
            ori = data["orientation"]
            config.orientation = OrientationConfig(
                roll=ori.get("roll", 0.0),
                pitch=ori.get("pitch", 90.0),
                yaw=ori.get("yaw", 0.0),
            )

        if "motion" in data:
            mot = data["motion"]
            config.motion = MotionConfig(
                default_speed=mot.get("default_speed", 0.3),
                singularity_check=mot.get("singularity_check", True),
                singularity_threshold=mot.get("singularity_threshold", 0.001),
            )

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "workspace": {
                "x_arm": {"min": self.workspace.x_arm.min, "max": self.workspace.x_arm.max},
                "y_arm": {"min": self.workspace.y_arm.min, "max": self.workspace.y_arm.max},
                "z_arm": {"min": self.workspace.z_arm.min, "max": self.workspace.z_arm.max},
            },
            "safe_height": {
                "strategy": self.safe_height.strategy,
                "base_z": self.safe_height.base_z,
                "distance_factor": self.safe_height.distance_factor,
                "z_min": self.safe_height.z_min,
                "z_max": self.safe_height.z_max,
            },
            "orientation": {
                "roll": self.orientation.roll,
                "pitch": self.orientation.pitch,
                "yaw": self.orientation.yaw,
            },
            "motion": {
                "default_speed": self.motion.default_speed,
                "singularity_check": self.motion.singularity_check,
                "singularity_threshold": self.motion.singularity_threshold,
            },
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        if yaml is None:
            raise ImportError("PyYAML is required: pip install pyyaml")

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class CameraMapping:
    """Maps normalized camera coordinates to arm workspace coordinates.

    Camera coordinate system (normalized 0-1):
        - Y_cam: horizontal axis (0=left, 1=right)
        - Z_cam: vertical axis (0=top, 1=bottom)

    Arm coordinate system (REP-103 convention, meters):
        - X_arm: forward/backward (前後)
        - Y_arm: left/right (左右)
        - Z_arm: up/down (高度)

    Mapping:
        - Camera Y → Arm Y (with inversion option)
        - Camera Z → Arm X (with inversion)
    """

    def __init__(
        self,
        config: CameraMappingConfig,
        invert_cam_y: bool = False,
        invert_cam_z: bool = True,
    ):
        """Initialize CameraMapping.

        Args:
            config: Camera mapping configuration
            invert_cam_y: If True, invert camera Y axis (0=right, 1=left)
            invert_cam_z: If True, invert camera Z axis (0=bottom, 1=top).
                          Default True maps camera top to arm front.
        """
        self.config = config
        self.invert_cam_y = invert_cam_y
        self.invert_cam_z = invert_cam_z

    def camera_to_arm_xy(
        self, y_cam: float, z_cam: float, clamp: bool = True
    ) -> Tuple[float, float]:
        """Convert normalized camera coordinates to arm X and Y.

        Args:
            y_cam: Camera horizontal coordinate (0-1)
            z_cam: Camera vertical coordinate (0-1)
            clamp: If True, clamp output to workspace bounds

        Returns:
            (x_arm, y_arm) in meters
        """
        # Normalize input to 0-1 range
        y_cam = max(0.0, min(1.0, y_cam))
        z_cam = max(0.0, min(1.0, z_cam))

        # Apply inversions
        if self.invert_cam_y:
            y_cam = 1.0 - y_cam
        if self.invert_cam_z:
            z_cam = 1.0 - z_cam

        # Map camera Y (0-1) → arm Y (left-right)
        y_arm = (
            self.config.workspace.y_arm.min
            + y_cam * self.config.workspace.y_arm.range()
        )

        # Map camera Z (0-1) → arm X (front-back)
        x_arm = (
            self.config.workspace.x_arm.min
            + z_cam * self.config.workspace.x_arm.range()
        )

        if clamp:
            x_arm = self.config.workspace.x_arm.clamp(x_arm)
            y_arm = self.config.workspace.y_arm.clamp(y_arm)

        return x_arm, y_arm

    def camera_to_arm_xz(
        self, y_cam: float, z_cam: float, clamp: bool = True
    ) -> Tuple[float, float]:
        """Convert normalized camera coordinates to arm X and Z.

        For XZ plane tracking (front-back + height).

        Args:
            y_cam: Camera horizontal coordinate (0-1)
            z_cam: Camera vertical coordinate (0-1)
            clamp: If True, clamp output to workspace bounds

        Returns:
            (x_arm, z_arm) in meters
        """
        # Normalize input to 0-1 range
        y_cam = max(0.0, min(1.0, y_cam))
        z_cam = max(0.0, min(1.0, z_cam))

        # Apply inversions
        if self.invert_cam_y:
            y_cam = 1.0 - y_cam
        if self.invert_cam_z:
            z_cam = 1.0 - z_cam

        # Map camera Y (horizontal, 0-1) → arm X (front-back)
        x_arm = (
            self.config.workspace.x_arm.min
            + y_cam * self.config.workspace.x_arm.range()
        )

        # Map camera Z (vertical, 0-1) → arm Z (height)
        z_arm = (
            self.config.workspace.z_arm.min
            + z_cam * self.config.workspace.z_arm.range()
        )

        if clamp:
            x_arm = self.config.workspace.x_arm.clamp(x_arm)
            z_arm = self.config.workspace.z_arm.clamp(z_arm)

        return x_arm, z_arm

    def arm_to_camera(self, x_arm: float, y_arm: float) -> Tuple[float, float]:
        """Convert arm X and Y coordinates back to normalized camera coordinates.

        Args:
            x_arm: Arm X coordinate in meters (forward/backward)
            y_arm: Arm Y coordinate in meters (left/right)

        Returns:
            (y_cam, z_cam) normalized 0-1
        """
        # Map arm Y → camera Y
        y_range = self.config.workspace.y_arm.range()
        if y_range > 0:
            y_cam = (y_arm - self.config.workspace.y_arm.min) / y_range
        else:
            y_cam = 0.5

        # Map arm X → camera Z
        x_range = self.config.workspace.x_arm.range()
        if x_range > 0:
            z_cam = (x_arm - self.config.workspace.x_arm.min) / x_range
        else:
            z_cam = 0.5

        # Undo inversions
        if self.invert_cam_y:
            y_cam = 1.0 - y_cam
        if self.invert_cam_z:
            z_cam = 1.0 - z_cam

        return y_cam, z_cam


class SafeHeightProjector:
    """Computes safe Z (height) values for arm positions (REP-103 convention).

    Ensures the end-effector maintains a safe height above the workspace
    while avoiding singularities.
    """

    def __init__(self, config: CameraMappingConfig):
        """Initialize SafeHeightProjector.

        Args:
            config: Camera mapping configuration
        """
        self.config = config

    def compute_z(self, x_arm: float, y_arm: float) -> float:
        """Compute safe Z value for given X and Y position.

        Args:
            x_arm: Target X position in meters (forward/backward)
            y_arm: Target Y position in meters (left/right)

        Returns:
            Safe Z value in meters (height)
        """
        return self.config.safe_height.compute_z(
            x_arm, y_arm, self.config.workspace
        )

    def compute_full_position(
        self, x_arm: float, y_arm: float
    ) -> Tuple[float, float, float]:
        """Compute full (X, Y, Z) position with safe Z height.

        Args:
            x_arm: Target X position in meters (forward/backward)
            y_arm: Target Y position in meters (left/right)

        Returns:
            (x_arm, y_arm, z_arm) in meters
        """
        z_arm = self.compute_z(x_arm, y_arm)
        return x_arm, y_arm, z_arm
