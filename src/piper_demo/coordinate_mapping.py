"""Coordinate mapping for camera-to-arm projection.

Maps normalized camera coordinates (0-1) to robot arm workspace coordinates,
with configurable workspace bounds and safe Y-axis projection strategies.

Coordinate mapping:
    - Camera Y (horizontal, 0→1) → Arm Z (left-right)
    - Camera Z (vertical, 0→1)   → Arm X (front-back, inverted)
    - Arm Y (height)             → Dynamically computed for safety
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
    """3D workspace boundary definition for the arm.

    Defines the reachable/safe workspace in arm coordinates.
    """
    x_arm: AxisBounds = field(default_factory=lambda: AxisBounds(0.10, 0.35))
    z_arm: AxisBounds = field(default_factory=lambda: AxisBounds(-0.15, 0.15))
    y_arm: AxisBounds = field(default_factory=lambda: AxisBounds(0.05, 0.25))

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
class SafePlaneConfig:
    """Configuration for safe Y-axis (height) projection.

    Strategies:
        - "constant": Fixed Y value (base_y)
        - "adaptive": Y varies based on XZ distance from center
    """
    strategy: str = "adaptive"
    # Constant strategy
    base_y: float = 0.10  # meters
    # Adaptive strategy parameters
    distance_factor: float = 0.05  # Y increase per meter from center
    y_min: float = 0.05  # meters
    y_max: float = 0.25  # meters

    def compute_y(
        self, x_arm: float, z_arm: float, workspace: WorkspaceBounds
    ) -> float:
        """Compute safe Y value based on strategy.

        Args:
            x_arm: Target X position in arm coordinates (meters)
            z_arm: Target Z position in arm coordinates (meters)
            workspace: Workspace bounds for reference

        Returns:
            Safe Y value in meters
        """
        if self.strategy == "constant":
            return self.base_y

        elif self.strategy == "adaptive":
            # Calculate distance from workspace center
            x_center = workspace.x_arm.center()
            z_center = workspace.z_arm.center()
            dx = x_arm - x_center
            dz = z_arm - z_center
            distance = math.sqrt(dx * dx + dz * dz)

            # Increase Y as we move further from center
            y = self.base_y + distance * self.distance_factor

            # Clamp to safe range
            return max(self.y_min, min(self.y_max, y))

        else:
            raise ValueError(f"Unknown safe plane strategy: {self.strategy}")


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

    Example YAML:
        workspace:
          z_arm: { min: -0.15, max: 0.15 }
          x_arm: { min: 0.10, max: 0.35 }
        safe_plane:
          strategy: "adaptive"
          base_y: 0.10
          distance_factor: 0.05
          y_min: 0.05
          y_max: 0.25
        orientation:
          roll: 0.0
          pitch: 90.0
          yaw: 0.0
        motion:
          default_speed: 0.3
          singularity_check: true
    """
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    safe_plane: SafePlaneConfig = field(default_factory=SafePlaneConfig)
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

        if "safe_plane" in data:
            sp = data["safe_plane"]
            config.safe_plane = SafePlaneConfig(
                strategy=sp.get("strategy", "adaptive"),
                base_y=sp.get("base_y", 0.10),
                distance_factor=sp.get("distance_factor", 0.05),
                y_min=sp.get("y_min", 0.05),
                y_max=sp.get("y_max", 0.25),
            )
            # Handle nested adaptive config
            if "adaptive" in sp:
                adaptive = sp["adaptive"]
                config.safe_plane.base_y = adaptive.get("base_y", config.safe_plane.base_y)
                config.safe_plane.distance_factor = adaptive.get(
                    "distance_factor", config.safe_plane.distance_factor
                )
                config.safe_plane.y_min = adaptive.get("y_min", config.safe_plane.y_min)
                config.safe_plane.y_max = adaptive.get("y_max", config.safe_plane.y_max)

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
                "z_arm": {"min": self.workspace.z_arm.min, "max": self.workspace.z_arm.max},
                "y_arm": {"min": self.workspace.y_arm.min, "max": self.workspace.y_arm.max},
            },
            "safe_plane": {
                "strategy": self.safe_plane.strategy,
                "base_y": self.safe_plane.base_y,
                "distance_factor": self.safe_plane.distance_factor,
                "y_min": self.safe_plane.y_min,
                "y_max": self.safe_plane.y_max,
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

    Arm coordinate system (meters):
        - X_arm: forward/backward
        - Y_arm: up/down (height)
        - Z_arm: left/right

    Mapping:
        - Camera Y → Arm Z (with inversion option)
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

    def camera_to_arm_xz(
        self, y_cam: float, z_cam: float, clamp: bool = True
    ) -> Tuple[float, float]:
        """Convert normalized camera coordinates to arm X and Z.

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

        # Map camera Y (0-1) → arm Z
        z_arm = (
            self.config.workspace.z_arm.min
            + y_cam * self.config.workspace.z_arm.range()
        )

        # Map camera Z (0-1) → arm X
        x_arm = (
            self.config.workspace.x_arm.min
            + z_cam * self.config.workspace.x_arm.range()
        )

        if clamp:
            x_arm = self.config.workspace.x_arm.clamp(x_arm)
            z_arm = self.config.workspace.z_arm.clamp(z_arm)

        return x_arm, z_arm

    def arm_to_camera(self, x_arm: float, z_arm: float) -> Tuple[float, float]:
        """Convert arm X and Z coordinates back to normalized camera coordinates.

        Args:
            x_arm: Arm X coordinate in meters
            z_arm: Arm Z coordinate in meters

        Returns:
            (y_cam, z_cam) normalized 0-1
        """
        # Map arm Z → camera Y
        z_range = self.config.workspace.z_arm.range()
        if z_range > 0:
            y_cam = (z_arm - self.config.workspace.z_arm.min) / z_range
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


class SafePlaneProjector:
    """Computes safe Y (height) values for arm positions.

    Ensures the end-effector maintains a safe height above the workspace
    while avoiding singularities.
    """

    def __init__(self, config: CameraMappingConfig):
        """Initialize SafePlaneProjector.

        Args:
            config: Camera mapping configuration
        """
        self.config = config

    def compute_y(self, x_arm: float, z_arm: float) -> float:
        """Compute safe Y value for given X and Z position.

        Args:
            x_arm: Target X position in meters
            z_arm: Target Z position in meters

        Returns:
            Safe Y value in meters
        """
        return self.config.safe_plane.compute_y(
            x_arm, z_arm, self.config.workspace
        )

    def compute_full_position(
        self, x_arm: float, z_arm: float
    ) -> Tuple[float, float, float]:
        """Compute full (X, Y, Z) position with safe Y.

        Args:
            x_arm: Target X position in meters
            z_arm: Target Z position in meters

        Returns:
            (x_arm, y_arm, z_arm) in meters
        """
        y_arm = self.compute_y(x_arm, z_arm)
        return x_arm, y_arm, z_arm
