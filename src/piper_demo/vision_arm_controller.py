"""High-level vision-to-arm controller.

Integrates coordinate mapping with motion control for vision-guided
robotic arm movements.

Example:
    from piper_demo import PiperConnection, JointReader, MotionController
    from piper_demo.vision_arm_controller import VisionArmController

    with PiperConnection(can_name="can0") as conn:
        conn.enable(go_home=True)
        reader = JointReader(conn.piper)
        motion = MotionController(conn.piper)

        controller = VisionArmController.from_yaml(
            "config/camera_workspace.yaml", reader, motion
        )
        result = controller.move_to_normalized(y_cam=0.5, z_cam=0.5)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .coordinate_mapping import (
    AxisBounds,
    CameraMappingConfig,
    CameraMapping,
    SafeHeightProjector,
)
from .kinematics import forward_kinematics, FKResult
from .inverse_kinematics import inverse_kinematics, IKConfig
from .jacobian import is_near_singularity, compute_jacobian, jacobian_determinant
from .joint_reader import JointReader
from .motion import MotionController


@dataclass
class MoveResult:
    """Result of a vision-guided move operation.

    Coordinate system follows ROS REP-103 convention:
        - x_arm: forward/backward (前後)
        - y_arm: left/right (左右)
        - z_arm: up/down (高度)
    """

    # Target position in arm coordinates (meters, REP-103)
    x_arm: float  # forward/backward
    y_arm: float  # left/right
    z_arm: float  # height

    # Target orientation (radians)
    roll: float
    pitch: float
    yaw: float

    # Normalized camera coordinates
    y_cam: float
    z_cam: float

    # IK solution
    joint_angles: List[float]
    ik_converged: bool
    ik_iterations: int
    position_error: float
    orientation_error: float

    # Motion status
    motion_completed: bool
    near_singularity: bool

    def __repr__(self) -> str:
        status = "OK" if self.motion_completed and self.ik_converged else "FAILED"
        return (
            f"MoveResult({status}, "
            f"cam=({self.y_cam:.2f}, {self.z_cam:.2f}), "
            f"arm=({self.x_arm * 1000:.1f}, {self.y_arm * 1000:.1f}, {self.z_arm * 1000:.1f})mm)"
        )

    def position_mm(self) -> Tuple[float, float, float]:
        """Get position in millimeters."""
        return (self.x_arm * 1000, self.y_arm * 1000, self.z_arm * 1000)


class VisionArmController:
    """High-level controller for vision-guided arm movements.

    Integrates:
        - CameraMapping: normalized camera coords → arm XY
        - SafeHeightProjector: compute safe Z height
        - IK solver: compute joint angles
        - MotionController: execute movement

    Coordinate system follows ROS REP-103 convention:
        - X: forward/backward
        - Y: left/right
        - Z: up/down (height)

    Example:
        controller = VisionArmController(config, reader, motion)
        result = controller.move_to_normalized(0.5, 0.5)
        if result.motion_completed:
            print(f"Moved to: {result.position_mm()}")
    """

    def __init__(
        self,
        config: CameraMappingConfig,
        reader: JointReader,
        motion: MotionController,
        invert_cam_y: bool = False,
        invert_cam_z: bool = True,
    ):
        """Initialize VisionArmController.

        Args:
            config: Camera mapping configuration
            reader: JointReader for position feedback
            motion: MotionController for movement
            invert_cam_y: Invert camera Y axis
            invert_cam_z: Invert camera Z axis (default True)
        """
        self.config = config
        self.reader = reader
        self.motion = motion

        self.camera_mapping = CameraMapping(
            config, invert_cam_y=invert_cam_y, invert_cam_z=invert_cam_z
        )
        self.projector = SafeHeightProjector(config)

        # Default IK configuration
        self.ik_config = IKConfig(
            max_iterations=100,
            damping_factor=0.05,
            position_tolerance=1e-4,
            orientation_tolerance=1e-3,
        )

    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        reader: JointReader,
        motion: MotionController,
        invert_cam_y: bool = False,
        invert_cam_z: bool = True,
    ) -> "VisionArmController":
        """Create controller from YAML configuration file.

        Args:
            config_path: Path to YAML configuration file
            reader: JointReader instance
            motion: MotionController instance
            invert_cam_y: Invert camera Y axis
            invert_cam_z: Invert camera Z axis

        Returns:
            VisionArmController instance
        """
        config = CameraMappingConfig.from_yaml(config_path)
        return cls(config, reader, motion, invert_cam_y, invert_cam_z)

    @classmethod
    def with_defaults(
        cls,
        reader: JointReader,
        motion: MotionController,
        invert_cam_y: bool = False,
        invert_cam_z: bool = True,
    ) -> "VisionArmController":
        """Create controller with default configuration.

        Args:
            reader: JointReader instance
            motion: MotionController instance
            invert_cam_y: Invert camera Y axis
            invert_cam_z: Invert camera Z axis

        Returns:
            VisionArmController instance
        """
        config = CameraMappingConfig()
        return cls(config, reader, motion, invert_cam_y, invert_cam_z)

    @classmethod
    def from_home_position(
        cls,
        home_joints: List[float],
        reader: JointReader,
        motion: MotionController,
        workspace_range_xz: float = 0.10,
        workspace_range_z: float = 0.05,
        invert_cam_y: bool = False,
        invert_cam_z: bool = True,
        override_z: Optional[float] = None,
    ) -> Tuple["VisionArmController", FKResult]:
        """Create controller with workspace centered on HOME_POSITION FK result.

        Automatically computes workspace bounds based on the forward kinematics
        of the given home joint angles, avoiding the common mismatch between
        hardcoded workspace bounds and the actual arm home position.

        Coordinate system follows ROS REP-103 convention:
            - X: forward/backward
            - Y: left/right (XY plane for camera mapping)
            - Z: up/down (height)

        Args:
            home_joints: 6 joint angles for home position (radians)
            reader: JointReader instance
            motion: MotionController instance
            workspace_range_xz: Half-range for X and Y axes (meters, default ±100mm)
            workspace_range_z: Half-range for Z axis/height (meters, default ±50mm)
            invert_cam_y: Invert camera Y axis
            invert_cam_z: Invert camera Z axis
            override_z: Use this Z height instead of home FK Z (meters).
                        Useful for maximizing XY reachable area. Use
                        scripts/find_optimal_z.py to determine the optimal value.

        Returns:
            Tuple of (VisionArmController instance, FKResult of home position)

        Example:
            from piper_demo import PiperConnection
            controller, home_fk = VisionArmController.from_home_position(
                PiperConnection.HOME_POSITION, reader, motion
            )
            print(f"Workspace centered at: {home_fk.position_mm()} mm")

            # With override_z for larger XY reach:
            controller, home_fk = VisionArmController.from_home_position(
                PiperConnection.HOME_POSITION, reader, motion,
                override_z=0.07  # 70mm, computed by find_optimal_z.py
            )
        """
        # Compute FK for home position
        home_fk = forward_kinematics(home_joints)

        # Use override_z if provided, otherwise use home FK Z (height)
        working_z = override_z if override_z is not None else home_fk.z

        # Build config centered on home FK
        config = CameraMappingConfig()

        # Workspace bounds centered on home position (XY) or working_z (Z)
        config.workspace.x_arm = AxisBounds(
            home_fk.x - workspace_range_xz,
            home_fk.x + workspace_range_xz,
        )
        config.workspace.y_arm = AxisBounds(
            home_fk.y - workspace_range_xz,
            home_fk.y + workspace_range_xz,
        )
        config.workspace.z_arm = AxisBounds(
            working_z - workspace_range_z,
            working_z + workspace_range_z,
        )

        # Use home FK orientation
        config.orientation.roll = math.degrees(home_fk.roll)
        config.orientation.pitch = math.degrees(home_fk.pitch)
        config.orientation.yaw = math.degrees(home_fk.yaw)

        # Use constant Z at working height for predictable motion
        config.safe_height.strategy = "constant"
        config.safe_height.base_z = working_z

        # Auto-adjust singularity threshold based on HOME position
        # If HOME is near singularity, use a smaller threshold to avoid false positives
        home_jacobian = compute_jacobian(list(home_joints))
        home_det = abs(jacobian_determinant(home_jacobian))
        if home_det < config.motion.singularity_threshold:
            # Set threshold to 10% of HOME determinant to allow movement near HOME
            config.motion.singularity_threshold = home_det * 0.1

        return cls(config, reader, motion, invert_cam_y, invert_cam_z), home_fk

    def compute_target_pose(
        self, y_cam: float, z_cam: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute target pose from normalized camera coordinates.

        YZ plane tracking: camera horizontal -> arm Y, camera vertical -> arm Z.
        X stays fixed at workspace center.

        Args:
            y_cam: Camera horizontal coordinate (0-1) -> Arm Y (left-right)
            z_cam: Camera vertical coordinate (0-1) -> Arm Z (height)

        Returns:
            (x, y, z, roll, pitch, yaw) in meters and radians (REP-103 convention)
        """
        # Map camera coords to arm YZ (left-right + height)
        y_arm, z_arm = self.camera_mapping.camera_to_arm_yz(y_cam, z_cam)

        # X stays fixed at workspace center
        x_arm = self.config.workspace.x_arm.center()

        # Get orientation
        roll, pitch, yaw = self.config.orientation.to_radians()

        return x_arm, y_arm, z_arm, roll, pitch, yaw

    def move_to_normalized(
        self,
        y_cam: float,
        z_cam: float,
        speed_factor: Optional[float] = None,
        wait: bool = True,
        initial_guess: Optional[List[float]] = None,
        timeout_sec: float = 15.0,
        tolerance_rad: float = 0.035,  # ~2 degrees
    ) -> MoveResult:
        """Move arm to position specified by normalized camera coordinates.

        Args:
            y_cam: Camera horizontal coordinate (0-1, left to right)
            z_cam: Camera vertical coordinate (0-1, top to bottom)
            speed_factor: Override default speed (0.0-1.0)
            wait: If True, wait for motion to complete
            initial_guess: Initial joint angles for IK solver
            timeout_sec: Motion timeout in seconds
            tolerance_rad: Joint position tolerance for wait

        Returns:
            MoveResult with status and position info
        """
        if speed_factor is None:
            speed_factor = self.config.motion.default_speed

        # Compute target pose
        x_arm, y_arm, z_arm, roll, pitch, yaw = self.compute_target_pose(y_cam, z_cam)

        # Get initial guess if not provided
        if initial_guess is None:
            joint_state = self.reader.read_joints()
            initial_guess = list(joint_state.positions)

        # Compute IK
        ik_result = inverse_kinematics(
            x_arm,
            y_arm,
            z_arm,
            roll,
            pitch,
            yaw,
            initial_guess=initial_guess,
            config=self.ik_config,
        )

        # Check singularity if enabled
        near_singularity = False
        if self.config.motion.singularity_check:
            near_singularity = is_near_singularity(
                ik_result.joint_angles,
                threshold=self.config.motion.singularity_threshold,
            )

        # Create result object
        result = MoveResult(
            x_arm=x_arm,
            y_arm=y_arm,
            z_arm=z_arm,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            y_cam=y_cam,
            z_cam=z_cam,
            joint_angles=ik_result.joint_angles,
            ik_converged=ik_result.converged,
            ik_iterations=ik_result.iterations,
            position_error=ik_result.position_error,
            orientation_error=ik_result.orientation_error,
            motion_completed=False,
            near_singularity=near_singularity,
        )

        # Abort if IK failed
        if not ik_result.converged:
            return result

        # Abort if near singularity and check is enabled
        if near_singularity and self.config.motion.singularity_check:
            return result

        # Execute motion
        self.motion.move_joint(ik_result.joint_angles, speed_factor=speed_factor)

        # Wait for motion to complete if requested
        if wait:
            motion_completed = self.reader.wait_for_position(
                ik_result.joint_angles,
                tolerance_rad=tolerance_rad,
                timeout_sec=timeout_sec,
            )
            result.motion_completed = motion_completed
        else:
            result.motion_completed = True  # Assume success for async

        return result

    def get_current_camera_coords(self) -> Tuple[float, float]:
        """Get current arm position as normalized camera coordinates.

        Useful for visualization or feedback display.

        Returns:
            (y_cam, z_cam) normalized 0-1
        """
        pose = self.reader.read_end_pose()
        return self.camera_mapping.arm_to_camera(pose.x, pose.y)

    def get_workspace_corners(self) -> List[Tuple[float, float, float]]:
        """Get workspace corner positions for visualization.

        Returns:
            List of (x, y, z) corner positions in meters (REP-103 convention)
        """
        corners = []
        ws = self.config.workspace

        for x in [ws.x_arm.min, ws.x_arm.max]:
            for y in [ws.y_arm.min, ws.y_arm.max]:
                z = self.projector.compute_z(x, y)
                corners.append((x, y, z))

        return corners

    def scan_workspace(
        self,
        grid_size: int = 3,
        speed_factor: float = 0.2,
        dwell_sec: float = 0.5,
    ) -> List[MoveResult]:
        """Scan workspace in a grid pattern for calibration or testing.

        Args:
            grid_size: Number of points per axis (grid_size x grid_size)
            speed_factor: Movement speed (0.0-1.0)
            dwell_sec: Dwell time at each point in seconds

        Returns:
            List of MoveResult for each grid point
        """
        import time

        results = []

        for i in range(grid_size):
            y_cam = i / (grid_size - 1) if grid_size > 1 else 0.5
            for j in range(grid_size):
                z_cam = j / (grid_size - 1) if grid_size > 1 else 0.5

                result = self.move_to_normalized(
                    y_cam,
                    z_cam,
                    speed_factor=speed_factor,
                    wait=True,
                )
                results.append(result)

                if dwell_sec > 0 and result.motion_completed:
                    time.sleep(dwell_sec)

        return results
