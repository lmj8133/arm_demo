"""Piper robotic arm demo package.

This package provides high-level interfaces for controlling the Piper robotic arm
via CAN bus using the piper_sdk.

Quick Start:
    from piper_demo import PiperConnection, JointReader, MotionController, GripperController

    with PiperConnection(can_name="can0") as conn:
        reader = JointReader(conn.piper)
        state = reader.read_joints()
        print(state)
"""

from .connection import PiperConnection
from .joint_reader import JointReader, JointState, EndPoseState
from .motion import MotionController
from .gripper import GripperController
from .kinematics import forward_kinematics, FKResult
from .jacobian import compute_jacobian, is_near_singularity
from .inverse_kinematics import inverse_kinematics, IKResult, IKConfig, pose_to_matrix
from .utils import check_can_interface, deg_to_rad, rad_to_deg, format_end_pose
from .coordinate_mapping import (
    CameraMappingConfig,
    CameraMapping,
    SafePlaneProjector,
    WorkspaceBounds,
    AxisBounds,
)
from .vision_arm_controller import VisionArmController, MoveResult

__all__ = [
    "PiperConnection",
    "JointReader",
    "JointState",
    "EndPoseState",
    "MotionController",
    "GripperController",
    "forward_kinematics",
    "FKResult",
    "compute_jacobian",
    "is_near_singularity",
    "inverse_kinematics",
    "IKResult",
    "IKConfig",
    "pose_to_matrix",
    "check_can_interface",
    "deg_to_rad",
    "rad_to_deg",
    "format_end_pose",
    # Coordinate mapping
    "CameraMappingConfig",
    "CameraMapping",
    "SafePlaneProjector",
    "WorkspaceBounds",
    "AxisBounds",
    # Vision arm controller
    "VisionArmController",
    "MoveResult",
]

__version__ = "0.1.0"
