"""Utility functions for Piper arm demo.

Provides CAN interface checking, angle conversions, and joint limit validation.
"""

import math
import subprocess
from typing import List, Tuple, Optional


# Piper arm joint limits (radians) - 6 joints + gripper
# Reference: piper_sdk documentation
JOINT_LIMITS_RAD: List[Tuple[float, float]] = [
    (-2.618, 2.618),  # Joint 1: -150° to 150°
    (0.0, 3.14),  # Joint 2: 0° to 180°
    (-2.967, 0.0),  # Joint 3: -170° to 0°
    (-1.745, 1.745),  # Joint 4: -100° to 100°
    (-1.22, 1.22),  # Joint 5: -70° to 70°
    (-2.094, 2.094),  # Joint 6: -120° to 120°
]

# Gripper limits (meters)
GRIPPER_LIMIT_M: Tuple[float, float] = (0.0, 0.08)  # 0 to 80mm


def check_can_interface(can_name: str = "can0") -> bool:
    """Check if CAN interface is active.

    Args:
        can_name: CAN interface name (default: "can0")

    Returns:
        True if interface is up, False otherwise
    """
    try:
        result = subprocess.run(
            ["ip", "link", "show", can_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "UP" in result.stdout and "LOWER_UP" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


def clamp_joint_position(joint_index: int, value_rad: float) -> float:
    """Clamp joint position to valid range.

    Args:
        joint_index: Joint index (0-5)
        value_rad: Desired position in radians

    Returns:
        Clamped position within joint limits
    """
    if joint_index < 0 or joint_index >= len(JOINT_LIMITS_RAD):
        raise ValueError(f"Invalid joint index: {joint_index} (must be 0-5)")

    min_val, max_val = JOINT_LIMITS_RAD[joint_index]
    return max(min_val, min(max_val, value_rad))


def clamp_gripper_position(value_m: float) -> float:
    """Clamp gripper position to valid range.

    Args:
        value_m: Desired gripper opening in meters

    Returns:
        Clamped position within gripper limits
    """
    min_val, max_val = GRIPPER_LIMIT_M
    return max(min_val, min(max_val, value_m))


def validate_joint_positions(positions: List[float]) -> Tuple[bool, Optional[str]]:
    """Validate all joint positions are within limits.

    Args:
        positions: List of 6 joint positions in radians

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(positions) != 6:
        return False, f"Expected 6 joint positions, got {len(positions)}"

    for i, pos in enumerate(positions):
        min_val, max_val = JOINT_LIMITS_RAD[i]
        if pos < min_val or pos > max_val:
            return False, (
                f"Joint {i + 1} position {rad_to_deg(pos):.1f}° "
                f"out of range [{rad_to_deg(min_val):.1f}°, {rad_to_deg(max_val):.1f}°]"
            )
    return True, None


def format_joint_state(
    positions: List[float], velocities: Optional[List[float]] = None
) -> str:
    """Format joint state for display.

    Args:
        positions: Joint positions in radians
        velocities: Optional joint velocities

    Returns:
        Formatted string for display
    """
    lines = ["Joint State:"]
    for i, pos in enumerate(positions):
        line = f"  J{i + 1}: {rad_to_deg(pos):7.2f}°"
        if velocities and i < len(velocities):
            line += f"  vel: {velocities[i]:6.3f} rad/s"
        lines.append(line)
    return "\n".join(lines)


def format_end_pose(pose) -> str:
    """Format end-effector pose for display.

    Args:
        pose: EndPoseState object with x, y, z, roll, pitch, yaw attributes

    Returns:
        Formatted string for display
    """
    x_mm, y_mm, z_mm = pose.x * 1000, pose.y * 1000, pose.z * 1000
    r_deg, p_deg, y_deg = (
        rad_to_deg(pose.roll),
        rad_to_deg(pose.pitch),
        rad_to_deg(pose.yaw),
    )
    lines = [
        "End Pose:",
        f"  Position: ({x_mm:7.1f}, {y_mm:7.1f}, {z_mm:7.1f}) mm",
        f"  Orientation: (R:{r_deg:6.1f}°, P:{p_deg:6.1f}°, Y:{y_deg:6.1f}°)",
    ]
    return "\n".join(lines)
