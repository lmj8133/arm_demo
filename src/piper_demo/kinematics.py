"""Forward kinematics for Piper arm.

Compute end-effector pose from joint angles using Modified DH parameters.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

# Type alias for 4x4 matrix
Matrix4 = List[List[float]]


@dataclass
class DHParams:
    """Modified DH parameters for a single joint."""
    alpha: float  # Link twist (rad)
    a: float      # Link length (m)
    d: float      # Link offset (m)
    theta_offset: float  # Joint angle offset (rad)


# Piper arm Modified DH parameters
# Reference: https://www.hackster.io/agilexrobotics/jacobian-magic-piper-arm-kinematics-unleashed-0d2f86
PIPER_DH_PARAMS: List[DHParams] = [
    DHParams(alpha=0,           a=0,         d=0.123,   theta_offset=0),
    DHParams(alpha=-math.pi/2,  a=0,         d=0,       theta_offset=math.radians(-172.22)),
    DHParams(alpha=0,           a=0.28503,   d=0,       theta_offset=math.radians(-102.78)),
    DHParams(alpha=math.pi/2,   a=-0.021984, d=0.25075, theta_offset=0),
    DHParams(alpha=-math.pi/2,  a=0,         d=0,       theta_offset=0),
    DHParams(alpha=math.pi/2,   a=0,         d=0.091,   theta_offset=0),
]


def _identity4() -> Matrix4:
    """Create 4x4 identity matrix."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_mult(A: Matrix4, B: Matrix4) -> Matrix4:
    """Multiply two 4x4 matrices."""
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += A[i][k] * B[k][j]
    return result


def _dh_transform_modified(alpha: float, a: float, d: float, theta: float) -> Matrix4:
    """Compute Modified DH transformation matrix.

    The Modified DH convention uses: T = Rx(α) * Dx(a) * Rz(θ) * Dz(d)
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)

    return [
        [ct,       -st,       0.0,    a],
        [st*ca,    ct*ca,    -sa,    -sa*d],
        [st*sa,    ct*sa,     ca,     ca*d],
        [0.0,      0.0,       0.0,    1.0]
    ]


def _rotation_to_euler_zyx(T: Matrix4) -> Tuple[float, float, float]:
    """Convert rotation matrix to ZYX Euler angles (roll, pitch, yaw)."""
    R00, R10, R20 = T[0][0], T[1][0], T[2][0]
    R11, R12 = T[1][1], T[1][2]
    R21, R22 = T[2][1], T[2][2]

    sy = math.sqrt(R00**2 + R10**2)

    if sy > 1e-6:
        roll = math.atan2(R21, R22)
        pitch = math.atan2(-R20, sy)
        yaw = math.atan2(R10, R00)
    else:
        roll = math.atan2(-R12, R11)
        pitch = math.atan2(-R20, sy)
        yaw = 0.0

    return roll, pitch, yaw


@dataclass
class FKResult:
    """Forward kinematics result."""
    x: float      # Position X (meters)
    y: float      # Position Y (meters)
    z: float      # Position Z (meters)
    roll: float   # Orientation roll (radians)
    pitch: float  # Orientation pitch (radians)
    yaw: float    # Orientation yaw (radians)

    def position_mm(self) -> Tuple[float, float, float]:
        """Get position in millimeters."""
        return (self.x * 1000, self.y * 1000, self.z * 1000)

    def orientation_deg(self) -> Tuple[float, float, float]:
        """Get orientation in degrees."""
        return (
            math.degrees(self.roll),
            math.degrees(self.pitch),
            math.degrees(self.yaw),
        )


def forward_kinematics(joint_angles: List[float]) -> FKResult:
    """Compute end-effector pose from joint angles.

    Args:
        joint_angles: 6 joint angles in radians

    Returns:
        FKResult with position (meters) and orientation (radians)

    Example:
        from piper_demo.kinematics import forward_kinematics
        result = forward_kinematics([0, 0, 0, 0, 0, 0])
        print(f"Position: {result.position_mm()} mm")
    """
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")

    T = _identity4()

    for theta, dh in zip(joint_angles, PIPER_DH_PARAMS):
        theta_total = theta + dh.theta_offset
        T_i = _dh_transform_modified(dh.alpha, dh.a, dh.d, theta_total)
        T = _mat4_mult(T, T_i)

    roll, pitch, yaw = _rotation_to_euler_zyx(T)

    return FKResult(
        x=T[0][3],
        y=T[1][3],
        z=T[2][3],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )
