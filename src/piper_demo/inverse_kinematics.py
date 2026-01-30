"""Inverse kinematics solver for Piper arm.

Uses Damped Least Squares (DLS) iterative method.
Reference: https://discourse.openrobotics.org/t/implementation-of-forward-and-inverse-kinematics-for-agilex-piper-robotic-arm-using-eigen-linear-algebra-library/50153
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from .kinematics import (
        PIPER_DH_PARAMS,
        Matrix4,
        _identity4,
        _mat4_mult,
        _dh_transform_modified,
    )
    from .jacobian import compute_jacobian_numerical, Jacobian6x6
    from .utils import JOINT_LIMITS_RAD
except ImportError:
    from kinematics import (
        PIPER_DH_PARAMS,
        Matrix4,
        _identity4,
        _mat4_mult,
        _dh_transform_modified,
    )
    from jacobian import compute_jacobian_numerical, Jacobian6x6
    from utils import JOINT_LIMITS_RAD


@dataclass
class IKConfig:
    """Configuration for IK solver."""

    max_iterations: int = 100
    position_tolerance: float = 1e-4  # meters
    orientation_tolerance: float = 1e-3  # radians
    damping_factor: float = 0.05  # λ for DLS
    step_limit: float = 0.1  # Max joint change per iteration (rad)


@dataclass
class IKResult:
    """Result of inverse kinematics computation."""

    joint_angles: List[float]  # Solution in radians
    converged: bool
    iterations: int
    position_error: float  # meters
    orientation_error: float  # radians

    def __repr__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return (
            f"IKResult({status}, iter={self.iterations}, "
            f"pos_err={self.position_error * 1000:.3f}mm, "
            f"ori_err={math.degrees(self.orientation_error):.3f}°)"
        )


def _fk_transform(joint_angles: List[float]) -> Matrix4:
    """Compute forward kinematics transformation matrix."""
    T = _identity4()
    for theta, dh in zip(joint_angles, PIPER_DH_PARAMS):
        theta_total = theta + dh.theta_offset
        T_i = _dh_transform_modified(dh.alpha, dh.a, dh.d, theta_total)
        T = _mat4_mult(T, T_i)
    return T


def _rotation_matrix_to_euler_zyx(R: Matrix4) -> Tuple[float, float, float]:
    """Convert rotation matrix to ZYX Euler angles (roll, pitch, yaw)."""
    sy = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)

    if sy > 1e-6:
        roll = math.atan2(R[2][1], R[2][2])
        pitch = math.atan2(-R[2][0], sy)
        yaw = math.atan2(R[1][0], R[0][0])
    else:
        roll = math.atan2(-R[1][2], R[1][1])
        pitch = math.atan2(-R[2][0], sy)
        yaw = 0.0

    return roll, pitch, yaw


def _euler_zyx_to_rotation_matrix(
    roll: float, pitch: float, yaw: float
) -> List[List[float]]:
    """Convert ZYX Euler angles to 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotation_error(
    R_current: Matrix4, R_target: List[List[float]]
) -> Tuple[float, float, float]:
    """Compute orientation error using axis-angle representation.

    Computes R_error = R_target * R_current^T, then extracts axis-angle.
    """
    # Build R_current^T (3x3)
    Rc_T = [
        [R_current[0][0], R_current[1][0], R_current[2][0]],
        [R_current[0][1], R_current[1][1], R_current[2][1]],
        [R_current[0][2], R_current[1][2], R_current[2][2]],
    ]

    # R_error = R_target * R_current^T (3x3 multiplication)
    R_err = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                R_err[i][j] += R_target[i][k] * Rc_T[k][j]

    # Extract axis-angle from R_error
    trace = R_err[0][0] + R_err[1][1] + R_err[2][2]
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))

    if abs(angle) < 1e-10:
        return (0.0, 0.0, 0.0)

    if abs(angle - math.pi) < 1e-6:
        # Near 180 degrees singularity
        axis = (
            math.sqrt(max(0, (R_err[0][0] + 1) / 2)),
            math.sqrt(max(0, (R_err[1][1] + 1) / 2)),
            math.sqrt(max(0, (R_err[2][2] + 1) / 2)),
        )
        return (axis[0] * angle, axis[1] * angle, axis[2] * angle)

    # General case: axis = (1 / 2sin(θ)) * [R32-R23, R13-R31, R21-R12]
    k = angle / (2.0 * math.sin(angle))
    return (
        k * (R_err[2][1] - R_err[1][2]),
        k * (R_err[0][2] - R_err[2][0]),
        k * (R_err[1][0] - R_err[0][1]),
    )


def _compute_pose_error(
    current_pose: Matrix4, target_pose: Matrix4
) -> Tuple[List[float], float, float]:
    """Compute 6D pose error between current and target.

    Returns:
        (error_vector, position_norm, orientation_norm)
    """
    # Position error
    e_p = [
        target_pose[0][3] - current_pose[0][3],
        target_pose[1][3] - current_pose[1][3],
        target_pose[2][3] - current_pose[2][3],
    ]

    # Orientation error
    e_o = _rotation_error(current_pose, target_pose)

    # Combine into 6D error
    error = [e_p[0], e_p[1], e_p[2], e_o[0], e_o[1], e_o[2]]

    pos_norm = math.sqrt(e_p[0] ** 2 + e_p[1] ** 2 + e_p[2] ** 2)
    ori_norm = math.sqrt(e_o[0] ** 2 + e_o[1] ** 2 + e_o[2] ** 2)

    return error, pos_norm, ori_norm


def _mat6_transpose(A: Jacobian6x6) -> Jacobian6x6:
    """Transpose 6x6 matrix."""
    return [[A[j][i] for j in range(6)] for i in range(6)]


def _mat6_mult(A: Jacobian6x6, B: Jacobian6x6) -> Jacobian6x6:
    """Multiply two 6x6 matrices."""
    result = [[0.0] * 6 for _ in range(6)]
    for i in range(6):
        for j in range(6):
            for k in range(6):
                result[i][j] += A[i][k] * B[k][j]
    return result


def _mat6_vec6_mult(A: Jacobian6x6, v: List[float]) -> List[float]:
    """Multiply 6x6 matrix by 6-vector."""
    return [sum(A[i][j] * v[j] for j in range(6)) for i in range(6)]


def _mat6_add_diagonal(A: Jacobian6x6, scalar: float) -> Jacobian6x6:
    """Add scalar to diagonal elements."""
    result = [row[:] for row in A]
    for i in range(6):
        result[i][i] += scalar
    return result


def _mat6_inverse(A: Jacobian6x6) -> Optional[Jacobian6x6]:
    """Compute inverse of 6x6 matrix using Gauss-Jordan elimination.

    Returns None if matrix is singular.
    """
    n = 6
    # Create augmented matrix [A | I]
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = A[i][j]
        aug[i][n + i] = 1.0

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        if max_val < 1e-12:
            return None  # Singular matrix

        # Swap rows
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


def _damped_pseudoinverse(J: Jacobian6x6, damping: float) -> Jacobian6x6:
    """Compute damped pseudoinverse: J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹."""
    J_T = _mat6_transpose(J)
    JJ_T = _mat6_mult(J, J_T)

    # Add damping: JJᵀ + λ²I
    lambda_sq = damping * damping
    JJ_T_damped = _mat6_add_diagonal(JJ_T, lambda_sq)

    # Invert
    JJ_T_inv = _mat6_inverse(JJ_T_damped)
    if JJ_T_inv is None:
        # Fallback: return J^T scaled down
        return [[J_T[i][j] * 0.01 for j in range(6)] for i in range(6)]

    # J⁺ = Jᵀ * (JJᵀ + λ²I)⁻¹
    return _mat6_mult(J_T, JJ_T_inv)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _clamp_joint_angles(angles: List[float]) -> List[float]:
    """Clamp joint angles to valid ranges."""
    result = []
    for i, angle in enumerate(angles):
        min_val, max_val = JOINT_LIMITS_RAD[i]
        clamped = max(min_val, min(max_val, angle))
        result.append(clamped)
    return result


def pose_to_matrix(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> Matrix4:
    """Create transformation matrix from position and Euler angles.

    Args:
        x, y, z: Position in meters
        roll, pitch, yaw: Orientation in radians (ZYX convention)

    Returns:
        4x4 transformation matrix
    """
    R = _euler_zyx_to_rotation_matrix(roll, pitch, yaw)
    return [
        [R[0][0], R[0][1], R[0][2], x],
        [R[1][0], R[1][1], R[1][2], y],
        [R[2][0], R[2][1], R[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def inverse_kinematics(
    target_x: float,
    target_y: float,
    target_z: float,
    target_roll: float,
    target_pitch: float,
    target_yaw: float,
    initial_guess: Optional[List[float]] = None,
    config: Optional[IKConfig] = None,
) -> IKResult:
    """Compute inverse kinematics using Damped Least Squares.

    Args:
        target_x, target_y, target_z: Target position in meters
        target_roll, target_pitch, target_yaw: Target orientation in radians
        initial_guess: Starting joint angles (defaults to zeros)
        config: IK solver configuration

    Returns:
        IKResult with solution and convergence info

    Example:
        from piper_demo.inverse_kinematics import inverse_kinematics
        result = inverse_kinematics(0.3, 0.0, 0.2, 0, 0, 0)
        if result.converged:
            print(f"Solution: {result.joint_angles}")
    """
    if config is None:
        config = IKConfig()

    if initial_guess is None:
        initial_guess = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]  # Reasonable starting pose

    if len(initial_guess) != 6:
        raise ValueError(f"Expected 6 initial joint angles, got {len(initial_guess)}")

    # Build target pose matrix
    target_pose = pose_to_matrix(
        target_x, target_y, target_z, target_roll, target_pitch, target_yaw
    )

    # Initialize
    joint_angles = list(initial_guess)
    pos_error = float("inf")
    ori_error = float("inf")

    for iteration in range(config.max_iterations):
        # Compute current pose
        current_pose = _fk_transform(joint_angles)

        # Compute error
        error, pos_error, ori_error = _compute_pose_error(current_pose, target_pose)

        # Check convergence
        if (
            pos_error < config.position_tolerance
            and ori_error < config.orientation_tolerance
        ):
            return IKResult(
                joint_angles=joint_angles,
                converged=True,
                iterations=iteration + 1,
                position_error=pos_error,
                orientation_error=ori_error,
            )

        # Compute Jacobian and pseudoinverse
        J = compute_jacobian_numerical(joint_angles)
        J_pinv = _damped_pseudoinverse(J, config.damping_factor)

        # Compute joint increment: Δθ = J⁺ * e
        delta_theta = _mat6_vec6_mult(J_pinv, error)

        # Apply step limiting
        max_delta = max(abs(d) for d in delta_theta)
        if max_delta > config.step_limit:
            scale = config.step_limit / max_delta
            delta_theta = [d * scale for d in delta_theta]

        # Update joint angles
        joint_angles = [
            _normalize_angle(joint_angles[i] + delta_theta[i]) for i in range(6)
        ]

        # Clamp to joint limits
        joint_angles = _clamp_joint_angles(joint_angles)

    # Did not converge
    return IKResult(
        joint_angles=joint_angles,
        converged=False,
        iterations=config.max_iterations,
        position_error=pos_error,
        orientation_error=ori_error,
    )


def inverse_kinematics_from_pose(
    target_pose: Matrix4,
    initial_guess: Optional[List[float]] = None,
    config: Optional[IKConfig] = None,
) -> IKResult:
    """Compute IK from transformation matrix.

    Args:
        target_pose: 4x4 transformation matrix
        initial_guess: Starting joint angles
        config: IK solver configuration

    Returns:
        IKResult with solution
    """
    # Extract position
    x, y, z = target_pose[0][3], target_pose[1][3], target_pose[2][3]

    # Extract Euler angles
    roll, pitch, yaw = _rotation_matrix_to_euler_zyx(target_pose)

    return inverse_kinematics(x, y, z, roll, pitch, yaw, initial_guess, config)


def verify_ik_solution(
    target_x: float,
    target_y: float,
    target_z: float,
    target_roll: float,
    target_pitch: float,
    target_yaw: float,
    joint_angles: List[float],
) -> Tuple[float, float]:
    """Verify IK solution by computing FK and comparing.

    Returns:
        (position_error, orientation_error) in meters and radians
    """
    current_pose = _fk_transform(joint_angles)
    target_pose = pose_to_matrix(
        target_x, target_y, target_z, target_roll, target_pitch, target_yaw
    )
    _, pos_error, ori_error = _compute_pose_error(current_pose, target_pose)
    return pos_error, ori_error
