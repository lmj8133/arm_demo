"""Jacobian matrix computation for Piper arm.

Provides numerical Jacobian calculation for velocity kinematics and IK.
"""

import math
from typing import List, Tuple

try:
    from .kinematics import (
        PIPER_DH_PARAMS,
        Matrix4,
        _identity4,
        _mat4_mult,
        _dh_transform_modified,
    )
except ImportError:
    from kinematics import (
        PIPER_DH_PARAMS,
        Matrix4,
        _identity4,
        _mat4_mult,
        _dh_transform_modified,
    )

# Type alias for 6x6 Jacobian matrix
Jacobian6x6 = List[List[float]]

# Finite difference step for numerical Jacobian
DELTA_THETA = 1e-6


def _zeros(rows: int, cols: int) -> List[List[float]]:
    """Create a zero matrix."""
    return [[0.0] * cols for _ in range(rows)]


def _compute_all_transforms(joint_angles: List[float]) -> List[Matrix4]:
    """Compute cumulative transformation matrices for all joints.

    Returns list of T_0_i for i = 0..6 (7 matrices total).
    T_0_0 is identity, T_0_6 is the end-effector pose.
    """
    transforms = [_identity4()]

    T_cumulative = _identity4()
    for theta, dh in zip(joint_angles, PIPER_DH_PARAMS):
        theta_total = theta + dh.theta_offset
        T_i = _dh_transform_modified(dh.alpha, dh.a, dh.d, theta_total)
        T_cumulative = _mat4_mult(T_cumulative, T_i)
        transforms.append([row[:] for row in T_cumulative])  # Deep copy

    return transforms


def _extract_position(T: Matrix4) -> Tuple[float, float, float]:
    """Extract position from transformation matrix."""
    return T[0][3], T[1][3], T[2][3]


def _extract_z_axis(T: Matrix4) -> Tuple[float, float, float]:
    """Extract z-axis (rotation axis for revolute joint) from transformation."""
    return T[0][2], T[1][2], T[2][2]


def _cross_product(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Compute cross product a × b."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _compute_rotation_axes(
    joint_angles: List[float],
) -> List[Tuple[float, float, float]]:
    """Compute rotation axes for Modified DH convention.

    In Modified DH, joint i rotates around the z-axis of frame i (after alpha rotation).
    This requires computing intermediate transforms that include the alpha rotation
    but NOT the joint rotation.
    """
    axes = []

    T = _identity4()
    for i, dh in enumerate(PIPER_DH_PARAMS):
        # For Modified DH: T_i = Rx(alpha) * Dx(a) * Rz(theta) * Dz(d)
        # The rotation axis for joint i is the z-axis AFTER applying Rx(alpha) * Dx(a)
        # to the previous frame.

        if i == 0:
            # First joint rotates around base z-axis
            axes.append((0.0, 0.0, 1.0))
        else:
            # Apply alpha rotation from previous joint's DH params
            prev_dh = PIPER_DH_PARAMS[i - 1]
            ca, sa = math.cos(prev_dh.alpha), math.sin(prev_dh.alpha)
            # z-axis after Rx(alpha): [0, -sin(alpha), cos(alpha)] in previous frame
            # Transform to world frame
            z_local = (0.0, -sa, ca)
            # Rotate by accumulated transformation T
            z_world = (
                T[0][0] * z_local[0] + T[0][1] * z_local[1] + T[0][2] * z_local[2],
                T[1][0] * z_local[0] + T[1][1] * z_local[1] + T[1][2] * z_local[2],
                T[2][0] * z_local[0] + T[2][1] * z_local[1] + T[2][2] * z_local[2],
            )
            axes.append(z_world)

        # Update cumulative transform
        theta = joint_angles[i] + dh.theta_offset
        T_i = _dh_transform_modified(dh.alpha, dh.a, dh.d, theta)
        T = _mat4_mult(T, T_i)

    return axes


def compute_jacobian_analytical(joint_angles: List[float]) -> Jacobian6x6:
    """Compute analytical Jacobian matrix for Modified DH convention.

    The Jacobian relates joint velocities to end-effector velocities:
        [v]   [Jv]
        [ω] = [Jω] * q̇

    For Modified DH revolute joints:
        Jv_i = z_i × (p_e - p_i)
        Jω_i = z_i
    where z_i is the rotation axis for joint i in world frame.

    Args:
        joint_angles: 6 joint angles in radians

    Returns:
        6x6 Jacobian matrix (rows: [vx, vy, vz, ωx, ωy, ωz], cols: joints)
    """
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")

    transforms = _compute_all_transforms(joint_angles)
    p_end = _extract_position(transforms[6])

    # Compute rotation axes for Modified DH
    rotation_axes = _compute_rotation_axes(joint_angles)

    J = _zeros(6, 6)

    for i in range(6):
        # Rotation axis for joint i+1 (0-indexed)
        z_i = rotation_axes[i]
        # Position of frame i origin
        p_i = _extract_position(transforms[i])

        # Vector from joint i to end-effector
        p_diff = (p_end[0] - p_i[0], p_end[1] - p_i[1], p_end[2] - p_i[2])

        # Linear velocity component: z_i × (p_end - p_i)
        jv = _cross_product(z_i, p_diff)

        # Fill Jacobian columns
        J[0][i] = jv[0]  # vx
        J[1][i] = jv[1]  # vy
        J[2][i] = jv[2]  # vz
        J[3][i] = z_i[0]  # ωx
        J[4][i] = z_i[1]  # ωy
        J[5][i] = z_i[2]  # ωz

    return J


def _rotation_to_axis_angle(T: Matrix4) -> Tuple[float, float, float]:
    """Convert rotation matrix to axis-angle representation (angle * axis)."""
    # Extract rotation matrix
    R00, R01, R02 = T[0][0], T[0][1], T[0][2]
    R10, R11, R12 = T[1][0], T[1][1], T[1][2]
    R20, R21, R22 = T[2][0], T[2][1], T[2][2]

    # Compute rotation angle
    trace = R00 + R11 + R22
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))

    if abs(angle) < 1e-10:
        return (0.0, 0.0, 0.0)

    if abs(angle - math.pi) < 1e-6:
        # Near 180 degrees, use diagonal elements
        axis = (
            math.sqrt(max(0, (R00 + 1) / 2)),
            math.sqrt(max(0, (R11 + 1) / 2)),
            math.sqrt(max(0, (R22 + 1) / 2)),
        )
        return (axis[0] * angle, axis[1] * angle, axis[2] * angle)

    # General case
    k = 1.0 / (2.0 * math.sin(angle))
    axis = (k * (R21 - R12), k * (R02 - R20), k * (R10 - R01))

    return (axis[0] * angle, axis[1] * angle, axis[2] * angle)


def _fk_transform(joint_angles: List[float]) -> Matrix4:
    """Compute forward kinematics transformation matrix."""
    T = _identity4()
    for theta, dh in zip(joint_angles, PIPER_DH_PARAMS):
        theta_total = theta + dh.theta_offset
        T_i = _dh_transform_modified(dh.alpha, dh.a, dh.d, theta_total)
        T = _mat4_mult(T, T_i)
    return T


def compute_jacobian_numerical(
    joint_angles: List[float], delta: float = DELTA_THETA
) -> Jacobian6x6:
    """Compute numerical Jacobian using finite differences.

    Uses central difference for better accuracy:
        J_ij ≈ (f(θ + δe_j) - f(θ - δe_j)) / (2δ)

    Args:
        joint_angles: 6 joint angles in radians
        delta: Finite difference step size

    Returns:
        6x6 Jacobian matrix
    """
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")

    J = _zeros(6, 6)

    for j in range(6):
        # Perturb joint j
        theta_plus = joint_angles[:]
        theta_minus = joint_angles[:]
        theta_plus[j] += delta
        theta_minus[j] -= delta

        # Compute FK at perturbed positions
        T_plus = _fk_transform(theta_plus)
        T_minus = _fk_transform(theta_minus)

        # Position difference
        p_plus = _extract_position(T_plus)
        p_minus = _extract_position(T_minus)

        # Orientation difference (axis-angle) in world frame
        # Compute relative rotation: R_diff = R_plus * R_minus^T
        R_diff = _mat4_mult(T_plus, _transpose_rotation(T_minus))
        omega = _rotation_to_axis_angle(R_diff)

        inv_2delta = 1.0 / (2.0 * delta)

        # Linear velocity columns
        J[0][j] = (p_plus[0] - p_minus[0]) * inv_2delta
        J[1][j] = (p_plus[1] - p_minus[1]) * inv_2delta
        J[2][j] = (p_plus[2] - p_minus[2]) * inv_2delta

        # Angular velocity columns
        J[3][j] = omega[0] * inv_2delta
        J[4][j] = omega[1] * inv_2delta
        J[5][j] = omega[2] * inv_2delta

    return J


def _transpose_rotation(T: Matrix4) -> Matrix4:
    """Create transformation with transposed rotation (inverse for orthogonal)."""
    return [
        [T[0][0], T[1][0], T[2][0], 0.0],
        [T[0][1], T[1][1], T[2][1], 0.0],
        [T[0][2], T[1][2], T[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def compute_jacobian(
    joint_angles: List[float], method: str = "numerical"
) -> Jacobian6x6:
    """Compute Jacobian matrix.

    Args:
        joint_angles: 6 joint angles in radians
        method: "numerical" (recommended) or "analytical" (experimental)

    Returns:
        6x6 Jacobian matrix

    Example:
        from piper_demo.jacobian import compute_jacobian
        J = compute_jacobian([0, 0.5, -0.5, 0, 0, 0])

    Note:
        The numerical method is recommended as it correctly handles the
        Modified DH convention used by the Piper arm. The analytical method
        is experimental and may have accuracy issues.
    """
    if method == "numerical":
        return compute_jacobian_numerical(joint_angles)
    elif method == "analytical":
        return compute_jacobian_analytical(joint_angles)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'numerical' or 'analytical'.")


def jacobian_determinant(J: Jacobian6x6) -> float:
    """Compute determinant of 6x6 Jacobian (for singularity detection).

    Note: Uses simple LU-style computation. For production, use numpy.
    """
    # Make a copy to avoid modifying original
    A = [row[:] for row in J]
    n = 6
    det = 1.0

    for i in range(n):
        # Find pivot
        max_val = abs(A[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > max_val:
                max_val = abs(A[k][i])
                max_row = k

        if max_val < 1e-12:
            return 0.0

        # Swap rows
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            det *= -1

        det *= A[i][i]

        # Eliminate
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]

    return det


def is_near_singularity(joint_angles: List[float], threshold: float = 0.001) -> bool:
    """Check if configuration is near a singularity.

    Args:
        joint_angles: 6 joint angles in radians
        threshold: Determinant threshold for singularity detection

    Returns:
        True if near singularity
    """
    J = compute_jacobian_numerical(joint_angles)
    det = abs(jacobian_determinant(J))
    return det < threshold
