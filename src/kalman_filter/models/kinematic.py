r"""Module: kalman_filter.models.kinematic

Description: Standard kinematic target-tracking and parameter estimation models:
    - Constant Acceleration (CA) model for maneuvering vehicles.
    - Coordinated Turn (CT) model with turn rate state estimation and bearing wrapping.
    - Random Walk model for parameter identification, bias drift, and static state filtering.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.models.linear import LinearModel
from kalman_filter.models.nonlinear import NonlinearModel
from kalman_filter.models.sensors import wrap_angle
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = [
    "constant_acceleration_model",
    "coordinated_turn_model",
    "random_walk_model",
]


def constant_acceleration_model(
    dim: int,
    dt: float,
    jerk_std: float,
    H: FloatArray,
    R: FloatArray,
) -> LinearModel:
    r"""Builds a constant-acceleration (CA) motion model for maneuvering targets.

    State vector has size ``3 * dim`` ordered as:
    ``[pos_1..d, vel_1..d, acc_1..d]``.
    Process noise is modeled as a continuous-time white-noise jerk integrated over ``dt``.

    Args:
        dim: Spatial dimension :math:`d \in \{1, 2, 3\}`.
        dt: Sample period in seconds.
        jerk_std: Standard deviation of process jerk noise :math:`\sigma_j`.
        H: Measurement matrix, shape ``(m, 3 * dim)``.
        R: Measurement noise covariance matrix, shape ``(m, m)``.

    Returns:
        Assembled ``LinearModel``.

    Raises:
        ValueError: If ``dim < 1`` or ``dt <= 0``.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    # 1D transition blocks
    eye_d = np.eye(dim)
    F = np.block(
        [
            [eye_d, dt * eye_d, 0.5 * dt**2 * eye_d],
            [np.zeros((dim, dim)), eye_d, dt * eye_d],
            [np.zeros((dim, dim)), np.zeros((dim, dim)), eye_d],
        ]
    )

    # Continuous white-noise jerk integrated over sample period dt
    q11 = dt**5 / 20.0
    q12 = dt**4 / 8.0
    q13 = dt**3 / 6.0
    q22 = dt**3 / 3.0
    q23 = dt**2 / 2.0
    q33 = dt

    Q_1d = np.array(
        [
            [q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33],
        ]
    )
    Q = jerk_std**2 * np.kron(Q_1d, eye_d)

    return LinearModel(F=F, H=H, Q=Q, R=R)


def random_walk_model(
    dim: int,
    q_var: float | FloatArray,
    r_var: float | FloatArray,
) -> LinearModel:
    r"""Builds a random-walk state-space model :math:`x_k = x_{k-1} + w, z_k = x_k + v`.

    Commonly used for online sensor bias tracking, parameter estimation, and calibration.

    Args:
        dim: Dimension of parameter/state vector ``n``.
        q_var: Process noise variance (scalar or 1D array of shape ``(dim,)``).
        r_var: Measurement noise variance (scalar or 1D array of shape ``(dim,)``).

    Returns:
        Assembled ``LinearModel`` with :math:`F = I_n, H = I_n`.
    """
    F = np.eye(dim)
    H = np.eye(dim)
    q = np.broadcast_to(np.asarray(q_var, dtype=float), (dim,))
    r = np.broadcast_to(np.asarray(r_var, dtype=float), (dim,))
    Q = np.diag(q)
    R = np.diag(r)
    return LinearModel(F=F, H=H, Q=Q, R=R)


def coordinated_turn_model(
    dt: float,
    accel_std: float,
    turn_rate_noise_std: float,
    stations: FloatArray | None = None,
    R: FloatArray | None = None,
) -> NonlinearModel:
    r"""Builds a 2D coordinated-turn (CT) model with turn rate state estimation.

    State vector is :math:`[x, y, v_x, v_y, \omega]^T`, where :math:`\omega` is the turn rate.
    If ``stations`` is provided (coordinates shape ``(S, 2)``), the measurement function
    computes range and bearing to each station:
    :math:`[r_1, \theta_1, \dots, r_S, \theta_S]^T` with bearing wrapped to :math:`[-\pi, \pi)`.
    Otherwise, directly measures Cartesian position :math:`[x, y]^T`.

    Args:
        dt: Sample period in seconds.
        accel_std: Standard deviation of linear acceleration noise.
        turn_rate_noise_std: Standard deviation of turn-rate random walk noise.
        stations: Optional sensor station coordinates, shape ``(S, 2)``.
        R: Measurement noise covariance matrix. If None, default identity matrix is used.

    Returns:
        Assembled ``NonlinearModel`` with analytic Jacobians and bearing residual wrapping.
    """

    def f_ct(x: FloatArray, u: FloatArray | None = None) -> FloatArray:
        px, py, vx, vy, w = x
        wt = w * dt
        # Use sinc to prevent division by zero as w -> 0
        s = dt * np.sinc(wt / np.pi)
        c = 0.5 * w * dt**2 * np.sinc(wt / (2.0 * np.pi)) ** 2
        cw, sw = np.cos(wt), np.sin(wt)
        return np.array(
            [
                px + s * vx - c * vy,
                py + c * vx + s * vy,
                cw * vx - sw * vy,
                sw * vx + cw * vy,
                w,
            ]
        )

    # Process noise covariance
    G_v = np.vstack([0.5 * dt**2 * np.eye(2), dt * np.eye(2)])
    Q_v = accel_std**2 * (G_v @ G_v.T)
    Q = np.zeros((5, 5))
    Q[:4, :4] = Q_v
    Q[4, 4] = (turn_rate_noise_std * dt) ** 2

    if stations is not None:
        st_arr = np.atleast_2d(np.asarray(stations, dtype=float))
        num_stations = st_arr.shape[0]
        m_dim = 2 * num_stations
        if R is None:
            R = np.eye(m_dim)

        def h_meas(x: FloatArray) -> FloatArray:
            d = x[None, :2] - st_arr
            ranges = np.hypot(d[:, 0], d[:, 1])
            bearings = np.arctan2(d[:, 1], d[:, 0])
            return np.column_stack([ranges, bearings]).ravel()

        def h_jac(x: FloatArray) -> FloatArray:
            jac = np.zeros((m_dim, 5))
            for i, (dx, dy) in enumerate(x[None, :2] - st_arr):
                r2 = max(dx**2 + dy**2, 1e-12)
                r = np.sqrt(r2)
                jac[2 * i, :2] = [dx / r, dy / r]
                jac[2 * i + 1, :2] = [-dy / r2, dx / r2]
            return jac

        def res_fn(z: FloatArray, z_pred: FloatArray) -> FloatArray:
            nu = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
            nu[1::2] = wrap_angle(nu[1::2])
            return nu

        return NonlinearModel(
            f=f_ct,
            h=h_meas,
            Q=Q,
            R=R,
            h_jac=h_jac,
            residual_fn=res_fn,
        )

    # Default: Cartesian position measurement
    if R is None:
        R = np.eye(2)

    def h_cart(x: FloatArray) -> FloatArray:
        return x[:2]

    def h_cart_jac(x: FloatArray) -> FloatArray:
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    return NonlinearModel(
        f=f_ct,
        h=h_cart,
        Q=Q,
        R=R,
        h_jac=h_cart_jac,
    )
