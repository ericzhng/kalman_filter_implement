r"""Module: kalman_filter.applications.radar_tracking

Description: Standard Kalman Filter applications in target & radar tracking.
    Estimates 2D/3D Cartesian kinematics (position, velocity, acceleration) from
    heterogeneous radar and sensor detections with Doppler velocity and dropout handling.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.models.linear import LinearModel

logger = logging.getLogger(__name__)


__all__ = ["RadarTrackingModel", "create_3d_radar_tracker"]


class RadarTrackingModel(LinearModel):
    r"""Linear multi-channel radar tracking model in 2D or 3D Cartesian coordinates.

    State vector:
        - 2D: :math:`[x, y, v_x, v_y, a_x, a_y]^T` (dimension 6)
        - 3D: :math:`[x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T` (dimension 9)

    Process noise models continuous white-noise jerk integrated over time step ``dt``.
    Measurements can include Cartesian position fixes and/or linear Doppler velocity readings.

    Args:
        dim: Spatial dimension (2 or 3).
        dt: Radar sampling interval in seconds.
        jerk_std: Process noise jerk intensity.
        measure_velocity: Whether radar sensors directly measure velocity (e.g. Doppler).
        pos_noise_std: Measurement noise standard deviation for position channels.
        vel_noise_std: Measurement noise standard deviation for velocity channels.
    """

    def __init__(
        self,
        dim: int = 3,
        dt: float = 0.1,
        jerk_std: float = 1.0,
        measure_velocity: bool = True,
        pos_noise_std: float = 5.0,
        vel_noise_std: float = 1.0,
    ) -> None:
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.spatial_dim = dim
        self.dt = dt

        # State transition: pos -> vel -> acc
        eye = np.eye(dim)
        F = np.block(
            [
                [eye, dt * eye, 0.5 * dt**2 * eye],
                [np.zeros((dim, dim)), eye, dt * eye],
                [np.zeros((dim, dim)), np.zeros((dim, dim)), eye],
            ]
        )

        # Process noise covariance (continuous white-noise jerk integrated)
        q11 = dt**5 / 20.0
        q12 = dt**4 / 8.0
        q13 = dt**3 / 6.0
        q22 = dt**3 / 3.0
        q23 = dt**2 / 2.0
        q33 = dt
        Q_block = jerk_std**2 * np.array(
            [
                [q11, q12, q13],
                [q12, q22, q23],
                [q13, q23, q33],
            ]
        )
        Q = np.kron(Q_block, eye)

        # Measurement matrix H and noise covariance R
        pos_H = np.block([eye, np.zeros((dim, 2 * dim))])
        if measure_velocity:
            vel_H = np.block([np.zeros((dim, dim)), eye, np.zeros((dim, dim))])
            H = np.vstack([pos_H, vel_H])
            r_diag = np.concatenate(
                [
                    np.full(dim, pos_noise_std**2),
                    np.full(dim, vel_noise_std**2),
                ]
            )
        else:
            H = pos_H
            r_diag = np.full(dim, pos_noise_std**2)

        R = np.diag(r_diag)
        super().__init__(F=F, H=H, Q=Q, R=R)


def create_3d_radar_tracker(
    dt: float = 0.05,
    pos_std: float = 10.0,
    vel_std: float = 1.0,
    jerk_std: float = 0.5,
) -> RadarTrackingModel:
    r"""Convenience factory creating a 3D radar tracking model with Doppler velocity channels.

    Args:
        dt: Radar update period in seconds.
        pos_std: Position measurement noise standard deviation in meters.
        vel_std: Doppler velocity measurement noise standard deviation in m/s.
        jerk_std: Process jerk noise spectral density in m/s^3.

    Returns:
        Configured ``RadarTrackingModel``.
    """
    return RadarTrackingModel(
        dim=3,
        dt=dt,
        jerk_std=jerk_std,
        measure_velocity=True,
        pos_noise_std=pos_std,
        vel_noise_std=vel_std,
    )
