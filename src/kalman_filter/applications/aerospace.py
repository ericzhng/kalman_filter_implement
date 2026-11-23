r"""Module: kalman_filter.applications.aerospace

Description: Extended Kalman Filter applications in aerospace and multi-target tracking:
    - 3D flight/drone navigation with aerodynamic drag and specific thrust controls.
    - Spherical radar tracking with Cartesian kinematics and 3D spherical observation
      geometry (range, azimuth, elevation, Doppler range-rate).

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging

import numpy as np

from kalman_filter.models.nonlinear import NonlinearModel
from kalman_filter.models.sensors import wrap_angle
from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = ["DroneFlightModel", "SphericalRadarTrackingModel"]


class SphericalRadarTrackingModel(NonlinearModel):
    r"""3D aerospace radar tracking with spherical observation geometry.

    State vector (dimension 6): :math:`[x, y, z, v_x, v_y, v_z]^T` in Cartesian coordinates.
    Radar is located at :math:`s_{\text{radar}} = [s_x, s_y, s_z]^T`.

    Measurements are spherical detections (dimension 4):
        - Range: :math:`r = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}`
        - Azimuth: :math:`\theta = \text{wrap}(\text{atan2}(\Delta y, \Delta x))`
        - Elevation: :math:`\phi = \text{atan2}(\Delta z, \sqrt{\Delta x^2 + \Delta y^2})`
        - Doppler: :math:`\dot{r} = (\Delta x v_x + \Delta y v_y + \Delta z v_z) / r`

    Includes full analytic Jacobians for spherical coordinates with elevation singularity
    protection and angular innovation wrapping on azimuth and elevation channels.

    Args:
        dt: Sample period in seconds.
        accel_noise_std: Process noise acceleration spectral density.
        radar_pos: Coordinates of the radar station, shape ``(3,)``.
        range_std: Range measurement noise standard deviation in meters.
        azimuth_std: Azimuth measurement noise standard deviation in radians.
        elevation_std: Elevation measurement noise standard deviation in radians.
        doppler_std: Doppler range-rate noise standard deviation in m/s.
    """

    def __init__(
        self,
        dt: float = 0.1,
        accel_noise_std: float = 1.0,
        radar_pos: FloatArray | None = None,
        range_std: float = 5.0,
        azimuth_std: float = 0.005,
        elevation_std: float = 0.005,
        doppler_std: float = 0.5,
    ) -> None:
        self.dt = dt
        self.radar_pos = np.zeros(3) if radar_pos is None else np.asarray(radar_pos, dtype=float)

        def f_kin(x: FloatArray, u: FloatArray | None = None) -> FloatArray:
            pos = x[:3] + x[3:] * dt
            vel = x[3:]
            return np.concatenate([pos, vel])

        def h_spherical(x: FloatArray) -> FloatArray:
            dx, dy, dz = x[:3] - self.radar_pos
            vx, vy, vz = x[3:]

            d_horiz = max(np.hypot(dx, dy), 1e-9)
            r = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-9)
            azimuth = np.arctan2(dy, dx)
            elevation = np.arctan2(dz, d_horiz)
            range_rate = (dx * vx + dy * vy + dz * vz) / r

            return np.array([r, azimuth, elevation, range_rate])

        def h_jac(x: FloatArray) -> FloatArray:
            dx, dy, dz = x[:3] - self.radar_pos
            vx, vy, vz = x[3:]

            d_h2 = max(dx**2 + dy**2, 1e-12)
            d_h = np.sqrt(d_h2)
            r2 = max(dx**2 + dy**2 + dz**2, 1e-12)
            r = np.sqrt(r2)
            r3 = r * r2

            H = np.zeros((4, 6))

            # Range row: dr/dpos = d/r, dr/dvel = 0
            H[0, :3] = [dx / r, dy / r, dz / r]

            # Azimuth row: d_az/dx = -dy/d_h2, d_az/dy = dx/d_h2
            H[1, 0] = -dy / d_h2
            H[1, 1] = dx / d_h2

            # Elevation row: d_el/dpos
            H[2, 0] = -dx * dz / (r2 * d_h)
            H[2, 1] = -dy * dz / (r2 * d_h)
            H[2, 2] = d_h / r2

            # Doppler range rate row:
            v_dot_d = dx * vx + dy * vy + dz * vz
            H[3, 0] = vx / r - dx * v_dot_d / r3
            H[3, 1] = vy / r - dy * v_dot_d / r3
            H[3, 2] = vz / r - dz * v_dot_d / r3
            H[3, 3:] = [dx / r, dy / r, dz / r]

            return H

        def res_spherical(z: FloatArray, z_pred: FloatArray) -> FloatArray:
            nu = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
            nu[1] = wrap_angle(nu[1])  # Azimuth
            nu[2] = wrap_angle(nu[2])  # Elevation
            return nu

        # Process noise covariance
        G_1d = np.array([[0.5 * dt**2], [dt]])
        Q_block = accel_noise_std**2 * (G_1d @ G_1d.T)
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = Q_block[0, 0]
            Q[i, i + 3] = Q_block[0, 1]
            Q[i + 3, i] = Q_block[1, 0]
            Q[i + 3, i + 3] = Q_block[1, 1]

        # Measurement noise covariance
        R = np.diag([range_std**2, azimuth_std**2, elevation_std**2, doppler_std**2])

        super().__init__(
            f=f_kin,
            h=h_spherical,
            Q=Q,
            R=R,
            h_jac=h_jac,
            residual_fn=res_spherical,
        )


class DroneFlightModel(NonlinearModel):
    r"""3D flight trajectory model for quadrotor/fixed-wing drones with aerodynamic drag.

    State vector (dimension 6): :math:`[x, y, z, v_x, v_y, v_z]^T`.
    Control input :math:`u = [a_x, a_y, a_z]^T` provides commanded specific acceleration/thrust.
    Kinematics include linear aerodynamic drag:
    :math:`\dot{v} = u - c_d \|v\| v + g`.

    Args:
        dt: Sample period in seconds.
        drag_coeff: Quadratic aerodynamic drag coefficient :math:`c_d \ge 0`.
        gravity: Earth gravity magnitude along negative z-axis.
        pos_noise_std: Measurement noise standard deviation for GPS/optical tracking.
        process_std: Standard deviation of unmodeled wind gusts / atmospheric turbulence.
    """

    def __init__(
        self,
        dt: float = 0.05,
        drag_coeff: float = 0.02,
        gravity: float = 9.81,
        pos_noise_std: float = 0.5,
        process_std: float = 0.2,
    ) -> None:
        self.dt = dt
        self.cd = drag_coeff
        self.g = np.array([0.0, 0.0, -gravity])

        def f_drone(x: FloatArray, u: FloatArray | None = None) -> FloatArray:
            pos = x[:3]
            vel = x[3:]
            thrust = np.zeros(3) if u is None else np.asarray(u, dtype=float)

            speed = np.linalg.norm(vel)
            drag = -self.cd * speed * vel
            acc = thrust + drag + self.g

            pos_next = pos + vel * dt + 0.5 * acc * dt**2
            vel_next = vel + acc * dt
            return np.concatenate([pos_next, vel_next])

        def h_drone(x: FloatArray) -> FloatArray:
            return x[:3]

        def h_jac(x: FloatArray) -> FloatArray:
            H = np.zeros((3, 6))
            H[:3, :3] = np.eye(3)
            return H

        Q = (process_std**2) * np.eye(6)
        R = (pos_noise_std**2) * np.eye(3)

        super().__init__(f=f_drone, h=h_drone, Q=Q, R=R, h_jac=h_jac)
