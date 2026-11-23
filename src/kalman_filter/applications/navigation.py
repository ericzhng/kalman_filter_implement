r"""Module: kalman_filter.applications.navigation

Description: Extended Kalman Filter applications in navigation & sensor fusion:
    - GPS/IMU integrated navigation with dynamic accelerometer bias estimation.
    - Attitude & heading estimation fusing gyroscope propagation with accelerometer
      and magnetometer vector corrections.

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

__all__ = ["AttitudeHeadingModel", "GPSIMUNavigationModel"]


class GPSIMUNavigationModel(NonlinearModel):
    r"""Integrated GPS/IMU navigation model with accelerometer bias tracking.

    State vector (dimension 7):
    :math:`[p_x, p_y, v_x, v_y, b_{ax}, b_{ay}, \psi]^T`
    where :math:`(p_x, p_y)` is 2D position, :math:`(v_x, v_y)` is velocity,
    :math:`(b_{ax}, b_{ay})` is body accelerometer bias, and :math:`\psi` is heading angle.

    Control input :math:`u = [a_{mx}, a_{my}, \omega_m]^T` consists of raw IMU body-frame
    accelerometer specific force and gyroscope yaw rate.

    Measurements: GPS position fixes and GPS ground track velocity:
    :math:`z = [p_x, p_y, v_x, v_y]^T`.

    Args:
        dt: IMU update rate in seconds.
        accel_noise_std: Accelerometer measurement noise intensity.
        gyro_noise_std: Gyroscope angular rate noise intensity.
        bias_walk_std: Accelerometer bias random-walk diffusion rate.
        gps_pos_std: GPS position measurement error standard deviation.
        gps_vel_std: GPS velocity measurement error standard deviation.
    """

    def __init__(
        self,
        dt: float = 0.02,
        accel_noise_std: float = 0.2,
        gyro_noise_std: float = 0.01,
        bias_walk_std: float = 1e-4,
        gps_pos_std: float = 2.5,
        gps_vel_std: float = 0.2,
    ) -> None:
        self.dt = dt

        def f_nav(x: FloatArray, u: FloatArray | None = None) -> FloatArray:
            px, py, vx, vy, bax, bay, psi = x
            if u is None:
                amx, amy, w_gyro = 0.0, 0.0, 0.0
            else:
                amx, amy, w_gyro = u

            # Subtract estimated sensor bias
            ax_body = amx - bax
            ay_body = amy - bay

            # Rotate to navigation frame
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)
            ax_nav = cos_psi * ax_body - sin_psi * ay_body
            ay_nav = sin_psi * ax_body + cos_psi * ay_body

            # Integrate kinematics
            px_next = px + vx * dt + 0.5 * ax_nav * dt**2
            py_next = py + vy * dt + 0.5 * ay_nav * dt**2
            vx_next = vx + ax_nav * dt
            vy_next = vy + ay_nav * dt
            bax_next = bax
            bay_next = bay
            psi_next = wrap_angle(psi + w_gyro * dt)

            return np.array([px_next, py_next, vx_next, vy_next, bax_next, bay_next, psi_next])

        def h_nav(x: FloatArray) -> FloatArray:
            return x[:4]

        def h_jac(x: FloatArray) -> FloatArray:
            H = np.zeros((4, 7))
            H[:4, :4] = np.eye(4)
            return H

        # Process noise covariance Q
        q_pos = (0.5 * accel_noise_std * dt**2) ** 2
        q_vel = (accel_noise_std * dt) ** 2
        q_bias = (bias_walk_std * dt) ** 2
        q_psi = (gyro_noise_std * dt) ** 2
        Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_bias, q_bias, q_psi])

        # Measurement noise covariance R
        R = np.diag([gps_pos_std**2, gps_pos_std**2, gps_vel_std**2, gps_vel_std**2])

        super().__init__(f=f_nav, h=h_nav, Q=Q, R=R, h_jac=h_jac)


class AttitudeHeadingModel(NonlinearModel):
    r"""Attitude and heading estimation model fusing gyroscopes, accelerometers, and magnetometers.

    State: :math:`[\phi, \theta, \psi, b_p, b_q, b_r]^T` representing roll, pitch, yaw
    (Euler angles) and 3-axis rate gyroscope biases.


    Control input :math:`u = [p, q, r]^T` contains angular rates from the 3-axis rate gyros.
    Measurements :math:`z = [a_x, a_y, a_z, m_x, m_y, m_z]^T` are 3-axis gravity acceleration
    and 3-axis geomagnetic field components in the body frame.

    Args:
        dt: Sample period in seconds.
        gyro_noise_std: Gyroscope white noise standard deviation.
        bias_drift_std: Gyroscope bias random-walk drift rate.
        accel_noise_std: Accelerometer noise standard deviation.
        mag_noise_std: Magnetometer noise standard deviation.
        gravity: Earth gravitational acceleration magnitude (m/s^2).
        mag_field: Reference Earth magnetic field vector in navigation frame
            :math:`[m_N, 0, m_D]^T`.

    """

    def __init__(
        self,
        dt: float = 0.01,
        gyro_noise_std: float = 0.005,
        bias_drift_std: float = 1e-4,
        accel_noise_std: float = 0.1,
        mag_noise_std: float = 0.05,
        gravity: float = 9.81,
        mag_field: FloatArray | None = None,
    ) -> None:
        self.dt = dt
        self.gravity = gravity
        self.mag_ref = (
            np.array([0.5, 0.0, 0.866]) if mag_field is None else np.asarray(mag_field, dtype=float)
        )

        def f_att(x: FloatArray, u: FloatArray | None = None) -> FloatArray:
            phi, theta, psi, bp, bq, br = x
            if u is None:
                p, q, r = 0.0, 0.0, 0.0
            else:
                p, q, r = u

            # Unbiased angular rates
            p_c = p - bp
            q_c = q - bq
            r_c = r - br

            # Euler angle kinematic rates
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            cos_th = max(np.cos(theta), 1e-4)
            tan_th = np.sin(theta) / cos_th

            d_phi = p_c + (q_c * sin_phi + r_c * cos_phi) * tan_th
            d_theta = q_c * cos_phi - r_c * sin_phi
            d_psi = (q_c * sin_phi + r_c * cos_phi) / cos_th

            phi_next = wrap_angle(phi + d_phi * dt)
            theta_next = np.clip(theta + d_theta * dt, -0.49 * np.pi, 0.49 * np.pi)
            psi_next = wrap_angle(psi + d_psi * dt)

            return np.array([phi_next, theta_next, psi_next, bp, bq, br])

        def h_att(x: FloatArray) -> FloatArray:
            phi, theta, psi = x[:3]
            c_p, s_p = np.cos(phi), np.sin(phi)
            c_t, s_t = np.cos(theta), np.sin(theta)
            c_y, s_y = np.cos(psi), np.sin(psi)

            # Rotation matrix from Navigation to Body frame (R_nb^T)
            # Gravity in body frame: R_nb^T * [0, 0, -g]^T
            h_acc = np.array(
                [-self.gravity * s_t, self.gravity * c_t * s_p, self.gravity * c_t * c_p]
            )


            # Magnetic field in body frame
            mN, _, mD = self.mag_ref
            h_mag = np.array(
                [
                    mN * c_t * c_y - mD * s_t,
                    mN * (s_p * s_t * c_y - c_p * s_y) + mD * s_p * c_t,
                    mN * (c_p * s_t * c_y + s_p * s_y) + mD * c_p * c_t,
                ]
            )
            return np.concatenate([h_acc, h_mag])

        Q = np.diag(
            [
                (gyro_noise_std * dt) ** 2,
                (gyro_noise_std * dt) ** 2,
                (gyro_noise_std * dt) ** 2,
                (bias_drift_std * dt) ** 2,
                (bias_drift_std * dt) ** 2,
                (bias_drift_std * dt) ** 2,
            ]
        )
        R = np.diag(
            [
                accel_noise_std**2,
                accel_noise_std**2,
                accel_noise_std**2,
                mag_noise_std**2,
                mag_noise_std**2,
                mag_noise_std**2,
            ]
        )

        super().__init__(f=f_att, h=h_att, Q=Q, R=R)
