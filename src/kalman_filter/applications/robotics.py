r"""Module: kalman_filter.applications.robotics

Description: Extended Kalman Filter applications in robotics and autonomous vehicles:
    - Wheeled dead reckoning with dynamic wheel slip estimation.
    - Feature-based 2D EKF-SLAM (Simultaneous Localization and Mapping).

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

__all__ = ["EKFSLAMModel", "WheeledRobotSlipModel"]


class WheeledRobotSlipModel(NonlinearModel):
    r"""Differential-drive wheeled mobile robot with dynamic wheel slip estimation.

    State vector (dimension 5):
    :math:`[x, y, \theta, s_L, s_R]^T`
    where :math:`(x, y)` is 2D position, :math:`\theta` is robot heading,
    and :math:`s_L, s_R \in (-1, 1)` are longitudinal wheel slip factors.

    Control input :math:`u = [v_L, v_R]^T` contains commanded wheel velocities.
    Effective wheel speeds after slip:
    :math:`v_{L,\text{eff}} = (1 - s_L) v_L`, :math:`v_{R,\text{eff}} = (1 - s_R) v_R`.

    Robot kinematics:
    .. math::
        v = 0.5 (v_{R,\text{eff}} + v_{L,\text{eff}}), \quad
        \omega = (v_{R,\text{eff}} - v_{L,\text{eff}}) / W

    where :math:`W` is track width / wheelbase.

    Measurements: External pose / beacon / GPS fixes :math:`z = [x, y, \theta]^T`.

    Args:
        wheelbase: Distance between left and right drive wheels :math:`W`.
        dt: Sample period in seconds.
        slip_drift_std: Slip random-walk diffusion rate standard deviation.
        meas_pos_std: External position measurement noise standard deviation.
        meas_heading_std: External heading measurement noise standard deviation.
    """

    def __init__(
        self,
        wheelbase: float = 0.5,
        dt: float = 0.05,
        slip_drift_std: float = 1e-3,
        meas_pos_std: float = 0.1,
        meas_heading_std: float = 0.02,
    ) -> None:
        self.W = wheelbase
        self.dt = dt

        def f_slip(x: FloatArray, u: FloatArray | None = None) -> FloatArray:
            rx, ry, th, sL, sR = x
            if u is None:
                vL, vR = 0.0, 0.0
            else:
                vL, vR = u

            vL_eff = (1.0 - sL) * vL
            vR_eff = (1.0 - sR) * vR
            v = 0.5 * (vR_eff + vL_eff)
            w = (vR_eff - vL_eff) / self.W

            rx_next = rx + v * dt * np.cos(th + 0.5 * w * dt)
            ry_next = ry + v * dt * np.sin(th + 0.5 * w * dt)
            th_next = wrap_angle(th + w * dt)
            sL_next = np.clip(sL, -0.9, 0.9)
            sR_next = np.clip(sR, -0.9, 0.9)

            return np.array([rx_next, ry_next, th_next, sL_next, sR_next])

        def h_pose(x: FloatArray) -> FloatArray:
            return x[:3]

        def res_pose(z: FloatArray, z_pred: FloatArray) -> FloatArray:
            nu = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
            nu[2] = wrap_angle(nu[2])
            return nu

        # Covariances
        Q = np.diag(
            [
                (0.01 * dt) ** 2,
                (0.01 * dt) ** 2,
                (0.005 * dt) ** 2,
                (slip_drift_std * dt) ** 2,
                (slip_drift_std * dt) ** 2,
            ]
        )
        R = np.diag([meas_pos_std**2, meas_pos_std**2, meas_heading_std**2])

        super().__init__(f=f_slip, h=h_pose, Q=Q, R=R, residual_fn=res_pose)


class EKFSLAMModel(NonlinearModel):
    r"""Feature-based 2D EKF-SLAM (Simultaneous Localization and Mapping).

    State vector (dimension :math:`3 + 2M`):
    :math:`[x_r, y_r, \theta_r, l_{1x}, l_{1y}, \dots, l_{Mx}, l_{My}]^T`
    composed of robot pose and positions of :math:`M` static landmarks.

    Control input :math:`u = [v, \omega]^T` contains linear forward velocity and yaw rate.
    Measurements :math:`z = [r_1, \phi_1, \dots, r_M, \phi_M]^T` contain range and bearing
    from robot to each visible landmark:
    .. math::
        r_i = \sqrt{(l_{ix} - x_r)^2 + (l_{iy} - y_r)^2}, \quad
        \phi_i = \text{wrap}(\text{atan2}(l_{iy} - y_r, l_{ix} - x_r) - \theta_r)

    Args:
        num_landmarks: Number of static landmarks :math:`M`.
        dt: Sample period in seconds.
        v_noise_std: Velocity control input noise standard deviation.
        w_noise_std: Angular rate control input noise standard deviation.
        range_noise_std: Range sensor noise standard deviation.
        bearing_noise_std: Bearing sensor noise standard deviation.
    """

    def __init__(
        self,
        num_landmarks: int = 5,
        dt: float = 0.1,
        v_noise_std: float = 0.1,
        w_noise_std: float = 0.05,
        range_noise_std: float = 0.2,
        bearing_noise_std: float = 0.02,
    ) -> None:
        self.num_landmarks = num_landmarks
        self.dt = dt
        n_state = 3 + 2 * num_landmarks
        m_meas = 2 * num_landmarks

        def f_slam(x: FloatArray, u: FloatArray | None = None) -> FloatArray:
            xr, yr, thr = x[:3]
            if u is None:
                v, w = 0.0, 0.0
            else:
                v, w = u

            xr_next = xr + v * dt * np.cos(thr)
            yr_next = yr + v * dt * np.sin(thr)
            thr_next = wrap_angle(thr + w * dt)

            out = x.copy()
            out[0] = xr_next
            out[1] = yr_next
            out[2] = thr_next
            # Landmarks are stationary: out[3:] remain unchanged
            return out

        def h_slam(x: FloatArray) -> FloatArray:
            xr, yr, thr = x[:3]
            meas_list = []
            for i in range(num_landmarks):
                lx = x[3 + 2 * i]
                ly = x[4 + 2 * i]
                dx = lx - xr
                dy = ly - yr
                r = np.hypot(dx, dy)
                bearing = wrap_angle(np.arctan2(dy, dx) - thr)
                meas_list.extend([r, bearing])
            return np.array(meas_list)

        def h_jac(x: FloatArray) -> FloatArray:
            xr, yr, thr = x[:3]
            H = np.zeros((m_meas, n_state))
            for i in range(num_landmarks):
                lx = x[3 + 2 * i]
                ly = x[4 + 2 * i]
                dx = lx - xr
                dy = ly - yr
                q = max(dx**2 + dy**2, 1e-12)
                sq = np.sqrt(q)

                # Row for range r_i
                H[2 * i, 0] = -dx / sq
                H[2 * i, 1] = -dy / sq
                H[2 * i, 2] = 0.0
                H[2 * i, 3 + 2 * i] = dx / sq
                H[2 * i, 4 + 2 * i] = dy / sq

                # Row for bearing phi_i
                H[2 * i + 1, 0] = dy / q
                H[2 * i + 1, 1] = -dx / q
                H[2 * i + 1, 2] = -1.0
                H[2 * i + 1, 3 + 2 * i] = -dy / q
                H[2 * i + 1, 4 + 2 * i] = dx / q

            return H

        def res_slam(z: FloatArray, z_pred: FloatArray) -> FloatArray:
            nu = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
            nu[1::2] = wrap_angle(nu[1::2])
            return nu

        # Covariances
        Q = np.zeros((n_state, n_state))
        Q[0, 0] = (v_noise_std * dt) ** 2
        Q[1, 1] = (v_noise_std * dt) ** 2
        Q[2, 2] = (w_noise_std * dt) ** 2
        # Stationary landmarks have zero process noise
        np.fill_diagonal(Q[3:, 3:], 1e-12)

        r_diag = []
        for _ in range(num_landmarks):
            r_diag.extend([range_noise_std**2, bearing_noise_std**2])
        R = np.diag(r_diag)

        super().__init__(f=f_slam, h=h_slam, Q=Q, R=R, h_jac=h_jac, residual_fn=res_slam)
