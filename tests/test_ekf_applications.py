r"""Module: tests.test_ekf_applications

Description: Unit tests for Extended Kalman Filter applications in navigation/sensor fusion,
    robotics/autonomous vehicles (wheel slip & SLAM), and aerospace/spherical radar tracking.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np

from kalman_filter import (
    AttitudeHeadingModel,
    DroneFlightModel,
    EKFSLAMModel,
    ExtendedKalmanFilter,
    GPSIMUNavigationModel,
    SphericalRadarTrackingModel,
    WheeledRobotSlipModel,
    simulate,
)


def test_gps_imu_integrated_navigation() -> None:
    r"""Checks integrated GPS/IMU navigation with accelerometer bias tracking."""
    dt = 0.05
    model = GPSIMUNavigationModel(dt=dt, gps_pos_std=1.0, gps_vel_std=0.1)
    assert model.state_dim == 7  # px, py, vx, vy, bax, bay, psi
    assert model.meas_dim == 4  # px, py, vx, vy

    # Constant forward acceleration in body frame: amx = 2.0 m/s^2
    u_seq = np.tile([2.0, 0.0, 0.05], (40, 1))
    x0 = np.array([0.0, 0.0, 5.0, 0.0, 0.1, -0.1, 0.0])
    states, z = simulate(model, x0, steps=40, rng=np.random.default_rng(1), controls=u_seq)

    ekf = ExtendedKalmanFilter(model, x0=np.zeros(7), P0=np.eye(7) * 2.0)
    res = ekf.run(z, controls=u_seq)

    # Position tracking check
    pos_err = np.linalg.norm(res.filtered_means[-1, :2] - states[-1, :2])
    assert pos_err < 3.0


def test_attitude_heading_estimation() -> None:
    r"""Checks attitude and heading estimation with gravity and magnetic field corrections."""
    model = AttitudeHeadingModel(dt=0.02)
    assert model.state_dim == 6  # roll, pitch, yaw, bp, bq, br
    assert model.meas_dim == 6  # 3 accel, 3 mag

    x0 = np.array([0.1, -0.05, 0.5, 0.01, -0.01, 0.0])
    u_seq = np.tile([0.0, 0.0, 0.1], (30, 1))  # Yaw turn
    states, z = simulate(model, x0, steps=30, rng=np.random.default_rng(2), controls=u_seq)

    ekf = ExtendedKalmanFilter(model, x0=np.zeros(6), P0=np.eye(6) * 0.5)
    res = ekf.run(z, controls=u_seq)
    assert res.filtered_means.shape == (30, 6)
    assert not np.any(np.isnan(res.filtered_means))


def test_wheeled_robot_slip_estimation() -> None:
    r"""Checks differential drive robot dead reckoning with dynamic wheel slip estimation."""
    model = WheeledRobotSlipModel(wheelbase=0.6, dt=0.05)
    assert model.state_dim == 5  # x, y, th, sL, sR
    assert model.meas_dim == 3  # x, y, th

    x0 = np.array([0.0, 0.0, 0.0, 0.1, 0.05])
    u_seq = np.tile([1.5, 1.2], (40, 1))  # Commanded wheel speeds
    states, z = simulate(model, x0, steps=40, rng=np.random.default_rng(3), controls=u_seq)

    ekf = ExtendedKalmanFilter(model, x0=np.zeros(5), P0=np.eye(5) * 0.2)
    res = ekf.run(z, controls=u_seq)

    # Heading error should remain low
    heading_err = abs(res.filtered_means[-1, 2] - states[-1, 2])
    assert heading_err < 0.2


def test_ekf_slam_feature_tracking() -> None:
    r"""Checks 2D EKF-SLAM landmark mapping and robot pose tracking."""
    num_landmarks = 3
    model = EKFSLAMModel(num_landmarks=num_landmarks, dt=0.1)
    assert model.state_dim == 3 + 2 * num_landmarks  # 9
    assert model.meas_dim == 2 * num_landmarks  # 6

    # Landmarks at (5, 5), (-5, 5), (0, 10)
    landmarks = np.array([5.0, 5.0, -5.0, 5.0, 0.0, 10.0])
    x0 = np.concatenate([[0.0, 0.0, 0.0], landmarks])
    u_seq = np.tile([1.0, 0.1], (30, 1))

    states, z = simulate(model, x0, steps=30, rng=np.random.default_rng(4), controls=u_seq)

    # Initial guess with uncertainty on landmarks
    x_init = x0.copy()
    x_init[3:] += np.random.default_rng(5).normal(0.0, 0.5, size=6)
    P0 = np.diag([0.1, 0.1, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    ekf = ExtendedKalmanFilter(model, x0=x_init, P0=P0)
    res = ekf.run(z, controls=u_seq)

    # Landmark covariance should shrink after observing them
    init_cov_trace = np.trace(P0[3:, 3:])
    final_cov_trace = np.trace(res.filtered_covs[-1, 3:, 3:])
    assert final_cov_trace < init_cov_trace


def test_spherical_radar_tracking_3d() -> None:
    r"""Checks 3D aerospace target tracking with spherical observation geometry."""
    model = SphericalRadarTrackingModel(dt=0.1, range_std=2.0, azimuth_std=0.005)
    assert model.state_dim == 6  # x, y, z, vx, vy, vz
    assert model.meas_dim == 4  # r, az, el, range_rate

    x0 = np.array([1000.0, 500.0, 300.0, -50.0, 20.0, -5.0])
    states, z = simulate(model, x0, steps=40, rng=np.random.default_rng(6))

    P0 = np.diag([100.0, 100.0, 100.0, 25.0, 25.0, 25.0])
    ekf = ExtendedKalmanFilter(model, x0=x0 + [10, -10, 5, 2, -2, 1], P0=P0)
    res = ekf.run(z)

    pos_err = np.linalg.norm(res.filtered_means[-1, :3] - states[-1, :3])
    assert pos_err < 15.0


def test_drone_flight_aerodynamic_drag() -> None:
    r"""Checks 3D drone flight model with drag and specific thrust input."""
    model = DroneFlightModel(dt=0.05, drag_coeff=0.05, gravity=9.81)
    assert model.state_dim == 6
    assert model.meas_dim == 3

    # Hovering / vertical climb thrust: counter gravity + upward acceleration
    u_seq = np.tile([1.0, 0.0, 11.81], (30, 1))
    x0 = np.array([0.0, 0.0, 10.0, 2.0, 0.0, 1.0])
    states, z = simulate(model, x0, steps=30, rng=np.random.default_rng(7), controls=u_seq)

    ekf = ExtendedKalmanFilter(model, x0=np.zeros(6), P0=np.eye(6) * 5.0)
    res = ekf.run(z, controls=u_seq)

    err = np.linalg.norm(res.filtered_means[-1, :3] - states[-1, :3])
    assert err < 3.0
