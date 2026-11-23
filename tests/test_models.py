r"""Module: tests.test_models

Description: Unit tests for target tracking, chaotic data assimilation, and sensor fusion models.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import numpy as np
import pytest

from kalman_filter import (
    Lorenz63Model,
    Lorenz96Model,
    SensorStream,
    constant_acceleration_model,
    coordinated_turn_model,
    merge_sensor_streams,
    random_walk_model,
    wrap_angle,
)


def test_constant_acceleration_model_shapes() -> None:
    r"""Checks dimensions and structure of Constant Acceleration model."""
    H = np.zeros((2, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    model = constant_acceleration_model(dim=2, dt=0.2, jerk_std=0.5, H=H, R=np.eye(2))
    assert model.state_dim == 6
    assert model.meas_dim == 2
    assert model.F.shape == (6, 6)
    assert model.Q.shape == (6, 6)


def test_random_walk_model() -> None:
    r"""Checks Random Walk model behavior."""
    model = random_walk_model(dim=3, q_var=0.1, r_var=0.5)
    assert model.state_dim == 3
    assert model.meas_dim == 3
    np.testing.assert_array_equal(model.F, np.eye(3))
    np.testing.assert_array_equal(model.H, np.eye(3))


def test_coordinated_turn_model_bearing_wrapping() -> None:
    r"""Checks Coordinated Turn model with station range-bearing measurements."""
    stations = np.array([[0.0, 0.0], [50.0, 50.0]])
    model = coordinated_turn_model(
        dt=0.1, accel_std=0.2, turn_rate_noise_std=0.05, stations=stations
    )
    assert model.state_dim == 5

    assert model.meas_dim == 4

    # Check innovation wraps angles
    z1 = np.array([10.0, np.pi - 0.1, 20.0, -np.pi + 0.1])
    z2 = np.array([10.0, -np.pi + 0.1, 20.0, np.pi - 0.1])
    res = model.residual(z1, z2)
    assert abs(res[1]) < 0.3  # Wrapped around pi


def test_lorenz96_properties() -> None:
    r"""Checks Lorenz-96 spatial model dimensions, distances, and localization."""
    lorenz = Lorenz96Model(dim=20, dt=0.05, forcing=8.0, obs_indices=list(range(0, 20, 2)))
    assert lorenz.state_dim == 20
    assert lorenz.meas_dim == 10

    # Distance matrix periodic symmetry
    dist = lorenz.compute_periodic_distances()
    assert dist.shape == (20, 20)
    assert dist[0, 19] == 1.0  # Periodic neighbor
    assert dist[0, 10] == 10.0

    # Localization matrix shape
    loc = lorenz.compute_localization_matrix(cutoff_radius=4.0)
    assert loc.shape == (20, 10)


def test_lorenz63_step() -> None:
    r"""Checks Lorenz-63 model propagation."""
    l63 = Lorenz63Model(dt=0.01)
    assert l63.state_dim == 3
    assert l63.meas_dim == 2
    x = np.array([1.0, 1.0, 1.0])
    x_next = l63.propagate(x)
    assert x_next.shape == (3,)
    assert not np.allclose(x, x_next)


def test_wrap_angle() -> None:
    r"""Checks angle wrapping into [-pi, pi)."""
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(3.0 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(-4.0 * np.pi) == pytest.approx(0.0)


def test_merge_sensor_streams() -> None:
    r"""Checks multi-rate sensor merging into synchronized matrix with NaNs."""
    # Sensor 1 at 2 Hz
    s1 = SensorStream(
        name="GPS",
        timestamps=np.array([0.5, 1.0, 1.5, 2.0]),
        values=np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]]),
        noise_std=np.array([0.1, 0.1]),
    )
    # Sensor 2 at 1 Hz
    s2 = SensorStream(
        name="Beacon",
        timestamps=np.array([1.0, 2.0]),
        values=np.array([[10.0], [20.0]]),
        noise_std=np.array([0.5]),
    )

    t_grid, meas, r_diag = merge_sensor_streams([s1, s2], dt=0.5, t_end=2.0)
    assert len(t_grid) == 4
    assert meas.shape == (4, 3)
    # At t = 0.5: GPS present, Beacon missing (NaN)
    assert not np.isnan(meas[0, 0])
    assert np.isnan(meas[0, 2])
    # At t = 1.0: both present
    assert not np.isnan(meas[1, 0])
    assert not np.isnan(meas[1, 2])
