r"""Module: kalman_filter.models.sensors

Description: Utilities for multi-sensor fusion: angular residual wrapping, multi-rate
    sensor data alignment, and asynchronous sensor stream merging with NaN-dropouts.

Author: Hui Zhang
Date: 2026-11-23
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kalman_filter.types import FloatArray

logger = logging.getLogger(__name__)

__all__ = [
    "SensorStream",
    "bearing_residual",
    "merge_sensor_streams",
    "wrap_angle",
]


def wrap_angle(angle: FloatArray | float) -> FloatArray | float:
    r"""Wraps an angle or array of angles into the half-open interval :math:`[-\pi, \pi)`.

    Args:
        angle: Angle or array of angles in radians.

    Returns:
        Wrapped angle(s) in :math:`[-\pi, \pi)`.
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def bearing_residual(
    z: FloatArray,
    z_pred: FloatArray,
    angle_indices: Sequence[int] | None = None,
) -> FloatArray:
    r"""Computes innovation :math:`z - z_{\text{pred}}` with angular wrapping on specified channels.

    Args:
        z: True measurement vector, shape ``(m,)``.
        z_pred: Predicted measurement vector, shape ``(m,)``.
        angle_indices: Sequence of 0-based indices corresponding to angular channels.
            If None, all channels are treated as linear.

    Returns:
        Innovation vector with angles wrapped to :math:`[-\pi, \pi)`.
    """
    res = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
    if angle_indices is not None and len(angle_indices) > 0:
        res = res.copy()
        idx = list(angle_indices)
        res[idx] = wrap_angle(res[idx])
    return res


@dataclass(frozen=True)
class SensorStream:
    r"""Timestamped readings from a single physical sensor.

    Attributes:
        name: Human-readable name or channel identifier.
        timestamps: 1D array of observation timestamps in seconds, shape ``(K,)``.
        values: 2D array of measurements, shape ``(K, m_sensor)``.
        noise_std: Standard deviation of measurement noise for each channel, shape ``(m_sensor,)``.
    """

    name: str
    timestamps: FloatArray
    values: FloatArray
    noise_std: FloatArray

    @property
    def meas_dim(self) -> int:
        r"""Number of measurement channels for this sensor.

        Returns:
            Measurement dimension.
        """
        vals = np.atleast_2d(self.values)
        return vals.shape[1]


def merge_sensor_streams(
    streams: Sequence[SensorStream],
    dt: float,
    t_end: float,
    t_start: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Merges multiple asynchronous sensor streams onto a uniform discrete-time grid.

    Time grid runs :math:`t_k = t_{\text{start}} + k \cdot dt` for :math:`k = 1, \dots, T`.
    If a sensor does not report an observation within :math:`[t_k - dt/2, t_k + dt/2)`,
    its corresponding channels are marked as ``NaN``.

    Args:
        streams: Sequence of ``SensorStream`` instances.
        dt: Discrete filtering time step in seconds.
        t_end: End time in seconds.
        t_start: Start time in seconds (default is 0.0).

    Returns:
        A tuple ``(time_grid, measurements, R_diag)``:
            - ``time_grid``: Array of time steps, shape ``(T,)``.
            - ``measurements``: Fused multi-channel measurements with NaNs, shape ``(T, m_total)``.
            - ``R_diag``: Diagonal vector of measurement noise variances, shape ``(m_total,)``.

    Raises:
        ValueError: If ``dt <= 0``, ``t_end <= t_start``, or streams is empty.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start")
    if not streams:
        raise ValueError("At least one SensorStream must be provided")

    time_grid = np.arange(t_start + dt, t_end + 1e-9, dt)
    T = len(time_grid)

    total_channels = sum(s.meas_dim for s in streams)
    measurements = np.full((T, total_channels), np.nan)
    r_diag_list = []

    channel_offset = 0
    for s in streams:
        m_s = s.meas_dim
        r_diag_list.extend(np.asarray(s.noise_std, dtype=float) ** 2)
        s_times = np.asarray(s.timestamps, dtype=float)
        s_vals = np.atleast_2d(np.asarray(s.values, dtype=float))

        for k, t_k in enumerate(time_grid):
            # Find readings falling in window [t_k - dt/2, t_k + dt/2)
            mask = (s_times >= t_k - 0.5 * dt) & (s_times < t_k + 0.5 * dt)
            matching_idx = np.flatnonzero(mask)
            if matching_idx.size > 0:
                # Use the closest reading to t_k
                closest = matching_idx[np.argmin(np.abs(s_times[matching_idx] - t_k))]
                measurements[k, channel_offset : channel_offset + m_s] = s_vals[closest]

        channel_offset += m_s

    return time_grid, measurements, np.array(r_diag_list)
