"""State estimation: inferring hidden states from noisy measurements.

These compute the posterior distributions of a state-space model's states,
parallelized over time with associative scans.
"""

from .kalman import (
    KalmanStatistics,
    kalman_em_statistics,
    kalman_filter,
    kalman_log_likelihood,
    kalman_smoother,
)

__all__ = [
    "KalmanStatistics",
    "kalman_em_statistics",
    "kalman_filter",
    "kalman_log_likelihood",
    "kalman_smoother",
]
