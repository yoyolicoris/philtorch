"""State estimation: inferring hidden states from noisy measurements.

These compute the posterior distributions of a state-space model's states,
parallelized over time with associative scans.
"""

from .kalman import kalman_filter, kalman_smoother

__all__ = [
    "kalman_filter",
    "kalman_smoother",
]
