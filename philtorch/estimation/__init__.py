"""State estimation: inferring hidden states from noisy measurements.

These compute the posterior distributions of a state-space model's states,
continuous (Kalman) or discrete (hidden Markov models), parallelized over
time with associative scans, and fit the models' parameters by
expectation-maximization.
"""

from ._em import em
from .hmm import hmm_em_step, hmm_filter, hmm_smoother, hmm_viterbi
from .kalman import (
    KalmanFilterResult,
    KalmanSmootherResult,
    kalman_em_step,
    kalman_filter,
    kalman_smoother,
)

__all__ = [
    "KalmanFilterResult",
    "KalmanSmootherResult",
    "em",
    "hmm_em_step",
    "hmm_filter",
    "hmm_smoother",
    "hmm_viterbi",
    "kalman_em_step",
    "kalman_filter",
    "kalman_smoother",
]
