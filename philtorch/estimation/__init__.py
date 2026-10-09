"""State estimation: inferring hidden states from noisy measurements.

These compute the posterior distributions of a state-space model's states,
continuous (Kalman) or discrete (hidden Markov models), parallelized over
time with associative scans.
"""

from .hmm import (
    HMMFilterResult,
    HMMSmootherResult,
    HMMViterbiResult,
    hmm_filter,
    hmm_smoother,
    hmm_viterbi,
)
from .kalman import KalmanFilterResult, KalmanSmootherResult, kalman_filter, kalman_smoother

__all__ = [
    "HMMFilterResult",
    "HMMSmootherResult",
    "HMMViterbiResult",
    "KalmanFilterResult",
    "KalmanSmootherResult",
    "hmm_filter",
    "hmm_smoother",
    "hmm_viterbi",
    "kalman_filter",
    "kalman_smoother",
]
