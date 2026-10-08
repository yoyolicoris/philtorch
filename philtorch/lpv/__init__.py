"""Filters and state-space models with time-varying coefficients.

These are linear parameter-varying (LPV) systems: the filter coefficients
and state matrices have a time dimension aligned with the input. For
constant coefficients, use :mod:`philtorch.lti`.

The Kalman filter and smoother are here too, and also take constant model
matrices: their gain varies over time even when the model doesn't.
"""

from .filtering import allpole, fir, lfilter
from .kalman import kalman_filter, kalman_smoother
from .recur import linear_recurrence
from .ssm import state_space, state_space_recursion

__all__ = [
    "lfilter",
    "linear_recurrence",
    "state_space",
    "state_space_recursion",
    "allpole",
    "fir",
    "kalman_filter",
    "kalman_smoother",
]
