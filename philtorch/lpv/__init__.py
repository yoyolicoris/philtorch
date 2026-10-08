"""Filters and state-space models with time-varying coefficients.

These are linear parameter-varying (LPV) systems: every coefficient tensor
has a time dimension aligned with the input. For constant coefficients,
use :mod:`philtorch.lti`.
"""

from .filtering import allpole, fir, lfilter
from .recur import linear_recurrence
from .ssm import state_space, state_space_recursion

__all__ = [
    "lfilter",
    "linear_recurrence",
    "state_space",
    "state_space_recursion",
    "allpole",
    "fir",
]
