"""Differentiable, time-domain linear digital filters for PyTorch.

Time-invariant filters live in :mod:`philtorch.lti`, time-varying (linear
parameter-varying) filters in :mod:`philtorch.lpv`, state estimation in
:mod:`philtorch.estimation`, and sequence alignment in :mod:`philtorch.align`.
"""

from pathlib import Path

import torch

from ._build_check import check_torch_build, installed_torch_requirement

try:
    from ._version import __version__ as __version__  # type: ignore
except ImportError:
    __version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text().strip()

# Fail clearly on a wheel installed next to a torch it wasn't built for, before
# loading the extension would fail with an undefined-symbol error.
check_torch_build(__version__, torch.__version__, torch.version.cuda, installed_torch_requirement())

# The compiled extension, then its ops' fake and autograd implementations.
from . import _C, _ops  # noqa: E402, F401
