"""Differentiable, time-domain linear digital filters for PyTorch.

Time-invariant filters live in :mod:`philtorch.lti`, time-varying (linear
parameter-varying) filters in :mod:`philtorch.lpv`, state estimation in
:mod:`philtorch.estimation`, and sequence alignment in :mod:`philtorch.align`.
"""

from pathlib import Path

import torch

from ._build_check import check_torch_build

try:
    from ._version import __version__ as __version__  # type: ignore
except ImportError:
    __version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text().strip()

# Fail clearly on a torch older than the floor, a CUDA wheel on another CUDA
# major, or a CUDA older than 12.9, before loading the extension would fail
# obscurely.
check_torch_build(__version__, torch.__version__, torch.version.cuda)

# The compiled extension, then its ops' fake and autograd implementations.
from . import _C, _ops, nl  # noqa: E402, F401
