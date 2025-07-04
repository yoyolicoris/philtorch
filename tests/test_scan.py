import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

from philtorch.lpv import scan as lpv_scan
from philtorch.lti import scan as lti_scan


def test_scan_equivalence():
    """Test that LPV scan is equivalent to LTI scan for scalar inputs."""
    batch_size = 3
    N = 100
    unroll_factor = 5

    a = torch.rand(batch_size) * 2 - 1
    x = torch.randn(batch_size, N)
    init = torch.randn(batch_size)

    # LPV scan
    lpv_output = lpv_scan(
        a.unsqueeze(1).expand(-1, N), init, x, unroll_factor=unroll_factor
    )
    # LTI scan
    lti_output = lti_scan(a, init, x, unroll_factor=unroll_factor)

    # Compare outputs
    assert torch.allclose(
        lpv_output, lti_output
    ), "LPV scan output does not match LTI scan output"
