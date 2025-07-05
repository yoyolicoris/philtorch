import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

from philtorch.lpv import linear_recurrence as lpv_linear_recurrence
from philtorch.lti import linear_recurrence as lti_linear_recurrence


def test_linear_recurrence_equivalence():
    """Test that LPV linear_recurrence is equivalent to LTI linear_recurrence for scalar inputs."""
    batch_size = 3
    N = 100
    unroll_factor = 5

    a = torch.rand(batch_size) * 2 - 1
    x = torch.randn(batch_size, N)
    init = torch.randn(batch_size)

    # LPV linear_recurrence
    lpv_output = lpv_linear_recurrence(
        a.unsqueeze(1).expand(-1, N), init, x, unroll_factor=unroll_factor
    )
    # LTI linear_recurrence
    lti_output = lti_linear_recurrence(a, init, x, unroll_factor=unroll_factor)

    # Compare outputs
    assert torch.allclose(
        lpv_output, lti_output
    ), "LPV linear_recurrence output does not match LTI linear_recurrence output"
