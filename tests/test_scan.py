import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

from philtorch.lpv import linear_recurrence as lpv_linear_recurrence
from philtorch.lti import linear_recurrence as lti_linear_recurrence


def test_linear_recurrence_equivalence():
    """Test that LPV linear_recurrence is equivalent to LTI linear_recurrence for scalar inputs."""
    batch_size = 3
    N = 101
    unroll_factor = 5

    a = torch.rand(batch_size).double() * 2 - 1
    x = torch.randn(batch_size, N).double()
    init = torch.randn(batch_size).double()

    # LPV linear_recurrence
    lpv_output = lpv_linear_recurrence(
        a.unsqueeze(1).expand(-1, N), init, x, unroll_factor=unroll_factor
    )
    # LTI linear_recurrence
    lti_output = lti_linear_recurrence(a, init, x, unroll_factor=unroll_factor)

    # Compare outputs
    assert torch.allclose(lpv_output, lti_output), torch.max(
        torch.abs(lpv_output - lti_output)
    )


def test_linear_recurrence_unrolling():
    """Test that unrolling works correctly in LPV linear_recurrence."""
    batch_size = 2
    N = 17
    unroll_factor = 3

    a = torch.rand(batch_size, N) * 2 - 1
    x = torch.randn(batch_size, N)
    init = torch.randn(batch_size)

    output_naive = lpv_linear_recurrence(
        a, init, x, unroll_factor=1
    )  # Naive implementation without unrolling
    output_unrolled = lpv_linear_recurrence(
        a, init, x, unroll_factor=unroll_factor
    )  # Unrolled implementation
    # Check output shape
    assert torch.allclose(output_naive, output_unrolled), torch.max(
        torch.abs(output_naive - output_unrolled)
    )
