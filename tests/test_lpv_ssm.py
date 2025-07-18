import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

from philtorch.lpv import state_space_recursion as lpv_state_space
from philtorch.lti import state_space_recursion as lti_state_space
from philtorch.mat import companion

from .test_lpv_filters import _generate_time_varying_coeffs, _generate_test_signal


def test_ssm_equivalence():
    """Test that LPV state_space is equivalent to LTI state_space for scalar inputs."""
    batch_size = 3
    N = 23
    unroll_factor = 7
    order = 2

    _, a = _generate_time_varying_coeffs(1, batch_size, order, order)
    a = a.squeeze(0)  # Remove batch dimension for scalar case
    x = _generate_test_signal(batch_size, N, "white_noise")

    A = companion(a)
    zi = torch.randn(batch_size, order)

    # LPV state_space
    lpv_output = lpv_state_space(
        A.unsqueeze(1).expand(-1, N, -1, -1),
        zi,
        x,
        unroll_factor=unroll_factor,
    )

    # LTI state_space
    lti_output = lti_state_space(A, zi, x, unroll_factor=unroll_factor)
    print(lti_output, lpv_output)
    # Compare outputs
    assert torch.allclose(lpv_output, lti_output), torch.max(
        torch.abs(lpv_output - lti_output)
    )


def test_ssm_unrolling():
    """Test that unrolling works correctly in LPV state_space."""
    batch_size = 2
    N = 17
    unroll_factor = 3
    order = 2

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    x = _generate_test_signal(batch_size, N, "white_noise")
    A = companion(a)
    zi = torch.randn(batch_size, order)

    output_naive = lpv_state_space(
        A, zi, x, unroll_factor=1
    )  # Naive implementation without unrolling
    output_unrolled = lpv_state_space(
        A, zi, x, unroll_factor=unroll_factor
    )  # Unrolled implementation
    # Check output shape
    assert torch.allclose(output_naive, output_unrolled), torch.max(
        torch.abs(output_naive - output_unrolled)
    )
