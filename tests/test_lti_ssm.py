import pytest
import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional

from philtorch.lti import ssm_recursion
from philtorch.prototype.utils import a2companion

from .test_lti_lfilter import _generate_random_signal


def _generate_random_filter_coeffs(order: int, B: int) -> np.ndarray:
    """Generate random filter coefficients"""

    a = np.random.randn(B, order)
    a = a / np.abs(a).sum(axis=-1, keepdims=True)

    return a


@pytest.mark.parametrize("B", [1, 8, 16])
@pytest.mark.parametrize("T", [17, 29, 101])
@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("unroll_factor", [None, 2, 5])
def test_time_invariant_filter(
    B: int,
    T: int,
    order: int,
    unroll_factor: Optional[int],
):
    """Test time-invariant filters against scipy.signal.lfilter"""

    # Generate test data
    a = _generate_random_filter_coeffs(order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    A = a2companion(a_torch)

    # Apply philtorch filter
    y_torch = ssm_recursion(A, x_torch, out_idx=0, unroll_factor=unroll_factor)

    # Apply scipy filter
    y_scipy = np.stack(
        [signal.lfilter([1.0], [1.0] + a[i].tolist(), x[i]) for i in range(B)],
    )

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy, atol=1e-6)


@pytest.mark.parametrize("order", [8])
@pytest.mark.parametrize("out_idx", [0, 1, 3, 7])
def test_out_idx(order: int, out_idx: int):
    """Test the out_idx functionality of ssm_recursion"""

    # Generate test data
    B, T = 2, 10
    a = _generate_random_filter_coeffs(order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    A = a2companion(a_torch)

    # Apply philtorch filter with out_idx
    y_torch = ssm_recursion(A, x_torch, out_idx=out_idx)[:, out_idx:]

    y_scipy = np.stack(
        [signal.lfilter([1.0], [1.0] + a[i].tolist(), x[i]) for i in range(B)],
    )
    if out_idx > 0:
        y_scipy = y_scipy[:, :-out_idx]

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy, atol=1e-6)
