import pytest
import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional

from philtorch.lti import state_space_recursion, state_space
from philtorch.mat import a2companion

from .test_lti_lfilter import _generate_random_signal


def _generate_random_filter_coeffs(order: int, B: int) -> np.ndarray:
    """Generate random filter coefficients"""

    a = np.random.randn(B, order)
    a = a / np.abs(a).sum(axis=-1, keepdims=True)

    return a


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("T", [17, 29, 101])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("unroll_factor", [1, 5])
def test_time_invariant_ssm(
    B: int,
    T: int,
    order: int,
    unroll_factor: int,
):
    """Test time-invariant filters against scipy.signal.lfilter"""

    # Generate test data
    a = _generate_random_filter_coeffs(order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    A = a2companion(a_torch).squeeze(0)
    zi = x_torch.new_zeros(B, A.size(-1))

    # Apply philtorch filter
    y_torch = state_space_recursion(
        A, zi, x_torch, out_idx=0, unroll_factor=unroll_factor
    )

    # Apply scipy filter
    y_scipy = np.stack(
        [signal.lfilter([1.0], [1.0] + a[i].tolist(), x[i]) for i in range(B)],
    )

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy)


@pytest.mark.parametrize("order", [8])
@pytest.mark.parametrize("out_idx", [0, 1, 3, 7])
def test_out_idx(order: int, out_idx: int):
    """Test the out_idx functionality of state_space_recursion"""

    # Generate test data
    B, T = 2, 10
    a = _generate_random_filter_coeffs(order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    A = a2companion(a_torch)
    zi = x_torch.new_zeros(B, A.size(-1))

    # Apply philtorch filter with out_idx
    y_torch = state_space_recursion(A, zi, x_torch, out_idx=out_idx)[:, out_idx:]

    y_scipy = np.stack(
        [signal.lfilter([1.0], [1.0] + a[i].tolist(), x[i]) for i in range(B)],
    )
    if out_idx > 0:
        y_scipy = y_scipy[:, :-out_idx]

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy)


@pytest.mark.parametrize(
    ("x_shape", "B_shape"),
    [
        ((5, 97), (3,)),
        ((5, 97), (5, 3)),
        ((5, 97), None),
        ((5, 97, 3), None),
        ((5, 97, 2), (3, 2)),
        ((5, 97, 2), (5, 3, 2)),
    ],
)
@pytest.mark.parametrize("A_shape", [(3, 3), (5, 3, 3)])
@pytest.mark.parametrize("C_shape", [None, (3,), (5, 3), (2, 3), (5, 2, 3)])
@pytest.mark.parametrize("D_shape", [None])
@pytest.mark.parametrize("zi_shape", [None, (3,), (5, 3)])
def test_ssm_shape_handling(x_shape, A_shape, B_shape, C_shape, D_shape, zi_shape):
    unroll_factor = 4

    x = torch.randn(*x_shape)
    A = torch.randn(*A_shape)
    B = torch.randn(*B_shape) if B_shape is not None else None
    C = torch.randn(*C_shape) if C_shape is not None else None
    if D_shape is None:
        D = None
    else:
        D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = state_space(A, x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

    if zi is not None:
        y, zf = result
        assert zf.shape[-1] == zi_shape[-1]
    else:
        y = result

    assert y.shape[:2] == x.shape[:2]

    if y.dim() == 3:
        if C_shape is None:
            assert y.shape[2] == A.shape[-1]
        elif len(C_shape) == 1:
            assert y.shape[2] == C_shape[0]
        elif len(C_shape) == 2 and C_shape[0] != x_shape[0]:
            assert y.shape[2] == C_shape[0]
        elif len(C_shape) == 3:
            assert y.shape[2] == C_shape[1]
        else:
            assert False, f"Unexpected C_shape: {C_shape}"


@pytest.mark.parametrize(
    ("D_shape", "x_shape", "B_shape", "C_shape"),
    [
        ((1,), (5, 97), (3,), (3,)),
        ((), (5, 97), (3,), (3,)),
        ((1,), (5, 97, 2), (3, 2), (2, 3)),
        ((), (5, 97, 2), (3, 2), (2, 3)),
        ((5,), (5, 97), (3,), (3,)),
        ((7, 2), (5, 97, 2), (3, 2), (7, 3)),
        ((5, 7, 2), (5, 97, 2), (3, 2), (5, 7, 3)),
    ],
)
@pytest.mark.parametrize("A_shape", [(3, 3)])
@pytest.mark.parametrize("zi_shape", [None, (3,), (5, 3)])
def test_ssm_D_shape_handling(x_shape, A_shape, B_shape, C_shape, D_shape, zi_shape):
    unroll_factor = 4

    x = torch.randn(*x_shape)
    A = torch.randn(*A_shape)
    B = torch.randn(*B_shape)
    C = torch.randn(*C_shape)
    D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = state_space(A, x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

    if zi is not None:
        y, zf = result
        assert zf.shape[-1] == zi_shape[-1]
    else:
        y = result

    assert y.shape[:2] == x.shape[:2]

    if y.dim() == 3:
        if len(D_shape) > 1:
            assert y.shape[2] == D_shape[-2]
        else:
            assert y.shape[2] == C_shape[-2]
