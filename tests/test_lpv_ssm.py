import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional
import pytest
from itertools import product, chain

from philtorch.lpv import state_space_recursion as lpv_state_space, state_space
from philtorch.lti import state_space_recursion as lti_state_space
from philtorch.mat import companion

from .test_lpv_filters import _generate_time_varying_coeffs, _generate_test_signal


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_ssm_equivalence(device):
    """Test that LPV state_space is equivalent to LTI state_space for scalar inputs."""
    batch_size = 3
    N = 23
    unroll_factor = 7
    order = 2

    _, a = _generate_time_varying_coeffs(1, batch_size, order, order)
    a = a.squeeze(0)  # Remove batch dimension for scalar case
    x = _generate_test_signal(batch_size, N, "white_noise").to(device)

    A = companion(a).to(device)
    zi = torch.randn(batch_size, order).to(device)

    # LPV state_space
    lpv_output = lpv_state_space(
        A.unsqueeze(1).expand(-1, N, -1, -1),
        zi,
        x,
        unroll_factor=unroll_factor,
    )

    # LTI state_space
    lti_output = lti_state_space(A, zi, x, unroll_factor=unroll_factor)
    # Compare outputs
    assert torch.allclose(lpv_output, lti_output, atol=1e-6), torch.max(
        torch.abs(lpv_output - lti_output)
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        # pytest.param(
        #     "cuda",
        #     marks=pytest.mark.skipif(
        #         not torch.cuda.is_available(), reason="CUDA not available"
        #     ),
        # ),
    ],
)
@pytest.mark.parametrize("order", [2, 3, 5])
def test_recurN_extension(device, order):
    """Test that the recur2 extension works correctly."""
    batch_size = 2
    N = 37

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    # x = _generate_test_signal(batch_size, N, "white_noise").cuda()
    x = torch.randn(batch_size, N, order).to(device)  # Simulated input
    A = companion(a).to(device)
    zi = torch.randn(batch_size, order).to(device)

    ext_output = torch.ops.philtorch.recurN(A, zi, x)
    torch_output = lpv_state_space(A, zi, x, unroll_factor=1)

    # Compare outputs
    assert torch.allclose(ext_output, torch_output, atol=1e-6), torch.max(
        torch.abs(ext_output - torch_output)
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_recur2_extension(device):
    """Test that the recur2 extension works correctly."""
    batch_size = 2
    N = 17
    order = 2

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    # x = _generate_test_signal(batch_size, N, "white_noise").cuda()
    x = torch.randn(batch_size, N, 2).to(device)  # Simulated input
    A = companion(a).to(device)
    zi = torch.randn(batch_size, order).to(device)

    ext_output = torch.ops.philtorch.recur2(A, zi, x)
    torch_output = lpv_state_space(A, zi, x, unroll_factor=1)

    # Compare outputs
    assert torch.allclose(ext_output, torch_output, atol=1e-6), torch.max(
        torch.abs(ext_output - torch_output)
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_ssm_unrolling(device):
    """Test that unrolling works correctly in LPV state_space."""
    batch_size = 2
    N = 17
    unroll_factor = 3
    order = 2

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    x = _generate_test_signal(batch_size, N, "white_noise").to(device)
    A = companion(a).to(device)
    zi = torch.randn(batch_size, order).to(device)

    output_naive = lpv_state_space(
        A, zi, x, unroll_factor=1
    )  # Naive implementation without unrolling
    output_unrolled = lpv_state_space(
        A, zi, x, unroll_factor=unroll_factor
    )  # Unrolled implementation
    # Check output shape
    assert torch.allclose(output_naive, output_unrolled, atol=1e-6), torch.max(
        torch.abs(output_naive - output_unrolled)
    )


@pytest.mark.parametrize(
    ("x_shape", "B_shape"),
    [
        ((5, 97), (3,)),
        ((5, 97), (5, 3)),
        ((5, 97), (5, 97, 3)),
        ((5, 97), (97, 3)),
        ((5, 97, 3), None),
        ((5, 97, 2), (3, 2)),
        ((5, 97, 2), (5, 3, 2)),
        ((5, 97, 2), (97, 3, 2)),
        ((5, 97, 2), (5, 97, 3, 2)),
    ],
)
@pytest.mark.parametrize("A_shape", [(97, 3, 3), (5, 97, 3, 3)])
@pytest.mark.parametrize(
    "C_shape",
    [
        None,
        (3,),
        (5, 3),
        (2, 3),
        (5, 2, 3),
        (97, 3),
        (5, 97, 3),
        (97, 2, 3),
        (5, 97, 2, 3),
    ],
)
@pytest.mark.parametrize("D_shape", [None])
@pytest.mark.parametrize("zi_shape", [None, (3,), (5, 3)])
def test_ssm_shape_handling(x_shape, A_shape, B_shape, C_shape, D_shape, zi_shape):
    unroll_factor = 3

    x = torch.randn(*x_shape)
    A = torch.randn(*A_shape)
    B = torch.randn(*B_shape) if B_shape is not None else None
    C = torch.randn(*C_shape) if C_shape is not None else None
    if D_shape is None:
        D = None
    else:
        D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = state_space(A=A, x=x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

    if zi is not None:
        y, zf = result
        assert zf.shape[-1] == zi_shape[-1]
    else:
        y = result

    assert y.shape[:2] == x.shape[:2]

    if y.dim() == 3:
        if C_shape is None:
            assert y.shape[2] == A.shape[-1]
        elif len(C_shape) == 2:
            assert y.shape[2] == C_shape[0]
        elif len(C_shape) == 3:
            assert y.shape[2] == C_shape[1]
        elif len(C_shape) == 4:
            assert y.shape[2] == C_shape[2]
        else:
            assert False, f"Unexpected C_shape: {C_shape}"


@pytest.mark.parametrize(
    ("D_shape", "x_shape", "B_shape", "C_shape"),
    chain(
        product(
            [(5,), (1,), (), (97,), (5, 97)],
            [(5, 97)],
            [(3,), (97, 3), (5, 3), (5, 97, 3)],
            [(3,), (97, 3), (5, 3), (5, 97, 3)],
        ),
        product(
            [(1,), (), (2, 2), (5, 2, 2), (5, 97, 2, 2)],
            [(5, 97, 2)],
            [(3, 2), (97, 3, 2), (5, 3, 2), (5, 97, 3, 2)],
            [(2, 3), (97, 2, 3), (5, 2, 3), (5, 97, 2, 3)],
        ),
        product(
            [(4,), (5, 4), (5, 97, 4), (97, 4)],
            [(5, 97)],
            [(3,), (97, 3), (5, 3), (5, 97, 3)],
            [(4, 3), (97, 4, 3), (5, 4, 3), (5, 97, 4, 3)],
        ),
        product(
            [(7, 2), (5, 7, 2), (5, 97, 7, 2)],
            [(5, 97, 2)],
            [(3, 2), (97, 3, 2), (5, 3, 2), (5, 97, 3, 2)],
            [(7, 3), (97, 7, 3), (5, 7, 3), (5, 97, 7, 3)],
        ),
        product(
            [(2,), (5, 2), (97, 2), (5, 97, 2)],
            [(5, 97, 2)],
            [(3, 2), (97, 3, 2), (5, 3, 2), (5, 97, 3, 2)],
            [(3,), (97, 3), (5, 3), (5, 97, 3)],
        ),
    ),
)
@pytest.mark.parametrize("A_shape", [(97, 3, 3), (5, 97, 3, 3)])
@pytest.mark.parametrize("zi_shape", [None, (3,), (5, 3)])
def test_ssm_D_shape_handling(x_shape, A_shape, B_shape, C_shape, D_shape, zi_shape):
    unroll_factor = 4

    x = torch.randn(*x_shape)
    A = torch.randn(*A_shape)
    B = torch.randn(*B_shape)
    C = torch.randn(*C_shape)
    D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = state_space(A=A, x=x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

    if zi is not None:
        y, zf = result
        assert zf.shape[-1] == zi_shape[-1]
    else:
        y = result

    assert y.shape[:2] == x.shape[:2]
