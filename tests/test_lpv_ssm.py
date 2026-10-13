from itertools import chain

import pytest
import torch

from philtorch.lpv import state_space
from philtorch.lpv import state_space_recursion as lpv_state_space
from philtorch.mat import companion

from ._cases import each_choice
from .test_lpv_filters import _generate_test_signal, _generate_time_varying_coeffs


# Each block multiplies unroll_factor - 1 matrices with matrices_cumdot, so
# 5, 9 and 13 cover products over 4 = 2 * 2, 8 = 2 * 2 * 2 and 12 = 2 * 2 * 3
# matrices, which it used to multiply in the wrong order.
@pytest.mark.parametrize("unroll_factor", [3, 5, 9, 13])
def test_ssm_unrolling(unroll_factor):
    """Test that unrolling works correctly in LPV state_space."""
    batch_size = 2
    N = 17
    order = 2

    _, a = _generate_time_varying_coeffs(batch_size, N, order, order)
    x = _generate_test_signal(batch_size, N, "white_noise").double()
    A = companion(a).double()
    zi = torch.randn(batch_size, order).double()

    output_naive = lpv_state_space(A, zi, x)  # The default kernels
    output_unrolled = lpv_state_space(
        A, zi, x, unroll_factor=unroll_factor
    )  # Unrolled implementation
    # Check output shape
    assert torch.allclose(output_naive, output_unrolled), torch.max(
        torch.abs(output_naive - output_unrolled)
    )


@pytest.mark.parametrize(
    ("x_B_shape", "A_shape", "C_shape", "zi_shape"),
    each_choice(
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
        [(97, 3, 3), (5, 97, 3, 3)],
        [None, (3,), (5, 3), (2, 3), (5, 2, 3), (97, 3), (5, 97, 3), (97, 2, 3), (5, 97, 2, 3)],
        [None, (3,), (5, 3)],
    ),
)
def test_ssm_shape_handling(x_B_shape, A_shape, C_shape, zi_shape):
    unroll_factor = 3
    (x_shape, B_shape), D_shape = x_B_shape, None

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


# Each group fixes x's dimensions and the D, B and C shapes that go with them.
_A_SHAPES = [(97, 3, 3), (5, 97, 3, 3)]
_ZI_SHAPES = [None, (3,), (5, 3)]


@pytest.mark.parametrize(
    ("D_shape", "x_shape", "B_shape", "C_shape", "A_shape", "zi_shape"),
    list(
        chain(
            each_choice(
                [(5,), (1,), (), (97,), (5, 97)],
                [(5, 97)],
                [(3,), (97, 3), (5, 3), (5, 97, 3)],
                [(3,), (97, 3), (5, 3), (5, 97, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(1,), (), (2, 2), (5, 2, 2), (97, 2, 2), (5, 97, 2, 2)],
                [(5, 97, 2)],
                [(3, 2), (97, 3, 2), (5, 3, 2), (5, 97, 3, 2)],
                [(2, 3), (97, 2, 3), (5, 2, 3), (5, 97, 2, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(4,), (5, 4), (5, 97, 4), (97, 4)],
                [(5, 97)],
                [(3,), (97, 3), (5, 3), (5, 97, 3)],
                [(4, 3), (97, 4, 3), (5, 4, 3), (5, 97, 4, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(7, 2), (5, 7, 2), (5, 97, 7, 2)],
                [(5, 97, 2)],
                [(3, 2), (97, 3, 2), (5, 3, 2), (5, 97, 3, 2)],
                [(7, 3), (97, 7, 3), (5, 7, 3), (5, 97, 7, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
            each_choice(
                [(2,), (5, 2), (97, 2), (5, 97, 2)],
                [(5, 97, 2)],
                [(3, 2), (97, 3, 2), (5, 3, 2), (5, 97, 3, 2)],
                [(3,), (97, 3), (5, 3), (5, 97, 3)],
                _A_SHAPES,
                _ZI_SHAPES,
            ),
        )
    ),
)
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
