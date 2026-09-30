import pytest
import numpy as np
import torch
from scipy import signal
from itertools import product, chain
from philtorch.lti import state_space_recursion, state_space, diag_state_space
from philtorch.lti.ssm import _ssm_B, _ssm_C_D
from philtorch.mat import companion

from .test_lti_lfilter import _generate_random_signal


def _generate_random_filter_coeffs(order: int, B: int) -> np.ndarray:
    """Generate random filter coefficients"""

    a = np.random.randn(B, order)
    a = a / np.abs(a).sum(axis=-1, keepdims=True)

    return a


def _generate_diagonalizable_matrix(shape: tuple[int, ...]) -> torch.Tensor:
    order = shape[-1]
    values = torch.arange(1, order + 1, dtype=torch.get_default_dtype())
    basis = torch.vander(values, N=order, increasing=True)
    Q, _ = torch.linalg.qr(basis)
    eigenvalues = torch.linspace(0.2, 0.8, order, dtype=Q.dtype)
    A = Q @ torch.diag(eigenvalues) @ Q.mT
    return A.expand(*shape[:-2], order, order).clone()


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
    A = companion(a_torch).squeeze(0)
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
    A = companion(a_torch)
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
@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_shape_handling(x_shape, A_shape, B_shape, C_shape, D_shape, zi_shape, ssm):
    unroll_factor = 4

    x = torch.randn(*x_shape)
    A = _generate_diagonalizable_matrix(A_shape)
    B = torch.randn(*B_shape) if B_shape is not None else None
    C = torch.randn(*C_shape) if C_shape is not None else None
    if D_shape is None:
        D = None
    else:
        D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = ssm(A=A, x=x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

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
        else:
            assert False, f"Unexpected C_shape: {C_shape}"


@pytest.mark.parametrize(
    ("D_shape", "x_shape", "B_shape", "C_shape"),
    chain(
        product(
            [(5,), (1,), ()],
            [(5, 97)],
            [(3,), (5, 3)],
            [(3,), (5, 3)],
        ),
        product(
            [(1,), (), (2, 2), (5, 2, 2)],
            [(5, 97, 2)],
            [(3, 2), (5, 3, 2)],
            [(2, 3), (5, 2, 3)],
        ),
        product(
            [(4,), (5, 4)],
            [(5, 97)],
            [(3,), (5, 3)],
            [(4, 3), (5, 4, 3)],
        ),
        product(
            [(7, 2), (5, 7, 2)],
            [(5, 97, 2)],
            [(3, 2), (5, 3, 2)],
            [(7, 3), (5, 7, 3)],
        ),
        product(
            [(2,), (5, 2)],
            [(5, 97, 2)],
            [(3, 2), (5, 3, 2)],
            [(3,), (5, 3)],
        ),
    ),
)
@pytest.mark.parametrize("A_shape", [(3, 3), (5, 3, 3)])
@pytest.mark.parametrize("zi_shape", [None, (3,), (5, 3)])
@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_D_shape_handling(
    x_shape, A_shape, B_shape, C_shape, D_shape, zi_shape, ssm
):
    unroll_factor = 4

    x = torch.randn(*x_shape)
    A = _generate_diagonalizable_matrix(A_shape)
    B = torch.randn(*B_shape)
    C = torch.randn(*C_shape)
    D = torch.randn(*D_shape) if len(D_shape) > 0 else torch.randn(1)
    zi = torch.randn(*zi_shape) if zi_shape is not None else None

    result = ssm(A=A, x=x, B=B, C=C, D=D, zi=zi, unroll_factor=unroll_factor)

    if zi is not None:
        y, zf = result
        assert zf.shape[-1] == zi_shape[-1]
    else:
        y = result

    assert y.shape[:2] == x.shape[:2]


_b, _N, _M, _F, _P = 3, 6, 4, 2, 5


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize(
    ("x_shape", "B_shape", "equation"),
    [
        ((_b, _N), (_M,), "bn,m->bnm"),
        ((_b, _N), (_b, _M), "bn,bm->bnm"),
        ((_b, _N, _F), (_M, _F), "bnf,mf->bnm"),
        ((_b, _N, _F), (_b, _M, _F), "bnf,bmf->bnm"),
        # batch == M: (M, F) must not be read as a batched (B, M) vector.
        ((2, _N, 2), (2, 2), "bnf,mf->bnm"),
        ((1, _N, 1), (1, 1), "bnf,mf->bnm"),
    ],
)
def test_ssm_B_layouts(x_shape, B_shape, equation, dtype):
    x = torch.randn(x_shape, dtype=dtype)
    B = torch.randn(B_shape, dtype=dtype)
    M = B_shape[-1] if len(x_shape) == 2 else B_shape[-2]
    expected = torch.einsum(equation, x, B)
    torch.testing.assert_close(_ssm_B(B, x, x_shape[0], M), expected)


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize(
    ("x_shape", "D_shape", "equation"),
    [
        ((_b, _N), (), "bn,->bn"),
        ((_b, _N), (1,), "bn,o->bn"),
        ((_b, _N), (_b,), "bn,b->bn"),
        ((_b, _N), (_P,), "bn,p->bnp"),
        ((_b, _N), (_b, _P), "bn,bp->bnp"),
        ((_b, _N, _F), (), "bnf,->bnf"),
        ((_b, _N, _F), (1,), "bnf,o->bnf"),
        ((_b, _N, _F), (_F,), "bnf,f->bn"),
        ((_b, _N, _F), (_b, _F), "bnf,bf->bn"),
        ((_b, _N, _F), (_P, _F), "bnf,pf->bnp"),
        ((_b, _N, _F), (_b, _P, _F), "bnf,bpf->bnp"),
        # batch == 1: a (1,) gain on vector input stays a scalar gain.
        ((1, _N, _F), (1,), "bnf,o->bnf"),
    ],
)
def test_ssm_D_layouts(x_shape, D_shape, equation, dtype):
    x = torch.randn(x_shape, dtype=dtype)
    D = torch.randn(D_shape, dtype=dtype)
    expected = torch.einsum(equation, x, D)
    y = _ssm_C_D(torch.zeros((), dtype=dtype), x, None, D, x_shape[0], _M)
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize(
    ("C_shape", "equation"),
    [
        (None, "bnm->bnm"),
        ((_M,), "bnm,m->bn"),
        ((_b, _M), "bnm,bm->bn"),
        ((_P, _M), "bnm,pm->bnp"),
        ((_b, _P, _M), "bnm,bpm->bnp"),
    ],
)
def test_ssm_C_layouts(C_shape, equation, dtype):
    h = torch.randn(_b, _N, _M, dtype=dtype)
    x = torch.randn(_b, _N, dtype=dtype)
    if C_shape is None:
        C, expected = None, h
    else:
        C = torch.randn(C_shape, dtype=dtype)
        expected = torch.einsum(equation, h, C)
    torch.testing.assert_close(_ssm_C_D(h, x, C, None, _b, _M), expected)


@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_unbatched_B_when_batch_equals_state_dim(ssm):
    # batch == M == F, so the (M, F) matrix has the same shape as a (batch, M) one.
    x = torch.randn(2, 17, 2, dtype=torch.double)
    A = _generate_diagonalizable_matrix((2, 2)).double()
    B = torch.randn(2, 2, dtype=torch.double)

    y = ssm(A=A, x=x, B=B)

    torch.testing.assert_close(y, ssm(A=A, x=x, B=B.expand(2, 2, 2)))


@pytest.mark.parametrize("batched_A", [False, True])
@pytest.mark.parametrize(
    ("x_shape", "B_shape"),
    [
        ((2, 17), (2,)),
        ((2, 17), (2, 2)),
        ((2, 17, 2), (2, 2)),
        ((2, 17, 2), (2, 2, 2)),
    ],
)
def test_diag_state_space_B_layouts_match_state_space(x_shape, B_shape, batched_A):
    # batch == M == F: every 2D B shape collides, so only x.dim() tells them apart.
    x = torch.randn(x_shape, dtype=torch.double)
    A = _generate_diagonalizable_matrix((2, 2, 2) if batched_A else (2, 2)).double()
    B = torch.randn(B_shape, dtype=torch.double)

    y = diag_state_space(A=A, x=x, B=B)

    # Without C, diag_state_space returns its complex eigenbasis result as is.
    torch.testing.assert_close(y.imag, torch.zeros_like(y.real))
    torch.testing.assert_close(y.real, state_space(A=A, x=x, B=B))


@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
@pytest.mark.parametrize(
    ("x_shape", "B_shape"), [((3, 17), (3, 2)), ((3, 17, 2), (3,))]
)
def test_ssm_rejects_invalid_B(ssm, x_shape, B_shape):
    A = _generate_diagonalizable_matrix((3, 3))
    with pytest.raises(ValueError, match=f"Input matrix B .* for {len(x_shape)}D"):
        ssm(A=A, x=torch.randn(x_shape), B=torch.randn(B_shape))


@pytest.mark.parametrize("ssm", [state_space, diag_state_space])
def test_ssm_scalar_D_with_single_batch_vector_input(ssm):
    x = torch.randn(1, 17, 2)
    A = _generate_diagonalizable_matrix((3, 3))
    B, C, D = torch.randn(3, 2), torch.randn(2, 3), torch.randn(1)

    y = ssm(A=A, x=x, B=B, C=C, D=D)

    torch.testing.assert_close(y, ssm(A=A, x=x, B=B, C=C, D=D[0]))


@pytest.mark.parametrize(
    ("x_shape", "B_shape"),
    [
        ((_b, _N), (_M, _F)),
        ((_b, _N), ()),
        ((_b, _N), (_M + 1,)),
        ((_b, _N, _F), (_M,)),
        ((_b, _N, _F), (_b, _M)),
        ((_b, _N, _F), (_M, _F + 1)),
    ],
)
def test_ssm_B_rejects_invalid_shapes(x_shape, B_shape):
    with pytest.raises(ValueError, match=f"Input matrix B .* for {len(x_shape)}D"):
        _ssm_B(torch.zeros(B_shape), torch.zeros(x_shape), _b, _M)


@pytest.mark.parametrize(
    ("x_shape", "D_shape"),
    [
        ((_b, _N), (_P, _F)),
        ((_b, _N), (_b, _P, _F)),
        ((_b, _N, _F), (_b,)),
        ((_b, _N, _F), (_P,)),
        ((_b, _N, _F), (_b, _P)),
        ((_b, _N, _F), (_b, _P, _F + 1)),
        ((_b, _N, _F), (1, 1, 1, 1)),
    ],
)
def test_ssm_D_rejects_invalid_shapes(x_shape, D_shape):
    with pytest.raises(ValueError, match=f"Input matrix D .* for {len(x_shape)}D"):
        _ssm_C_D(
            torch.zeros(()), torch.zeros(x_shape), None, torch.zeros(D_shape), _b, _M
        )


@pytest.mark.parametrize("C_shape", [(), (_M + 1,), (_b, _P, _M + 1), (1, 1, 1, _M)])
def test_ssm_C_rejects_invalid_shapes(C_shape):
    with pytest.raises(ValueError, match="Output matrix C"):
        _ssm_C_D(
            torch.zeros(_b, _N, _M),
            torch.zeros(_b, _N),
            torch.zeros(C_shape),
            None,
            _b,
            _M,
        )
