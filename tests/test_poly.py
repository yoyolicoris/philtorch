import numpy as np
import pytest
import torch

from philtorch.poly import polydiv, polymul


@pytest.mark.parametrize(("dividend", "divisor"), [(6, 3), (4, 4), (5, 1)])
def test_polydiv_matches_numpy(dividend: int, divisor: int):
    torch.manual_seed(0)
    u = torch.randn(3, dividend, dtype=torch.double)
    v = torch.randn(3, divisor, dtype=torch.double)
    q, r = polydiv(u, v)
    for row in range(3):
        expected_q, expected_r = np.polydiv(u[row].numpy(), v[row].numpy())
        np.testing.assert_allclose(q[row].numpy(), expected_q)
        # NumPy trims the remainder's leading zeros, but returns [0.] for an
        # empty one; polydiv keeps the zeros and returns no coefficients.
        # NumPy's remainder can hold rounding error where it is exactly zero.
        remainder = np.concatenate([np.zeros(1), r[row].numpy()])
        np.testing.assert_allclose(
            remainder[remainder.size - expected_r.size :], expected_r, atol=1e-12
        )


def test_polydiv_gradcheck():
    # Gradients reach both the dividend and the divisor.
    u = torch.randn(2, 6, dtype=torch.double, requires_grad=True)
    v = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(polydiv, (u, v))


def _random_complex(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.cdouble)


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize("shape_a, shape_b", [((3, 4), (3, 5)), ((4,), (6,)), ((1,), (3, 1))])
def test_polymul_matches_numpy(shape_a, shape_b, dtype):
    torch.manual_seed(0)
    a = torch.randn(*shape_a, dtype=dtype)
    b = torch.randn(*shape_b, dtype=dtype)
    c = polymul(a, b)
    a_b, b_b = torch.broadcast_tensors(a[..., :1], b[..., :1])
    batch = a_b.shape[:-1]
    assert c.shape == (*batch, a.size(-1) + b.size(-1) - 1)
    a_full = a.expand(*batch, -1).reshape(-1, a.size(-1))
    b_full = b.expand(*batch, -1).reshape(-1, b.size(-1))
    for row, (x, y) in enumerate(zip(a_full.numpy(), b_full.numpy())):
        np.testing.assert_allclose(c.reshape(-1, c.size(-1))[row].numpy(), np.polymul(x, y))


def test_polymul_gradcheck():
    a = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    b = torch.randn(1, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(polymul, (a, b))
    a = _random_complex(2, 3).requires_grad_()
    b = _random_complex(2, 2).requires_grad_()
    assert torch.autograd.gradcheck(polymul, (a, b))
    assert torch.autograd.gradgradcheck(polymul, (a, b))
