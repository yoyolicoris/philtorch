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


@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize("shape_a, shape_b", [((3, 4), (3, 5)), ((4,), (6,)), ((1,), (3, 1))])
def test_polymul_matches_numpy(shape_a, shape_b, dtype):
    torch.manual_seed(0)
    a = torch.randn(*shape_a, dtype=dtype)
    b = torch.randn(*shape_b, dtype=dtype)
    c = polymul(a, b)
    batch = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    assert c.shape == (*batch, a.size(-1) + b.size(-1) - 1)
    a_full = a.expand(*batch, -1).reshape(-1, a.size(-1))
    b_full = b.expand(*batch, -1).reshape(-1, b.size(-1))
    for row, (x, y) in enumerate(zip(a_full.numpy(), b_full.numpy())):
        np.testing.assert_allclose(c.reshape(-1, c.size(-1))[row].numpy(), np.polymul(x, y))
