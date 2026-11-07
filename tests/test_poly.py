import numpy as np
import pytest
import torch

from philtorch.poly import polydiv


@pytest.mark.parametrize(("dividend", "divisor"), [(6, 3), (4, 4), (5, 1)])
def test_polydiv_matches_numpy(dividend: int, divisor: int):
    u = torch.randn(3, dividend, dtype=torch.double)
    v = torch.randn(3, divisor, dtype=torch.double)
    q, r = polydiv(u, v)
    for row in range(3):
        expected_q, expected_r = np.polydiv(u[row].numpy(), v[row].numpy())
        np.testing.assert_allclose(q[row].numpy(), expected_q)
        # NumPy trims the remainder's leading zeros, but returns [0.] for an
        # empty one; polydiv keeps the zeros and returns no coefficients.
        remainder = np.concatenate([np.zeros(1), r[row].numpy()])
        np.testing.assert_allclose(remainder[remainder.size - expected_r.size :], expected_r)


def test_polydiv_gradcheck():
    # Gradients reach both the dividend and the divisor.
    u = torch.randn(2, 6, dtype=torch.double, requires_grad=True)
    v = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(polydiv, (u, v))
