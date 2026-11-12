import numpy as np
import pytest
import torch
from scipy import signal

from philtorch.poly import polydiv, polymul, unique_roots


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


def _check_unique_roots(p: np.ndarray, tol: float, rtype: str):
    expected_u, expected_m = signal.unique_roots(p, tol=tol, rtype=rtype)
    u, m = unique_roots(torch.from_numpy(p), tol=tol, rtype=rtype)
    assert m.dtype == torch.int64
    np.testing.assert_allclose(u.numpy(), expected_u)
    np.testing.assert_array_equal(m.numpy(), expected_m)


@pytest.mark.parametrize("rtype", ["max", "maximum", "min", "minimum", "avg", "mean"])
def test_unique_roots_matches_scipy(rtype):
    # SciPy's docstring example
    _check_unique_roots(np.array([0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3]), 2e-2, rtype)
    # a ~ b ~ c, with a and c farther apart than tol: a takes b, c stays alone
    _check_unique_roots(np.array([0.0, 0.0015, 0.003, 1.0]), 2e-3, rtype)
    _check_unique_roots(np.array([0.003, 0.0015, 0.0, 1.0]), 2e-3, rtype)
    # complex roots, compared by real part, then imaginary part
    z = np.array([0.5 + 0.3j, 0.5 - 0.3j, 0.5 + 0.3001j, 0.2j, 0.2 - 0.0004j, 0.5 - 0.2999j])
    _check_unique_roots(z, 1e-2, rtype)
    rng = np.random.default_rng(0)
    base = rng.standard_normal(5) + 1j * rng.standard_normal(5)
    z = np.concatenate([base, base + 1e-4, base - 1e-4j])[rng.permutation(15)]
    _check_unique_roots(z, 1e-3, rtype)
    _check_unique_roots(z.real.copy(), 1e-3, rtype)


@pytest.mark.parametrize("rtype", ["min", "max", "avg"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_unique_roots_batched(rtype, dtype):
    rng = np.random.default_rng(1)
    base = rng.standard_normal((4, 6)).astype(dtype)
    if dtype is np.complex128:
        base = base + 1j * rng.standard_normal((4, 6))
    # each row repeats a different number of its roots, in a shuffled order
    shift = np.where(np.arange(4)[None] <= np.arange(4)[:, None], 1e-4, 1.0)
    p = np.stack([np.concatenate([base[b], base[b, :4] + shift[b]]) for b in range(4)])
    p = np.stack([row[rng.permutation(10)] for row in p])
    u, m, index = unique_roots(torch.from_numpy(p), rtype=rtype)
    assert u.dtype == torch.from_numpy(p).dtype
    assert m.dtype == index.dtype == torch.int64
    expected = [signal.unique_roots(row, rtype=rtype) for row in p]
    np.testing.assert_array_equal(index.numpy(), np.repeat(range(4), [len(e[0]) for e in expected]))
    np.testing.assert_allclose(u.numpy(), np.concatenate([e[0] for e in expected]))
    np.testing.assert_array_equal(m.numpy(), np.concatenate([e[1] for e in expected]))


def test_unique_roots_gradcheck():
    # Rows with groups of 2 and 1, and of 3: the mean's gradient is 1 / size.
    p = torch.tensor([[0.0, 0.0001, 1.0], [2.0, 2.0002, 2.0001]], dtype=torch.double)
    p.requires_grad_()
    assert torch.autograd.gradcheck(lambda p: unique_roots(p, rtype="avg")[0], (p,))
    p = p.detach().to(torch.cdouble).requires_grad_()
    assert torch.autograd.gradcheck(lambda p: unique_roots(p, rtype="avg")[0], (p,))
    # a minimum takes the gradient of the selected root alone
    p = torch.tensor([0.1, 0.0, 0.2], dtype=torch.double, requires_grad=True)
    unique_roots(p, tol=1.0, rtype="min")[0].sum().backward()
    torch.testing.assert_close(p.grad, torch.tensor([0.0, 1.0, 0.0], dtype=torch.double))
