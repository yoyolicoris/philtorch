from functools import reduce

import numpy as np
import pytest
import torch
from scipy import signal

from philtorch.lti import invresz, residuez, sos2tf, unique_roots

F64 = torch.float64


def _random_sos(*lead: int, K: int) -> np.ndarray:
    """Sections b0 b1 b2 a1 a2 of shape (*lead, K, 5), poles inside the unit circle."""
    shape = (*lead, K)
    r = np.random.rand(*shape) * 0.8 + 0.1
    theta = np.random.rand(*shape) * np.pi
    a1, a2 = -2 * r * np.cos(theta), r**2
    b = np.random.randn(*shape, 3)
    return np.concatenate([b, a1[..., None], a2[..., None]], axis=-1)


def _a(*poles: complex) -> np.ndarray:
    """The denominator coefficients without the leading 1 of a polynomial with these roots."""
    a = np.poly(poles)[1:]
    return a.real if np.allclose(a.imag, 0) else a


def _canonical(r: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order poles that SciPy's sort by magnitude leaves tied, such as a conjugate pair.

    Equal poles keep their order, which is that of their powers.
    """
    order = np.lexsort((np.round(p.imag, 4), np.round(p.real, 4)))
    return r[order], p[order]


def _check_residuez(b: np.ndarray, a: np.ndarray, **kwargs):
    """Compare residuez with SciPy, row by row when batched."""
    r, p, k = residuez(torch.from_numpy(b), torch.from_numpy(a), **kwargs)
    assert r.is_complex() and p.is_complex()
    b, a, r, p, k = (np.atleast_2d(t) for t in (b, a, r.numpy(), p.numpy(), k.numpy()))
    for i in range(len(r)):
        # an unbatched b or a is shared by the rows
        row_b, row_a = b[i if len(b) > 1 else 0], a[i if len(a) > 1 else 0]
        expected_r, expected_p, expected_k = signal.residuez(
            row_b, np.concatenate([[1.0], row_a]), **kwargs
        )
        got_r, got_p = _canonical(r[i], p[i])
        expected_r, expected_p = _canonical(expected_r, expected_p)
        np.testing.assert_allclose(got_p, expected_p, atol=1e-8)
        np.testing.assert_allclose(got_r, expected_r, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(k[i], expected_k, rtol=1e-6, atol=1e-8)


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


@pytest.mark.parametrize("K", [1, 2, 3, 5])
@pytest.mark.parametrize("batched", [False, True])
def test_sos2tf_matches_scipy(K, batched):
    np.random.seed(K)
    sos = _random_sos(3, K=K) if batched else _random_sos(K=K)
    b, a = sos2tf(torch.from_numpy(sos))
    assert b.shape == (*sos.shape[:-2], 2 * K + 1)
    assert a.shape == (*sos.shape[:-2], 2 * K)
    for i, row in enumerate(sos if batched else sos[None]):
        expected_b, expected_a = signal.sos2tf(np.insert(row, 3, 1.0, axis=-1))
        np.testing.assert_allclose((b[i] if batched else b).numpy(), expected_b, atol=1e-12)
        np.testing.assert_allclose((a[i] if batched else a).numpy(), expected_a[1:], atol=1e-12)


@pytest.mark.parametrize(
    "b_length, a",
    [
        (3, _a(0.5, -0.3, 0.2)),  # simple real poles
        (3, _a(0.6 + 0.2j, 0.6 - 0.2j, 0.1)),  # conjugate pair
        (3, _a(0.5 + 0.2j, 0.3 - 0.1j, -0.4j)),  # complex coefficients
        (5, _a(0.5, 0.5, 0.3)),  # repeated real poles
        (5, _a(0.3 + 0.4j, 0.3 - 0.4j, 0.3 + 0.4j, 0.3 - 0.4j, 0.7)),  # repeated complex poles
        (5, _a(0.5, 0.5, 0.3 + 0.4j, 0.3 - 0.4j, 0.3 + 0.4j, 0.3 - 0.4j)),  # both
        (3, _a(0.5, 0.2, 0.3 + 0.4j, 0.3 - 0.4j)),  # numerator shorter than the denominator
        (4, _a(0.5, 0.2, 0.3 + 0.4j, 0.3 - 0.4j)),  # as long as the denominator
        (5, _a(0.5, 0.2, 0.3 + 0.4j, 0.3 - 0.4j)),  # one longer: a constant k
        (9, _a(0.5, 0.5, 0.3 + 0.4j, 0.3 - 0.4j, 0.3 + 0.4j, 0.3 - 0.4j)),  # longer
    ],
)
@pytest.mark.parametrize("rtype", ["avg", "min"])
def test_residuez_matches_scipy(b_length, a, rtype):
    rng = np.random.default_rng(b_length)
    b = rng.standard_normal(b_length)
    if np.iscomplexobj(a):
        b = b + 1j * rng.standard_normal(b_length)
    _check_residuez(b, a, rtype=rtype)


def test_residuez_triple_pole():
    # the roots of a triple pole are only known to about 1e-5, so "min" would
    # choose among noise; the average is stable
    b = np.random.default_rng(0).standard_normal(6)
    _check_residuez(b, _a(0.5, 0.5, 0.5, -0.2))


def test_residuez_batched_multiplicities():
    # one filter per row, each with a different multiplicity of the same order
    rows = [
        _a(0.5, 0.5, 0.3),
        _a(0.5, 0.4, 0.3),
        _a(0.5, 0.5, 0.5),
        _a(0.6, 0.3 + 0.4j, 0.3 - 0.4j),
    ]
    a = np.stack(rows)
    b = np.random.default_rng(0).standard_normal((4, 5))
    _check_residuez(b, a)
    # an unbatched b is shared by the rows
    _check_residuez(b[0], a)


def test_residuez_unbatched_denominator_with_batched_numerator():
    a = _a(0.5, 0.5, 0.3)
    b = np.random.default_rng(0).standard_normal((3, 5))
    r, p, k = residuez(torch.from_numpy(b), torch.from_numpy(a))
    assert r.shape == p.shape == (3, 3)
    assert k.shape == (3, 2)
    _check_residuez(b, a)


def test_residuez_without_poles():
    b = torch.arange(3.0, dtype=F64)
    r, p, k = residuez(b, torch.zeros(0, dtype=F64))
    assert r.shape == p.shape == (0,)
    torch.testing.assert_close(k, b)


def _check_invresz(r: np.ndarray, p: np.ndarray, k: np.ndarray, **kwargs):
    r, p, k = (np.ascontiguousarray(x) for x in (r, p, k))  # SciPy's k has a negative stride
    b, a = invresz(torch.from_numpy(r), torch.from_numpy(p), torch.from_numpy(k), **kwargs)
    expected_b, expected_a = signal.invresz(r, p, k, **kwargs)
    assert b.is_complex() and a.is_complex()
    assert b.shape[-1] == len(r) + len(k)
    # SciPy's numerator can be longer than ours by zeros, never shorter
    expected_b = np.pad(expected_b, (0, b.size(-1) - len(expected_b)))
    np.testing.assert_allclose(b.numpy(), expected_b, atol=1e-10)
    np.testing.assert_allclose(a.numpy(), expected_a[1:], atol=1e-10)


@pytest.mark.parametrize("b_length", [3, 6, 9])
@pytest.mark.parametrize(
    "a",
    [
        _a(0.5, -0.3, 0.2, 0.1),
        _a(0.5, 0.5, 0.3, -0.2),
        _a(0.5, 0.5, 0.3 + 0.4j, 0.3 - 0.4j, 0.3 + 0.4j, 0.3 - 0.4j),
    ],
)
@pytest.mark.parametrize("rtype", ["avg", "min", "max"])
def test_invresz_matches_scipy_and_round_trips(b_length, a, rtype):
    rng = np.random.default_rng(b_length)
    b = rng.standard_normal(b_length)
    r, p, k = signal.residuez(b, np.concatenate([[1.0], a]))
    _check_invresz(r, p, k, rtype=rtype)

    # invresz(residuez(b, a)) is (b, a), a numerator shorter than a padded
    got_b, got_a = invresz(*residuez(torch.from_numpy(b), torch.from_numpy(a)))
    np.testing.assert_allclose(got_a.numpy(), a, atol=1e-10)
    np.testing.assert_allclose(got_b.numpy(), np.pad(b, (0, got_b.size(-1) - b_length)), atol=1e-9)


def test_invresz_batched():
    rows = [_a(0.5, 0.5, 0.3), _a(0.5, 0.4, 0.3), _a(0.5, 0.5, 0.5)]
    parts = [
        signal.residuez(np.random.default_rng(i).standard_normal(5), [1.0, *a])
        for i, a in enumerate(rows)
    ]
    r, p, k = (torch.from_numpy(np.stack([part[i] for part in parts])) for i in range(3))
    b, a = invresz(r, p, k)
    assert b.shape == (3, 5) and a.shape == (3, 3)
    for i, part in enumerate(parts):
        expected_b, expected_a = signal.invresz(*part)
        np.testing.assert_allclose(
            b[i].numpy(), np.pad(expected_b, (0, 5 - len(expected_b))), atol=1e-10
        )
        np.testing.assert_allclose(a[i].numpy(), expected_a[1:], atol=1e-10)
    # a shared k
    b_shared, _ = invresz(r, p, k[0])
    assert b_shared.shape == (3, 5)


def test_invresz_groups_poles_by_the_previous_pole():
    # Each pole of the first three is within tol of the one before, but the
    # first and the third are not: one run of three, then a simple pole. The
    # run's value is the average of its members, 0.0015.
    p = np.array([0.0, 0.0015, 0.003, 1.0])
    r = np.array([1.0, 2.0, 3.0, 4.0])
    b, a = invresz(torch.from_numpy(r), torch.from_numpy(p), torch.zeros(0, dtype=F64), tol=2e-3)

    def poly(*roots):  # coefficients of the product of (1 - x w), ascending
        return reduce(np.convolve, [[1.0, -x] for x in roots])

    q = 0.0015
    denominator = poly(q, q, q, 1.0)
    # r1 / (1 - q w) + r2 / (1 - q w)^2 + r3 / (1 - q w)^3 + r4 / (1 - w)
    terms = [poly(q, q, 1.0), poly(q, 1.0), poly(1.0), poly(q, q, q)]
    numerator = sum(c * np.pad(t, (0, 4 - len(t))) for c, t in zip(r, terms))
    np.testing.assert_allclose(a.numpy(), denominator[1:], atol=1e-12)
    np.testing.assert_allclose(b.numpy(), numerator, atol=1e-12)


def test_invresz_without_poles():
    k = torch.arange(3.0, dtype=F64)
    b, a = invresz(torch.zeros(0, dtype=F64), torch.zeros(0, dtype=F64), k)
    assert a.shape == (0,)
    torch.testing.assert_close(b.real, k)


def test_sos2tf_gradcheck():
    sos = torch.from_numpy(_random_sos(2, K=3)).requires_grad_()
    assert torch.autograd.gradcheck(sos2tf, (sos,))


@pytest.mark.parametrize(
    "a",
    [
        # distinct magnitudes, so that a perturbation doesn't reorder the poles
        _a(0.6, -0.3, 0.2, -0.1),
        _a(0.6, 0.2 + 0.3j, 0.1 - 0.4j, -0.3),
    ],
)
def test_residuez_gradcheck(a):
    b = torch.randn(6, dtype=F64, requires_grad=True)
    a = torch.from_numpy(a).requires_grad_()
    assert torch.autograd.gradcheck(residuez, (b, a))


def test_invresz_gradcheck():
    r = torch.randn(4, dtype=torch.cdouble, requires_grad=True)
    p = torch.tensor([0.6, -0.3, 0.2 + 0.3j, 0.1], dtype=torch.cdouble, requires_grad=True)
    k = torch.randn(2, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(invresz, (r, p, k))
