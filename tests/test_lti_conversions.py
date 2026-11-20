from functools import reduce

import numpy as np
import pytest
import torch
from scipy import signal

from philtorch.lti import (
    cascade_ss,
    invresz,
    pfefilt,
    residuez,
    sos2pfe,
    sos2ss,
    sos2tf,
    sosfilt,
    state_space,
    unique_roots,
)

from .test_lti_sosfilt import _flat

F64 = torch.float64


def _random_sos(*lead: int, K: int) -> np.ndarray:
    """Sections b0 b1 b2 a1 a2 of shape (*lead, K, 5), poles inside the unit circle."""
    shape = (*lead, K)
    r = np.random.rand(*shape) * 0.8 + 0.1
    theta = np.random.rand(*shape) * np.pi
    a1, a2 = -2 * r * np.cos(theta), r**2
    b = np.random.randn(*shape, 3)
    return np.concatenate([b, a1[..., None], a2[..., None]], axis=-1)


def _ragged_rows() -> list[np.ndarray]:
    """Rows of 3, 1, 0 and 2 sections, with a real section and a section without poles."""
    np.random.seed(0)
    rows = [_random_sos(K=3), _random_sos(K=1), np.zeros((0, 5)), _random_sos(K=2)]
    rows[0][1, 3:] = [-0.8, 0.15]
    rows[3][0, 3:] = 0.0
    return rows


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


def _delayed_residuez(b: np.ndarray, a: np.ndarray, **kwargs):
    """SciPy's residuez in the delayed form, with a = [1, a1, ...].

    The direct term is the first D samples of the impulse response, and the
    residues are SciPy's of the rest, H - k, advanced by D, which has no
    direct term.
    """
    D = max(len(b) - len(a) + 1, 0)
    if D == 0:
        return signal.residuez(b, a, **kwargs)
    k = signal.lfilter(b, a, np.eye(1, D)[0])
    rest = (b - np.convolve(k, a)[: len(b)])[D:]
    r, p, _ = signal.residuez(rest, a, **kwargs)
    return r, p, k


def _delayed_invresz(r: np.ndarray, p: np.ndarray, k: np.ndarray, **kwargs):
    """SciPy's invresz of the delayed form: k a, plus SciPy's numerator delayed by D."""
    b_rest, a = signal.invresz(r, p, [], **kwargs)
    D = len(k)
    b = np.zeros(len(r) + D, dtype=complex)
    if D:
        b += np.convolve(k, a)
    b[D : D + len(b_rest)] += b_rest
    return b, a


def _check_residuez(b: np.ndarray, a: np.ndarray, delayed_form: bool = False, **kwargs):
    """Compare residuez with SciPy, row by row when batched."""
    r, p, k = residuez(*map(torch.from_numpy, (b, a)), delayed_form=delayed_form, **kwargs)
    reference = _delayed_residuez if delayed_form else signal.residuez
    assert r.is_complex() and p.is_complex()
    b, a, r, p, k = (np.atleast_2d(t) for t in (b, a, r.numpy(), p.numpy(), k.numpy()))
    for i in range(len(r)):
        # an unbatched b or a is shared by the rows
        row_b, row_a = b[i if len(b) > 1 else 0], a[i if len(a) > 1 else 0]
        expected_r, expected_p, expected_k = reference(
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


@pytest.mark.parametrize("rtype", ["max", "min", "avg"])
def test_unique_roots_matches_scipy(rtype):
    # SciPy's docstring example
    _check_unique_roots(np.array([0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3]), 2e-2, rtype)
    # a chain a ~ b ~ c, with a and c farther apart than tol, is one group in
    # either order, which SciPy's greedy grouping splits
    for chain in ([0.0, 0.0015, 0.003, 1.0], [0.003, 0.0015, 0.0, 1.0]):
        _, m = unique_roots(torch.tensor(chain, dtype=F64), tol=2e-3, rtype=rtype)
        assert m.tolist() == [3, 1]
    # complex roots, compared by real part, then imaginary part
    z = np.array([0.5 + 0.3j, 0.5 - 0.3j, 0.5 + 0.3001j, 0.2j, 0.2 - 0.0004j, 0.5 - 0.2999j])
    _check_unique_roots(z, 1e-2, rtype)
    rng = np.random.default_rng(0)
    base = rng.standard_normal(5) + 1j * rng.standard_normal(5)
    z = np.concatenate([base, base + 1e-4, base - 1e-4j])[rng.permutation(15)]
    _check_unique_roots(z, 1e-3, rtype)
    _check_unique_roots(z.real.copy(), 1e-3, rtype)


def test_unique_roots_without_roots():
    u, m = unique_roots(torch.zeros(0, dtype=F64))
    assert u.shape == m.shape == (0,)


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


@pytest.mark.parametrize("K", [1, 2, 3, 5])
@pytest.mark.parametrize("batched", [False, True])
def test_sos2tf_matches_scipy(K, batched):
    np.random.seed(K)
    sos = _random_sos(3, K=K) if batched else _random_sos(K=K)
    batch_index = torch.arange(3).repeat_interleave(K) if batched else None
    b, a = sos2tf(torch.from_numpy(sos.reshape(-1, 5)), batch_index)
    assert b.shape == (*sos.shape[:-2], 2 * K + 1)
    assert a.shape == (*sos.shape[:-2], 2 * K)
    rows = zip(
        sos.reshape(-1, K, 5), b.reshape(-1, 2 * K + 1).numpy(), a.reshape(-1, 2 * K).numpy()
    )
    for row, got_b, got_a in rows:
        expected_b, expected_a = signal.sos2tf(np.insert(row, 3, 1.0, axis=-1))
        np.testing.assert_allclose(got_b, expected_b, atol=1e-12)
        np.testing.assert_allclose(got_a, expected_a[1:], atol=1e-12)


@pytest.mark.parametrize("K", [1, 3])
def test_sos2ss_matches_sosfilt(K):
    np.random.seed(K)
    sos = torch.from_numpy(_random_sos(K=K))
    x, zi = torch.randn(4, 16, dtype=F64), torch.randn(4, K, 2, dtype=F64)
    A, B, C, D = sos2ss(sos)
    y, zf = state_space(A, x, B, C, D, zi=zi.flatten(1))
    expected, expected_zf = sosfilt(sos, x, zi=zi)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(zf, expected_zf.flatten(1))


def test_sos2ss_ragged_rows_equal_sosfilt_with_batch_index():
    sos, batch_index = _flat(_ragged_rows(), interleave=True)
    x = torch.randn(4, 16, dtype=F64)
    A, B, C, D = sos2ss(sos, batch_index)
    assert A.shape == (4, 6, 6)
    expected = sosfilt(sos, x, batch_index=batch_index)
    torch.testing.assert_close(state_space(A, x, B, C, D), expected)


def test_sos2tf_ragged_rows_equal_each_row_alone_padded_with_zeros():
    rows = _ragged_rows()
    b, a = sos2tf(*_flat(rows, interleave=True))
    assert b.shape == (4, 7) and a.shape == (4, 6)
    for i, row in enumerate(rows):
        expected_b, expected_a = (
            sos2tf(torch.from_numpy(row))
            if len(row)
            else (torch.ones(1, dtype=F64), torch.zeros(0, dtype=F64))
        )
        torch.testing.assert_close(
            b[i], torch.nn.functional.pad(expected_b, (0, 7 - expected_b.numel()))
        )
        torch.testing.assert_close(
            a[i], torch.nn.functional.pad(expected_a, (0, 6 - expected_a.numel()))
        )


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
@pytest.mark.parametrize("delayed_form", [False, True])
def test_residuez_matches_scipy(b_length, a, delayed_form):
    # rtype stays "avg": rounding splits a double pole by about 1e-8,
    # into two real roots or a conjugate pair, depending on the LAPACK build, and
    # "min" would choose among that noise
    rng = np.random.default_rng(b_length)
    b = rng.standard_normal(b_length)
    if np.iscomplexobj(a):
        b = b + 1j * rng.standard_normal(b_length)
    _check_residuez(b, a, delayed_form=delayed_form)


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
    # an unbatched b is shared by the rows, and so is an unbatched a
    _check_residuez(b[0], a)
    _check_residuez(b, a[0])
    r, p, k = residuez(torch.from_numpy(b), torch.from_numpy(a[0]))
    assert r.shape == p.shape == (4, 3)
    assert k.shape == (4, 2)


def test_residuez_orders_a_conjugate_pair_by_angle():
    # the solver returns real input's conjugates exactly, so their magnitudes tie
    a = torch.from_numpy(_a(0.3 + 0.4j, 0.3 - 0.4j))
    _, p, _ = residuez(torch.ones(1, dtype=F64), a)
    torch.testing.assert_close(p, torch.tensor([0.3 - 0.4j, 0.3 + 0.4j], dtype=torch.complex128))


@pytest.mark.parametrize("b_length", [2, 4])
def test_residuez_delayed_form_takes_a_pole_at_zero(b_length):
    # a first-order filter padded to second order: poles 0.5 and 0
    b = torch.arange(1.0, b_length + 1, dtype=F64)
    a = torch.tensor([-0.5, 0.0], dtype=F64)
    r, p, k = residuez(b, a, delayed_form=True)
    # the expansion's impulse response is the filter's
    # (NumPy's 0j ** 0 is 1, torch's is nan)
    h = (r.numpy()[:, None] * p.numpy()[:, None] ** np.arange(20)).sum(0).real
    h = np.concatenate([k.numpy(), h])[:20]
    x = np.eye(1, 20)[0]
    np.testing.assert_allclose(h, signal.lfilter(b.numpy(), [1.0, -0.5, 0.0], x), atol=1e-12)
    b_back, a_back = invresz(r, p, k, delayed_form=True)
    torch.testing.assert_close(b_back.real, b)
    torch.testing.assert_close(a_back.real, a)


def test_residuez_without_poles():
    b = torch.arange(3.0, dtype=F64)
    r, p, k = residuez(b, torch.zeros(0, dtype=F64))
    assert r.shape == p.shape == (0,)
    torch.testing.assert_close(k, b)


def _check_invresz(r, p, k, delayed_form: bool = False, **kwargs):
    r, p, k = (np.ascontiguousarray(x) for x in (r, p, k))  # SciPy's k has a negative stride
    b, a = invresz(*map(torch.from_numpy, (r, p, k)), delayed_form=delayed_form, **kwargs)
    reference = _delayed_invresz if delayed_form else signal.invresz
    expected_b, expected_a = reference(r, p, k, **kwargs)
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
@pytest.mark.parametrize("delayed_form", [False, True])
def test_invresz_matches_scipy_and_round_trips(b_length, a, delayed_form):
    rng = np.random.default_rng(b_length)
    b = rng.standard_normal(b_length)
    reference = _delayed_residuez if delayed_form else signal.residuez
    r, p, k = reference(b, np.concatenate([[1.0], a]))
    _check_invresz(r, p, k, delayed_form)

    # invresz(residuez(b, a)) is (b, a), a numerator shorter than a padded
    expansion = residuez(torch.from_numpy(b), torch.from_numpy(a), delayed_form=delayed_form)
    got_b, got_a = invresz(*expansion, delayed_form=delayed_form)
    np.testing.assert_allclose(got_a.numpy(), a, atol=1e-10)
    np.testing.assert_allclose(got_b.numpy(), np.pad(b, (0, got_b.size(-1) - b_length)), atol=1e-9)


@pytest.mark.parametrize("delayed_form", [False, True])
def test_invresz_keeps_a_long_direct_term_beside_a_growing_pole(delayed_form):
    # the fraction's response grows as 2^n, which would round the ones of k away
    _check_invresz(np.array([1.0]), np.array([2.0]), np.ones(60), delayed_form)


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
    torch.testing.assert_close(invresz(r, p, k[0]), invresz(r, p, k[0].expand(3, -1)))


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


def _check_sos2pfe(sos: np.ndarray, atol: float):
    """Compare sos2pfe with SciPy's residue of the sections' product, row by row."""
    flat = torch.from_numpy(sos.reshape(-1, 5))
    r, p, g, batch_index, multiplicity = sos2pfe(
        flat,
        batch_index=torch.arange(sos.shape[0]).repeat_interleave(sos.shape[1])
        if sos.ndim == 3
        else None,
    )
    U = p.numel()
    assert p.shape == multiplicity.shape == (U,)
    assert r.shape == (int(multiplicity.sum()),)
    assert (batch_index is None) == (sos.ndim == 2)
    assert batch_index is None or batch_index.shape == (U,)
    assert g.shape == sos.shape[:-2]
    index = torch.zeros(U, dtype=torch.long) if batch_index is None else batch_index
    for b, (row, got_g) in enumerate(zip(sos.reshape(-1, sos.shape[-2], 5), g.numpy().ravel())):
        mine = index == b
        # each pole's residues, one entry per power: the pole is listed as often
        got_r = r[mine.repeat_interleave(multiplicity)].numpy()
        got_p = p[mine].repeat_interleave(multiplicity[mine]).numpy()
        if sos.dtype.kind != "c":
            # the half form lists one pole of each conjugate pair
            pair = got_p.imag != 0
            got_r = np.concatenate([got_r, got_r[pair].conj()])
            got_p = np.concatenate([got_p, got_p[pair].conj()])
        expected_r, expected_p, expected_k = signal.residue(
            *signal.sos2tf(np.insert(row, 3, 1.0, axis=-1))
        )
        got_r, got_p = _canonical(got_r, got_p)
        expected_r, expected_p = _canonical(expected_r, expected_p)
        np.testing.assert_allclose(got_p, expected_p, atol=atol)
        np.testing.assert_allclose(got_r, expected_r, rtol=atol, atol=atol)
        np.testing.assert_allclose(got_g, expected_k[0], rtol=1e-12)


def test_sos2pfe_matches_scipy_for_simple_poles():
    np.random.seed(0)
    sos = _random_sos(K=4)
    # a section with real poles, and an odd-order design, whose last section has a pole at 0
    sos[1, 3:] = [-0.8, 0.15]
    _check_sos2pfe(sos, 1e-8)
    _check_sos2pfe(np.delete(signal.butter(5, 0.2, output="sos"), 3, 1), 1e-8)
    _check_sos2pfe(_random_sos(K=3) + 0.2j * _random_sos(K=3), 1e-10)
    # a complex sos with a real pole, whose residue is complex
    _check_sos2pfe(np.array([[1.0, 1j, 0.0, -0.5, 0.0]]), 1e-12)


@pytest.mark.parametrize("rounding", [0.0, 1e-9])
def test_sos2pfe_lists_one_pole_of_a_conjugate_pair(monkeypatch, rounding):
    # CUDA's complex sqrt of a negative real has a real part that isn't 0 (2e-8 in complex64)
    sqrt = torch.sqrt
    monkeypatch.setattr(torch, "sqrt", lambda z: sqrt(z) + rounding)
    sos = torch.tensor([[1.0, 0.5, 0.2, -0.6, 0.45]], dtype=F64)
    r, p, g, _, multiplicity = sos2pfe(sos)
    assert p.shape == (1,) and p[0].imag > 0
    x = torch.randn(8, dtype=F64)
    torch.testing.assert_close(pfefilt(r, p, g, x, multiplicity=multiplicity), sosfilt(sos, x))


@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_sos2pfe_default_tol_merges_a_double_root_split_by_rounding(dtype):
    # (z - 0.7)^2, whose roots rounding splits by about sqrt(eps)
    _, p, _, _, multiplicity = sos2pfe(torch.tensor([[1.0, 0.3, 0.2, -1.4, 0.49]], dtype=dtype))
    assert p.shape == (1,)
    assert multiplicity.tolist() == [2]


def test_sos2pfe_matches_scipy_for_repeated_poles():
    double_real = [1.0, 0.5, -0.3, -1.0, 0.25]
    complex_pair = np.delete(signal.butter(2, 0.3, output="sos"), 3, 1)[0]
    _check_sos2pfe(np.stack([complex_pair, complex_pair, complex_pair, double_real]), 1e-10)


def test_sos2pfe_fir_section():
    # 1 + 2 / z + 3 / z^2: a double pole at 0 with residues 2 and 3
    r, p, g, batch_index, multiplicity = sos2pfe(
        torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]], dtype=F64)
    )
    torch.testing.assert_close(r, torch.tensor([2.0, 3.0], dtype=torch.complex128))
    torch.testing.assert_close(p, torch.zeros(1, dtype=torch.complex128))
    torch.testing.assert_close(g, torch.tensor(1.0, dtype=F64))
    assert multiplicity.tolist() == [2] and batch_index is None


def test_sos2pfe_ragged_rows_equal_each_row_alone():
    rows = _ragged_rows()
    sos, batch_index = _flat(rows, interleave=True)
    r, p, g, index, multiplicity = sos2pfe(sos, batch_index=batch_index)
    assert g.shape == (4,)
    # the row without a section: no term and a gain of 1
    assert g[2] == 1 and (index != 2).all()
    for i, row in enumerate(rows):
        if not len(row):
            continue
        expected_r, expected_p, expected_g, _, expected_multiplicity = sos2pfe(
            torch.from_numpy(row)
        )
        torch.testing.assert_close(r[(index == i).repeat_interleave(multiplicity)], expected_r)
        torch.testing.assert_close(p[index == i], expected_p)
        torch.testing.assert_close(g[i], expected_g)
        assert (multiplicity[index == i] == expected_multiplicity).all()
    # in the order of the rows
    assert (index.diff() >= 0).all()


def test_sos2pfe_gives_no_term_for_the_padding_of_a_shorter_row():
    # one section among rows of 3: padded with identity sections, whose double
    # pole at 0 must not join a pole at 0 of a live section, which here is
    # a section without poles
    np.random.seed(0)
    live = np.array([[1.0, 2.0, 3.0, 0.0, 0.0]])
    rows = [_random_sos(K=3), live, _random_sos(K=3)]
    sos, batch_index = _flat(rows, interleave=True)
    r, p, g, index, multiplicity = sos2pfe(sos, batch_index=batch_index)
    expected_r, expected_p, _, _, expected_multiplicity = sos2pfe(torch.from_numpy(live))
    torch.testing.assert_close(r[(index == 1).repeat_interleave(multiplicity)], expected_r)
    torch.testing.assert_close(p[index == 1], expected_p)
    assert (multiplicity[index == 1] == expected_multiplicity).all()
    assert (index == 1).sum() == 1


def test_sos2pfe_batched_multiplicities():
    # one cascade per row, each with a different multiplicity of the same order
    double = [1.0, 0.5, -0.3, -1.0, 0.25]
    pair = np.delete(signal.butter(2, 0.3, output="sos"), 3, 1)[0]
    other = [0.7, 0.1, 0.2, -0.3, 0.02]
    sos = np.stack([[double, double], [double, other], [pair, pair], [pair, other]])
    _check_sos2pfe(sos, 1e-10)


@pytest.mark.parametrize(
    "design, atol",
    [
        (lambda: signal.butter(24, 0.2, output="sos"), 5e-10),
        (lambda: signal.cheby1(24, 1, 0.2, output="sos"), 1e-11),
        # poles 1.2e-4 apart, which SciPy's tol of 1e-3 would merge
        (lambda: signal.ellip(24, 0.1, 100, 0.05, output="sos"), 1e-12),
    ],
)
def test_sos2pfe_impulse_response_matches_sosfilt(design, atol):
    sos = torch.from_numpy(np.delete(design(), 3, 1))
    r, p, g, batch_index, multiplicity = sos2pfe(sos)
    x = torch.eye(1, 200, dtype=F64)[0]
    h = pfefilt(r, p, g, x, batch_index=batch_index, multiplicity=multiplicity)
    expected = sosfilt(sos, torch.eye(1, 200, dtype=F64)[0])
    assert ((h - expected).abs().max() / expected.abs().max()).item() < atol


def test_sos2pfe_groups_a_cluster_of_conjugate_pairs_symmetrically():
    # pairs 0.5 +- 0.0004j and 0.5006 +- 0.0008j: each pole of the second is
    # within tol of the first's on its side, which are within tol of each
    # other, so the four are one group, its own conjugate, with a real pole
    poles = [0.5 + 0.0004j, 0.5006 + 0.0008j]
    sos = torch.tensor([[1.0, 0.3, 0.2, -2 * q.real, abs(q) ** 2] for q in poles], dtype=F64)
    r, p, g, _, multiplicity = sos2pfe(sos, tol=1e-3)
    assert p.shape == (1,) and p[0].imag == 0
    assert multiplicity.tolist() == [4]


# Input and output sizes between the stages of a cascade, 0 for a scalar signal; the sizes of
# the stages' states follow.
_CASCADES = {
    1: ([0, 0], [3]),
    2: ([2, 0, 3], [2, 3]),
    5: ([0, 2, 0, 3, 1, 0], [2, 3, 1, 4, 2]),
}


def _random_stage(M: int, F: int, P: int, lead: tuple[int, ...]):
    """A stable stage with A (M, M), B (M, F), C (P, M) and D (P, F), a size of 0 dropping its
    dimension, and `lead` batch dimensions."""
    A = torch.randn((*lead, M, M), dtype=F64)
    A = A * (0.8 / torch.linalg.eigvals(A).abs().amax(-1))[..., None, None]
    B = torch.randn((*lead, M, *([F] if F else [])), dtype=F64)
    C = torch.randn((*lead, *([P] if P else []), M), dtype=F64)
    D = torch.randn((*lead, *([P] if P else []), *([F] if F else [])), dtype=F64)
    return A, B, C, D


@pytest.mark.parametrize(("K", "batched"), [(1, False), (2, False), (5, True)])
def test_cascade_ss_matches_the_stages_in_series(K, batched):
    torch.manual_seed(K)
    sizes, orders = _CASCADES[K]
    # more signals than any size, so that no shape of a stage reads as another
    batch, steps = 5, 12
    lead = (batch,) if batched else ()
    stages = [
        _random_stage(M, F, P, lead) for M, F, P in zip(orders, sizes[:-1], sizes[1:], strict=True)
    ]
    zi = torch.randn(batch, sum(orders), dtype=F64)
    x = torch.randn(batch, steps, *([sizes[0]] if sizes[0] else []), dtype=F64)

    A, B, C, D = cascade_ss(*zip(*stages))
    y, zf = state_space(A, x, B, C, D, zi=zi)

    expected, finals = x, []
    for stage, z in zip(stages, zi.split(orders, 1), strict=True):
        expected, final = state_space(stage[0], expected, *stage[1:], zi=z)
        finals.append(final)
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(zf, torch.cat(finals, 1))
