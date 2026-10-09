import math

import pytest
import torch

import philtorch.estimation.kalman as kalman
from philtorch.estimation import kalman_filter, kalman_smoother
from philtorch.lpv import state_space_recursion as lpv_state_space_recursion

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    ),
]


def _transitions(N, prior):
    """How many steps A and Q have.

    "first_state" is the functions' model, with the prior on the first
    measured state. The references also take "before_first_step", a prior
    one step before it, to check the docstring's conversion.
    """
    return N if prior == "before_first_step" else max(N - 1, 0)


def _model(batch_size, N, M, P, *, prior="first_state", seed=0, dtype=torch.float64):
    """A random time-varying model, as in Sec. VI-A of arXiv:2511.10363."""
    gen = torch.Generator().manual_seed(seed)
    T = _transitions(N, prior)

    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, generator=gen)

    def covariance(*shape):
        X = randn(*shape)
        return X @ X.mH + 0.1 * torch.eye(shape[-1], dtype=dtype)

    A = 0.99 * torch.linalg.qr(randn(batch_size, T, M, M))[0]
    C = randn(batch_size, N, P, M)
    Q = covariance(batch_size, T, M, M)
    R = covariance(batch_size, N, P, P)
    m0 = randn(batch_size, M)
    P0 = covariance(batch_size, M, M)
    y = randn(batch_size, N, P)
    return y, A, C, Q, R, m0, P0


def _reference(y, A, C, Q, R, m0, P0, prior, u=None, d=None):
    """The sequential Kalman filter and Rauch-Tung-Striebel smoother.

    Returns the filtered and smoothed moments of the measured states. The
    optional u and d are known offsets of the state (one per step of A) and
    the measurements.
    """
    N = y.size(1)
    u = torch.zeros_like(A[..., 0]) if u is None else u
    d = torch.zeros_like(y) if d is None else d
    # The step into the state y[n] measures, if there is one.
    step = (lambda n: n) if prior == "before_first_step" else (lambda n: n - 1)
    m, P = m0, P0
    means, covs = [], []
    for n in range(N):
        if step(n) >= 0:
            k = step(n)
            m = (A[:, k] @ m[..., None]).squeeze(-1) + u[:, k]
            P = A[:, k] @ P @ A[:, k].mH + Q[:, k]
        S = C[:, n] @ P @ C[:, n].mH + R[:, n]
        K = torch.linalg.solve(S, C[:, n] @ P).mH
        innovation = y[:, n] - d[:, n] - (C[:, n] @ m[..., None]).squeeze(-1)
        m = m + (K @ innovation[..., None]).squeeze(-1)
        P = P - K @ S @ K.mH
        means.append(m)
        covs.append(P)

    smoothed_means, smoothed_covs = [means[-1]], [covs[-1]]
    for n in range(N - 2, -1, -1):
        k = step(n + 1)
        m_pred = (A[:, k] @ means[n][..., None]).squeeze(-1) + u[:, k]
        P_pred = A[:, k] @ covs[n] @ A[:, k].mH + Q[:, k]
        G = torch.linalg.solve(P_pred, A[:, k] @ covs[n]).mH
        smoothed_means.append(means[n] + (G @ (smoothed_means[-1] - m_pred)[..., None]).squeeze(-1))
        smoothed_covs.append(covs[n] + G @ (smoothed_covs[-1] - P_pred) @ G.mH)
    return (
        torch.stack(means, 1),
        torch.stack(covs, 1),
        torch.stack(smoothed_means[::-1], 1),
        torch.stack(smoothed_covs[::-1], 1),
    )


def _check_against_reference(args, rtol=1e-9, atol=1e-9):
    expected = _reference(*args, "first_state")
    filtered = kalman_filter(*args)
    smoothed = kalman_smoother(*args)
    actual = (filtered.means, filtered.covs, smoothed.means, smoothed.covs)
    names = ["filter mean", "filter cov", "smoother mean", "smoother cov"]
    for name, a, e in zip(names, actual, expected):
        torch.testing.assert_close(a, e, rtol=rtol, atol=atol, msg=name)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("N", [1, 2, 7, 64])
@pytest.mark.parametrize("M, P", [(1, 1), (4, 2), (2, 3)])
def test_kalman_matches_sequential(device, N, M, P):
    args = [t.to(device) for t in _model(3, N, M, P)]
    _check_against_reference(args)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("M, P", [(1, 1), (4, 2), (2, 3)])
def test_kalman_complex_matches_sequential(device, M, P):
    args = [t.to(device) for t in _model(3, 7, M, P, dtype=torch.complex128)]
    _check_against_reference(args)


def _from_before_first_step(y, A, C, Q, R, m0, P0):
    """The docstring's conversion of a prior one step before the first measured state."""
    m0 = (A[:, 0] @ m0.unsqueeze(-1)).squeeze(-1)
    P0 = A[:, 0] @ P0 @ A[:, 0].mH + Q[:, 0]
    return y, A[:, 1:], C, Q[:, 1:], R, m0, P0


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_kalman_prior_before_the_first_step(dtype):
    args = _model(3, 6, 3, 2, prior="before_first_step", dtype=dtype)
    expected = _reference(*args, "before_first_step")
    converted = _from_before_first_step(*args)
    filtered, smoothed = kalman_filter(*converted), kalman_smoother(*converted)
    actual = (filtered.means, filtered.covs, smoothed.means, smoothed.covs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-9)
    # The dense posterior of the model with the extra state, x[0] before y[0].
    means, covs, log_p = _dense_posterior(*args, "before_first_step")
    torch.testing.assert_close(smoothed.means, means[:, 1:], rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(smoothed.log_likelihood, log_p, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("fn", [kalman_filter, kalman_smoother])
@pytest.mark.parametrize("batch_size, N", [(0, 5), (2, 0), (2, 1)])
def test_kalman_empty_inputs(fn, batch_size, N):
    y, A, C, Q, R, m0, P0 = _model(2, 5, 3, 2)
    y = y.new_zeros(batch_size, N, 2)
    result = fn(y, A[0, 0], C[0, 0], Q[0, 0], R[0, 0], m0[0], P0[0])
    assert result.means.shape == (batch_size, N, 3)
    assert result.covs.shape == (batch_size, N, 3, 3)
    assert result.log_likelihood.shape == (batch_size,)
    if fn is kalman_smoother:
        assert result.cross_covs.shape == (batch_size, max(N - 1, 0), 3, 3)
    if N == 0 and batch_size:
        # With no measurements, log p(y) = 0.
        torch.testing.assert_close(result.log_likelihood, y.new_zeros(batch_size))


@pytest.mark.parametrize("fn", [kalman_filter, kalman_smoother])
def test_kalman_coefficient_shapes(fn):
    batch_size, N, M, P = 2, 5, 3, 2
    T = N - 1
    y, A, C, Q, R, m0, P0 = _model(batch_size, N, M, P)

    expected = fn(y, A, C, Q, R, m0, P0)

    # Time-varying shared coefficients, and a shared prior.
    shared = fn(y[:1].expand(batch_size, -1, -1), A[0], C[0], Q[0], R[0], m0[0], P0[0])
    full = fn(
        y[:1].expand(batch_size, -1, -1),
        *(t[:1].expand(batch_size, *t.shape[1:]) for t in (A, C, Q, R, m0, P0)),
    )
    for a, e in zip(shared, full):
        torch.testing.assert_close(a, e)

    # Constant per-signal and constant shared coefficients.
    def over_time(t, steps):
        return t.unsqueeze(-3).expand(*t.shape[:-2], steps, -1, -1)

    steps = (T, N, T, N)
    per_signal = fn(y, A[:, 0], C[:, 0], Q[:, 0], R[:, 0], m0, P0)
    expanded = fn(y, *(over_time(t[:, 0], k) for t, k in zip((A, C, Q, R), steps)), m0, P0)
    for a, e in zip(per_signal, expanded):
        torch.testing.assert_close(a, e)
    constant = fn(y, A[0, 0], C[0, 0], Q[0, 0], R[0, 0], m0, P0)
    expanded = fn(y, *(over_time(t[0, 0], k) for t, k in zip((A, C, Q, R), steps)), m0, P0)
    for a, e in zip(constant, expanded):
        torch.testing.assert_close(a, e)

    for a, e in zip(fn(y, A, C, Q, R, m0, P0), expected):
        torch.testing.assert_close(a, e)


def test_kalman_reads_n_equal_b_as_time_varying():
    # B = 4 signals of N = 5 measurements, so A has 4 transitions.
    y, A, C, Q, R, m0, P0 = _model(4, 5, 2, 1)
    means = kalman_filter(y, A[0], C, Q, R, m0, P0).means
    expected = kalman_filter(y, A[:1].expand(4, -1, -1, -1), C, Q, R, m0, P0).means
    torch.testing.assert_close(means, expected)


@pytest.mark.parametrize(
    "name, shape",
    [
        ("A", (3, 2, 2)),
        ("A", (5, 2, 2)),
        ("Q", (2, 5, 2, 2)),
        ("C", (2, 2)),
        ("C", (4, 1, 2)),
        ("Q", (3, 4, 2, 2)),
        ("R", (1, 2)),
        ("m0", (3,)),
        ("P0", (3, 2, 2)),
    ],
)
def test_kalman_rejects_unsupported_shapes(name, shape):
    # N = 5 measurements, 4 transitions, 2 signals.
    y, A, C, Q, R, m0, P0 = _model(2, 5, 2, 1)
    args = dict(y=y, A=A, C=C, Q=Q, R=R, m0=m0, P0=P0)
    args[name] = torch.ones(shape, dtype=torch.float64)
    with pytest.raises(ValueError, match=name):
        kalman_filter(**args)


@pytest.mark.parametrize("fn", [kalman_filter, kalman_smoother])
def test_kalman_gradcheck(fn):
    y, A, C, Q, R, m0, P0 = _model(2, 5, 2, 1)
    # Covariances as factors, so that gradcheck's perturbations keep them valid.
    factors = [torch.linalg.cholesky(t) for t in (Q, R, P0)]

    def run(y, A, C, Q_factor, R_factor, m0, P0_factor):
        Q, R, P0 = (L @ L.mH for L in (Q_factor, R_factor, P0_factor))
        return tuple(fn(y, A, C, Q, R, m0, P0))

    inputs = [t.clone().requires_grad_() for t in (y, A, C, factors[0], factors[1], m0, factors[2])]
    assert torch.autograd.gradcheck(run, inputs)


@pytest.mark.parametrize("fn", [kalman_filter, kalman_smoother])
def test_kalman_known_inputs_recipe(fn):
    """The recipe in kalman_filter's docstring, from shapes it says to expand."""
    B, N, M, P = 3, 9, 3, 2
    T = N - 1
    y, A, C, Q, R, m0, P0 = _model(B, N, M, P)
    gen = torch.Generator().manual_seed(1)
    u = torch.randn(B, T, M, dtype=torch.float64, generator=gen)
    d = torch.randn(B, N, P, dtype=torch.float64, generator=gen)
    # A constant A, a constant C per signal, and a shared prior mean.
    A, C, m0 = A[0, 0], C[:, 0], m0[0]
    A_full, C_full = A.expand(B, T, M, M), C[:, None].expand(B, N, P, M)

    x_u = lpv_state_space_recursion(A_full, y.new_zeros(B, M), u)
    x_u = torch.cat([y.new_zeros(B, 1, M), x_u], dim=1)  # x_u[0] = 0
    y_s = y - d - (C_full @ x_u.unsqueeze(-1)).squeeze(-1)
    result = fn(y_s, A_full, C_full, Q, R, m0, P0)
    means = result.means + x_u

    expected = _reference(y, A_full, C_full, Q, R, m0.expand(B, M), P0, "first_state", u, d)
    expected = expected[:2] if fn is kalman_filter else expected[2:]
    torch.testing.assert_close(means, expected[0], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(result.covs, expected[1], rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kalman_float32_long_sequence():
    args = _model(2, 4096, 4, 2)
    expected = _reference(*args, "first_state")
    filtered = kalman_filter(*(t.float().cuda() for t in args))
    smoothed = kalman_smoother(*(t.float().cuda() for t in args))
    actual = (filtered.means, filtered.covs, smoothed.means, smoothed.covs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.double().cpu(), e, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "torch_version, supported",
    [
        ("2.4.1", False),
        # 2.5 needs every leaf to have one shape; 2.6 and 2.7 always compile
        # the scan, which fails on these elements; and vmapped solve fails on
        # the scan's empty slices before 2.11.
        ("2.5.1+cpu", False),
        ("2.7.1", False),
        ("2.10.0", False),
        ("2.11.0", True),
        ("2.13.0+cu130", True),
        ("3.0.0", True),
    ],
)
def test_kalman_generic_scan_support_by_version(torch_version, supported):
    assert kalman._supports_generic_scan(torch_version) is supported


def test_kalman_rejects_pytorch_before_2_11(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    with pytest.raises(RuntimeError, match="PyTorch 2.11"):
        kalman_filter(*_model(1, 3, 1, 1))


def _dense_posterior(y, A, C, Q, R, m0, P0, prior):
    """Condition the joint Gaussian of all the states and y directly.

    Returns the posterior mean (B, S, M) and covariance (B, S, M, S, M) of
    the S states, and log p(y), (B,): a reference that shares nothing with
    the Kalman recursions.
    """
    B, N, P = y.shape
    M = m0.size(-1)
    offset = 1 if prior == "before_first_step" else 0  # y[n] measures x[n + offset]
    S = N + offset
    D = M * S
    means, log_ps, covs = [], [], []
    for b in range(B):
        # x = T z with z = (x[0], w[0], ..., w[S - 2]); x[s + 1] = A[s] x[s] + w[s].
        T = y.new_zeros(D, D)
        T[:M, :M] = torch.eye(M, dtype=y.dtype)
        for s in range(S - 1):
            rows, prev = slice(M * (s + 1), M * (s + 2)), slice(M * s, M * (s + 1))
            T[rows] = A[b, s] @ T[prev]
            T[rows, M * (s + 1) : M * (s + 2)] += torch.eye(M, dtype=y.dtype)
        cov_z = torch.block_diag(P0[b], *Q[b, : S - 1])
        mean_x = T[:, :M] @ m0[b]
        cov_x = T @ cov_z @ T.mH
        H = y.new_zeros(N * P, D)
        for n in range(N):
            s = n + offset
            H[P * n : P * (n + 1), M * s : M * (s + 1)] = C[b, n]
        mean_y = H @ mean_x
        cov_y = H @ cov_x @ H.mH + torch.block_diag(*R[b])
        cov_xy = cov_x @ H.mH
        residual = y[b].reshape(-1) - mean_y
        gain = torch.linalg.solve(cov_y, cov_xy.mH).mH
        means.append((mean_x + gain @ residual).reshape(S, M))
        covs.append((cov_x - gain @ cov_xy.mH).reshape(S, M, S, M))
        quadratic = (residual.conj() @ torch.linalg.solve(cov_y, residual)).real
        log_det = torch.linalg.slogdet(cov_y)[1]
        if y.is_complex():
            log_ps.append(-(N * P * math.log(math.pi) + log_det + quadratic))
        else:
            log_ps.append(-0.5 * (N * P * math.log(2 * math.pi) + log_det + quadratic))
    return torch.stack(means), torch.stack(covs), torch.stack(log_ps)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("N", [1, 2, 6])
@pytest.mark.parametrize("M, P", [(1, 1), (3, 2), (2, 3)])
def test_kalman_smoother_matches_dense_posterior(device, dtype, N, M, P):
    args = [t.to(device) for t in _model(2, N, M, P, dtype=dtype)]
    means, covs, log_p = _dense_posterior(*[t.cpu() for t in args], "first_state")
    result = kalman_smoother(*args)
    steps = torch.arange(means.size(1))
    torch.testing.assert_close(result.means.cpu(), means, rtol=1e-8, atol=1e-8)
    expected_covs = covs[:, steps, :, steps].transpose(0, 1)
    torch.testing.assert_close(result.covs.cpu(), expected_covs, rtol=1e-8, atol=1e-8)
    # covs[:, s + 1, :, s] is Cov(x[s + 1], x[s] | y).
    cross = covs[:, steps[1:], :, steps[:-1]].transpose(0, 1)
    torch.testing.assert_close(result.cross_covs.cpu(), cross, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(result.log_likelihood.cpu(), log_p, rtol=1e-9, atol=1e-9)
    filtered = kalman_filter(*args)
    torch.testing.assert_close(filtered.log_likelihood, result.log_likelihood)


def _em_step(y, result):
    """The M-step of kalman_smoother's docstring, as written there."""
    m, V, V10, _ = result
    outer = V + m.unsqueeze(-1) @ m.unsqueeze(-2).conj()  # E[x[n] x[n]^H]
    cross = V10 + m[:, 1:].unsqueeze(-1) @ m[:, :-1].unsqueeze(-2).conj()
    S00 = outer[:, :-1].sum((0, 1))  # sums of E[x[n] x[n]^H], n < N - 1
    S11 = outer[:, 1:].sum((0, 1))  # sums of E[x[n + 1] x[n + 1]^H]
    S10 = cross.sum((0, 1))  # sums of E[x[n + 1] x[n]^H]
    A = torch.linalg.solve(S00, S10.mH).mH  # S10 S00^-1
    Q = (S11 - A @ S10.mH) / (y.size(0) * (y.size(1) - 1))
    Syx = (y.unsqueeze(-1) @ m.unsqueeze(-2).conj()).sum((0, 1))
    C = torch.linalg.solve(outer.sum((0, 1)), Syx.mH).mH
    yy = (y.unsqueeze(-1) @ y.unsqueeze(-2).conj()).sum((0, 1))
    R = (yy - C @ Syx.mH) / (y.size(0) * y.size(1))
    m0 = m[:, 0].mean(0)
    P0 = outer[:, 0].mean(0) - m0.outer(m0.conj())
    # Exactly Hermitian, despite rounding, for the next step's factorizations.
    Q, R, P0 = (Q + Q.mH) / 2, (R + R.mH) / 2, (P0 + P0.mH) / 2
    return A, C, Q, R, m0, P0


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_kalman_em_increases_likelihood(dtype):
    # Data from a known model, fitted from a poor initial guess.
    B, N, M, P = 4, 50, 2, 2
    gen = torch.Generator().manual_seed(3)
    true_A = torch.tensor([[0.9, 0.2], [-0.2, 0.9]], dtype=dtype)
    x = torch.zeros(B, M, dtype=dtype)
    ys = []
    for _ in range(N):
        x = x @ true_A.mT + 0.3 * torch.randn(B, M, generator=gen, dtype=dtype)
        ys.append(x + 0.5 * torch.randn(B, P, generator=gen, dtype=dtype))
    y = torch.stack(ys, 1)
    eye = torch.eye(M, dtype=dtype)
    params = (0.5 * eye, eye.clone(), eye.clone(), eye.clone(), torch.zeros(M, dtype=dtype), eye)
    log_ps = []
    for _ in range(20):
        result = kalman_smoother(y, *params)
        log_ps.append(result.log_likelihood.sum().item())
        params = _em_step(y, result)
    assert all(b >= a - 1e-8 for a, b in zip(log_ps, log_ps[1:]))
    assert log_ps[-1] > log_ps[0] + 10


def test_kalman_known_initial_state_with_singular_noise():
    """An AR(2) model from a known zero state, one step before its first measurement.

    Converted as in the docstring, its prior on the first measured state has
    the singular covariance Q.
    """
    dtype = torch.float64
    A = torch.tensor([[1.2, -0.5], [1.0, 0.0]], dtype=dtype)
    C = torch.tensor([[1.0, 0.0]], dtype=dtype)
    Q = torch.diag(torch.tensor([1.0, 0.0], dtype=dtype))
    R = torch.tensor([[0.1]], dtype=dtype)
    y = torch.randn(2, 6, 1, dtype=dtype, generator=torch.Generator().manual_seed(0))
    m0, P0 = torch.zeros(2, 2, dtype=dtype), torch.zeros(2, 2, 2, dtype=dtype)
    args = (y, *(t.expand(2, 6, *t.shape) for t in (A, C, Q, R)), m0, P0)
    means, covs, log_p = _dense_posterior(*args, "before_first_step")
    result = kalman_smoother(*_from_before_first_step(*args))
    torch.testing.assert_close(result.means, means[:, 1:], rtol=1e-8, atol=1e-8)
    steps = torch.arange(1, 7)
    expected_covs = covs[:, steps, :, steps].transpose(0, 1)
    torch.testing.assert_close(result.covs, expected_covs, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(result.log_likelihood, log_p, rtol=1e-9, atol=1e-9)
