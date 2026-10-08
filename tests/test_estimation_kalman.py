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


def _model(batch_size, N, M, P, *, seed=0, dtype=torch.float64):
    """A random time-varying model, as in Sec. VI-A of arXiv:2511.10363."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, generator=gen)

    def covariance(*shape):
        X = randn(*shape)
        return X @ X.mH + 0.1 * torch.eye(shape[-1], dtype=dtype)

    A = 0.99 * torch.linalg.qr(randn(batch_size, N, M, M))[0]
    C = randn(batch_size, N, P, M)
    Q = covariance(batch_size, N, M, M)
    R = covariance(batch_size, N, P, P)
    m0 = randn(batch_size, M)
    P0 = covariance(batch_size, M, M)
    y = randn(batch_size, N, P)
    return y, A, C, Q, R, m0, P0


def _reference(y, A, C, Q, R, m0, P0, u=None, d=None):
    """The sequential Kalman filter and Rauch-Tung-Striebel smoother.

    The optional u and d are known offsets of the state and the measurements.
    """
    N = y.size(1)
    u = torch.zeros_like(A[..., 0]) if u is None else u
    d = torch.zeros_like(y) if d is None else d
    m, P = m0, P0
    means, covs = [], []
    for n in range(N):
        m = (A[:, n] @ m[..., None]).squeeze(-1) + u[:, n]
        P = A[:, n] @ P @ A[:, n].mH + Q[:, n]
        S = C[:, n] @ P @ C[:, n].mH + R[:, n]
        K = torch.linalg.solve(S, C[:, n] @ P).mH
        innovation = y[:, n] - d[:, n] - (C[:, n] @ m[..., None]).squeeze(-1)
        m = m + (K @ innovation[..., None]).squeeze(-1)
        P = P - K @ S @ K.mH
        means.append(m)
        covs.append(P)

    smoothed_means, smoothed_covs = [means[-1]], [covs[-1]]
    for n in range(N - 2, -1, -1):
        m_pred = (A[:, n + 1] @ means[n][..., None]).squeeze(-1) + u[:, n + 1]
        P_pred = A[:, n + 1] @ covs[n] @ A[:, n + 1].mH + Q[:, n + 1]
        G = torch.linalg.solve(P_pred, A[:, n + 1] @ covs[n]).mH
        smoothed_means.append(means[n] + (G @ (smoothed_means[-1] - m_pred)[..., None]).squeeze(-1))
        smoothed_covs.append(covs[n] + G @ (smoothed_covs[-1] - P_pred) @ G.mH)
    return (
        torch.stack(means, 1),
        torch.stack(covs, 1),
        torch.stack(smoothed_means[::-1], 1),
        torch.stack(smoothed_covs[::-1], 1),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("N", [1, 2, 7, 64])
@pytest.mark.parametrize("M, P", [(1, 1), (4, 2), (2, 3)])
def test_kalman_matches_sequential(device, N, M, P):
    args = [t.to(device) for t in _model(3, N, M, P)]
    expected = _reference(*args)
    actual = (*kalman_filter(*args), *kalman_smoother(*args))
    for name, a, e in zip(
        ["filter mean", "filter cov", "smoother mean", "smoother cov"], actual, expected
    ):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-9, msg=name)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("M, P", [(1, 1), (4, 2), (2, 3)])
def test_kalman_complex_matches_sequential(device, M, P):
    args = [t.to(device) for t in _model(3, 7, M, P, dtype=torch.complex128)]
    expected = _reference(*args)
    actual = (*kalman_filter(*args), *kalman_smoother(*args))
    for name, a, e in zip(
        ["filter mean", "filter cov", "smoother mean", "smoother cov"], actual, expected
    ):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-9, msg=name)


@pytest.mark.parametrize("fn", [kalman_filter, kalman_smoother])
@pytest.mark.parametrize("batch_size, N", [(0, 5), (2, 0)])
def test_kalman_empty_inputs(fn, batch_size, N):
    y, A, C, Q, R, m0, P0 = _model(2, 5, 3, 2)
    y = y.new_zeros(batch_size, N, 2)
    means, covs = fn(y, A[0, 0], C[0, 0], Q[0, 0], R[0, 0], m0[0], P0[0])
    assert means.shape == (batch_size, N, 3)
    assert covs.shape == (batch_size, N, 3, 3)


@pytest.mark.parametrize("fn", [kalman_filter, kalman_smoother])
def test_kalman_coefficient_shapes(fn):
    batch_size, N, M, P = 2, 5, 3, 2
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
    def over_time(t):
        return t.unsqueeze(-3).expand(*t.shape[:-2], N, -1, -1)

    per_signal = fn(y, A[:, 0], C[:, 0], Q[:, 0], R[:, 0], m0, P0)
    expanded = fn(y, *(over_time(t[:, 0]) for t in (A, C, Q, R)), m0, P0)
    for a, e in zip(per_signal, expanded):
        torch.testing.assert_close(a, e)
    constant = fn(y, A[0, 0], C[0, 0], Q[0, 0], R[0, 0], m0, P0)
    expanded = fn(y, *(over_time(t[0, 0]) for t in (A, C, Q, R)), m0, P0)
    for a, e in zip(constant, expanded):
        torch.testing.assert_close(a, e)

    for a, e in zip(fn(y, A, C, Q, R, m0, P0), expected):
        torch.testing.assert_close(a, e)


def test_kalman_reads_n_equal_b_as_time_varying():
    y, A, C, Q, R, m0, P0 = _model(4, 4, 2, 1)
    means, _ = kalman_filter(y, A[0], C, Q, R, m0, P0)
    expected, _ = kalman_filter(y, A[:1].expand(4, -1, -1, -1), C, Q, R, m0, P0)
    torch.testing.assert_close(means, expected)


@pytest.mark.parametrize(
    "name, shape",
    [
        ("A", (3, 2, 2)),
        ("A", (4, 2, 2)),
        ("Q", (2, 4, 2, 2)),
        ("C", (2, 2)),
        ("Q", (3, 5, 2, 2)),
        ("R", (1, 2)),
        ("m0", (3,)),
        ("P0", (3, 2, 2)),
    ],
)
def test_kalman_rejects_unsupported_shapes(name, shape):
    y, A, C, Q, R, m0, P0 = _model(2, 5, 2, 1)
    args = dict(y=y, A=A, C=C, Q=Q, R=R, m0=m0, P0=P0)
    args[name] = torch.ones(shape, dtype=torch.float64)
    with pytest.raises(ValueError, match=name):
        kalman_filter(**args)


@pytest.mark.parametrize("fn", [kalman_filter, kalman_smoother])
def test_kalman_gradcheck(fn):
    y, A, C, Q, R, m0, P0 = _model(2, 5, 2, 1)
    Q_factor = torch.linalg.cholesky(Q)

    def run(y, A, C, Q_factor, m0):
        return fn(y, A, C, Q_factor @ Q_factor.mT, R, m0, P0)

    inputs = [t.clone().requires_grad_() for t in (y, A, C, Q_factor, m0)]
    assert torch.autograd.gradcheck(run, inputs)


@pytest.mark.parametrize(
    "fn, outputs", [(kalman_filter, slice(0, 2)), (kalman_smoother, slice(2, 4))]
)
def test_kalman_known_inputs_recipe(fn, outputs):
    """The recipe in kalman_filter's docstring, from shapes it says to expand."""
    B, N, M, P = 3, 9, 3, 2
    y, A, C, Q, R, m0, P0 = _model(B, N, M, P)
    gen = torch.Generator().manual_seed(1)
    u = torch.randn(B, N, M, dtype=torch.float64, generator=gen)
    d = torch.randn(B, N, P, dtype=torch.float64, generator=gen)
    # A constant A, a constant C per signal, and a shared prior mean.
    A, C, m0 = A[0, 0], C[:, 0], m0[0]
    A_full, C_full = A.expand(B, N, M, M), C[:, None].expand(B, N, P, M)

    x_u = lpv_state_space_recursion(A_full, y.new_zeros(B, M), u)
    y_s = y - d - (C_full @ x_u.unsqueeze(-1)).squeeze(-1)
    means, covs = fn(y_s, A_full, C_full, Q, R, m0, P0)
    means = means + x_u

    expected = _reference(y, A_full, C_full, Q, R, m0.expand(B, M), P0, u, d)[outputs]
    torch.testing.assert_close(means, expected[0], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(covs, expected[1], rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kalman_float32_long_sequence():
    args = _model(2, 4096, 4, 2)
    expected = _reference(*args)
    actual = (
        *kalman_filter(*(t.float().cuda() for t in args)),
        *kalman_smoother(*(t.float().cuda() for t in args)),
    )
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
