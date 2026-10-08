import pytest
import torch

import philtorch.lpv.kalman as kalman
from philtorch.lpv import kalman_filter, kalman_smoother

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
        return X @ X.mT + 0.1 * torch.eye(shape[-1], dtype=dtype)

    A = 0.99 * torch.linalg.qr(randn(batch_size, N, M, M))[0]
    C = randn(batch_size, N, P, M)
    Q = covariance(batch_size, N, M, M)
    R = covariance(batch_size, N, P, P)
    m0 = randn(batch_size, M)
    P0 = covariance(batch_size, M, M)
    y = randn(batch_size, N, P)
    return y, A, C, Q, R, m0, P0


def _reference(y, A, C, Q, R, m0, P0):
    """The sequential Kalman filter and Rauch-Tung-Striebel smoother."""
    N = y.size(1)
    m, P = m0, P0
    means, covs = [], []
    for n in range(N):
        m = (A[:, n] @ m[..., None]).squeeze(-1)
        P = A[:, n] @ P @ A[:, n].mT + Q[:, n]
        S = C[:, n] @ P @ C[:, n].mT + R[:, n]
        K = torch.linalg.solve(S, C[:, n] @ P).mT
        m = m + (K @ (y[:, n] - (C[:, n] @ m[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        P = P - K @ S @ K.mT
        means.append(m)
        covs.append(P)

    smoothed_means, smoothed_covs = [means[-1]], [covs[-1]]
    for n in range(N - 2, -1, -1):
        m_pred = (A[:, n + 1] @ means[n][..., None]).squeeze(-1)
        P_pred = A[:, n + 1] @ covs[n] @ A[:, n + 1].mT + Q[:, n + 1]
        G = torch.linalg.solve(P_pred, A[:, n + 1] @ covs[n]).mT
        smoothed_means.append(means[n] + (G @ (smoothed_means[-1] - m_pred)[..., None]).squeeze(-1))
        smoothed_covs.append(covs[n] + G @ (smoothed_covs[-1] - P_pred) @ G.mT)
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


def test_kalman_requires_generic_associative_scan(monkeypatch):
    monkeypatch.setattr(kalman, "_HAS_GENERIC_SCAN", False)
    with pytest.raises(RuntimeError, match="PyTorch 2.5"):
        kalman_filter(*_model(1, 3, 1, 1))
