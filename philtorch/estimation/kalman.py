"""Kalman filters and smoothers parallelized over time with associative scans.

These follow Särkkä and García-Fernández, "Temporal parallelization of
Bayesian smoothers" (IEEE TAC, 2021), as evaluated on GPUs in Särkkä and
García-Fernández, "On the performance of prefix-sum parallel Kalman filters
and smoothers on GPUs" (arXiv:2511.10363).
"""

import math
import re
from typing import NamedTuple

import torch
from torch import Tensor

try:
    from torch._higher_order_ops.associative_scan import associative_scan as _associative_scan
except ImportError:  # pragma: no cover - PyTorch without associative_scan
    _associative_scan = None

# The generic mode of associative_scan takes any combine function. It runs
# eagerly from PyTorch 2.8: the combine function is vmapped over the scan
# dimension and applied O(log N) times, so autograd follows it as usual.
# Earlier releases fail on these elements: 2.5 requires every leaf to have the
# same shape, and these mix matrices and vectors, while 2.6 and 2.7 always run
# the scan under torch.compile, which fails on the batched matrix products.
# The scan also calls the combine function on empty slices, and vmapped
# torch.linalg.solve fails on those before PyTorch 2.11.
_MIN_TORCH_VERSION = (2, 11)


def _supports_generic_scan(torch_version: str) -> bool:
    major, minor = re.match(r"(\d+)\.(\d+)", torch_version).groups()
    return _associative_scan is not None and (int(major), int(minor)) >= _MIN_TORCH_VERSION


def _scan(combine_fn, xs: tuple[Tensor, ...], reverse: bool = False) -> tuple[Tensor, ...]:
    if not _supports_generic_scan(torch.__version__):
        raise RuntimeError(
            "Kalman filtering needs the generic mode of PyTorch's associative_scan "
            f"from PyTorch 2.11; this is PyTorch {torch.__version__}."
        )
    # With no signals, steps or states there is nothing to combine, and the
    # scan itself rejects an empty scan dimension and fails on an empty batch.
    if any(x.numel() == 0 for x in xs):
        return xs
    return tuple(_associative_scan(combine_fn, xs, dim=1, reverse=reverse, combine_mode="generic"))


def _coefficient(name: str, t: Tensor, base: tuple[int, ...], batch_size: int, N: int) -> Tensor:
    """View a coefficient so it broadcasts against (batch_size, N, *base)."""
    if t.dim() >= len(base) and tuple(t.shape[t.dim() - len(base) :]) == base:
        match tuple(t.shape[: t.dim() - len(base)]):
            case ():
                return t
            case (n,) if n == N:
                return t
            case (b,) if b == batch_size:
                return t.unsqueeze(1)
            case (b, n) if b == batch_size and n == N:
                return t
    raise ValueError(
        f"{name} must be of shape {base}, {(N, *base)}, {(batch_size, *base)}, "
        f"or {(batch_size, N, *base)}, got {tuple(t.shape)}"
    )


def _parse(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, ...]:
    assert y.dim() == 3, f"Measurements y must be 3D (batch, time, features), got {y.shape}"
    batch_size, N, P = y.shape
    assert A.dim() >= 2 and A.size(-1) == A.size(-2), f"A must be square, got {A.shape}"
    M = A.size(-1)

    A = _coefficient("A", A, (M, M), batch_size, N)
    C = _coefficient("C", C, (P, M), batch_size, N)
    Q = _coefficient("Q", Q, (M, M), batch_size, N)
    R = _coefficient("R", R, (P, P), batch_size, N)
    match tuple(m0.shape):
        case (m,) if m == M:
            pass
        case (b, m) if b == batch_size and m == M:
            m0 = m0.unsqueeze(1)
        case _:
            raise ValueError(f"m0 must be of shape {(M,)} or {(batch_size, M)}, got {m0.shape}")
    match tuple(P0.shape):
        case (m, k) if m == k == M:
            pass
        case (b, m, k) if b == batch_size and m == k == M:
            P0 = P0.unsqueeze(1)
        case _:
            raise ValueError(
                f"P0 must be of shape {(M, M)} or {(batch_size, M, M)}, got {P0.shape}"
            )

    def full(t: Tensor, *base: int) -> Tensor:
        return t.expand(batch_size, N, *base)

    return full(A, M, M), full(C, P, M), full(Q, M, M), full(R, P, P), m0, P0


def _mv(A: Tensor, x: Tensor) -> Tensor:
    return (A @ x.unsqueeze(-1)).squeeze(-1)


def _filtering_elements(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, ...]:
    """Build the elements (F, b, G, eta, J) of the parallel Kalman filter.

    These are the paper's (A, b, C, eta, J), renamed so they don't clash with
    the model's A and C.

    Element n + 1 describes step n (eqs. 42-45 of the GPU paper, with their
    k = n + 1): x[n + 1] | x[n], y[n] ~ N(F x[n] + b, G), and the likelihood
    of y[n] given x[n] in information form (eta, J). Element 0 is the prior,
    which doesn't depend on any earlier state, so its F, eta and J are zero.
    """
    batch_size, M = y.size(0), A.size(-1)
    CA = C @ A
    S = C @ Q @ C.mH + R
    K = torch.linalg.solve(S, C @ Q).mH
    F = A - K @ CA
    b = _mv(K, y)
    G = Q - K @ S @ K.mH
    eta = _mv(CA.mH, torch.linalg.solve(S, y))
    J = CA.mH @ torch.linalg.solve(S, CA)

    m0 = m0.expand(batch_size, 1, M)
    P0 = P0.expand(batch_size, 1, M, M)
    zero = torch.zeros_like(P0)
    return (
        torch.cat([zero, F], dim=1),
        torch.cat([m0, b], dim=1),
        torch.cat([P0, G], dim=1),
        torch.cat([torch.zeros_like(m0), eta], dim=1),
        torch.cat([zero, J], dim=1),
    )


def _filtering_operator(
    earlier: tuple[Tensor, ...], later: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    """Combine two filtering elements (Lemma 3 of the GPU paper)."""
    F_i, b_i, G_i, eta_i, J_i = earlier
    F_j, b_j, G_j, eta_j, J_j = later
    eye = torch.eye(F_i.size(-1), dtype=F_i.dtype, device=F_i.device)
    # X = F_j (I + G_i J_j)^-1 and Y = F_i^H (I + J_j G_i)^-1, as left solves.
    X = torch.linalg.solve((eye + G_i @ J_j).mH, F_j.mH).mH
    Y = torch.linalg.solve((eye + J_j @ G_i).mH, F_i).mH
    return (
        X @ F_i,
        _mv(X, b_i + _mv(G_i, eta_j)) + b_j,
        X @ G_i @ F_j.mH + G_j,
        _mv(Y, eta_j - _mv(J_j, b_i)) + eta_i,
        Y @ J_j @ F_i + J_i,
    )


def _filter(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, Tensor]:
    """The filtered moments of x[0], ..., x[N]: the prior, then each step's result."""
    elements = _filtering_elements(y, A, C, Q, R, m0, P0)
    _, means, covs, _, _ = _scan(_filtering_operator, elements)
    return means, covs


def _predict(A: Tensor, Q: Tensor, means: Tensor, covs: Tensor) -> tuple[Tensor, Tensor]:
    """The moments of x[n + 1] given y[0], ..., y[n - 1], from those of x[n] given them."""
    return _mv(A, means), A @ covs @ A.mH + Q


def _log_likelihood(
    y: Tensor, C: Tensor, R: Tensor, predicted_means: Tensor, predicted_covs: Tensor
) -> Tensor:
    """log p(y[0], ..., y[N - 1]) = sum over n of log p(y[n] | y[0], ..., y[n - 1]).

    Each term is a Gaussian density of y[n], with the mean and covariance of
    C[n] x[n + 1] + v[n] given the earlier measurements: real, or circularly
    symmetric complex for complex tensors.
    """
    residual = y - _mv(C, predicted_means)
    S = C @ predicted_covs @ C.mH + R
    factor = torch.linalg.cholesky(S)
    whitened = torch.linalg.solve_triangular(factor, residual.unsqueeze(-1), upper=False)
    quadratic = whitened.squeeze(-1).abs().square().sum(-1)
    log_det = 2 * factor.diagonal(dim1=-2, dim2=-1).real.log().sum(-1)
    P = y.size(-1)
    if y.is_complex():
        log_densities = -(P * math.log(math.pi) + log_det + quadratic)
    else:
        log_densities = -0.5 * (P * math.log(2 * math.pi) + log_det + quadratic)
    return log_densities.sum(1)


def _smoothing_operator(
    later: tuple[Tensor, ...], earlier: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    """Combine two smoothing elements (Lemma 4 of the GPU paper).

    The scan runs in reverse, so the element later in time comes first.
    """
    E_j, g_j, L_j = later
    E_i, g_i, L_i = earlier
    return E_i @ E_j, _mv(E_i, g_j) + g_i, E_i @ L_j @ E_i.mH + L_i


def _smooth(A: Tensor, Q: Tensor, means: Tensor, covs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Smooth the filtered moments of x[0], ..., x[N] from :func:`_filter`.

    Returns the smoothed moments of x[0], ..., x[N] and the gains E[n] of
    the smoothing elements, which make Cov(x[n], x[n + 1] | y) =
    E[n] Cov(x[n + 1] | y).
    """
    # Element n describes x[n] given x[n + 1] and y[0], ..., y[n - 1]
    # (eqs. 48-50 of the GPU paper): x[n] | x[n + 1] ~ N(E x[n + 1] + g, L),
    # through A[n] and Q[n]. The last one is the filtering result itself.
    m_n, P_n = means[:, :-1], covs[:, :-1]
    _, P_pred = _predict(A, Q, m_n, P_n)
    E = torch.linalg.solve(P_pred, A @ P_n).mH
    g = m_n - _mv(E @ A, m_n)
    L = P_n - E @ P_pred @ E.mH
    elements = (
        torch.cat([E, torch.zeros_like(covs[:, -1:])], dim=1),
        torch.cat([g, means[:, -1:]], dim=1),
        torch.cat([L, covs[:, -1:]], dim=1),
    )
    _, means, covs = _scan(_smoothing_operator, elements, reverse=True)
    return means, covs, E


class KalmanStatistics(NamedTuple):
    r"""The expectations of the E-step of expectation-maximization (EM).

    Returned by :func:`kalman_em_statistics`, for the states
    :math:`\mathbf{x}[0], \dots, \mathbf{x}[N]`, prior state included.
    """

    #: Smoothed means :math:`E[\mathbf{x}[n] \mid \mathbf{y}]`, of shape :math:`(B, N + 1, M)`.
    means: Tensor
    #: Smoothed covariances :math:`\mathrm{Cov}(\mathbf{x}[n] \mid \mathbf{y})`, of shape
    #: :math:`(B, N + 1, M, M)`.
    covs: Tensor
    #: Lag-one cross-covariances
    #: :math:`\mathrm{Cov}(\mathbf{x}[n + 1], \mathbf{x}[n] \mid \mathbf{y})`, for
    #: :math:`n = 0, \dots, N - 1`, of shape :math:`(B, N, M, M)`.
    cross_covs: Tensor
    #: Each signal's log marginal likelihood :math:`\log p(\mathbf{y})`, of shape :math:`(B)`.
    log_likelihood: Tensor


def kalman_filter(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Filter a linear Gaussian state-space model in parallel over time.

    For the model

    .. math::
        \mathbf{x}[n + 1] &= A[n] \mathbf{x}[n] + \mathbf{w}[n],
        & \mathbf{w}[n] &\sim \mathcal{N}(\mathbf{0}, Q[n]), \\
        \mathbf{y}[n] &= C[n] \mathbf{x}[n + 1] + \mathbf{v}[n],
        & \mathbf{v}[n] &\sim \mathcal{N}(\mathbf{0}, R[n]),

    with prior :math:`\mathbf{x}[0] \sim \mathcal{N}(\mathbf{m}_0, P_0)`,
    this returns the mean and covariance of each state given the measurements
    up to it, :math:`p(\mathbf{x}[n + 1] \mid \mathbf{y}[0], \dots,
    \mathbf{y}[n])`, for the states :math:`\mathbf{x}[1], \dots,
    \mathbf{x}[N]`. These are the results of the sequential Kalman filter,
    computed instead with an associative scan of depth :math:`O(\log N)`, as
    in Särkkä and García-Fernández (2021). The scan is PyTorch's
    ``associative_scan`` in its generic mode, which needs PyTorch 2.11 or later.

    Each of :attr:`A`, :attr:`C`, :attr:`Q` and :attr:`R` may be constant or
    time-varying, and shared or one per signal: its base shape below can be
    prefixed with :math:`N` for time-varying values, :math:`B` for one per
    signal, or :math:`(B, N)` for both. When two readings fit, such as
    :math:`N = B`, the time-varying one is taken. Even with constant
    matrices, the Kalman gain varies over time. Complex tensors describe
    circularly symmetric complex Gaussian noise, with conjugate transposes in
    place of transposes.

    Note:
        As in :func:`~philtorch.lpv.state_space_recursion`, the prior is the
        state before the first step, and step :math:`n` returns
        :math:`\mathbf{x}[n + 1]`. So :math:`\mathbf{y}[n]` measures the state
        after :math:`A[n]`, unlike the output of
        :func:`~philtorch.lpv.state_space`, which reads :math:`\mathbf{x}[n]`.

    Note:
        Known inputs need no extra arguments. For
        :math:`\mathbf{x}[n + 1] = A[n] \mathbf{x}[n] + \mathbf{u}[n] +
        \mathbf{w}[n]` and :math:`\mathbf{y}[n] = C[n] \mathbf{x}[n + 1] +
        \mathbf{d}[n] + \mathbf{v}[n]`, filter the part of the state that
        :math:`\mathbf{u}` doesn't drive, then add the part it does, which
        :func:`~philtorch.lpv.state_space_recursion` computes from zero. With
        :attr:`u` of shape :math:`(B, N, M)`, :attr:`d` of shape
        :math:`(B, N, P)`, and :attr:`A` and :attr:`C` expanded to their full
        shapes :math:`(B, N, M, M)` and :math:`(B, N, P, M)`, such as
        ``A.expand(B, N, M, M)`` for a constant :attr:`A`::

            from philtorch.lpv import state_space_recursion

            x_u = state_space_recursion(A, y.new_zeros(B, M), u)
            y_s = y - d - (C @ x_u.unsqueeze(-1)).squeeze(-1)
            means, covs = kalman_filter(y_s, A, C, Q, R, m0, P0)
            means = means + x_u

        This is exact, as the model is linear, and the covariances need no
        correction. The same works for :func:`kalman_smoother`.

    Args:
        y (Tensor): measurements :math:`\mathbf{y}[n]`, of shape
            :math:`(B, N, P)`.
        A (Tensor): state transition matrices :math:`A[n]`, taking
            :math:`\mathbf{x}[n]` to :math:`\mathbf{x}[n + 1]`, of base shape
            :math:`(M, M)`.
        C (Tensor): measurement matrices, of base shape :math:`(P, M)`.
        Q (Tensor): process noise covariances, of base shape :math:`(M, M)`.
        R (Tensor): measurement noise covariances, of base shape
            :math:`(P, P)`.
        m0 (Tensor): the prior mean of :math:`\mathbf{x}[0]`, of shape
            :math:`(M)` or :math:`(B, M)`.
        P0 (Tensor): the prior covariance of :math:`\mathbf{x}[0]`, of shape
            :math:`(M, M)` or :math:`(B, M, M)`.

    Returns:
        tuple of Tensor: the filtered means, of shape :math:`(B, N, M)`, and
        covariances, of shape :math:`(B, N, M, M)`.

    Raises:
        ValueError: if a coefficient has an unsupported shape.
        AssertionError: if :attr:`y` is not 3-D or :attr:`A` is not square.
        RuntimeError: on PyTorch older than 2.11.

    Example::

        >>> from philtorch.estimation import kalman_filter
        >>> # A random walk with unit steps, measured with noise variance 2.
        >>> # Its prior covariance is the steady state, so the gain stays 0.5.
        >>> y = torch.ones(1, 3, 1, dtype=torch.float64)
        >>> one = torch.ones(1, 1, dtype=torch.float64)
        >>> m0 = torch.zeros(1, dtype=torch.float64)
        >>> means, covs = kalman_filter(y, one, one, one, 2 * one, m0, one)
        >>> means.squeeze()
        tensor([0.5000, 0.7500, 0.8750], dtype=torch.float64)
    """
    means, covs = _filter(y, *_parse(y, A, C, Q, R, m0, P0))
    # Drop the prior: the scan's first output is x[0] before any measurement.
    return means[:, 1:], covs[:, 1:]


def kalman_smoother(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Smooth a linear Gaussian state-space model in parallel over time.

    For the model of :func:`kalman_filter`, this returns the mean and
    covariance of each state given all the measurements,
    :math:`p(\mathbf{x}[n + 1] \mid \mathbf{y}[0], \dots,
    \mathbf{y}[N - 1])`, for the states :math:`\mathbf{x}[1], \dots,
    \mathbf{x}[N]`.
    These are the results of the Rauch--Tung--Striebel smoother, computed
    with a reverse associative scan after :func:`kalman_filter`, as in
    Särkkä and García-Fernández (2021). The arguments are those of
    :func:`kalman_filter`, and the same PyTorch 2.11 requirement applies.

    Args:
        y (Tensor): measurements, of shape :math:`(B, N, P)`.
        A (Tensor): state transition matrices :math:`A[n]`, taking
            :math:`\mathbf{x}[n]` to :math:`\mathbf{x}[n + 1]`, of base shape
            :math:`(M, M)`.
        C (Tensor): measurement matrices, of base shape :math:`(P, M)`.
        Q (Tensor): process noise covariances, of base shape :math:`(M, M)`.
        R (Tensor): measurement noise covariances, of base shape
            :math:`(P, P)`.
        m0 (Tensor): the prior mean, of shape :math:`(M)` or :math:`(B, M)`.
        P0 (Tensor): the prior covariance, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.

    Returns:
        tuple of Tensor: the smoothed means, of shape :math:`(B, N, M)`, and
        covariances, of shape :math:`(B, N, M, M)`.

    Raises:
        ValueError: if a coefficient has an unsupported shape.
        AssertionError: if :attr:`y` is not 3-D or :attr:`A` is not square.
        RuntimeError: on PyTorch older than 2.11.

    Example::

        >>> from philtorch.estimation import kalman_smoother
        >>> # The random walk of :func:`kalman_filter`'s example.
        >>> y = torch.ones(1, 3, 1, dtype=torch.float64)
        >>> one = torch.ones(1, 1, dtype=torch.float64)
        >>> m0 = torch.zeros(1, dtype=torch.float64)
        >>> means, covs = kalman_smoother(y, one, one, one, 2 * one, m0, one)
        >>> means.squeeze()  # the last state has no later measurements
        tensor([0.6562, 0.8125, 0.8750], dtype=torch.float64)
    """
    A, C, Q, R, m0, P0 = _parse(y, A, C, Q, R, m0, P0)
    means, covs, _ = _smooth(A, Q, *_filter(y, A, C, Q, R, m0, P0))
    # Drop the prior, x[0] given all the measurements.
    return means[:, 1:], covs[:, 1:]


def kalman_log_likelihood(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> Tensor:
    r"""The log marginal likelihood of a linear Gaussian state-space model.

    For the model of :func:`kalman_filter`, this returns

    .. math::
        \log p(\mathbf{y}[0], \dots, \mathbf{y}[N - 1])
        = \sum_{n=0}^{N-1} \log \mathcal{N}\big(\mathbf{y}[n];
        C[n] \hat{\mathbf{x}}[n + 1], S[n]\big),

    where :math:`\hat{\mathbf{x}}[n + 1]` is the mean of
    :math:`\mathbf{x}[n + 1]` given :math:`\mathbf{y}[0], \dots,
    \mathbf{y}[n - 1]` and :math:`S[n] = C[n] \hat{P}[n + 1] C[n]^H + R[n]`
    the covariance of the innovation, both one prediction step from the
    output of :func:`kalman_filter`, so it costs one filter. It is
    differentiable, for fitting the model's matrices by gradient ascent,
    including matrices predicted by a network. Complex tensors use the
    circularly symmetric complex Gaussian density. The arguments are those
    of :func:`kalman_filter`, and the same PyTorch 2.11 requirement applies.

    Args:
        y (Tensor): measurements, of shape :math:`(B, N, P)`.
        A (Tensor): state transition matrices :math:`A[n]`, taking
            :math:`\mathbf{x}[n]` to :math:`\mathbf{x}[n + 1]`, of base shape
            :math:`(M, M)`.
        C (Tensor): measurement matrices, of base shape :math:`(P, M)`.
        Q (Tensor): process noise covariances, of base shape :math:`(M, M)`.
        R (Tensor): measurement noise covariances, of base shape
            :math:`(P, P)`.
        m0 (Tensor): the prior mean, of shape :math:`(M)` or :math:`(B, M)`.
        P0 (Tensor): the prior covariance, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.

    Returns:
        Tensor: the log-likelihood of each signal, of shape :math:`(B)`.

    Raises:
        ValueError: if a coefficient has an unsupported shape.
        AssertionError: if :attr:`y` is not 3-D or :attr:`A` is not square.
        RuntimeError: on PyTorch older than 2.11.

    Example::

        >>> from philtorch.estimation import kalman_log_likelihood
        >>> # One step of the random walk of :func:`kalman_filter`'s example:
        >>> # y[0] ~ N(0, P0 + Q + R) = N(0, 4).
        >>> y = torch.zeros(1, 1, 1, dtype=torch.float64)
        >>> one = torch.ones(1, 1, dtype=torch.float64)
        >>> m0 = torch.zeros(1, dtype=torch.float64)
        >>> log_p = kalman_log_likelihood(y, one, one, one, 2 * one, m0, one)
        >>> torch.allclose(log_p, -0.5 * torch.log(2 * torch.pi * torch.tensor(4.0)).double())
        True
    """
    A, C, Q, R, m0, P0 = _parse(y, A, C, Q, R, m0, P0)
    means, covs = _filter(y, A, C, Q, R, m0, P0)
    return _log_likelihood(y, C, R, *_predict(A, Q, means[:, :-1], covs[:, :-1]))


def kalman_em_statistics(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> KalmanStatistics:
    r"""The expectations for fitting a linear Gaussian state-space model by EM.

    For the model of :func:`kalman_filter`, the E-step of
    expectation-maximization needs the smoothed moments of every state,
    including the prior state :math:`\mathbf{x}[0]`, and the lag-one
    cross-covariances :math:`\mathrm{Cov}(\mathbf{x}[n + 1], \mathbf{x}[n]
    \mid \mathbf{y})`. This returns those and the log-likelihood of
    :func:`kalman_log_likelihood` from one filter and one smoother scan. The
    cross-covariances come from the smoother's gains: :math:`\mathrm{Cov}(
    \mathbf{x}[n], \mathbf{x}[n + 1] \mid \mathbf{y}) = E[n] \,
    \mathrm{Cov}(\mathbf{x}[n + 1] \mid \mathbf{y})`, where :math:`E[n] =
    P[n] A[n]^H (A[n] P[n] A[n]^H + Q[n])^{-1}` with :math:`P[n]` the
    filtered covariance of :math:`\mathbf{x}[n]`. The arguments are those of
    :func:`kalman_filter`, and the same PyTorch 2.11 requirement applies.

    With constant matrices, the M-step has a closed form (Shumway and
    Stoffer, 1982), here for the measurements ``y`` of shape
    :math:`(B, N, P)`, fitting one model to all the signals::

        from philtorch.estimation import kalman_em_statistics

        stats = kalman_em_statistics(y, A, C, Q, R, m0, P0)
        m, V, V10 = stats.means, stats.covs, stats.cross_covs
        outer = V + m.unsqueeze(-1) @ m.unsqueeze(-2).conj()  # E[x[n] x[n]^H]
        cross = V10 + m[:, 1:].unsqueeze(-1) @ m[:, :-1].unsqueeze(-2).conj()
        S00 = outer[:, :-1].sum((0, 1))  # sums of E[x[n] x[n]^H], n < N
        S11 = outer[:, 1:].sum((0, 1))  # sums of E[x[n + 1] x[n + 1]^H]
        S10 = cross.sum((0, 1))  # sums of E[x[n + 1] x[n]^H]
        count = y.size(0) * y.size(1)
        A = torch.linalg.solve(S00, S10.mH).mH  # S10 S00^-1
        Q = (S11 - A @ S10.mH) / count
        Syx = (y.unsqueeze(-1) @ m[:, 1:].unsqueeze(-2).conj()).sum((0, 1))
        C = torch.linalg.solve(S11, Syx.mH).mH  # Syx S11^-1
        R = ((y.unsqueeze(-1) @ y.unsqueeze(-2).conj()).sum((0, 1)) - C @ Syx.mH) / count
        m0 = m[:, 0].mean(0)
        P0 = outer[:, 0].mean(0) - m0.outer(m0.conj())

    Each such step increases the log-likelihood until it converges. Average
    ``Q`` and ``R`` with their conjugate transposes to keep them exactly
    Hermitian.

    Args:
        y (Tensor): measurements, of shape :math:`(B, N, P)`.
        A (Tensor): state transition matrices :math:`A[n]`, taking
            :math:`\mathbf{x}[n]` to :math:`\mathbf{x}[n + 1]`, of base shape
            :math:`(M, M)`.
        C (Tensor): measurement matrices, of base shape :math:`(P, M)`.
        Q (Tensor): process noise covariances, of base shape :math:`(M, M)`.
        R (Tensor): measurement noise covariances, of base shape
            :math:`(P, P)`.
        m0 (Tensor): the prior mean, of shape :math:`(M)` or :math:`(B, M)`.
        P0 (Tensor): the prior covariance, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.

    Returns:
        KalmanStatistics: the smoothed means and covariances of
        :math:`\mathbf{x}[0], \dots, \mathbf{x}[N]`, of shapes
        :math:`(B, N + 1, M)` and :math:`(B, N + 1, M, M)`, the lag-one
        cross-covariances, of shape :math:`(B, N, M, M)`, and the
        log-likelihood, of shape :math:`(B)`.

    Raises:
        ValueError: if a coefficient has an unsupported shape.
        AssertionError: if :attr:`y` is not 3-D or :attr:`A` is not square.
        RuntimeError: on PyTorch older than 2.11.
    """
    A, C, Q, R, m0, P0 = _parse(y, A, C, Q, R, m0, P0)
    filtered_means, filtered_covs = _filter(y, A, C, Q, R, m0, P0)
    log_likelihood = _log_likelihood(
        y, C, R, *_predict(A, Q, filtered_means[:, :-1], filtered_covs[:, :-1])
    )
    means, covs, E = _smooth(A, Q, filtered_means, filtered_covs)
    # Cov(x[n + 1], x[n] | y) = Cov(x[n], x[n + 1] | y)^H = Cov(x[n + 1] | y) E[n]^H.
    cross_covs = covs[:, 1:] @ E.mH
    return KalmanStatistics(means, covs, cross_covs, log_likelihood)
