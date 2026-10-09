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
    """Validate the model, and return it as the scan's N steps.

    Each scan step is a transition, then a measurement of the state it
    reaches. The model's first state, which the prior describes, is measured
    before any transition, so the first step is the identity without noise,
    followed by the model's N - 1 transitions: step n ends at x[n].
    """
    assert y.dim() == 3, f"Measurements y must be 3D (batch, time, features), got {y.shape}"
    batch_size, N, P = y.shape
    assert A.dim() >= 2 and A.size(-1) == A.size(-2), f"A must be square, got {A.shape}"
    M = A.size(-1)
    transitions = max(N - 1, 0)

    A = _coefficient("A", A, (M, M), batch_size, transitions)
    C = _coefficient("C", C, (P, M), batch_size, N)
    Q = _coefficient("Q", Q, (M, M), batch_size, transitions)
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

    def full(t: Tensor, steps: int, *base: int) -> Tensor:
        return t.expand(batch_size, steps, *base)

    A, Q = full(A, transitions, M, M), full(Q, transitions, M, M)
    eye = torch.eye(M, dtype=A.dtype, device=A.device).expand(batch_size, 1, M, M)
    A = torch.cat([eye, A], dim=1)[:, :N]
    Q = torch.cat([torch.zeros_like(eye), Q], dim=1)[:, :N]
    return A, full(C, N, P, M), Q, full(R, N, P, P), m0, P0


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
    """Smooth filtered moments: those of the states, A and Q of the steps out of each.

    Returns the smoothed moments and the gains E[n] of the smoothing
    elements, which make Cov(x[n], x[n + 1] | y) = E[n] Cov(x[n + 1] | y).
    """
    # Element n describes x[n] given x[n + 1] and y[0], ..., y[n] (eqs. 48-50
    # of the GPU paper): x[n] | x[n + 1] ~ N(E x[n + 1] + g, L), through A[n]
    # and Q[n]. The last one is the filtering result itself.
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


class KalmanFilterResult(NamedTuple):
    """The result of :func:`kalman_filter`."""

    #: Filtered means, of shape :math:`(B, N, M)`.
    means: Tensor
    #: Filtered covariances, of shape :math:`(B, N, M, M)`.
    covs: Tensor
    #: Each signal's log marginal likelihood :math:`\log p(\mathbf{y})`, of shape :math:`(B)`.
    log_likelihood: Tensor


class KalmanSmootherResult(NamedTuple):
    """The result of :func:`kalman_smoother`."""

    #: Smoothed means, of shape :math:`(B, N, M)`.
    means: Tensor
    #: Smoothed covariances, of shape :math:`(B, N, M, M)`.
    covs: Tensor
    #: Lag-one cross-covariances
    #: :math:`\mathrm{Cov}(\mathbf{x}[n + 1], \mathbf{x}[n] \mid \mathbf{y})`, of shape
    #: :math:`(B, N - 1, M, M)`.
    cross_covs: Tensor
    #: Each signal's log marginal likelihood :math:`\log p(\mathbf{y})`, of shape :math:`(B)`.
    log_likelihood: Tensor


def kalman_filter(
    y: Tensor,
    A: Tensor,
    C: Tensor,
    Q: Tensor,
    R: Tensor,
    m0: Tensor,
    P0: Tensor,
) -> KalmanFilterResult:
    r"""Filter a linear Gaussian state-space model in parallel over time.

    For the linear Gaussian state-space model

    .. math::
        \mathbf{x}[n + 1] &= A[n] \mathbf{x}[n] + \mathbf{w}[n],
        & \mathbf{w}[n] &\sim \mathcal{N}(\mathbf{0}, Q[n]), \\
        \mathbf{y}[n] &= C[n] \mathbf{x}[n] + \mathbf{v}[n],
        & \mathbf{v}[n] &\sim \mathcal{N}(\mathbf{0}, R[n]),

    with prior :math:`\mathbf{x}[0] \sim \mathcal{N}(\mathbf{m}_0, P_0)` on
    the first state, which :math:`\mathbf{y}[0]` measures, the states are
    :math:`\mathbf{x}[0], \dots, \mathbf{x}[N - 1]`, and :attr:`A` and
    :attr:`Q` have the :math:`N - 1` transitions between them.

    This returns the mean and covariance of each state given the
    measurements up to it, and the log marginal likelihood

    .. math::
        \log p(\mathbf{y}[0], \dots, \mathbf{y}[N - 1]) = \sum_n \log
        \mathcal{N}\big(\mathbf{y}[n];\ C[n] \hat{\mathbf{x}}_n,\
        C[n] \hat{P}_n C[n]^H + R[n]\big),

    with :math:`\hat{\mathbf{x}}_n` and :math:`\hat{P}_n` the predicted
    moments of the state :math:`\mathbf{y}[n]` measures, which is
    differentiable, for fitting the model's matrices by gradient ascent,
    including matrices predicted by a network. These are the results of the
    sequential Kalman filter, computed instead with an associative scan of
    depth :math:`O(\log N)`, as in Särkkä and García-Fernández (2021). The
    scan is PyTorch's ``associative_scan`` in its generic mode, which needs
    PyTorch 2.11 or later.

    Each of :attr:`A`, :attr:`C`, :attr:`Q` and :attr:`R` may be constant or
    time-varying, and shared or one per signal: its base shape below can be
    prefixed with its number of steps :math:`T` (:math:`N` for :attr:`C` and
    :attr:`R`, :math:`N - 1` for :attr:`A` and :attr:`Q`) for time-varying
    values, :math:`B` for one per signal, or
    :math:`(B, T)` for both. When two readings fit, such as :math:`T = B`, the
    time-varying one is taken. Even with constant matrices, the Kalman gain
    varies over time. Complex tensors describe circularly symmetric complex
    Gaussian noise, with conjugate transposes in place of transposes.

    Note:
        Known inputs need no extra arguments. For
        :math:`\mathbf{x}[n + 1] = A[n] \mathbf{x}[n] + \mathbf{u}[n] +
        \mathbf{w}[n]` and :math:`\mathbf{y}[n] = C[n] \mathbf{x}[n] +
        \mathbf{d}[n] + \mathbf{v}[n]`, filter the part of the state that
        :math:`\mathbf{u}` doesn't drive, then add the part it does, which
        :func:`~philtorch.lpv.state_space_recursion` computes from zero. With
        :attr:`u` of shape :math:`(B, N - 1, M)`, :attr:`d` of shape
        :math:`(B, N, P)`, and :attr:`A` and :attr:`C` expanded to their full
        shapes :math:`(B, N - 1, M, M)` and :math:`(B, N, P, M)`, such as
        ``A.expand(B, N - 1, M, M)`` for a constant :attr:`A`::

            from philtorch.lpv import state_space_recursion

            x_u = state_space_recursion(A, y.new_zeros(B, M), u)
            x_u = torch.cat([y.new_zeros(B, 1, M), x_u], dim=1)  # x_u[0] = 0
            y_s = y - d - (C @ x_u.unsqueeze(-1)).squeeze(-1)
            result = kalman_filter(y_s, A, C, Q, R, m0, P0)
            means = result.means + x_u

        This is exact, as the model is linear, and the covariances need no
        correction. The same works for :func:`kalman_smoother`.

    Note:
        A prior on the state before the first measured one, as ``zi`` in
        :func:`~philtorch.lpv.state_space_recursion`, with :math:`N`
        transitions :attr:`A` and :attr:`Q` into the measured states, is a
        prior on the first measured state after one prediction step::

            m0 = (A[..., 0, :, :] @ m0.unsqueeze(-1)).squeeze(-1)
            P0 = A[..., 0, :, :] @ P0 @ A[..., 0, :, :].mH + Q[..., 0, :, :]
            result = kalman_filter(y, A[..., 1:, :, :], C, Q[..., 1:, :, :], R, m0, P0)

        for time-varying :attr:`A` and :attr:`Q` of shape
        :math:`(B, N, M, M)`.

    Args:
        y (Tensor): measurements :math:`\mathbf{y}[n]`, of shape
            :math:`(B, N, P)`.
        A (Tensor): state transition matrices, each taking a state to the
            next, of base shape :math:`(M, M)`.
        C (Tensor): measurement matrices, of base shape :math:`(P, M)`.
        Q (Tensor): process noise covariances, of base shape :math:`(M, M)`.
        R (Tensor): measurement noise covariances, of base shape
            :math:`(P, P)`.
        m0 (Tensor): the prior mean of :math:`\mathbf{x}[0]`, of shape
            :math:`(M)` or :math:`(B, M)`.
        P0 (Tensor): the prior covariance of :math:`\mathbf{x}[0]`, of shape
            :math:`(M, M)` or :math:`(B, M, M)`.

    Returns:
        KalmanFilterResult: the filtered means, of shape :math:`(B, N, M)`,
        and covariances, of shape :math:`(B, N, M, M)`, and the
        log-likelihood, of shape :math:`(B)`.

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
        >>> result = kalman_filter(y, one, one, one, 2 * one, m0, 2 * one)
        >>> result.means.squeeze()
        tensor([0.5000, 0.7500, 0.8750], dtype=torch.float64)
    """
    A, C, Q, R, m0, P0 = _parse(y, A, C, Q, R, m0, P0)
    means, covs = _filter(y, A, C, Q, R, m0, P0)
    log_likelihood = _log_likelihood(y, C, R, *_predict(A, Q, means[:, :-1], covs[:, :-1]))
    # The scan's first output is the identity step's input, x[0] again.
    return KalmanFilterResult(means[:, 1:], covs[:, 1:], log_likelihood)


def kalman_smoother(
    y: Tensor,
    A: Tensor,
    C: Tensor,
    Q: Tensor,
    R: Tensor,
    m0: Tensor,
    P0: Tensor,
) -> KalmanSmootherResult:
    r"""Smooth a linear Gaussian state-space model in parallel over time.

    For the linear Gaussian state-space model

    .. math::
        \mathbf{x}[n + 1] &= A[n] \mathbf{x}[n] + \mathbf{w}[n],
        & \mathbf{w}[n] &\sim \mathcal{N}(\mathbf{0}, Q[n]), \\
        \mathbf{y}[n] &= C[n] \mathbf{x}[n] + \mathbf{v}[n],
        & \mathbf{v}[n] &\sim \mathcal{N}(\mathbf{0}, R[n]),

    with prior :math:`\mathbf{x}[0] \sim \mathcal{N}(\mathbf{m}_0, P_0)` on
    the first state, which :math:`\mathbf{y}[0]` measures, the states are
    :math:`\mathbf{x}[0], \dots, \mathbf{x}[N - 1]`, and :attr:`A` and
    :attr:`Q` have the :math:`N - 1` transitions between them.

    This returns the mean and covariance of each state given all the
    measurements, the lag-one cross-covariances
    :math:`\mathrm{Cov}(\mathbf{x}[n + 1], \mathbf{x}[n] \mid \mathbf{y})`,
    and the log-likelihood of :func:`kalman_filter`: the results of the
    Rauch--Tung--Striebel smoother, computed with a reverse associative scan
    after the filter's, as in Särkkä and García-Fernández (2021), with the
    same PyTorch 2.11 requirement. The cross-covariances come from the
    smoother's gains, :math:`\mathrm{Cov}(\mathbf{x}[n], \mathbf{x}[n + 1]
    \mid \mathbf{y}) = E[n] \, \mathrm{Cov}(\mathbf{x}[n + 1] \mid
    \mathbf{y})`.

    Each of :attr:`A`, :attr:`C`, :attr:`Q` and :attr:`R` may be constant or
    time-varying, and shared or one per signal: its base shape below can be
    prefixed with its number of steps :math:`T` (:math:`N` for :attr:`C` and
    :attr:`R`, :math:`N - 1` for :attr:`A` and :attr:`Q`) for time-varying
    values, :math:`B` for one per signal, or
    :math:`(B, T)` for both. When two readings fit, such as :math:`T = B`, the
    time-varying one is taken. Even with constant matrices, the Kalman gain
    varies over time. Complex tensors describe circularly symmetric complex
    Gaussian noise, with conjugate transposes in place of transposes.

    These are the expectations of the E-step of expectation-maximization.
    With constant matrices, the M-step has a closed form (Shumway and
    Stoffer, 1982), here fitting one model to all the signals::

        from philtorch.estimation import kalman_smoother

        m, V, V10, _ = kalman_smoother(y, A, C, Q, R, m0, P0)
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

    Each such step increases the log-likelihood until it converges. Average
    ``Q`` and ``R`` with their conjugate transposes to keep them exactly
    Hermitian.

    Args:
        y (Tensor): measurements :math:`\mathbf{y}[n]`, of shape
            :math:`(B, N, P)`.
        A (Tensor): state transition matrices, each taking a state to the
            next, of base shape :math:`(M, M)`.
        C (Tensor): measurement matrices, of base shape :math:`(P, M)`.
        Q (Tensor): process noise covariances, of base shape :math:`(M, M)`.
        R (Tensor): measurement noise covariances, of base shape
            :math:`(P, P)`.
        m0 (Tensor): the prior mean of :math:`\mathbf{x}[0]`, of shape
            :math:`(M)` or :math:`(B, M)`.
        P0 (Tensor): the prior covariance of :math:`\mathbf{x}[0]`, of shape
            :math:`(M, M)` or :math:`(B, M, M)`.

    Returns:
        KalmanSmootherResult: the smoothed means and covariances, of shapes
        :math:`(B, N, M)` and :math:`(B, N, M, M)`, the cross-covariances, of
        shape :math:`(B, N - 1, M, M)`, and the log-likelihood, of shape
        :math:`(B)`.

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
        >>> result = kalman_smoother(y, one, one, one, 2 * one, m0, 2 * one)
        >>> result.means.squeeze()  # the last state has no later measurements
        tensor([0.6562, 0.8125, 0.8750], dtype=torch.float64)
    """
    A, C, Q, R, m0, P0 = _parse(y, A, C, Q, R, m0, P0)
    filtered_means, filtered_covs = _filter(y, A, C, Q, R, m0, P0)
    log_likelihood = _log_likelihood(
        y, C, R, *_predict(A, Q, filtered_means[:, :-1], filtered_covs[:, :-1])
    )
    # Drop the identity step's input, x[0] again, and smooth through the
    # model's own transitions.
    means, covs, E = _smooth(A[:, 1:], Q[:, 1:], filtered_means[:, 1:], filtered_covs[:, 1:])
    # Cov(x[n + 1], x[n] | y) = Cov(x[n], x[n + 1] | y)^H = Cov(x[n + 1] | y) E[n]^H.
    cross_covs = covs[:, 1:] @ E.mH
    return KalmanSmootherResult(means, covs, cross_covs, log_likelihood)
