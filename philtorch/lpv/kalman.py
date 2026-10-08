"""Kalman filters and smoothers parallelized over time with associative scans.

These follow Särkkä and García-Fernández, "Temporal parallelization of
Bayesian smoothers" (IEEE TAC, 2021), as evaluated on GPUs in Särkkä and
García-Fernández, "On the performance of prefix-sum parallel Kalman filters
and smoothers on GPUs" (arXiv:2511.10363).
"""

import inspect

import torch
from torch import Tensor

try:
    from torch._higher_order_ops.associative_scan import associative_scan as _associative_scan
except ImportError:  # pragma: no cover - PyTorch without associative_scan
    _associative_scan = None

# combine_mode="generic", which takes any combine function, arrived in
# PyTorch 2.5. It runs eagerly: the combine function is vmapped over the scan
# dimension and applied O(log N) times, so autograd follows it as usual.
_HAS_GENERIC_SCAN = _associative_scan is not None and (
    "combine_mode" in inspect.signature(_associative_scan).parameters
)


def _scan(combine_fn, xs: tuple[Tensor, ...], reverse: bool = False) -> tuple[Tensor, ...]:
    if not _HAS_GENERIC_SCAN:
        raise RuntimeError(
            "Kalman filtering needs the generic mode of PyTorch's associative_scan, "
            f"added in PyTorch 2.5; this is PyTorch {torch.__version__}."
        )
    # Positional, since PyTorch 2.5 calls the second argument `input`, not `xs`.
    return tuple(_associative_scan(combine_fn, xs, 1, reverse, "generic"))


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

    # N measurements have N - 1 transitions between them.
    A = _coefficient("A", A, (M, M), batch_size, N - 1)
    C = _coefficient("C", C, (P, M), batch_size, N)
    Q = _coefficient("Q", Q, (M, M), batch_size, N - 1)
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

    return (
        full(A, N - 1, M, M),
        full(C, N, P, M),
        full(Q, N - 1, M, M),
        full(R, N, P, P),
        m0,
        P0,
    )


def _mv(A: Tensor, x: Tensor) -> Tensor:
    return (A @ x.unsqueeze(-1)).squeeze(-1)


def _filtering_elements(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, ...]:
    """Build the elements (F, b, G, eta, J) of the parallel Kalman filter.

    These are the paper's (A, b, C, eta, J), renamed so they don't clash with
    the model's A and C.

    Element n describes x[n] given x[n - 1] and y[n] (eqs. 42-45 of the GPU
    paper, one index earlier): x[n] | x[n - 1], y[n] ~ N(F x[n - 1] + b, G),
    and the likelihood of y[n] given x[n - 1] in information form (eta, J).
    Element 0 holds the update of the prior with y[0].
    """
    batch_size, M = y.size(0), A.size(-1)
    # Steps 1, ..., N - 1 predict through A[n - 1] and Q[n - 1], then update.
    C_n, R_n, y_n = C[:, 1:], R[:, 1:], y[:, 1:]
    CA = C_n @ A
    S = C_n @ Q @ C_n.mT + R_n
    K = torch.linalg.solve(S, C_n @ Q).mT
    F = A - K @ CA
    b = _mv(K, y_n)
    G = Q - K @ S @ K.mT
    eta = _mv(CA.mT, torch.linalg.solve(S, y_n))
    J = CA.mT @ torch.linalg.solve(S, CA)

    # Step 0 updates the prior N(m0, P0) with y[0]; it doesn't depend on any
    # earlier state, so its F, eta and J are zero.
    m0 = m0.expand(batch_size, 1, M)
    P0 = P0.expand(batch_size, 1, M, M)
    C_0, R_0, y_0 = C[:, :1], R[:, :1], y[:, :1]
    S_0 = C_0 @ P0 @ C_0.mT + R_0
    K_0 = torch.linalg.solve(S_0, C_0 @ P0).mT
    b_0 = m0 + _mv(K_0, y_0 - _mv(C_0, m0))
    G_0 = P0 - K_0 @ S_0 @ K_0.mT

    def first(t0: Tensor, t: Tensor) -> Tensor:
        return torch.cat([t0, t], dim=1)

    return (
        first(torch.zeros_like(P0), F),
        first(b_0, b),
        first(G_0, G),
        first(torch.zeros_like(m0), eta),
        first(torch.zeros_like(P0), J),
    )


def _filtering_operator(
    earlier: tuple[Tensor, ...], later: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    """Combine two filtering elements (Lemma 3 of the GPU paper)."""
    F_i, b_i, G_i, eta_i, J_i = earlier
    F_j, b_j, G_j, eta_j, J_j = later
    eye = torch.eye(F_i.size(-1), dtype=F_i.dtype, device=F_i.device)
    # X = F_j (I + G_i J_j)^-1 and Y = F_i^T (I + J_j G_i)^-1. These use inv,
    # not solve: the generic scan also calls this on empty slices, which
    # solve's vmap batching rule fails on.
    X = F_j @ torch.linalg.inv(eye + G_i @ J_j)
    Y = F_i.mT @ torch.linalg.inv(eye + J_j @ G_i)
    return (
        X @ F_i,
        _mv(X, b_i + _mv(G_i, eta_j)) + b_j,
        X @ G_i @ F_j.mT + G_j,
        _mv(Y, eta_j - _mv(J_j, b_i)) + eta_i,
        Y @ J_j @ F_i + J_i,
    )


def _filter(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, Tensor]:
    elements = _filtering_elements(y, A, C, Q, R, m0, P0)
    _, means, covs, _, _ = _scan(_filtering_operator, elements)
    return means, covs


def _smoothing_operator(
    later: tuple[Tensor, ...], earlier: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    """Combine two smoothing elements (Lemma 4 of the GPU paper).

    The scan runs in reverse, so the element later in time comes first.
    """
    E_j, g_j, L_j = later
    E_i, g_i, L_i = earlier
    return E_i @ E_j, _mv(E_i, g_j) + g_i, E_i @ L_j @ E_i.mT + L_i


def kalman_filter(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Filter a linear Gaussian state-space model in parallel over time.

    For the model

    .. math::
        \mathbf{x}[n + 1] &= A[n] \mathbf{x}[n] + \mathbf{w}[n],
        & \mathbf{w}[n] &\sim \mathcal{N}(\mathbf{0}, Q[n]), \\
        \mathbf{y}[n] &= C[n] \mathbf{x}[n] + \mathbf{v}[n],
        & \mathbf{v}[n] &\sim \mathcal{N}(\mathbf{0}, R[n]),

    with prior :math:`\mathbf{x}[0] \sim \mathcal{N}(\mathbf{m}_0, P_0)`,
    this returns the mean and covariance of each state given the measurements
    up to it, :math:`p(\mathbf{x}[n] \mid \mathbf{y}[0], \dots,
    \mathbf{y}[n])`. These are the results of the sequential Kalman filter,
    computed instead with an associative scan of depth :math:`O(\log N)`, as
    in Särkkä and García-Fernández (2021). The scan is PyTorch's
    ``associative_scan`` in its generic mode, which needs PyTorch 2.5 or later.

    Each of :attr:`A`, :attr:`C`, :attr:`Q` and :attr:`R` may be constant or
    time-varying, and shared or one per signal: its base shape below can be
    prefixed with the number of steps for time-varying values, :math:`B` for
    one per signal, or both. That number is :math:`N` for :attr:`C` and
    :attr:`R`, one per measurement, and :math:`N - 1` for :attr:`A` and
    :attr:`Q`, one per transition between measurements. When two readings
    fit, such as :math:`N = B`, the time-varying one is taken. Even with
    constant matrices, the Kalman gain varies over time.

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
        RuntimeError: on PyTorch older than 2.5.

    Example::

        >>> from philtorch.lpv import kalman_filter
        >>> # A random walk with unit steps, measured with noise variance 2.
        >>> y = torch.ones(1, 3, 1, dtype=torch.float64)
        >>> one = torch.ones(1, 1, dtype=torch.float64)
        >>> m0 = torch.zeros(1, dtype=torch.float64)
        >>> means, covs = kalman_filter(y, one, one, one, 2 * one, m0, one)
        >>> means.squeeze()
        tensor([0.3333, 0.6364, 0.8140], dtype=torch.float64)
    """
    return _filter(y, *_parse(y, A, C, Q, R, m0, P0))


def kalman_smoother(
    y: Tensor, A: Tensor, C: Tensor, Q: Tensor, R: Tensor, m0: Tensor, P0: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Smooth a linear Gaussian state-space model in parallel over time.

    For the model of :func:`kalman_filter`, this returns the mean and
    covariance of each state given all the measurements,
    :math:`p(\mathbf{x}[n] \mid \mathbf{y}[0], \dots, \mathbf{y}[N - 1])`.
    These are the results of the Rauch--Tung--Striebel smoother, computed
    with a reverse associative scan after :func:`kalman_filter`, as in
    Särkkä and García-Fernández (2021). The arguments are those of
    :func:`kalman_filter`, and the same PyTorch 2.5 requirement applies.

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
        RuntimeError: on PyTorch older than 2.5.

    Example::

        >>> from philtorch.lpv import kalman_smoother
        >>> # The random walk of :func:`kalman_filter`'s example.
        >>> y = torch.ones(1, 3, 1, dtype=torch.float64)
        >>> one = torch.ones(1, 1, dtype=torch.float64)
        >>> m0 = torch.zeros(1, dtype=torch.float64)
        >>> means, covs = kalman_smoother(y, one, one, one, 2 * one, m0, one)
        >>> means.squeeze()  # the last state has no later measurements
        tensor([0.4884, 0.7209, 0.8140], dtype=torch.float64)
    """
    A, C, Q, R, m0, P0 = _parse(y, A, C, Q, R, m0, P0)
    means, covs = _filter(y, A, C, Q, R, m0, P0)

    # Element n describes x[n] given x[n + 1] and y[0], ..., y[n]
    # (eqs. 48-50): x[n] | x[n + 1] ~ N(E x[n + 1] + g, L). The last one is
    # the filtering result itself.
    m_n, P_n = means[:, :-1], covs[:, :-1]
    P_pred = A @ P_n @ A.mT + Q
    E = torch.linalg.solve(P_pred, A @ P_n).mT
    g = m_n - _mv(E @ A, m_n)
    L = P_n - E @ P_pred @ E.mT
    elements = (
        torch.cat([E, torch.zeros_like(covs[:, -1:])], dim=1),
        torch.cat([g, means[:, -1:]], dim=1),
        torch.cat([L, covs[:, -1:]], dim=1),
    )
    _, means, covs = _scan(_smoothing_operator, elements, reverse=True)
    return means, covs
