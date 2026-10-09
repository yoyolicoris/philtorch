"""Dynamic time warping (DTW) and soft-DTW."""

from typing import Literal

import torch
from torch import Tensor

StepPattern = Literal["symmetric", "asymmetric", "orthogonal"]


def _shear(cost: Tensor, width: int) -> Tensor:
    """The asymmetric steps' grid as the orthogonal steps' one.

    With k = n - m, the asymmetric steps (1, 0) and (1, 1) are (0, 1) and
    (1, 0) in (m, k), and a path from (0, 0) to (N - 1, M - 1) is one from
    (0, 0) to (M - 1, N - M) over the same cells: sheared[b, m, k] =
    cost[b, m + k, m], for k < width. Indices past N - 1 are clamped; those
    cells come after every cell a path of an item can use.
    """
    N, M = cost.shape[-2:]
    m = torch.arange(M, device=cost.device)[:, None]
    k = torch.arange(width, device=cost.device)
    return cost[:, (m + k).clamp(max=N - 1), m]


def _band_mask(N: int, M: int, n_len: Tensor, m_len: Tensor, band: float) -> Tensor:
    """Cells within ``band`` columns of the straight line between each item's corners."""
    i = torch.arange(N, device=n_len.device, dtype=torch.float64)[:, None]
    j = torch.arange(M, device=n_len.device, dtype=torch.float64)
    slope = (m_len - 1).double() / (n_len - 1).clamp(min=1).double()
    return (j - i * slope[:, None, None]).abs() <= band


def dtw(
    cost: Tensor,
    gamma: float = 0.0,
    *,
    step_pattern: StepPattern = "symmetric",
    diagonal_weight: float = 1.0,
    lengths: tuple[Tensor, Tensor] | None = None,
    band: float | None = None,
) -> Tensor:
    r"""The DTW or soft-DTW distance between pairs of sequences, from their costs.

    For a cost matrix :math:`c[n, m]` of matching frame :math:`n` of one
    sequence with frame :math:`m` of another, the accumulated costs

    .. math::
        D[n, m] = c[n, m] + \min_\gamma\big(D[n - 1, m],\ D[n, m - 1],\
        D[n - 1, m - 1] + (w - 1)\, c[n, m]\big),

    with :math:`D[0, 0] = c[0, 0]`, give the distance :math:`D[N - 1, M - 1]`:
    the cost of the cheapest monotonic path from the first pair of frames to
    the last. With :math:`\gamma = 0`, :math:`\min_\gamma` is the minimum (DTW);
    with :math:`\gamma > 0` it is the soft minimum
    :math:`-\gamma \log \sum_i e^{-x_i / \gamma}` (soft-DTW, Cuturi and Blondel,
    2017), and the distance is :math:`-\gamma \log` of the sum of
    :math:`e^{-\text{cost} / \gamma}` over all paths.

    The gradient of the distance with respect to :attr:`cost` is the
    alignment: the 0/1 indicator of the optimal path for DTW, and for
    soft-DTW the expected alignment, each cell's probability of being on a
    path under the Gibbs distribution. Both come from autograd, with no
    backtracking, and the gradient is itself differentiable, to any order.

    Rows of the accumulated costs are computed one after another, each with
    a parallel scan over its cells (after Xiao et al., "Parallelizing Dynamic
    Time Warping Algorithm Using Prefix Computations on GPU", HPCC 2013), in
    one Triton kernel per batch: :math:`\min(N, M)` sequential steps for the
    symmetric and orthogonal steps, :math:`M` for the asymmetric steps, and
    :math:`O(NM)` work and memory. The cost can be any tensor, from any
    differentiable function of the sequences, such as the squared Euclidean
    distances between their frames.

    Args:
        cost (Tensor): the costs :math:`c[n, m]`, of shape :math:`(B, N, M)`,
            on a CUDA device.
        gamma (float): 0 for DTW, positive for soft-DTW. Default: 0.
        step_pattern (str): the steps a path may take, as (row, column)
            increments: ``"symmetric"`` for (1, 0), (0, 1) and (1, 1);
            ``"asymmetric"`` for (1, 0) and (1, 1) only, so that every frame of
            the first sequence matches exactly one frame of the second
            (needs :math:`N \ge M`); ``"orthogonal"`` for (1, 0) and (0, 1)
            only, so that every path has :math:`N + M - 1` cells. Default:
            ``"symmetric"``.
        diagonal_weight (float): :math:`w`, the weight of a diagonal step's
            cost in the symmetric steps. 1 favors diagonal paths, which add
            fewer costs; 2 (the "symmetric2" pattern of Sakoe and Chiba) makes
            a diagonal step cost as much as a horizontal and a vertical one.
            Default: 1.
        lengths (tuple of Tensor, optional): each pair's lengths
            :math:`(N_b, M_b)`, two integer tensors of shape :math:`(B)`, for
            batches of sequences padded to :math:`N` and :math:`M`; the
            distance is then :math:`D[N_b - 1, M_b - 1]`. Padding costs don't
            affect it. Default: the full lengths.
        band (float, optional): a Sakoe-Chiba band: only cells within this
            many columns of the straight line from :math:`(0, 0)` to
            :math:`(N_b - 1, M_b - 1)` may be on a path. Default: no band.

    Returns:
        Tensor: the distances, of shape :math:`(B)`; ``inf`` for a pair with
        no path, such as the asymmetric steps with :math:`N_b < M_b`.

    Raises:
        ValueError: if :attr:`cost` is not on a CUDA device, or
            :attr:`diagonal_weight` is not 1 with steps that have no weighted
            diagonal.

    Example::

        >>> import torch
        >>> from philtorch.align import dtw
        >>> x = torch.randn(4, 100, 3, device="cuda")  # 4 sequences of 3-D frames
        >>> y = torch.randn(4, 80, 3, device="cuda", requires_grad=True)
        >>> cost = torch.cdist(x, y) ** 2
        >>> distance = dtw(cost, gamma=0.1)
        >>> alignment, = torch.autograd.grad(distance.sum(), cost, retain_graph=True)
        >>> distance.sum().backward()  # gradients with respect to y
    """
    assert cost.dim() == 3, f"cost must be (B, N, M), got {tuple(cost.shape)}"
    if not cost.is_cuda:
        raise ValueError("dtw runs Triton kernels, which need a CUDA tensor.")
    if diagonal_weight != 1.0 and step_pattern != "symmetric":
        raise ValueError(f"diagonal_weight needs the symmetric steps, not {step_pattern!r}.")
    from ._dtw_kernels import dtw_dp

    B, N, M = cost.shape
    if lengths is None:
        n_len = torch.full((B,), N, device=cost.device)
        m_len = torch.full((B,), M, device=cost.device)
    else:
        n_len, m_len = (t.to(cost.device, torch.long) for t in lengths)
    scale = 1.0 / gamma if gamma > 0 else 1.0
    cost = cost * scale
    if band is not None:
        # A large finite cost, not inf, keeps soft-min's arithmetic finite; a
        # path sums at most w (N + M) of them.
        big = torch.finfo(cost.dtype).max / (8 * max(diagonal_weight, 1.0) * (N + M))
        cost = torch.where(_band_mask(N, M, n_len, m_len, band), cost, big)

    no_path = torch.zeros(B, dtype=torch.bool, device=cost.device)
    if step_pattern == "asymmetric":
        no_path = n_len < m_len
        width = int((n_len - m_len).max()) + 1 if B else 1
        if width <= 0:
            return torch.full((B,), float("inf"), dtype=cost.dtype, device=cost.device)
        cost = _shear(cost, width)
        n_len, m_len = m_len, (n_len - m_len + 1).clamp(min=1)
    if cost.size(1) > cost.size(2):
        cost, n_len, m_len = cost.mT, m_len, n_len
    # After the shear, the asymmetric steps are orthogonal ones.
    D = dtw_dp(cost, gamma > 0, step_pattern == "symmetric", float(diagonal_weight))
    distance = D[torch.arange(B, device=D.device), n_len - 1, m_len - 1]
    if band is not None:
        no_path = no_path | (distance >= big / 2)
    return torch.where(no_path, float("inf"), distance) / scale


def soft_dtw_divergence(
    cost_xy: Tensor,
    cost_xx: Tensor,
    cost_yy: Tensor,
    gamma: float,
    *,
    lengths: tuple[Tensor, Tensor] | None = None,
    **kwargs,
) -> Tensor:
    r"""The soft-DTW divergence between pairs of sequences, from their costs.

    .. math::
        \mathrm{SDTW}_\gamma(x, y) - \tfrac{1}{2}\big(\mathrm{SDTW}_\gamma(x, x)
        + \mathrm{SDTW}_\gamma(y, y)\big),

    from Blondel, Mensch and Vert, "Differentiable Divergences Between Time
    Series" (AISTATS 2021). Unlike soft-DTW itself, it is zero for identical
    sequences, and with the squared Euclidean cost it is non-negative.

    Args:
        cost_xy (Tensor): costs between the sequences, of shape
            :math:`(B, N, M)`.
        cost_xx (Tensor): costs of the first sequences with themselves, of
            shape :math:`(B, N, N)`.
        cost_yy (Tensor): costs of the second sequences with themselves, of
            shape :math:`(B, M, M)`.
        gamma (float): the soft-min temperature, positive.
        lengths (tuple of Tensor, optional): each pair's lengths
            :math:`(N_b, M_b)`, as in :func:`dtw`.
        **kwargs: the other keyword arguments of :func:`dtw`.

    Returns:
        Tensor: the divergences, of shape :math:`(B)`.
    """
    assert gamma > 0, "the soft-DTW divergence needs gamma > 0"
    n_len, m_len = (None, None) if lengths is None else lengths
    xx = None if lengths is None else (n_len, n_len)
    yy = None if lengths is None else (m_len, m_len)
    return dtw(cost_xy, gamma, lengths=lengths, **kwargs) - 0.5 * (
        dtw(cost_xx, gamma, lengths=xx, **kwargs) + dtw(cost_yy, gamma, lengths=yy, **kwargs)
    )
