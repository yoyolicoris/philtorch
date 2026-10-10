"""Dynamic time warping (DTW) and soft-DTW."""

import math
from typing import Literal

import torch
from torch import Tensor

from .._triton import check_cuda_triton

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
    """Cells within ``band`` cells of the straight line between each pair's corners.

    The distance is measured along the longer side, and the band is widened
    where needed for consecutive rows (or columns) of it to overlap, so that
    it always holds a path.
    """
    i = torch.arange(N, device=n_len.device, dtype=torch.float64)[:, None]
    j = torch.arange(M, device=n_len.device, dtype=torch.float64)
    n = (n_len - 1).double()[:, None, None]
    m = (m_len - 1).double()[:, None, None]
    rows_longer = n > m
    # Along the longer side, the line advances `slope` cells per cell of the other.
    slope = torch.where(rows_longer, n, m) / torch.where(rows_longer, m, n).clamp(min=1)
    offset = torch.where(rows_longer, i - j * slope, j - i * slope).abs()
    width = slope / 2  # the least that leaves a step between consecutive rows
    one_line = torch.minimum(n, m) == 0  # a single row or column: no band to keep
    return (offset <= torch.maximum(width, torch.full_like(width, band))) | one_line


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
    :math:`-\gamma \log \sum_i e^{-x_i / \gamma}`, and the distance is
    :math:`-\gamma \log` of the sum of :math:`e^{-\text{cost} / \gamma}` over
    all paths: soft-DTW, from `Soft-DTW: a Differentiable Loss Function for
    Time-Series`_ (Cuturi and Blondel, 2017).

    The gradient of the distance with respect to :attr:`cost` is the
    alignment: the 0/1 indicator of the optimal path for DTW, and for
    soft-DTW the expected alignment, each cell's probability of being on a
    path under the Gibbs distribution; with a diagonal weight :math:`w`, a
    cell entered diagonally counts :math:`w` times. Both come from autograd,
    with no backtracking, and the gradient is itself differentiable, to any
    order. Costs may be ``inf`` to forbid cells.

    Rows of the accumulated costs are computed one after another, each with
    a parallel scan over its cells, after `Parallelizing Dynamic Time
    Warping Algorithm Using Prefix Computations on GPU`_ (Xiao et al.,
    2013), in one Triton kernel per batch: :math:`\min(N, M)` sequential
    steps for the symmetric and orthogonal steps, at most :math:`M` for the
    asymmetric steps, and :math:`O(NM)` work and memory. The cost can be any
    tensor, from any differentiable function of the sequences, such as the
    squared Euclidean distances between their frames.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        cost (Tensor): the costs :math:`c[n, m]`, of shape :math:`(B, N, M)`.
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
            many cells of the straight line from :math:`(0, 0)` to
            :math:`(N_b - 1, M_b - 1)`, measured along the longer sequence,
            may be on a path. Where the line is steeper than the band allows
            a step to follow, the band widens to keep a path. Default: no band.

    Returns:
        Tensor: the distances, of shape :math:`(B)`; ``inf`` for a pair with
        no path, such as the asymmetric steps with :math:`N_b < M_b`.

    Raises:
        ValueError: if :attr:`cost` is not a CUDA tensor of shape
            :math:`(B, N, M)` with :math:`N, M > 0`, at most
            :math:`2^{31} - 1` cells per pair and :math:`\max(N, M) \le
            2^{18}`, if :attr:`step_pattern` is unknown, if
            :attr:`diagonal_weight` is not 1 with steps that have no weighted
            diagonal, or if :attr:`lengths` are not of shape :math:`(B)`
            within :math:`1 \le N_b \le N` and :math:`1 \le M_b \le M`.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import dtw
        >>> x = torch.randn(4, 100, 3, device="cuda")  # 4 sequences of 3-D frames
        >>> y = torch.randn(4, 80, 3, device="cuda", requires_grad=True)
        >>> cost = torch.cdist(x, y) ** 2
        >>> distance = dtw(cost, gamma=0.1)
        >>> alignment, = torch.autograd.grad(distance.sum(), cost, retain_graph=True)
        >>> distance.sum().backward()  # gradients with respect to y

    .. _Soft-DTW\: a Differentiable Loss Function for Time-Series:
        https://proceedings.mlr.press/v70/cuturi17a.html
    .. _Parallelizing Dynamic Time Warping Algorithm Using Prefix Computations on GPU:
        https://doi.org/10.1109/HPCC.and.EUC.2013.50
    """
    _validate("dtw", cost, step_pattern, diagonal_weight, lengths)
    return _dtw(cost, gamma, step_pattern, diagonal_weight, lengths, band)[0]


def _validate(
    name: str,
    cost: Tensor,
    step_pattern: str,
    diagonal_weight: float,
    lengths: tuple[Tensor, Tensor] | None,
) -> None:
    if cost.dim() != 3:
        raise ValueError(f"cost must be (B, N, M), got {tuple(cost.shape)}")
    if step_pattern not in ("symmetric", "asymmetric", "orthogonal"):
        raise ValueError(f"unknown step_pattern {step_pattern!r}")
    B, N, M = cost.shape
    if N == 0 or M == 0:
        raise ValueError(f"cost has an empty sequence: shape {tuple(cost.shape)}")
    if diagonal_weight != 1.0 and step_pattern != "symmetric":
        raise ValueError(f"diagonal_weight needs the symmetric steps, not {step_pattern!r}.")
    if lengths is not None:
        n_len, m_len = lengths
        if n_len.shape != (B,) or m_len.shape != (B,):
            raise ValueError(f"lengths must be two tensors of shape ({B},)")
        if bool((n_len < 1).any() | (n_len > N).any() | (m_len < 1).any() | (m_len > M).any()):
            raise ValueError(f"lengths must be within 1 to {N} and 1 to {M}")
    check_cuda_triton(name, cost)


def _dtw(cost, gamma, step_pattern, diagonal_weight, lengths, band):
    """dtw's distances, and for dtw_path the kernels' grid: (distance, grid), the
    grid None if no pair has a path, else (D, its costs, steps, each pair's last
    cell on it (B, 2), whether it was sheared, whether then transposed)."""
    # Imported here so that philtorch.align imports without Triton.
    from ._dtw_kernels import dtw_dp

    B, N, M = cost.shape
    if lengths is None:
        n_len = torch.full((B,), N, device=cost.device)
        m_len = torch.full((B,), M, device=cost.device)
    else:
        n_len, m_len = (t.to(cost.device, torch.long) for t in lengths)
    # The kernels' soft-min is in base 2: costs in units of gamma / log2(e).
    scale = 1.0 / (gamma * math.log(2)) if gamma > 0 else 1.0
    cost = cost * scale
    if band is not None:
        # A large finite cost, not inf, keeps soft-min's arithmetic finite; a
        # path sums at most w (N + M) of them.
        big = torch.finfo(cost.dtype).max / (8 * max(diagonal_weight, 1.0) * (N + M))
        cost = torch.where(_band_mask(N, M, n_len, m_len, band), cost, big)

    no_path = torch.zeros(B, dtype=torch.bool, device=cost.device)
    sheared = step_pattern == "asymmetric"
    if sheared:
        no_path = n_len < m_len
        width = int((n_len - m_len).max()) + 1 if B else 1
        if width <= 0:
            # No pair has a path; stay in the graph, with zero gradients.
            return torch.where(no_path, float("inf"), cost[:, 0, 0]), None
        cost = _shear(cost, width)
        n_len, m_len = m_len, (n_len - m_len + 1).clamp(min=1)
    transposed = cost.size(1) > cost.size(2)
    if transposed:
        cost, n_len, m_len = cost.mT, m_len, n_len
    # After the shear, the asymmetric steps are orthogonal ones.
    steps = "symmetric" if step_pattern == "symmetric" else "orthogonal"
    D = dtw_dp(cost, None, gamma > 0, steps, float(diagonal_weight))
    distance = D[torch.arange(B, device=D.device), n_len - 1, m_len - 1]
    if band is not None:
        no_path = no_path | (distance >= big / 2)
    distance = torch.where(no_path, float("inf"), distance) / scale
    ends = torch.stack([n_len, m_len], -1) - 1
    return distance, (D, cost, steps, ends, sheared, transposed)


def dtw_path(
    cost: Tensor,
    *,
    step_pattern: StepPattern = "symmetric",
    diagonal_weight: float = 1.0,
    lengths: tuple[Tensor, Tensor] | None = None,
    band: float | None = None,
) -> tuple[Tensor, Tensor]:
    r"""The DTW distance and the optimal warping path between pairs of sequences.

    The path is DTW's (:func:`dtw` with :math:`\gamma = 0`) as index pairs
    :math:`(n, m)` from :math:`(0, 0)` to :math:`(N_b - 1, M_b - 1)`, as
    Viterbi decoding gives a hidden Markov model's states: backtracked over
    the forward pass's accumulated costs, one step per cell of the path, the
    same path as the distance's gradient marks. Under ties, it is one of the
    optimal paths. The distance is
    :func:`dtw`'s, differentiable to any order, its gradient the path.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        cost (Tensor): the costs :math:`c[n, m]`, of shape :math:`(B, N, M)`.
        step_pattern (str): as in :func:`dtw`.
        diagonal_weight (float): as in :func:`dtw`.
        lengths (tuple of Tensor, optional): as in :func:`dtw`.
        band (float, optional): as in :func:`dtw`.

    Returns:
        tuple of Tensor: the distances, of shape :math:`(B)`, ``inf`` for a
        pair with no path, and the paths, an integer tensor of shape
        :math:`(B, N + M - 1, 2)` of the cells in order, padded with -1 past
        each pair's path, all -1 for a pair with none.

    Raises:
        ValueError: as :func:`dtw` does.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import dtw_path
        >>> x = torch.tensor([[0.0, 1.0, 2.0]], device="cuda")
        >>> y = torch.tensor([[0.0, 0.0, 1.0, 2.0]], device="cuda")
        >>> distance, path = dtw_path((x[..., None] - y[:, None]) ** 2)
        >>> distance
        tensor([0.], device='cuda:0')
        >>> path[0]  # the cells, (0, 0) first, then -1 past the path
        tensor([[ 0,  0],
                [ 0,  1],
                [ 1,  2],
                [ 2,  3],
                [-1, -1],
                [-1, -1]], device='cuda:0')
    """
    _validate("dtw_path", cost, step_pattern, diagonal_weight, lengths)
    from ._dtw_kernels import backtrack

    B, N, M = cost.shape
    distance, grid = _dtw(cost, 0.0, step_pattern, diagonal_weight, lengths, band)
    path = torch.full((B, N + M - 1, 2), -1, dtype=torch.long, device=cost.device)
    if grid is None:
        return distance, path
    D, grid_cost, steps, ends, sheared, transposed = grid
    with torch.no_grad():
        cells = backtrack(D, grid_cost, ends, steps, float(diagonal_weight))
        # From the kernels' grid back to (n, m): undo the transpose, then the
        # shear, sheared[m, k] = cost[m + k, m].
        if transposed:
            cells = cells.flip(-1)
        if sheared:
            cells = torch.stack([cells[..., 0] + cells[..., 1], cells[..., 0]], -1)
        valid = (cells[..., :1] >= 0) & distance.isfinite()[:, None, None]
        path[:, : cells.size(1)] = torch.where(valid, cells, -1)
    return distance, path


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

    from `Differentiable Divergences Between Time Series`_ (Blondel, Mensch
    and Vert, 2021). Unlike soft-DTW itself, it is zero for identical
    sequences, and with the squared Euclidean cost it is non-negative.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

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

    Raises:
        ValueError: if :attr:`gamma` is not positive, if :attr:`cost_xx`
            and :attr:`cost_yy` are not of shapes :math:`(B, N, N)` and
            :math:`(B, M, M)`, or as :func:`dtw` does.
        RuntimeError: if Triton is not installed.

    .. _Differentiable Divergences Between Time Series:
        https://proceedings.mlr.press/v130/blondel21a.html
    """
    if not gamma > 0:
        raise ValueError(f"the soft-DTW divergence needs gamma > 0, got {gamma}")
    if cost_xy.dim() == 3:
        B, N, M = cost_xy.shape
        if cost_xx.shape != (B, N, N) or cost_yy.shape != (B, M, M):
            raise ValueError(
                f"cost_xx and cost_yy must be ({B}, {N}, {N}) and ({B}, {M}, {M}), "
                f"got {tuple(cost_xx.shape)} and {tuple(cost_yy.shape)}"
            )
    n_len, m_len = (None, None) if lengths is None else lengths
    xx = None if lengths is None else (n_len, n_len)
    yy = None if lengths is None else (m_len, m_len)
    return dtw(cost_xy, gamma, lengths=lengths, **kwargs) - 0.5 * (
        dtw(cost_xx, gamma, lengths=xx, **kwargs) + dtw(cost_yy, gamma, lengths=yy, **kwargs)
    )
