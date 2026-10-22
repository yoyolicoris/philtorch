"""Dynamic time warping (DTW) and soft-DTW."""

import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import pad

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


def _band_windows(N: int, M: int, n_len: Tensor, m_len: Tensor, band: float) -> Tensor:
    """Each row's allowed columns, (B, N, 2) int32 [lo, hi], lo > hi if none.

    The cells within ``band`` cells of the straight line between each pair's
    corners. The distance is measured along the longer side, and the band is
    widened where needed for consecutive rows (or columns) of it to overlap,
    so that it always holds a path.
    """
    i = torch.arange(N, device=n_len.device, dtype=torch.float64)
    n = (n_len - 1).double()[:, None]
    m = (m_len - 1).double()[:, None]
    rows_longer = n > m
    # Along the longer side, the line advances `slope` cells per cell of the other.
    slope = torch.where(rows_longer, n, m) / torch.where(rows_longer, m, n).clamp(min=1)
    reach = torch.maximum(slope / 2, torch.full_like(slope, band))  # slope / 2: a step between rows
    one_line = torch.minimum(n, m) == 0  # a single row or column: no band to keep

    def inside(j):
        offset = torch.where(rows_longer, i - j * slope, j - i * slope).abs()
        return ((offset <= reach) | one_line) & (j >= 0) & (j < M)

    # In closed form, then corrected by the test itself against rounding.
    lo = torch.where(rows_longer, (i - reach) / slope, i * slope - reach).ceil()
    hi = torch.where(rows_longer, (i + reach) / slope, i * slope + reach).floor()
    lo = torch.where(one_line, 0, lo.clamp(0, M))
    hi = torch.where(one_line, M - 1, hi.clamp(-1, M - 1))
    lo = torch.where(inside(lo - 1), lo - 1, lo + (~inside(lo)).double())
    hi = torch.where(inside(hi + 1), hi + 1, hi - (~inside(hi)).double())
    return torch.stack([lo, hi], -1).int()


def _band_mask(N: int, M: int, n_len: Tensor, m_len: Tensor, band: float) -> Tensor:
    """The cells of the windows, (B, N, M)."""
    window = _band_windows(N, M, n_len, m_len, band)
    j = torch.arange(M, device=n_len.device)
    return (j >= window[..., :1]) & (j <= window[..., 1:])


def _check_features(name: str, cost: Tensor | None, x: Tensor | None, y: Tensor | None) -> None:
    """Checks that there is either the ``cost`` or ``x`` and ``y``, as frames."""
    if (x is None) != (y is None):
        raise ValueError(f"{name} needs x and y together")
    if (cost is None) == (x is None):
        raise ValueError(f"{name} needs either the cost or x and y, not both or neither")
    if cost is not None:
        return
    if x.dim() != 3 or y.dim() != 3 or x.shape[::2] != y.shape[::2]:
        raise ValueError(
            f"x and y must be (B, N, F) and (B, M, F), got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    if x.dtype != y.dtype or not x.is_floating_point():
        raise ValueError(f"x and y must have one floating dtype, got {x.dtype} and {y.dtype}")
    check_cuda_triton(name, x, y)


def dtw(
    cost: Tensor | None = None,
    gamma: float = 0.0,
    *,
    x: Tensor | None = None,
    y: Tensor | None = None,
    step_pattern: StepPattern = "symmetric",
    diagonal_weight: float = 1.0,
    lengths: tuple[Tensor, Tensor] | None = None,
    band: float | None = None,
) -> Tensor:
    r"""The DTW or soft-DTW distance between pairs of sequences, from their costs.

    For a cost matrix :math:`c[n, m]` of matching frame :math:`n` of one
    sequence with frame :math:`m` of another, the accumulated costs

    .. math::
        D[n, m] = \min_\gamma\big(D[n - 1, m] + c[n, m],\
        D[n, m - 1] + c[n, m],\ D[n - 1, m - 1] + w\, c[n, m]\big),

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
    2013), in tiles of the grid run as a wavefront, so that a single long
    pair fills the GPU: :math:`O(NM)` work, and the memory of the cost and
    the accumulated costs, or from frames, without gradients,
    :math:`O(B (N + M))`. The cost can be any
    tensor, from any differentiable function of the sequences, such as the
    squared Euclidean distances between their frames.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        cost (Tensor, optional): the costs :math:`c[n, m]`, of shape
            :math:`(B, N, M)`. Give either it or both :attr:`x` and :attr:`y`.
        gamma (float): 0 for DTW, positive for soft-DTW. Default: 0.
        x (Tensor, optional): the first sequences' frames, of shape
            :math:`(B, N, F)`. With :attr:`y`, the cost is the squared
            Euclidean distances between their frames, computed in the kernels
            from the frames as direct differences, never stored (memory: the
            accumulated costs only; without gradients not even those, but
            :math:`O(B (N + M))`).
        y (Tensor, optional): the second sequences' frames, of shape
            :math:`(B, M, F)`, with the dtype of :attr:`x`.
        step_pattern (str): the steps a path may take, as (row, column)
            increments: ``"symmetric"`` for (1, 0), (0, 1) and (1, 1);
            ``"asymmetric"`` for (1, 0) and (1, 1) only, so that every frame of
            the first sequence matches exactly one frame of the second
            (needs :math:`N \ge M`); ``"orthogonal"`` for (1, 0) and (0, 1)
            only, so that every path has :math:`N + M - 1` cells. Default:
            ``"symmetric"``.
        diagonal_weight (float): :math:`w`, the weight of a diagonal step's
            cost in the symmetric steps. 1 favors diagonal paths, which sum
            fewer costs; 2, Sakoe and Chiba's symmetric form ("symmetric2" in
            Giorgino's dtw package), weighs each step's cost by how far it
            advances, :math:`\Delta n + \Delta m`, so that every path has the
            same total weight and diagonal steps are not favored. Default: 1.
        lengths (tuple of Tensor, optional): each pair's lengths
            :math:`(N_b, M_b)`, two integer tensors of shape :math:`(B)`, for
            batches of sequences padded to :math:`N` and :math:`M`; the
            distance is then :math:`D[N_b - 1, M_b - 1]`. Padding costs don't
            affect it. With the asymmetric steps, their values size the
            computation, which breaks a :func:`torch.compile` graph. Default:
            the full lengths.
        band (float, optional): a Sakoe-Chiba band: only cells within this
            many cells of the straight line from :math:`(0, 0)` to
            :math:`(N_b - 1, M_b - 1)`, measured along the longer sequence,
            may be on a path. Where the line is steeper than the band allows
            a step to follow, the band widens to keep a path. Default: no band.

    Returns:
        Tensor: the distances, of shape :math:`(B)`; ``inf`` for a pair with
        no path, such as the asymmetric steps with :math:`N_b < M_b`.

    Raises:
        ValueError: if not exactly one of :attr:`cost` and (:attr:`x`,
            :attr:`y`) is given, if :attr:`x` and :attr:`y` are not CUDA
            tensors of shapes :math:`(B, N, F)` and :math:`(B, M, F)` of one
            floating dtype, if :attr:`cost` is not a CUDA tensor of shape
            :math:`(B, N, M)` with :math:`N, M > 0` and at most
            :math:`2^{31} - 1` cells per pair, if :attr:`step_pattern` is
            unknown, if :attr:`diagonal_weight` is not 1 with steps that have
            no weighted diagonal, or if :attr:`lengths` are not of shape
            :math:`(B)` within :math:`1 \le N_b \le N` and
            :math:`1 \le M_b \le M`; the lengths' values are checked only in
            eager mode, as they would break a :func:`torch.compile` graph.
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
        >>> same = dtw(x=x, y=y, gamma=0.1)  # the same cost, from the frames

    .. _Soft-DTW\: a Differentiable Loss Function for Time-Series:
        https://proceedings.mlr.press/v70/cuturi17a.html
    .. _Parallelizing Dynamic Time Warping Algorithm Using Prefix Computations on GPU:
        https://doi.org/10.1109/HPCC.and.EUC.2013.50
    """
    _validate("dtw", cost, x, y, step_pattern, diagonal_weight, lengths)
    return _dtw(cost, x, y, gamma, step_pattern, diagonal_weight, lengths, band)[0]


def _validate(
    name: str,
    cost: Tensor | None,
    x: Tensor | None,
    y: Tensor | None,
    step_pattern: str,
    diagonal_weight: float,
    lengths: tuple[Tensor, Tensor] | None,
) -> tuple[int, int, int]:
    """Checks the arguments; returns the shape (B, N, M) of the costs."""
    _check_features(name, cost, x, y)
    if cost is None:
        shape = (x.size(0), x.size(1), y.size(1))
    elif cost.dim() != 3:
        raise ValueError(f"cost must be (B, N, M), got {tuple(cost.shape)}")
    else:
        shape = tuple(cost.shape)
    if step_pattern not in ("symmetric", "asymmetric", "orthogonal"):
        raise ValueError(f"unknown step_pattern {step_pattern!r}")
    B, N, M = shape
    if N == 0 or M == 0:
        raise ValueError(f"cost has an empty sequence: shape {shape}")
    if diagonal_weight != 1.0 and step_pattern != "symmetric":
        raise ValueError(f"diagonal_weight needs the symmetric steps, not {step_pattern!r}.")
    if lengths is not None:
        n_len, m_len = lengths
        if n_len.shape != (B,) or m_len.shape != (B,):
            raise ValueError(f"lengths must be two tensors of shape ({B},)")
        # In eager mode only: the values would break a torch.compile graph.
        if not torch.compiler.is_compiling() and bool(
            (n_len < 1).any() | (n_len > N).any() | (m_len < 1).any() | (m_len > M).any()
        ):
            raise ValueError(f"lengths must be within 1 to {N} and 1 to {M}")
    if cost is not None:
        check_cuda_triton(name, cost)
    return shape


class _Grid(NamedTuple):
    """What the kernels run on: the grid's costs (B, R, L) as the tensor
    ``cost``, or in feature mode (None) as ``window``, ``big`` and ``scale``
    over x and y; the steps; whether the grid is sheared, then transposed;
    each pair's end cell (B, 2); and the pairs known to have no path."""

    cost: Tensor | None
    window: Tensor | None
    big: float
    scale: float
    steps: str
    diag_weight: float
    sheared: bool
    transposed: bool
    R: int
    L: int
    ends: Tensor
    no_path: Tensor


def _grid(cost, x, y, gamma, step_pattern, diagonal_weight, lengths, band) -> _Grid | None:
    """The kernels' grid of the cost, or of the frames x and y, which they
    compare in place; None if no pair has a path."""
    features = cost is None
    ref = x if features else cost
    B, N = ref.shape[:2]
    M = y.size(1) if features else cost.size(2)
    if lengths is None:
        n_len = torch.full((B,), N, device=ref.device)
        m_len = torch.full((B,), M, device=ref.device)
    else:
        n_len, m_len = (t.to(ref.device, torch.long) for t in lengths)
    # The kernels' soft-min is in base 2: costs in units of gamma / log2(e).
    scale = 1.0 / (gamma * math.log(2)) if gamma > 0 else 1.0
    # A large finite cost outside the band, not inf, keeps soft-min's
    # arithmetic finite; a path sums at most w (N + M) of them.
    dtype = ref.dtype if ref.is_floating_point() else torch.get_default_dtype()  # cost * scale's
    big = torch.finfo(dtype).max / (8 * max(diagonal_weight, 1.0) * (N + M))
    window = None
    if features:
        if band is not None:
            window = _band_windows(N, M, n_len, m_len, band)
    else:
        cost = cost * scale
        if band is not None:
            cost = torch.where(_band_mask(N, M, n_len, m_len, band), cost, big)

    no_path = torch.zeros(B, dtype=torch.bool, device=ref.device)
    sheared = step_pattern == "asymmetric"
    R, L = N, M
    if sheared:
        no_path = n_len < m_len
        # From the shapes without lengths; from the lengths' values otherwise,
        # which breaks a torch.compile graph.
        if lengths is None:
            width = N - M + 1
        else:
            width = int((n_len - m_len).max()) + 1 if B else 1
        if width <= 0:
            return None
        R, L = M, width
        n_len, m_len = m_len, (n_len - m_len + 1).clamp(min=1)
    transposed = R > L
    if transposed:
        R, L, n_len, m_len = L, R, m_len, n_len
    if not features:
        if sheared:
            cost = _shear(cost, width)
        if transposed:
            cost = cost.mT
    # After the shear, the asymmetric steps are orthogonal ones.
    steps = "symmetric" if step_pattern == "symmetric" else "orthogonal"
    ends = torch.stack([n_len, m_len], -1) - 1
    return _Grid(
        cost, window, big, scale, steps, float(diagonal_weight), sheared, transposed, R, L, ends,
        no_path,
    )  # fmt: skip


def _dtw(cost, x, y, gamma, step_pattern, diagonal_weight, lengths, band):
    """dtw's distances, the kernels' grid (None if no pair has a path), and,
    from the cost, its accumulated costs (else None).

    From frames, without gradients, each pair's grid is advanced a row at a
    time up to its end cell and the accumulated costs are never stored."""
    # Imported here so that philtorch.align imports without Triton.
    from ._dtw_kernels import dtw_dp, dtw_dp_features, frontier

    features = cost is None
    if features and lengths is not None:
        # Frames past each pair's length as zeros: any value there, inf
        # included, would reach the gradients' products.
        x, y = (
            torch.where((torch.arange(t.size(1), device=t.device) < n[:, None])[..., None], t, 0)
            for t, n in zip((x, y), (n.to(x.device) for n in lengths))
        )
    g = _grid(cost, x, y, gamma, step_pattern, diagonal_weight, lengths, band)
    if g is None:
        # No pair has a path; stay in the graph, with zero gradients.
        first = x.sum((1, 2)) + y.sum((1, 2)) if features else cost[:, 0, 0]
        return torch.where(torch.ones_like(first, dtype=torch.bool), float("inf"), first), g, None
    batch = torch.arange(g.ends.size(0), device=g.ends.device)
    end_row, end_col = g.ends.unbind(-1)
    soft = gamma > 0
    D = None
    if not features:
        D = dtw_dp(g.cost, None, soft, g.steps, g.diag_weight)
        distance = D[batch, end_row, end_col]
    elif torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
        distance = dtw_dp_features(
            x, y, g.window, g.big, g.scale, soft, g.steps, g.diag_weight, g.sheared, g.transposed,
            g.R, g.L,
        )[batch, end_row, end_col]  # fmt: skip
    else:
        zero = torch.zeros_like(batch)
        items = torch.stack([batch, zero, zero, end_row + 1, end_col + 1], -1)
        row = frontier(
            x, y, g.window, g.big, g.scale, items, soft, g.steps, g.diag_weight, g.sheared,
            g.transposed, False, g.R, g.L,
        )  # fmt: skip
        distance = row.gather(1, end_col[:, None])[:, 0]
    no_path = g.no_path
    if band is not None:
        no_path = no_path | (distance >= g.big / 2)
    return torch.where(no_path, float("inf"), distance) / g.scale, g, D


# The linear-memory path: items of at most this many cells, or this many rows,
# are backtracked whole, in chunks of at most this many cells in all.
_LEAF_CELLS = 2**22
_LEAF_ROWS = 64
_CHUNK_CELLS = 2**26


def _row_costs(g: _Grid, x, y, p, r, col):
    """The costs of the grid's cells (r, col) of pairs p, as the kernels compute them."""
    a, s = (col, r) if g.transposed else (r, col)
    n = (a + s if g.sheared else a).clamp(max=x.size(1) - 1)
    m = (a if g.sheared else s).clamp(max=y.size(1) - 1)
    d = x[p, n] - y[p, m]
    c = (d * d).sum(-1) * g.scale
    if g.window is not None:
        lo, hi = g.window[p, n].unbind(-1)
        c = torch.where((m < lo) | (m > hi), g.big, c)
    return c


def _lines(items: Tensor) -> Tensor:
    """The cells (pair, row, column) of items of one row or column: a straight path."""
    p, r0, l0, h, w = items.unbind(-1)
    count = h + w - 1
    item = torch.repeat_interleave(torch.arange(len(items), device=items.device), count)
    k = torch.arange(len(item), device=items.device) - (count.cumsum(0) - count)[item]
    return torch.stack([p[item], r0[item] + k * (w[item] == 1), l0[item] + k * (h[item] == 1)], -1)


def _leaves(g: _Grid, x, y, items: Tensor) -> Tensor:
    """The cells (pair, row, column) of the optimal path of each item: the item
    as a grid of its own, padded to the largest of the batch, on the frames it
    needs: with a shear, the grid's cells (r, l) are x's frame r + l and y's r
    (l if transposed); else x's r and y's l (swapped if transposed)."""
    from ._dtw_kernels import backtrack_features, dtw_dp_features

    p, r0, l0, h, w = items.unbind(-1)
    if g.sheared:
        x0, nx = r0 + l0, h + w - 1
        y0, ny = (l0, w) if g.transposed else (r0, h)
    elif g.transposed:
        x0, nx, y0, ny = l0, w, r0, h
    else:
        x0, nx, y0, ny = r0, h, l0, w

    def frames(f, start, count):
        at = start[:, None] + torch.arange(int(count.max()), device=f.device)
        at_clamped = at.clamp(max=f.size(1) - 1)
        live = (at - start[:, None]) < count[:, None]
        return torch.where(live[..., None], f[p[:, None], at_clamped], 0.0), at_clamped

    fx, ix = frames(x, x0, nx)
    fy, _ = frames(y, y0, ny)
    window = None if g.window is None else g.window[p[:, None], ix] - y0[:, None, None].int()
    given = (fx, fy, window, g.big, g.scale)
    options = (g.steps, g.diag_weight, g.sheared, g.transposed)
    D = dtw_dp_features(*given, False, *options, int(h.max()), int(w.max()))
    cells = backtrack_features(D, *given, torch.stack([h - 1, w - 1], -1), *options)
    live = cells[..., 0] >= 0
    shift = torch.stack([p, r0, l0], -1)[:, None]
    return (shift + pad(cells, (1, 0)))[live]


def _split(g: _Grid, x, y, items: Tensor) -> Tensor:
    """Each item as two, split by the row after its middle row's crossing.

    Frontiers give the best cost to each cell of the middle row, from the
    item's first cell, and from each cell of the next row, to its last; the
    cheapest step across, up or diagonal, is on an optimal path."""
    from ._dtw_kernels import frontier

    p, r0, l0, h, w = items.unbind(-1)
    top = h // 2
    below = r0 + top  # the first row of the second half
    given = (x, y, g.window, g.big, g.scale)
    options = (False, g.steps, g.diag_weight, g.sheared, g.transposed)
    h_max, w_max = (int(t.max()) for t in (h - h // 2, w))
    D = frontier(*given, torch.stack([p, r0, l0, top, w], -1), *options, False, h_max, w_max)
    G = frontier(*given, torch.stack([p, below, l0, h - top, w], -1), *options, True, h_max, w_max)
    j = torch.arange(D.size(1), device=x.device)
    c = _row_costs(g, x, y, p[:, None], below[:, None], l0[:, None] + j)
    live = j < w[:, None]
    across = [torch.where(live, D + c + G, torch.inf)]
    if g.steps == "symmetric":
        diagonal = D[:, :-1] + g.diag_weight * c[:, 1:] + G[:, 1:]
        across.append(pad(torch.where(live[:, 1:], diagonal, torch.inf), (0, 1), value=torch.inf))
    # The first cheapest: the cell of the middle row, and the step from it.
    best = torch.stack(across, -1).flatten(1).argmin(1)
    column = best // len(across)
    next_column = column + best % len(across)
    return torch.cat(
        [
            torch.stack([p, r0, l0, top, column + 1], -1),
            torch.stack([p, below, l0 + next_column, h - top, w - next_column], -1),
        ]
    )


def _linear_path(g: _Grid, x, y, finite: Tensor) -> Tensor:
    """Each pair's optimal path on the kernels' grid, as :func:`backtrack` gives
    it, in memory linear in the grid's sides: Hirschberg's divide and conquer.

    An item, a rectangle of the grid between two cells of the path, is a
    straight line, a leaf backtracked whole, or split in two across its middle
    rows (:func:`_split`). Each round handles all items together."""
    B = g.ends.size(0)
    pairs = finite.nonzero()[:, 0]
    end_row, end_col = g.ends[pairs].unbind(-1)
    zero = torch.zeros_like(pairs)
    work = torch.stack([pairs, zero, zero, end_row + 1, end_col + 1], -1)
    found = [work.new_empty(0, 3)]
    while len(work):
        h, w = work[:, 3], work[:, 4]
        line = (h == 1) | (w == 1)
        leaf = ~line & ((h * w <= _LEAF_CELLS) | (h <= _LEAF_ROWS))
        found.append(_lines(work[line]))
        # The leaves by size, in chunks.
        leaves = work[leaf]
        leaves = leaves[(leaves[:, 3] * leaves[:, 4]).argsort(descending=True)]
        start, H, W = 0, 0, 0
        for i, (h, w) in enumerate(leaves[:, 3:].tolist()):
            if i > start and (i + 1 - start) * max(H, h) * max(W, w) > _CHUNK_CELLS:
                found.append(_leaves(g, x, y, leaves[start:i]))
                start, H, W = i, 0, 0
            H, W = max(H, h), max(W, w)
        if len(leaves):
            found.append(_leaves(g, x, y, leaves[start:]))
        splits = work[~(line | leaf)]
        work = _split(g, x, y, splits) if len(splits) else splits
    # Each pair's cells in path order: by row, then column.
    cells = torch.cat(found)
    cells = cells[((cells[:, 0] * g.R + cells[:, 1]) * g.L + cells[:, 2]).argsort()]
    count = torch.bincount(cells[:, 0], minlength=B)
    at = torch.arange(len(cells), device=x.device) - (count.cumsum(0) - count)[cells[:, 0]]
    walk = cells.new_full((B, g.R + g.L - 1, 2), -1)
    walk[cells[:, 0], at] = cells[:, 1:]
    return walk


def dtw_path(
    cost: Tensor | None = None,
    *,
    x: Tensor | None = None,
    y: Tensor | None = None,
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

    From :attr:`x` and :attr:`y`, the path is found by divide and conquer, in
    :math:`O(B (N + M))` memory and a few times the forward work: the best
    crossing of a rectangle's middle rows splits it in two between cells of
    the path, down to rectangles backtracked whole.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        cost (Tensor, optional): as in :func:`dtw`.
        x (Tensor, optional): as in :func:`dtw`.
        y (Tensor, optional): as in :func:`dtw`.
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
    B, N, M = _validate("dtw_path", cost, x, y, step_pattern, diagonal_weight, lengths)
    distance, g, D = _dtw(cost, x, y, 0.0, step_pattern, diagonal_weight, lengths, band)
    path = torch.full((B, N + M - 1, 2), -1, dtype=torch.long, device=distance.device)
    if g is None:
        return distance, path
    with torch.no_grad():
        if cost is None:
            cells = _linear_path(g, x, y, distance.isfinite())
        else:
            from ._dtw_kernels import backtrack

            cells = backtrack(D, g.cost, g.ends, g.steps, g.diag_weight)
        # From the kernels' grid back to (n, m): undo the transpose, then the
        # shear, sheared[m, k] = cost[m + k, m].
        if g.transposed:
            cells = cells.flip(-1)
        if g.sheared:
            cells = torch.stack([cells[..., 0] + cells[..., 1], cells[..., 0]], -1)
        valid = (cells[..., :1] >= 0) & distance.isfinite()[:, None, None]
        path[:, : cells.size(1)] = torch.where(valid, cells, -1)
    return distance, path


def soft_dtw_divergence(
    cost_xy: Tensor | None = None,
    cost_xx: Tensor | None = None,
    cost_yy: Tensor | None = None,
    gamma: float | None = None,
    *,
    x: Tensor | None = None,
    y: Tensor | None = None,
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
        cost_xy (Tensor, optional): costs between the sequences, of shape
            :math:`(B, N, M)`.
        cost_xx (Tensor, optional): costs of the first sequences with
            themselves, of shape :math:`(B, N, N)`.
        cost_yy (Tensor, optional): costs of the second sequences with
            themselves, of shape :math:`(B, M, M)`.
        gamma (float): the soft-min temperature, positive. Required.
        x (Tensor, optional): the first sequences' frames, of shape
            :math:`(B, N, F)`. Give either the three costs or both :attr:`x`
            and :attr:`y`, from which they are the squared Euclidean
            distances, as in :func:`dtw`.
        y (Tensor, optional): the second sequences' frames, of shape
            :math:`(B, M, F)`.
        lengths (tuple of Tensor, optional): each pair's lengths
            :math:`(N_b, M_b)`, as in :func:`dtw`.
        **kwargs: the other keyword arguments of :func:`dtw`.

    Returns:
        Tensor: the divergences, of shape :math:`(B)`.

    Raises:
        ValueError: if :attr:`gamma` is missing or not positive, if the three
            costs and (:attr:`x`, :attr:`y`) are both given or neither, if
            :attr:`cost_xx` and :attr:`cost_yy` are not of shapes
            :math:`(B, N, N)` and :math:`(B, M, M)`, or as :func:`dtw` does.
        RuntimeError: if Triton is not installed.

    .. _Differentiable Divergences Between Time Series:
        https://proceedings.mlr.press/v130/blondel21a.html
    """
    if gamma is None or not gamma > 0:
        raise ValueError(f"the soft-DTW divergence needs gamma > 0, got {gamma}")
    name = "soft_dtw_divergence"
    _check_features(name, cost_xy, x, y)
    _check_features(name, cost_xx, x, x)
    _check_features(name, cost_yy, y, y)
    n_len, m_len = (None, None) if lengths is None else lengths
    if x is None:
        if cost_xy.dim() == 3:
            B, N, M = cost_xy.shape
            if cost_xx.shape != (B, N, N) or cost_yy.shape != (B, M, M):
                raise ValueError(
                    f"cost_xx and cost_yy must be ({B}, {N}, {N}) and ({B}, {M}, {M}), "
                    f"got {tuple(cost_xx.shape)} and {tuple(cost_yy.shape)}"
                )
        inputs = (dict(cost=cost_xy), dict(cost=cost_xx), dict(cost=cost_yy))
    else:
        inputs = (dict(x=x, y=y), dict(x=x, y=x), dict(x=y, y=y))
    xx = None if lengths is None else (n_len, n_len)
    yy = None if lengths is None else (m_len, m_len)
    d_xy, d_xx, d_yy = (
        dtw(gamma=gamma, lengths=ls, **kw, **kwargs) for kw, ls in zip(inputs, (lengths, xx, yy))
    )
    return d_xy - 0.5 * (d_xx + d_yy)
