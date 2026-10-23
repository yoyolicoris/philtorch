"""Dynamic time warping (DTW) and soft-DTW."""

import math
from typing import Literal, NamedTuple

import torch
import torch.utils.checkpoint
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


def _band_windows(
    N: int, M: int, n_len: Tensor, m_len: Tensor, band: float, orthogonal: bool
) -> Tensor:
    """Each row's allowed columns, (B, N, 2) int32 [lo, hi], lo > hi if none.

    The cells within ``band`` cells of the straight line between each pair's
    corners. The distance is measured along the longer side, and the band is
    widened where needed so that it always holds a path: for consecutive rows
    (or columns) of it to touch, or, with the ``orthogonal`` steps, to overlap.
    """
    i = torch.arange(N, device=n_len.device, dtype=torch.float64)
    n = (n_len - 1).double()[:, None]
    m = (m_len - 1).double()[:, None]
    rows_longer = n > m
    # Along the longer side, the line advances `slope` cells per other cell;
    # a step between rows reaches slope / 2, and an orthogonal one, which
    # keeps its column, (slope + 1) / 2. Scaled by the shorter side, the offset
    # of (i, j) is |i m - j n|, an integer: exact, however compiled.
    longer, shorter = torch.maximum(n, m), torch.minimum(n, m)
    slope = longer / shorter.clamp(min=1)
    reach = torch.maximum((slope + orthogonal) / 2, torch.full_like(slope, band))
    twice = torch.maximum(longer + orthogonal * shorter, 2 * band * shorter)
    one_line = shorter == 0  # a row or column: no band to keep

    def inside(j):
        offset = 2 * (i * m - j * n).abs()
        return ((offset <= twice) | one_line) & (j >= 0) & (j < M)

    # In closed form, then corrected by the exact test.
    lo = torch.where(rows_longer, (i - reach) / slope, i * slope - reach).ceil()
    hi = torch.where(rows_longer, (i + reach) / slope, i * slope + reach).floor()
    lo = torch.where(one_line, 0, lo.clamp(0, M))
    hi = torch.where(one_line, M - 1, hi.clamp(-1, M - 1))
    lo = torch.where(inside(lo - 1), lo - 1, lo + (~inside(lo)).double())
    hi = torch.where(inside(hi + 1), hi + 1, hi - (~inside(hi)).double())
    return torch.stack([lo, hi], -1).int()


def _check_features(name: str, cost: Tensor | None, x: Tensor | None, y: Tensor | None) -> None:
    """Check that there is either the cost or frames x and y, and the frames."""
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
    checkpoint: int | None = None,
) -> Tensor:
    r"""Compute the DTW or soft-DTW distance between pairs of sequences.

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
    cell entered diagonally counts :math:`w` times. From a cost, both come
    from autograd, with no backtracking. The gradient is itself
    differentiable, to any order. Costs may be ``inf`` to forbid cells.

    Rows of the accumulated costs are computed one after another, each with
    a parallel scan over its cells, after `Parallelizing Dynamic Time
    Warping Algorithm Using Prefix Computations on GPU`_ (Xiao et al.,
    2013), in tiles of the grid run as a wavefront, so that a single long
    pair fills the GPU: :math:`O(BNM)` work. From a cost, the memory is that
    of the cost and the accumulated costs; from frames, :math:`O(B (N + M))`,
    but for soft-DTW's gradient that of the accumulated costs (fewer of them
    with :attr:`checkpoint`). The cost can be any tensor, from any
    differentiable function of the sequences, such as the squared Euclidean
    distances between their frames.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        cost (Tensor, optional): the costs :math:`c[n, m]`, of shape
            :math:`(B, N, M)`. Give either it or both :attr:`x` and :attr:`y`.
        gamma (float, optional): the soft-min temperature :math:`\gamma`: 0 for
            DTW, positive for soft-DTW. Default: 0.
        x (Tensor, optional): the first sequences' frames, of shape
            :math:`(B, N, F)`. With :attr:`y`, the cost is the squared
            Euclidean distances between their frames, computed in the kernels
            from direct differences and never stored, but for soft-DTW's
            second derivatives, which rebuild them. DTW's gradient is then
            that of its path's cost, the path found by divide and conquer as
            :func:`dtw_path` finds it.
        y (Tensor, optional): the second sequences' frames, of shape
            :math:`(B, M, F)`, with the dtype of :attr:`x`.
        step_pattern (str, optional): the steps a path may take, as (row,
            column) increments: ``"symmetric"`` for (1, 0), (0, 1) and (1, 1);
            ``"asymmetric"`` for (1, 0) and (1, 1) only, so that every frame of
            the first sequence matches exactly one frame of the second (a
            pair with :math:`N_b < M_b` has no path); ``"orthogonal"`` for
            (1, 0) and (0, 1) only, so that every path has :math:`N + M - 1`
            cells. Default: ``"symmetric"``.
        diagonal_weight (float, optional): :math:`w > 0`, the weight of a
            diagonal step's cost in the symmetric steps. 1 favors diagonal
            paths, which sum fewer costs; 2, Sakoe and Chiba's symmetric form
            ("symmetric2" in Giorgino's dtw package), weighs each step's cost
            by how far it advances, :math:`\Delta n + \Delta m`, so that every
            path has the same total weight and diagonal steps are not favored.
            Default: 1.
        lengths (tuple of Tensor, optional): each pair's lengths
            :math:`(N_b, M_b)`, two integer tensors of shape :math:`(B)`, for
            batches of sequences padded to :math:`N` and :math:`M`; the
            distance is then :math:`D[N_b - 1, M_b - 1]`. Padding costs don't
            affect it. With the asymmetric steps, their values size the
            computation, which breaks a :func:`torch.compile` graph. Default:
            ``None``, the full lengths.
        band (float, optional): a Sakoe-Chiba band, :math:`\ge 0`: only cells
            at most this many cells off the straight line from :math:`(0, 0)`
            to :math:`(N_b - 1, M_b - 1)`, measured along the longer
            sequence, may be on a path. Where the line is steeper than the
            band allows a step to follow, the band widens to keep a path.
            From :attr:`x` and :attr:`y` only, where the kernels skip the
            tiles of the grid outside it, for DTW and soft-DTW alike; with a
            cost, make the cost ``inf`` outside it instead. Default:
            ``None``, no band.
        checkpoint (int, optional): the number of rows of the kernels' grid
            between stored rows of the accumulated costs, which soft-DTW's
            gradient needs. The grid is :math:`R` by :math:`L` cells, with
            :math:`R \le L`: :math:`R = \min(N, M)` for the symmetric and
            orthogonal steps. The accumulated costs are not stored whole,
            :math:`O(B R L)`, but recomputed in the backward pass. A grid of
            several tile columns is cut along its wavefront, into strips of
            tile anti-diagonals of about :math:`k` rows, but of at least 62,
            so that the strips follow one another along it: with a seed per
            strip, the cells the next one reads, the memory is
            :math:`O(B L ((R + L / 8) / k + k))`, for tiles of 256 columns, as
            for :math:`B < 8` (wider tiles give less than :math:`L / 8`), and
            the passes take about 1.7 times as long as without
            :attr:`checkpoint`, plus a few milliseconds of launches per strip
            (for :math:`B = 1`, :math:`N = M = 16384` and :math:`F = 16`,
            about 2, 4 and 6 times as long for 1024, 256 and 128 rows). A
            grid of one tile column is cut into blocks of :math:`k` rows,
            each a pass of its own, filling the wavefront anew: the memory is
            :math:`O(B L (R / k + k))`, and a small :math:`k` is much slower.
            From :attr:`x` and :attr:`y` only: a cost is :math:`O(B R L)`
            anyway. Default: ``None``, all rows stored.

    Returns:
        Tensor: the distances, of shape :math:`(B)`, floating: of the dtype of
        the costs or frames, the default dtype for an integer cost; ``inf`` for
        a pair with no path, such as the asymmetric steps with
        :math:`N_b < M_b`.

    Raises:
        ValueError: if not exactly one of :attr:`cost` and (:attr:`x`,
            :attr:`y`) is given, if :attr:`x` and :attr:`y` are not of
            shapes :math:`(B, N, F)` and :math:`(B, M, F)` and one floating
            dtype, if :attr:`cost` is not of shape :math:`(B, N, M)`, if the
            inputs are not on one CUDA device, if :math:`N` or :math:`M` is
            0, if the accumulated costs stored for a pair (from a cost, or
            for soft-DTW's gradient from frames) exceed :math:`2^{31} - 1`
            cells, if :attr:`step_pattern` is unknown, if
            :attr:`diagonal_weight` is not positive, or not 1 with steps that
            have no weighted diagonal, if :attr:`gamma` or :attr:`band` is
            negative, if :attr:`lengths` are not of shape
            :math:`(B)` within :math:`1 \le N_b \le N` and
            :math:`1 \le M_b \le M`, if :attr:`band` comes with
            :attr:`cost`, or if :attr:`checkpoint` is not an integer of at
            least 1 or comes with :attr:`cost`. The lengths' values are
            checked only in eager mode, as they would break a
            :func:`torch.compile` graph.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import dtw
        >>> x = torch.randn(4, 100, 3, device="cuda")  # 3-D frames
        >>> y = torch.randn(4, 80, 3, device="cuda", requires_grad=True)
        >>> cost = torch.cdist(x, y) ** 2
        >>> distance = dtw(cost, gamma=0.1)
        >>> alignment, = torch.autograd.grad(
        ...     distance.sum(), cost, retain_graph=True
        ... )
        >>> distance.sum().backward()  # gradients with respect to y
        >>> same = dtw(x=x, y=y, gamma=0.1)  # the same cost, from the frames

    .. _Soft-DTW\: a Differentiable Loss Function for Time-Series:
        https://proceedings.mlr.press/v70/cuturi17a.html
    .. _Parallelizing Dynamic Time Warping Algorithm Using Prefix Computations on GPU:
        https://doi.org/10.1109/HPCC.and.EUC.2013.50
    """
    if not gamma >= 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if checkpoint is not None and (type(checkpoint) is not int or checkpoint < 1 or x is None):
        raise ValueError(f"checkpoint must be a positive integer, with x and y, got {checkpoint!r}")
    _validate("dtw", cost, x, y, step_pattern, diagonal_weight, lengths, band)
    return _dtw(cost, x, y, gamma, step_pattern, diagonal_weight, lengths, band, checkpoint)[0]


def _validate(
    name: str,
    cost: Tensor | None,
    x: Tensor | None,
    y: Tensor | None,
    step_pattern: str,
    diagonal_weight: float,
    lengths: tuple[Tensor, Tensor] | None,
    band: float | None,
) -> tuple[int, int, int]:
    """Check the arguments; return the shape (B, N, M) of the costs."""
    if band is not None and not band >= 0:
        raise ValueError(f"band must be non-negative, got {band}")
    _check_features(name, cost, x, y)
    if band is not None and cost is not None:
        raise ValueError(f"{name}'s band needs x and y; with a cost, make it inf outside the band")
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
    if not diagonal_weight > 0:
        raise ValueError(f"diagonal_weight must be positive, got {diagonal_weight}")
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
    # arithmetic finite; a path sums at most w (N + M) of them, in the dtype of
    # cost * scale.
    dtype = ref.dtype if ref.is_floating_point() else torch.get_default_dtype()
    big = torch.finfo(dtype).max / (8 * max(diagonal_weight, 1.0) * (N + M))
    window = None
    if features:
        if band is not None:
            window = _band_windows(N, M, n_len, m_len, band, step_pattern == "orthogonal")
    elif gamma > 0 or not cost.is_floating_point():
        cost = cost * scale  # in units of gamma; integers become floats

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


def _dtw(
    cost, x, y, gamma, step_pattern, diagonal_weight, lengths, band, checkpoint=None, path=False
):  # fmt: skip
    """dtw's distances, the kernels' grid (None if no pair has a path), from the
    cost its accumulated costs (else None), and from frames, for DTW with
    gradients or ``path``, each pair's path on the grid (else None)."""
    # Imported here so that philtorch.align imports without Triton.
    from ._dtw_kernels import (
        _seed_cells,
        dtw_dp,
        dtw_dp_features,
        dtw_dp_strip,
        frontier,
        strip_tile,
    )

    features = cost is None
    if features and lengths is not None:
        # Frames past each pair's length as the pair's mean valid frame: any
        # value there, inf included, would reach the gradients' products, and
        # their centring on the mean of all frames includes the padding.
        valid = [
            (torch.arange(t.size(1), device=t.device) < n.to(t.device)[:, None])[..., None]
            for t, n in zip((x, y), lengths)
        ]
        with torch.no_grad():
            acc = torch.promote_types(x.dtype, torch.float32)
            total = sum(torch.where(v, t, 0).sum(1, dtype=acc) for v, t in zip(valid, (x, y)))
            mean = (total / sum(v.sum(1) for v in valid)).to(x.dtype)
        x, y = (torch.where(v, t, mean[:, None]) for v, t in zip(valid, (x, y)))
    g = _grid(cost, x, y, gamma, step_pattern, diagonal_weight, lengths, band)
    if g is None:
        # No pair has a path; stay in the graph, with zero gradients.
        first = x.sum((1, 2)) + y.sum((1, 2)) if features else cost[:, 0, 0]
        infinite = torch.where(torch.ones_like(first, dtype=torch.bool), float("inf"), first)
        return infinite, g, None, None
    batch = torch.arange(g.ends.size(0), device=g.ends.device)
    end_row, end_col = g.ends.unbind(-1)
    soft = gamma > 0
    D = walk = None
    grad = features and torch.is_grad_enabled() and (x.requires_grad or y.requires_grad)
    if features and not soft and (grad or path):
        with torch.no_grad():
            walk = linear_path(
                x, y, g.window, g.big, g.scale, g.steps, g.diag_weight, g.sheared, g.transposed,
                g.R, g.L, g.ends, g.no_path,
            )  # fmt: skip
            distance = _path_cost(g, x, y, walk)
        # False for inf, NaN and, with a window, its no path.
        finite = distance.isfinite() if g.window is None else distance < g.big / 2
        finite &= _is_walk(g, walk)
        walk = torch.where(finite[:, None, None], walk, -1)
        # With gradients, the cost again, of the walk kept, in the graph.
        distance = torch.where(finite, _path_cost(g, x, y, walk) if grad else distance, torch.inf)
    elif features and not (grad and soft):
        zero = torch.zeros_like(batch)
        items = torch.stack([batch, zero, zero, end_row + 1, end_col + 1], -1)
        with torch.no_grad():
            row = frontier(
                x, y, g.window, g.big, g.scale, items, soft, g.steps, g.diag_weight, g.sheared,
                g.transposed, False, g.R, g.L,
            )  # fmt: skip
        distance = row.gather(1, end_col[:, None])[:, 0]
    elif not features:
        D = dtw_dp(g.cost, None, soft, g.steps, g.diag_weight)
        distance = D[batch, end_row, end_col]
    else:

        def dp(row0, rows, seed, x, y):
            """The grid's rows row0 to row0 + rows - 1 of D; the first is the
            seed, if given."""
            return dtw_dp_features(
                x, y, g.window, g.big, g.scale, soft, g.steps, g.diag_weight, g.sheared,
                g.transposed, rows, g.L, seed, row0,
            )  # fmt: skip

        def block(a, b, seed, x, y):
            """Rows a to b - 1 of D: the last one, and the distances of the
            pairs ending in them."""
            row0 = max(a - 1, 0)  # the seed's row, for a > 0
            D = dp(row0, b - row0, seed, x, y)
            picked = D[batch, (end_row - row0).clamp(0, b - row0 - 1), end_col]
            return D[:, -1].clone(), torch.where((end_row >= a) & (end_row < b), picked, 0.0)

        # The tile anti-diagonals of a strip: about ``checkpoint`` rows, at least 2.
        strip = None if checkpoint is None else strip_tile(g.ends.size(0), g.R, g.L)
        if strip is not None:
            TR, TL, n_diag = strip
            step = max(2, -(-checkpoint // TR))
        if checkpoint is None or checkpoint >= g.R or (strip is not None and step >= n_diag):
            distance = dp(0, g.R, None, x, y)[batch, end_row, end_col]
        elif strip is not None:

            def strip_dp(d0, seed, x, y):
                """The strip of tile anti-diagonals from d0 of D: the seed of
                the next (None after the last), and the distances of the pairs
                ending in it, 0 for those ending elsewhere."""
                D = dtw_dp_strip(
                    x, y, g.window, g.big, g.scale, soft, g.steps, g.diag_weight, g.sheared,
                    g.transposed, g.R, g.L, seed, step, d0,
                )  # fmt: skip
                B, RB, L = D.shape
                flat = D.flatten(1)
                at = end_row + end_col // TL * TR - d0 * TR + TR + 1  # the end cell's skewed row
                inside = (at > TR) & (at < RB)
                cell = torch.where(inside, at * L + end_col, 0)
                found = torch.where(inside, flat.gather(1, cell[:, None])[:, 0], 0.0)
                if d0 + step >= n_diag:
                    return None, found
                # The next seed: this one's cells, in the buffer's last rows.
                cells = _seed_cells(L, TR, TL, D.device) + (RB - TR - 1) * L
                return flat.gather(1, cells.expand(B, -1)), found

            distance, seed = 0, None
            for d0 in range(0, n_diag, step):
                seed, found = torch.utils.checkpoint.checkpoint(
                    strip_dp, d0, seed, x, y, use_reentrant=False, preserve_rng_state=False
                )
                distance = distance + found
        else:
            distance, seed = 0, None
            for a in range(0, g.R, checkpoint):
                # Autograd keeps each block's seed and outputs, not its D.
                seed, found = torch.utils.checkpoint.checkpoint(
                    block, a, min(a + checkpoint, g.R), seed, x, y, use_reentrant=False,
                    preserve_rng_state=False,
                )  # fmt: skip
                distance = distance + found
    no_path = g.no_path
    if band is not None:
        no_path = no_path | (distance >= g.big / 2)
    return torch.where(no_path, float("inf"), distance) / g.scale, g, D, walk


# The linear-memory path: items of at most this many cells, or this many rows,
# are backtracked whole, in chunks of at most this many cells in all, the rows
# rounded up to 16. A leaf's steps back take 2 bits a cell, so a chunk's are at
# most 256 MiB, but for a leaf of at most _LEAF_ROWS rows, chunked alone however
# wide.
_LEAF_CELLS = 2**28
_LEAF_ROWS = 64
_CHUNK_CELLS = 2**30


def _is_walk(g: _Grid, walk: Tensor) -> Tensor:
    """Whether each pair's walk, (B, R + L - 1, 2) with -1 past its cells, runs
    from the grid's first cell to the pair's end cell by the step set's steps:
    a finite cost does not prove it, as a row of infinite costs (an infinite
    frame) can stop the leaves' backtrack short."""
    r, col = walk.unbind(-1)
    valid = r >= 0
    last = walk.gather(1, (valid.sum(1) - 1).clamp(min=0)[:, None, None].expand(-1, 1, 2))[:, 0]
    dr, dc = r[:, 1:] - r[:, :-1], col[:, 1:] - col[:, :-1]
    step = ((dr == 1) & (dc == 0)) | ((dr == 0) & (dc == 1))
    if g.steps == "symmetric":
        step |= (dr == 1) & (dc == 1)
    return (walk[:, 0] == 0).all(-1) & (last == g.ends).all(-1) & (step | ~valid[:, 1:]).all(1)


def _path_cost(g: _Grid, x, y, walk: Tensor) -> Tensor:
    """The cost of each pair's path on the grid, (B, R + L - 1, 2) with -1 past
    it: its cells' costs from the frames, by PyTorch ops, a diagonal step's
    times its weight."""
    from ._dtw_kernels import _cell_costs

    r, col = walk.unbind(-1)
    p = torch.arange(walk.size(0), device=walk.device)[:, None].expand_as(r)
    c = _cell_costs(
        x, y, g.window, g.big, g.scale, g.sheared, g.transposed, p, r.clamp(min=0),
        col.clamp(min=0), r >= 0,
    )  # fmt: skip
    diagonal = (r[:, 1:] - r[:, :-1] == 1) & (col[:, 1:] - col[:, :-1] == 1)
    weight = pad(
        torch.where(diagonal, c.new_tensor(g.diag_weight), c.new_tensor(1.0)), (1, 0), value=1.0
    )
    return torch.where(r >= 0, weight * c, 0.0).sum(1)


def _lines(items: Tensor) -> Tensor:
    """The cells (pair, row, column) of items of one row or column: lines."""
    p, r0, l0, h, w = items.unbind(-1)
    count = h + w - 1
    item = torch.repeat_interleave(torch.arange(len(items), device=items.device), count)
    k = torch.arange(len(item), device=items.device) - (count.cumsum(0) - count)[item]
    return torch.stack([p[item], r0[item] + k * (w[item] == 1), l0[item] + k * (h[item] == 1)], -1)


def _leaves(g: _Grid, x, y, items: Tensor) -> Tensor:
    """The cells (pair, row, column) of the optimal path of each item: the
    frontier's wavefront over the items, which keeps each cell's best step back
    in 2 bits, and a walk from the item's last cell."""
    from ._dtw_kernels import leaf_paths

    p, r0, l0, h, w = items.unbind(-1)
    cells = leaf_paths(
        x, y, g.window, g.big, g.scale, items, g.steps, g.diag_weight, g.sheared, g.transposed,
        int(h.max()), int(w.max()),
    )  # fmt: skip
    live = cells[..., 0] >= 0
    shift = torch.stack([p, r0, l0], -1)[:, None]
    return (shift + pad(cells, (1, 0)))[live]


def _split(g: _Grid, x, y, items: Tensor) -> Tensor:
    """All the items split as :func:`_cross` splits one, by width, in chunks of
    it whose padding to the widest and tallest at most doubles their sides in
    all: a round's memory is linear in its items' sides, however skewed."""
    items = items[items[:, 4].argsort(descending=True)]
    parts, start, H, W, rows, cols = [items.new_empty(0, 5)], 0, 0, 0, 0, 0
    for i, (h, w) in enumerate(items[:, 3:].tolist()):
        h -= h // 2  # the second half's rows, the taller
        n = i + 1 - start
        if n > 1 and (n * max(H, h) > 2 * (rows + h) or n * W > 2 * (cols + w)):
            parts.append(_cross(g, x, y, items[start:i]))
            start, H, W, rows, cols, n = i, 0, 0, 0, 0, 1
        W = W or w  # the widest, the chunk's first
        H, rows, cols = max(H, h), rows + h, cols + w
    return torch.cat([*parts, _cross(g, x, y, items[start:])])


def _cross(g: _Grid, x, y, items: Tensor) -> Tensor:
    """Each item as two, split by the row after its middle row's crossing.

    Frontiers give the best cost to each cell of the middle row, from the
    item's first cell, and from each cell of the next row, to its last; the
    cheapest step across, up or diagonal, is on an optimal path."""
    from ._dtw_kernels import _cell_costs, frontier

    p, r0, l0, h, w = items.unbind(-1)
    top = h // 2
    below = r0 + top  # the first row of the second half
    given = (x, y, g.window, g.big, g.scale)
    options = (False, g.steps, g.diag_weight, g.sheared, g.transposed)
    h_max, w_max = (int(t.max()) for t in (h - h // 2, w))
    first = torch.stack([p, r0, l0, top, w], -1)
    second = torch.stack([p, below, l0, h - top, w], -1)
    # The reverse frontier on a side stream, joined back after the forward.
    main, side = torch.cuda.current_stream(x.device), torch.cuda.Stream(x.device)
    side.wait_stream(main)
    for t in (x, y, g.window, second):
        if t is not None:
            t.record_stream(side)
    with torch.cuda.stream(side):
        G = frontier(*given, second, *options, True, h_max, w_max)
    D = frontier(*given, first, *options, False, h_max, w_max)
    main.wait_stream(side)
    G.record_stream(main)
    j = torch.arange(D.size(1), device=x.device)
    c = _cell_costs(
        x, y, g.window, g.big, g.scale, g.sheared, g.transposed, p[:, None], below[:, None],
        l0[:, None] + j,
    )  # fmt: skip
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


# The path search syncs with the host, which a CUDA graph can't capture: the
# tag, from PyTorch 2.8, keeps the op out of the graphs of reduce-overhead.
_UNSAFE = getattr(torch.Tag, "cudagraph_unsafe", None)


@torch.library.custom_op(
    "philtorch::dtw_linear_path", mutates_args=(), **({"tags": (_UNSAFE,)} if _UNSAFE else {})
)
def linear_path(
    x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float, steps: str,
    diag_weight: float, sheared: bool, transposed: bool, R: int, L: int, ends: Tensor,
    no_path: Tensor,
) -> Tensor:  # fmt: skip
    """Each pair's optimal path (B, R + L - 1, 2) on the :class:`_Grid` of these
    fields, in :func:`backtrack`'s format, in memory linear in the grid's
    sides: Hirschberg's divide and conquer, for the pairs ``no_path`` is False
    for. An op because it reads sizes back to the host, which a
    :func:`torch.compile` graph can't hold.

    An item, a rectangle of the grid between two cells of the path, is a
    straight line, a leaf backtracked whole, or split in two across its middle
    rows (:func:`_split`). Each round handles all items together."""
    g = _Grid(
        None, window, big, scale, steps, diag_weight, sheared, transposed, R, L, ends, no_path
    )
    B = g.ends.size(0)
    pairs = (~no_path).nonzero()[:, 0]
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
            rows = -(-max(H, h) // 16) * 16  # the steps back's words, 16 rows each
            if i > start and (i + 1 - start) * rows * max(W, w) > _CHUNK_CELLS:
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


@linear_path.register_fake
def _(x, y, window, big, scale, steps, diag_weight, sheared, transposed, R, L, ends, no_path):
    return x.new_empty(ends.size(0), R + L - 1, 2, dtype=torch.long)


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
    r"""Find the DTW distance and optimal warping path of pairs of sequences.

    The path is DTW's (:func:`dtw` with :math:`\gamma = 0`) as index pairs
    :math:`(n, m)` from :math:`(0, 0)` to :math:`(N_b - 1, M_b - 1)`, the
    same path as the distance's gradient marks; under ties, one of the
    optimal paths. The distance is :func:`dtw`'s, differentiable to any
    order, its gradient the path.

    From :attr:`cost`, the path is walked back over the forward pass's
    accumulated costs, up to a block of a row's cells per step, as Viterbi
    decoding gives a hidden Markov model's states. From :attr:`x` and
    :attr:`y`, it is found by divide and conquer, after `Exact, Parallelizable
    Dynamic Time Warping Alignment with Linear Memory`_ (Tralie and Dempsey,
    2020), in :math:`O(B (N + M))` memory and a few times the forward work:
    the best crossing of a rectangle's middle rows splits it in two between
    cells of the path, down to rectangles backtracked whole from each cell's
    best step back, kept in 2 bits, after `A Study of Parallelizable
    Alternatives to Dynamic Time Warping for Aligning Long Sequences`_ (Yang
    et al., 2022); the distance is then the path's cost, from the frames,
    which autograd differentiates.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        cost (Tensor, optional): as in :func:`dtw`.
        x (Tensor, optional): as in :func:`dtw`.
        y (Tensor, optional): as in :func:`dtw`.
        step_pattern (str, optional): as in :func:`dtw`.
        diagonal_weight (float, optional): as in :func:`dtw`.
        lengths (tuple of Tensor, optional): as in :func:`dtw`.
        band (float, optional): as in :func:`dtw`.

    Returns:
        tuple of Tensor: the distances, of shape :math:`(B)`, ``inf`` for a
        pair with no path, and the paths, an integer tensor of shape
        :math:`(B, N + M - 1, 2)` of the cells in order, padded with -1 past
        each pair's path, all -1 for a pair with none.

    Raises:
        ValueError: as :func:`dtw` does for these arguments.
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

    .. _Exact, Parallelizable Dynamic Time Warping Alignment with Linear Memory:
        https://doi.org/10.5281/zenodo.4245470
    .. _A Study of Parallelizable Alternatives to Dynamic Time Warping for Aligning Long Sequences:
        https://doi.org/10.1109/TASLP.2022.3180673
    """
    B, N, M = _validate("dtw_path", cost, x, y, step_pattern, diagonal_weight, lengths, band)
    distance, g, D, walk = _dtw(
        cost, x, y, 0.0, step_pattern, diagonal_weight, lengths, band, path=True
    )
    path = torch.full((B, N + M - 1, 2), -1, dtype=torch.long, device=distance.device)
    if g is None:
        return distance, path
    with torch.no_grad():
        if cost is None:
            cells = walk
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
    r"""Compute the soft-DTW divergence between pairs of sequences.

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
        gamma (float): the soft-min temperature :math:`\gamma`, positive.
        x (Tensor, optional): the first sequences' frames, of shape
            :math:`(B, N, F)`. Give either the three costs or both :attr:`x`
            and :attr:`y`, from which they are the squared Euclidean
            distances, as in :func:`dtw`.
        y (Tensor, optional): the second sequences' frames, of shape
            :math:`(B, M, F)`, with the dtype of :attr:`x`.
        lengths (tuple of Tensor, optional): each pair's lengths
            :math:`(N_b, M_b)`, as in :func:`dtw`. Default: ``None``, the
            full lengths.
        **kwargs: the other keyword arguments of :func:`dtw`.

    Returns:
        Tensor: the divergences, of shape :math:`(B)`.

    Raises:
        ValueError: if :attr:`gamma` is missing or not positive, if not either
            all three costs or both :attr:`x` and :attr:`y` are given, if
            :attr:`cost_xx` and :attr:`cost_yy` are not of shapes
            :math:`(B, N, N)` and :math:`(B, M, M)`, or as :func:`dtw` does.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import soft_dtw_divergence
        >>> x = torch.randn(4, 100, 3, device="cuda")
        >>> y = torch.randn(4, 80, 3, device="cuda", requires_grad=True)
        >>> divergence = soft_dtw_divergence(gamma=0.1, x=x, y=y)
        >>> divergence.sum().backward()

    .. _Differentiable Divergences Between Time Series:
        https://proceedings.mlr.press/v130/blondel21a.html
    """
    if gamma is None or not gamma > 0:
        raise ValueError(f"the soft-DTW divergence needs gamma > 0, got {gamma}")
    name = "soft_dtw_divergence"
    _check_features(name, cost_xy, x, y)
    _check_features(name, cost_xx, x, x)
    _check_features(name, cost_yy, y, y)
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
    xx, yy = (None, None) if lengths is None else ((lengths[0],) * 2, (lengths[1],) * 2)
    d_xy, d_xx, d_yy = (
        dtw(gamma=gamma, lengths=ls, **kw, **kwargs) for kw, ls in zip(inputs, (lengths, xx, yy))
    )
    return d_xy - 0.5 * (d_xx + d_yy)
