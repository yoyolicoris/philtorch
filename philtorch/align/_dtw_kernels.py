"""Triton kernels for DTW, soft-DTW and CTC, differentiable to any order.

The accumulated costs D of a cost matrix (B, R, L) follow, with soft-min for
soft-DTW, in base 2 (-log2 of a sum of powers of 2: the hardware's exp2 and
log2, the multiplies of exp and log saved; callers scale costs by log2 e),

    D(i, j) = min(D(i - 1, j) + c(i, j), D(i, j - 1) + c(i, j),
                  D(i - 1, j - 1) + w c(i, j), D(i - 2, j - 1) + c(i, j)),

over one of three step sets: "orthogonal", up and left; "symmetric", also
the diagonal, weighted by w; and "ctc", left, the diagonal and the skip from
two rows up, the last only into the rows a mask allows. One program per
batch item holds a whole row and loops over the rows. Within row i, with t(j)
the minimum over the steps from earlier rows, the row is D(j) = min(t(j),
c(j) + D(j - 1)): a scan of the maps x -> min(A, C + x). The diagonal and
skip terms read earlier rows one column to the left, from the D already
written.

Derivatives. With pi(s, p) the share of each cell s's minimum taken from its
predecessor p (a one-hot of the best for DTW, a softmax of the negated
arguments for soft-DTW), the gradient of D(s) is the reverse accumulation

    e(s) = g(s) + sum over successors t of s of pi(t, s) e(t),

and the cost's gradient is e(s) times sum over p of pi(s, p) w(s, p), the
step weights of the arguments, which is e(s) itself when w = 1. The shares
are recomputed from the stored D and the cost, not taken from the scan's own
minima, which can differ from the stored D in the last bit.

Two implementations of that gradient:

* for first derivatives, :func:`_dtw_backward`, which computes the shares as
  it scans each row from the right, or for a few long rows
  :func:`_dtw_backward_parallel`, which computes them over all cells first
  and then accumulates;
* :func:`accumulate`, the linear accumulation over the grid in either
  direction, and PyTorch ops, which autograd differentiates again, for
  higher derivatives. accumulate's backward is itself in the other
  direction.

dtw_dp's backward takes the second when gradients of the gradient are
requested (``create_graph=True``) and the first otherwise.

Two kernels do all of it. ``_dag_kernel`` is the row-by-row recursion in
either semiring, min-plus (or soft-min-plus) for the forward pass and
linear for every accumulation, the first-order backward's included;
``_shares_kernel`` computes the shares over all cells at once. Both take
a ``STEPS`` constant, and with the ctc steps the (B, R) skip mask.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

_INF = float("inf")
_STEPS = {"orthogonal": 0, "symmetric": 1, "ctc": 2}
# Triton kernels can only read module constants made with tl.constexpr.
_KERNEL_INF = tl.constexpr(_INF)
_ORTHOGONAL, _CTC = (tl.constexpr(_STEPS[name]) for name in ("orthogonal", "ctc"))


@triton.jit
def _softmin(x, y):
    """-log2(2^-x + 2^-y) as min(x, y) - log2(1 + 2^-|x - y|), and inf if both are."""
    smaller = tl.minimum(x, y)
    return tl.where(smaller == _KERNEL_INF, smaller, smaller - tl.log2(1 + tl.exp2(-tl.abs(x - y))))


@triton.jit
def _min(x, y, SOFT: tl.constexpr):
    if SOFT:
        return _softmin(x, y)
    return tl.minimum(x, y)


# A row's maps, composed left (earlier) first; one function per semiring, as
# associative_scan takes no constants.


@triton.jit
def _compose_min(a_l, c_l, a_r, c_r):
    """x -> min(A, C + x)."""
    return tl.minimum(a_r, c_r + a_l), c_r + c_l


@triton.jit
def _compose_softmin(a_l, c_l, a_r, c_r):
    """x -> softmin(A, C + x)."""
    return _softmin(a_r, c_r + a_l), c_r + c_l


@triton.jit
def _compose_affine(a_l, b_l, a_r, b_r):
    """x -> a x + b, the multiplier first: as fast as the others' order or faster."""
    return a_r * a_l, a_r * b_l + b_r


# The kernels take contiguous (B, R, L) tensors, one program per batch item
# with a whole row in BLOCK >= L lanes. In the forward kernels, lanes past L
# hold padding; the scans run left to right, so it never reaches the real
# columns.


@triton.jit
def _shares(
    D_ptr, cost_ptr, skip_ptr, b, r, cols, R, L, diag_weight,
    SOFT: tl.constexpr, STEPS: tl.constexpr,
):  # fmt: skip
    """The shares of cells (r, cols) from their up, left, up-left and skip predecessors.

    Zero for cells outside the grid and for predecessors outside it or not
    among the steps.
    """
    valid = (cols >= 0) & (cols < L) & (r >= 0) & (r < R)
    offset = b * R * L + r * L + cols
    c = tl.load(cost_ptr + offset, mask=valid, other=0.0)
    absent = tl.full(c.shape, _KERNEL_INF, c.dtype)
    up = absent
    if STEPS != _CTC:
        up = tl.load(D_ptr + offset - L, mask=valid & (r > 0), other=_KERNEL_INF) + c
    left = tl.load(D_ptr + offset - 1, mask=valid & (cols > 0), other=_KERNEL_INF) + c
    diag = absent
    if STEPS != _ORTHOGONAL:
        up_left = tl.load(
            D_ptr + offset - L - 1, mask=valid & (r > 0) & (cols > 0), other=_KERNEL_INF
        )
        diag = up_left + diag_weight * c
    skip = absent
    if STEPS == _CTC:
        allowed = valid & (r > 1) & (cols > 0)
        allowed &= tl.load(skip_ptr + b * R + r, mask=(r >= 0) & (r < R), other=0) != 0
        skip = tl.load(D_ptr + offset - 2 * L - 1, mask=allowed, other=_KERNEL_INF) + c
    best = tl.minimum(tl.minimum(up, left), tl.minimum(diag, skip))
    none = best == _KERNEL_INF  # cell (0, 0), or outside the grid
    if SOFT:
        # Only the steps in the set: an absent one's exp would be 0, at a cost.
        safe = tl.where(none, 0.0, best)
        e_left = tl.exp2(safe - left)
        zero = tl.zeros(c.shape, c.dtype)
        e_up = zero
        e_diag = zero
        e_skip = zero
        if STEPS != _CTC:
            e_up = tl.exp2(safe - up)
        if STEPS != _ORTHOGONAL:
            e_diag = tl.exp2(safe - diag)
        if STEPS == _CTC:
            e_skip = tl.exp2(safe - skip)
        scale = tl.where(none, 0.0, 1.0 / (e_up + e_left + e_diag + e_skip))
        w_up, w_left, w_diag, w_skip = e_up * scale, e_left * scale, e_diag * scale, e_skip * scale
    else:
        # The first best of up, left, up-left and skip, as torch.argmin picks.
        is_up = (up == best) & ~none
        is_left = (left == best) & ~is_up & ~none
        is_diag = (diag == best) & ~is_up & ~is_left & ~none
        is_skip = (skip == best) & ~is_up & ~is_left & ~is_diag & ~none
        w_up = is_up.to(c.dtype)
        w_left = is_left.to(c.dtype)
        w_diag = is_diag.to(c.dtype)
        w_skip = is_skip.to(c.dtype)
    return w_up, w_left, w_diag, w_skip


@triton.jit
def _shares_kernel(
    D_ptr, cost_ptr, skip_ptr, up_ptr, left_ptr, diag_ptr, skip_share_ptr, R, L, diag_weight,
    SOFT: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """Every cell's shares of the steps in the set, one program per BLOCK cells of a row."""
    batch_row = tl.program_id(0)
    b = (batch_row // R).to(tl.int64)
    r = batch_row % R
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    up, left, diag, skip = _shares(
        D_ptr, cost_ptr, skip_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS
    )
    offset = b * R * L + r * L + cols
    mask = cols < L
    if STEPS != _CTC:
        tl.store(up_ptr + offset, up, mask=mask)
    tl.store(left_ptr + offset, left, mask=mask)
    if STEPS != _ORTHOGONAL:
        tl.store(diag_ptr + offset, diag, mask=mask)
    if STEPS == _CTC:
        tl.store(skip_share_ptr + offset, skip, mask=mask)


@triton.jit
def _neighbour(ptr, row, L, ROWS: tl.constexpr, COLS: tl.constexpr, REVERSE: tl.constexpr):
    """ptr at the cell ROWS rows and COLS columns after row's with REVERSE, else before.

    An edge's weight is stored at its successor, so it is
    ``_neighbour(ptr, row, L, ROWS * REVERSE, COLS * REVERSE, REVERSE)``: the
    neighbour with REVERSE, else row's own cell.
    """
    if REVERSE:
        return ptr + row + ROWS * L + COLS
    return ptr + row - ROWS * L - COLS


@triton.jit
def _dag_kernel(
    cost_ptr, D_ptr, skip_rows_ptr, up_ptr, left_ptr, diag_ptr, skip_ptr, x_ptr, y_ptr,
    out_ptr, R, L, diag_weight, SOFT: tl.constexpr, STEPS: tl.constexpr,
    REVERSE: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """y(s) = x(s) (+) the sum over steps of W(s, p) (x) y(p), predecessors p; with
    REVERSE, the transpose, over successors t of W(t, s) (x) y(t). Each edge's
    weight is stored at its successor.

    Three uses, by the pointers given:

    * the forward pass, with ``cost_ptr`` alone: min-plus, or soft-min-plus
      with ``SOFT``; every step's weight is the cost of the cell it enters,
      the diagonal's times ``diag_weight``, the ctc steps' skips gated by
      the (B, R) mask ``skip_rows_ptr``; D(0, 0) = c(0, 0);
    * the first-order backward, with ``cost_ptr`` and the forward pass's
      ``D_ptr``, REVERSE: linear, x the gradient of D, the weights the
      shares, computed from D and the cost (soft with ``SOFT``) as each row
      is reached and kept for the rows above; ``out_ptr``, if not None,
      takes the cost's gradient when the diagonal is weighted;
    * the linear accumulation, with x and the weights of the steps in the
      set, the others None.

    Row by row, the left step makes each row a scan of the maps
    x -> A (+) (C (x) x), from the right with REVERSE, where lane t holds
    column L - 1 - t. The other rows' terms come from y already written,
    after a barrier. The neighbours' offsets are constants: offsets computed
    from REVERSE would cost a fifth of the time on long rows.
    """
    SHARES: tl.constexpr = D_ptr is not None
    FORWARD: tl.constexpr = cost_ptr is not None and not SHARES
    b = tl.program_id(0).to(tl.int64)
    batch = b * R * L
    lanes = tl.arange(0, BLOCK)
    mask = lanes < L
    cols = (L - 1 - lanes) if REVERSE else lanes
    zero = _KERNEL_INF if FORWARD else 0.0  # the semiring's zero
    dtype = y_ptr.dtype.element_ty
    if FORWARD:
        # Row 0 takes the left steps alone: D(0, j) = c(0, 0) + ... + c(0, j).
        c = tl.load(cost_ptr + batch + cols, mask=mask, other=0.0)
        y_near = tl.cumsum(c, axis=0)
        tl.store(y_ptr + batch + cols, y_near, mask=mask)
    else:
        y_near = tl.full([BLOCK], zero, dtype)  # the up neighbour, in this lane
    if SHARES:
        # The shares of the rows below, kept from earlier rows: row r + 1
        # straight down at j and diagonally at j + 1, row r + 2 by a skip at
        # j + 1. None below the last row.
        up_below = tl.zeros([BLOCK], dtype)
        diag_below_next = tl.zeros([BLOCK], dtype)
        skip_below_next = tl.zeros([BLOCK], dtype)
        skip_below2_next = tl.zeros([BLOCK], dtype)
    for k in range(1 if FORWARD else 0, R):
        r = (R - 1 - k) if REVERSE else k
        row = batch + r * L + cols
        rows_1 = (r + 1 < R) if REVERSE else (r > 0)
        # Recomputed per row: kept live, it costs registers that long rows lack.
        side = mask & ((cols + 1 < L) if REVERSE else (cols > 0))
        if FORWARD:
            c = tl.load(cost_ptr + row, mask=mask, other=0.0)
        else:
            base = tl.load(x_ptr + row, mask=mask, other=0.0)
        if STEPS != _CTC:
            if FORWARD:
                base = c + y_near
            elif SHARES:
                base += up_below * y_near
            else:
                w = tl.load(
                    _neighbour(up_ptr, row, L, 1 * REVERSE, 0 * REVERSE, REVERSE),
                    mask=mask & rows_1,
                    other=0.0,
                )
                base += w * y_near
        if STEPS != _ORTHOGONAL:
            tl.debug_barrier()
            ok = side & rows_1
            if FORWARD:
                y = tl.load(_neighbour(y_ptr, row, L, 1, 1, REVERSE), mask=ok, other=zero)
                term = y + diag_weight * c
                # The ctc steps have no up step: the diagonal starts the sum.
                base = term if STEPS == _CTC else _min(base, term, SOFT)
            else:
                if SHARES:
                    w = diag_below_next
                else:
                    w = tl.load(
                        _neighbour(diag_ptr, row, L, 1 * REVERSE, 1 * REVERSE, REVERSE),
                        mask=ok,
                        other=0.0,
                    )
                base += w * tl.load(_neighbour(y_ptr, row, L, 1, 1, REVERSE), mask=ok, other=0.0)
            if STEPS == _CTC:
                ok &= (r + 2 < R) if REVERSE else (r > 1)
                if FORWARD:
                    ok &= tl.load(skip_rows_ptr + b * R + r) != 0
                    y = tl.load(_neighbour(y_ptr, row, L, 2, 1, REVERSE), mask=ok, other=zero)
                    base = _min(base, y + c, SOFT)
                else:
                    if SHARES:
                        w = skip_below2_next
                    else:
                        w = tl.load(
                            _neighbour(skip_ptr, row, L, 2 * REVERSE, 1 * REVERSE, REVERSE),
                            mask=ok,
                            other=0.0,
                        )
                    base += w * tl.load(
                        _neighbour(y_ptr, row, L, 2, 1, REVERSE), mask=ok, other=0.0
                    )
        if FORWARD:
            a = c
        elif SHARES:
            # Row r's shares at j, for the rows above, and at j + 1, whose left
            # share takes e(r, j + 1) into e(r, j).
            up_here, left_here, diag_here, _ = _shares(
                D_ptr, cost_ptr, skip_rows_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS
            )
            _, a, diag_next, skip_next = _shares(
                D_ptr, cost_ptr, skip_rows_ptr, b, r, cols + 1, R, L, diag_weight, SOFT, STEPS
            )
        else:
            a = tl.load(
                _neighbour(left_ptr, row, L, 0 * REVERSE, 1 * REVERSE, REVERSE),
                mask=side,
                other=0.0,
            )
        if not FORWARD:
            y_near = tl.associative_scan((a, base), 0, _compose_affine)[1]
        elif SOFT:
            y_near = tl.associative_scan((base, a), 0, _compose_softmin)[0]
        else:
            y_near = tl.associative_scan((base, a), 0, _compose_min)[0]
        tl.store(y_ptr + row, y_near, mask=mask)
        if SHARES:
            if out_ptr is not None:
                none = (up_here + left_here + diag_here) == 0  # cell (0, 0)
                factor = tl.where(none, 1.0, up_here + left_here + diag_weight * diag_here)
                tl.store(out_ptr + row, y_near * factor, mask=mask)
            up_below = up_here
            diag_below_next = diag_next
            skip_below2_next = skip_below_next
            skip_below_next = skip_next


@triton.jit
def _backtrack_kernel(
    D_ptr, cost_ptr, skip_ptr, ends_ptr, path_ptr, R, L, P, diag_weight,
    STEPS: tl.constexpr,
):  # fmt: skip
    """Each pair's optimal path, walked from its end cell back to (0, 0).

    At each cell, the first best of its up, left, up-left and skip
    predecessors, from D and the cost as the hard shares choose, so the
    path is the hard gradient's. The cells, the end first, go to path
    (B, P, 2) as (row, column), -1 past (0, 0). One program per pair: each
    step waits on the last, but the walk stays in cache.
    """
    b = tl.program_id(0).to(tl.int64)
    batch = b * R * L
    out = path_ptr + b * P * 2
    i = tl.load(ends_ptr + 2 * b)
    j = tl.load(ends_ptr + 2 * b + 1)
    for t in range(0, P):
        active = i >= 0
        tl.store(out + 2 * t, i)
        tl.store(out + 2 * t + 1, j)
        cell = batch + i * L + j
        c = tl.load(cost_ptr + cell, mask=active, other=0.0)
        best = _KERNEL_INF
        best_i = i
        best_j = j
        if STEPS != _CTC:
            up = tl.load(D_ptr + cell - L, mask=active & (i > 0), other=_KERNEL_INF) + c
            best_i = tl.where(up < best, i - 1, best_i)
            best = tl.minimum(up, best)
        left = tl.load(D_ptr + cell - 1, mask=active & (j > 0), other=_KERNEL_INF) + c
        best_i = tl.where(left < best, i, best_i)
        best_j = tl.where(left < best, j - 1, best_j)
        best = tl.minimum(left, best)
        if STEPS != _ORTHOGONAL:
            diag = (
                tl.load(D_ptr + cell - L - 1, mask=active & (i > 0) & (j > 0), other=_KERNEL_INF)
                + diag_weight * c
            )
            best_i = tl.where(diag < best, i - 1, best_i)
            best_j = tl.where(diag < best, j - 1, best_j)
            best = tl.minimum(diag, best)
        if STEPS == _CTC:
            allowed = active & (i > 1) & (j > 0)
            allowed &= tl.load(skip_ptr + b * R + i, mask=active, other=0) != 0
            skip = tl.load(D_ptr + cell - 2 * L - 1, mask=allowed, other=_KERNEL_INF) + c
            best_i = tl.where(skip < best, i - 2, best_i)
            best_j = tl.where(skip < best, j - 1, best_j)
            best = tl.minimum(skip, best)
        # (0, 0), and any cell no path reaches, has no predecessor: the walk ends.
        i = tl.where(best == _KERNEL_INF, -1, best_i)
        j = tl.where(best == _KERNEL_INF, -1, best_j)


def backtrack(
    D: Tensor, cost: Tensor, skip: Tensor | None, ends: Tensor, steps: str, diag_weight: float
) -> Tensor:
    """Each pair's optimal path on the kernels' grid, from its end cell ``ends`` (B, 2)
    to (0, 0), as (row, column) pairs (B, R + L - 1, 2) in order, -1 past the path."""
    B, R, L = D.shape
    P = R + L - 1
    walk = ends.new_empty(B, P, 2)
    _backtrack_kernel[(B,)](
        D, cost.contiguous(), skip, ends.contiguous(), walk, R, L, P, diag_weight,
        _STEPS[steps], num_warps=1,
    )  # fmt: skip
    # The walk runs from the end: each path's cells, reversed.
    count = (walk[..., 0] >= 0).sum(1, keepdim=True)
    t = torch.arange(P, device=D.device)
    path = walk.gather(1, (count - 1 - t).clamp(min=0)[..., None].expand(-1, -1, 2))
    return torch.where((t < count)[..., None], path, -1)


def _num_warps(L: int, backward: bool = False, diag: bool = False) -> int:
    """Warps per program by row length, measured on an RTX 5060 Ti.

    The backward kernel keeps more of each row live, and with the diagonal
    step its rows past 4096 lanes spill at 16 warps.
    """
    if not backward:
        warps = 4 if L <= 256 else 8 if L <= 1024 else 16
    else:
        warps = 4 if L <= 256 else 32 if L > 4096 and diag else 16
    # A program has at most 1024 threads, and AMD's datacenter GPUs have
    # 64-thread warps.
    return min(warps, 16) if torch.version.hip else warps


# Outputs are allocated contiguous: torch.empty_like keeps a permuted input's
# strides, and the kernels write rows contiguously. Offsets within a pair are
# 32-bit, which keeps long rows fast; the batch offset is 64-bit.
_MAX_CELLS = 2**31 - 1


def _launch(kernel, B: int, L: int, *args, num_warps: int | None = None):
    num_warps = _num_warps(L) if num_warps is None else num_warps
    kernel[(B,)](*args, BLOCK=triton.next_power_of_2(L), num_warps=num_warps)


def _empty(t: Tensor) -> Tensor:
    return torch.empty_like(t, memory_format=torch.contiguous_format)


def _dtw_backward(
    D: Tensor, cost: Tensor, skip: Tensor | None, grad: Tensor, soft: bool, steps: str,
    diag_weight: float,
) -> Tensor:  # fmt: skip
    B, R, L = D.shape
    weighted = diag_weight != 1.0
    e = _empty(D)
    out = _empty(D) if weighted else e
    _launch(
        _dag_kernel, B, L, cost, D, skip, None, None, None, None, grad.contiguous(), e,
        out if weighted else None, R, L, diag_weight, soft, _STEPS[steps], True,
        num_warps=_num_warps(L, backward=True, diag=steps != "orthogonal"),
    )  # fmt: skip
    return out


def _weigh(e: Tensor, up: Tensor, left: Tensor, diag: Tensor, diag_weight: float) -> Tensor:
    """The cost's gradient from e, with a weighted diagonal: e times the cell's weight."""
    none = (up + left + diag) == 0  # cell (0, 0)
    return e * torch.where(none, 1.0, up + left + diag_weight * diag)


def _dtw_backward_parallel(
    D: Tensor, cost: Tensor, skip: Tensor | None, grad: Tensor, soft: bool, steps: str,
    diag_weight: float,
) -> Tensor:  # fmt: skip
    """As :func:`_dtw_backward`, with the shares computed by a kernel over all cells.

    For few long rows: there the fused kernel computes every share in one
    program per batch item, leaving most of the GPU idle.
    """
    B, R, L = D.shape
    up = _empty(D) if steps != "ctc" else None
    left = _empty(D)
    diag = _empty(D) if steps != "orthogonal" else None
    skip_share = _empty(D) if steps == "ctc" else None
    block = 1024
    _shares_kernel[(B * R, triton.cdiv(L, block))](
        D, cost, skip, up, left, diag, skip_share, R, L, diag_weight, soft, _STEPS[steps],
        BLOCK=block, num_warps=4,
    )  # fmt: skip
    e = _accumulate(up, left, diag, skip_share, grad, True)
    return _weigh(e, up, left, diag, diag_weight) if diag_weight != 1.0 else e


def _shift(t: Tensor, rows: int, cols: int, fill: float = 0.0) -> Tensor:
    """t[..., i - rows, j - cols], filled where that is outside the grid."""
    R, L = t.shape[-2:]
    return F.pad(t, (cols, 0, rows, 0), value=fill)[..., :R, :L]


def _accumulate(
    w_up: Tensor | None, w_left: Tensor, w_diag: Tensor | None, w_skip: Tensor | None,
    x: Tensor, reverse: bool,
) -> Tensor:  # fmt: skip
    B, R, L = x.shape
    steps = "ctc" if w_up is None else "orthogonal" if w_diag is None else "symmetric"
    y = _empty(x)
    weights = (None if w is None else w.contiguous() for w in (w_up, w_left, w_diag, w_skip))
    _launch(
        _dag_kernel, B, L, None, None, None, *weights, x.contiguous(), y, None, R, L, 1.0,
        False, _STEPS[steps], reverse,
    )  # fmt: skip
    return y


@torch.library.custom_op("philtorch::dtw_accumulate", mutates_args=())
def accumulate(
    w_up: Tensor | None, w_left: Tensor, w_diag: Tensor | None, w_skip: Tensor | None,
    x: Tensor, reverse: bool,
) -> Tensor:  # fmt: skip
    """y(i, j) = x(i, j) + W_up y(i - 1, j) + W_left y(i, j - 1) + W_diag y(i - 1, j - 1)
    + W_skip y(i - 2, j - 1), or with ``reverse`` its transpose.

    Each edge's weight is stored at its successor; the weights of absent
    steps are None.
    """
    return _accumulate(w_up, w_left, w_diag, w_skip, x, reverse)


@accumulate.register_fake
def _(w_up, w_left, w_diag, w_skip, x, reverse):
    return _empty(x)


# The (row, column) offsets of the up, left, up-left and skip steps.
_OFFSETS = ((1, 0), (0, 1), (1, 1), (2, 1))


def _accumulate_setup(ctx, inputs, output):
    ctx.reverse = inputs[-1]
    ctx.save_for_backward(*inputs[:4], output)


def _accumulate_backward(ctx, grad_y):
    *weights, y = ctx.saved_tensors
    # y = (I - W)^-1 x, so x's gradient is the accumulation the other way, and
    # an edge's is that gradient at its successor times y at its predecessor,
    # or, the other way, the gradient at its predecessor times y at its successor.
    grad_x = accumulate(*weights, grad_y, not ctx.reverse)
    grads = [
        None if w is None
        else _shift(grad_x, *step) * y if ctx.reverse
        else grad_x * _shift(y, *step)
        for w, step in zip(weights, _OFFSETS)
    ]  # fmt: skip
    return (*grads, grad_x, None)


accumulate.register_autograd(_accumulate_backward, setup_context=_accumulate_setup)


def shares(
    D: Tensor, cost: Tensor, skip: Tensor | None, soft: bool, steps: str, diag_weight: float
) -> list[Tensor | None]:
    """The shares of each cell from its up, left, up-left and skip predecessors.

    As the kernels compute them, with PyTorch ops that autograd
    differentiates: a softmax of the negated arguments, in base 2, for soft-DTW, a
    one-hot of the first best for DTW; zero for predecessors outside the
    grid, and all zero for cell (0, 0). None for the steps not in the set.
    """
    present = (steps != "ctc", True, steps != "orthogonal", steps == "ctc")
    args = []
    if present[0]:
        args.append(_shift(D, 1, 0, _INF) + cost)
    args.append(_shift(D, 0, 1, _INF) + cost)
    if present[2]:
        args.append(_shift(D, 1, 1, _INF) + diag_weight * cost)
    if present[3]:
        args.append(torch.where(skip.bool()[..., None], _shift(D, 2, 1, _INF) + cost, _INF))
    args = torch.stack(args)
    outside = torch.isinf(args)
    if soft:
        none = outside.all(0, keepdim=True)
        weights = torch.softmax(torch.where(none, 0.0, -args) * math.log(2), dim=0)
    else:
        weights = F.one_hot(args.argmin(0), len(args)).movedim(-1, 0).to(D.dtype)
    weights = iter(torch.where(outside, 0.0, weights))
    return [next(weights) if p else None for p in present]


@torch.library.custom_op("philtorch::dtw_dp", mutates_args=())
def dtw_dp(cost: Tensor, skip: Tensor | None, soft: bool, steps: str, diag_weight: float) -> Tensor:
    """The accumulated costs D of cost (B, R, L) over the steps ``steps``, the
    diagonal weighted by ``diag_weight``, soft-min in base 2 with ``soft``;
    ``skip`` is the (B, R) mask of the rows the ctc steps may skip into, and
    None for the others. See the module docstring."""
    B, R, L = cost.shape
    if R * L > _MAX_CELLS:
        raise ValueError(
            f"the DTW kernels take at most 2^31 - 1 cells per pair, got {tuple(cost.shape)}"
        )
    D = _empty(cost)
    _launch(
        _dag_kernel, B, L, cost.contiguous(), None, skip, None, None, None, None, None, D, None,
        R, L, diag_weight, soft, _STEPS[steps], False,
    )  # fmt: skip
    return D


@dtw_dp.register_fake
def _(cost, skip, soft, steps, diag_weight):
    return _empty(cost)


def _dtw_dp_setup(ctx, inputs, output):
    cost, skip, *ctx.options = inputs
    ctx.save_for_backward(cost, skip, output)


def _dtw_dp_backward(ctx, grad_D):
    cost, skip, D = ctx.saved_tensors
    soft, steps, diag_weight = ctx.options
    if not torch.is_grad_enabled():
        # The fused kernel runs one program per pair; for a few long rows, the
        # shares are better computed over the whole GPU first.
        B, _, L = D.shape
        backward = _dtw_backward_parallel if B <= 4 and L > 4096 else _dtw_backward
        e = backward(D, cost.contiguous(), skip, grad_D, soft, steps, diag_weight)
        return e, None, None, None, None
    # Gradients of this gradient are wanted: build it from ops that autograd
    # differentiates.
    w_up, w_left, w_diag, w_skip = shares(D, cost, skip, soft, steps, diag_weight)
    e = accumulate(w_up, w_left, w_diag, w_skip, grad_D, True)
    if diag_weight != 1.0:
        e = _weigh(e, w_up, w_left, w_diag, diag_weight)
    return e, None, None, None, None


dtw_dp.register_autograd(_dtw_dp_backward, setup_context=_dtw_dp_setup)
