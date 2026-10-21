"""Triton kernels for DTW, soft-DTW and CTC, differentiable to any order.

The accumulated costs D of a cost matrix (B, R, L) follow, with soft-min for
soft-DTW, in base 2 (-log2 of a sum of powers of 2: the hardware's exp2 and
log2, the multiplies of exp and log saved; callers scale costs by log2 e),

    D(i, j) = min(D(i - 1, j) + c(i, j), D(i, j - 1) + c(i, j),
                  D(i - 1, j - 1) + w c(i, j), D(i - 2, j - 1) + c(i, j)),

over one of three step sets: "orthogonal", up and left; "symmetric", also
the diagonal, weighted by w; and "ctc", left, the diagonal and the skip from
two rows up, the last only into the rows a mask allows. The grid is cut into
tiles of TR rows by TL columns, run as a wavefront: one launch per tile
anti-diagonal, each tile a program, so that a long pair fills the GPU. A
tile loops over its rows. Within row i, with t(j) the minimum over the steps
from earlier rows, the row is D(j) = min(t(j), c(j) + D(j - 1)): a scan of
the maps x -> min(A, C + x) over the tile's columns, applied to the D of
the column before the tile, already written by an earlier launch. The
diagonal and skip terms read earlier rows one column to the left, from the
D already written.

Derivatives. With pi(s, p) the share of each cell s's minimum taken from its
predecessor p (a one-hot of the best for DTW, a softmax of the negated
arguments for soft-DTW), the gradient of D(s) is the reverse accumulation

    e(s) = g(s) + sum over successors t of s of pi(t, s) e(t),

and the cost's gradient is e(s) times sum over p of pi(s, p) w(s, p), the
step weights of the arguments, which is e(s) itself when w = 1. The shares
are recomputed from the stored D and the cost, not taken from the scan's own
minima, which can differ from the stored D in the last bit.

dtw_dp's backward is one op, :func:`dtw_dp_backward`, in eager and compiled
code alike: :func:`_dtw_backward`, which computes the shares as it scans each
row from the right, without storing them. Its own backward, for higher
derivatives, rebuilds the same gradient from :func:`shares`, PyTorch ops, and
:func:`accumulate`, the linear accumulation over the grid in either
direction, whose backward is itself in the other direction, and
differentiates that.

Two kernels do all of it. ``_dag_kernel`` is the row-by-row recursion in
either semiring, min-plus (or soft-min-plus) for the forward pass and
linear for every accumulation, the first-order backward's included;
``_shares_kernel`` computes, for CTC's forced alignment, each cell's best
step back over all cells at once; with the ctc steps, both take the (B, R)
skip mask.

Feature mode. With the features x (B, N, F) and y (B, M, F), the kernels
compute each cost they need on the spot, as the squared distance of the
two rows from direct differences, which don't cancel as |x|^2 + |y|^2 - 2 x.y
does, so that no (B, N, M) cost is stored. A window of allowed columns per
row of x masks it by a large finite cost, and the shear, for the asymmetric
steps, and the transpose, to keep the rows the shorter side, map the
kernels' cells to those of x and y.
"""

import math
import struct

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


# The kernels take contiguous (B, R, L) tensors. In _dag_kernel, lanes past L
# hold padding; the scans run left to right, so it never reaches the real
# columns.


@triton.jit
def _cost(
    cost_ptr, fx_ptr, fy_ptr, win_ptr, b, r, cols, valid, R, L, N, M, F, big_bits, scale_bits,
    SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr, BF: tl.constexpr,
):  # fmt: skip
    """The costs of cells (r, cols) of pair b, 0 where not valid: from the cost
    tensor, or in feature mode, with ``fx_ptr``, from the features.

    The cell is (a, s) = (cols, r) with TRANSPOSE, else (r, cols), of rows a
    and columns s; with SHEAR it is x's row a + s, clamped to N - 1, and y's a,
    else x's a and y's s. The cost is the squared distance times the scale
    (the bits of a float64: a kernel's float scalars are float32), ``big``
    (likewise) outside the window, columns ``lo`` to ``hi`` of each row of x.
    """
    if fx_ptr is None:
        c = tl.load(cost_ptr + b * R * L + r * L + cols, mask=valid, other=0.0)
    else:
        a = cols if TRANSPOSE else r
        s = r if TRANSPOSE else cols
        n = tl.broadcast_to(tl.minimum(a + s if SHEAR else a, N - 1), cols.shape)
        m = tl.broadcast_to(a if SHEAR else s, cols.shape)
        f_x = fx_ptr + b * N * F + n[:, None].to(tl.int64) * F
        f_y = fy_ptr + b * M * F + m[:, None].to(tl.int64) * F
        c = tl.zeros(cols.shape, fx_ptr.dtype.element_ty)
        for f0 in range(0, F, BF):
            f = (f0 + tl.arange(0, BF))[None, :]
            ok = valid[:, None] & (f < F)
            d = tl.load(f_x + f, mask=ok, other=0.0) - tl.load(f_y + f, mask=ok, other=0.0)
            c += tl.sum(d * d, axis=1)
        dtype = c.dtype
        c *= scale_bits.to(tl.int64).to(tl.float64, bitcast=True).to(dtype)
        if win_ptr is not None:
            at = win_ptr + (b * N + n) * 2
            outside = (m < tl.load(at, mask=valid, other=0)) | (
                m > tl.load(at + 1, mask=valid, other=0)
            )
            c = tl.where(outside, big_bits.to(tl.int64).to(tl.float64, bitcast=True).to(dtype), c)
        c = tl.where(valid, c, 0.0)
    return c


@triton.jit
def _shares(
    D_ptr, cost_ptr, fx_ptr, fy_ptr, win_ptr, skip_ptr, b, r, cols, R, L, N, M, F, big_bits,
    scale_bits, diag_weight, SOFT: tl.constexpr, STEPS: tl.constexpr, SHEAR: tl.constexpr,
    TRANSPOSE: tl.constexpr, BF: tl.constexpr,
):  # fmt: skip
    """The shares of cells (r, cols) from their up, left, up-left and skip predecessors.

    Zero for cells outside the grid and for predecessors outside it or not
    among the steps.
    """
    valid = (cols >= 0) & (cols < L) & (r >= 0) & (r < R)
    c = _cost(
        cost_ptr, fx_ptr, fy_ptr, win_ptr, b, r, cols, valid, R, L, N, M, F, big_bits, scale_bits,
        SHEAR, TRANSPOSE, BF,
    )  # fmt: skip
    return _shares_of(D_ptr, c, skip_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS)


@triton.jit
def _shares_of(
    D_ptr, c, skip_ptr, b, r, cols, R, L, diag_weight, SOFT: tl.constexpr, STEPS: tl.constexpr
):
    """:func:`_shares` from the cells' costs c."""
    valid = (cols >= 0) & (cols < L) & (r >= 0) & (r < R)
    offset = b * R * L + r * L + cols
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
def _shares_kernel(D_ptr, cost_ptr, skip_ptr, back_ptr, R, L, BLOCK: tl.constexpr):
    """Every cell's best step back under the hard ctc steps, as int8, one program per
    BLOCK cells of a row: 0 to stay in its row, 1 to advance from the row above, 2 to
    skip from two rows above."""
    batch_row = tl.program_id(0)
    b = (batch_row // R).to(tl.int64)
    r = batch_row % R
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    _, _, diag, skip = _shares(
        D_ptr, cost_ptr, None, None, None, skip_ptr, b, r, cols, R, L, 0, 0, 0, 0, 0, 1.0, False,
        _CTC, False, False, 1,
    )  # fmt: skip
    offset = b * R * L + r * L + cols
    tl.store(back_ptr + offset, (diag + 2 * skip).to(tl.int8), mask=cols < L)


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


@triton.jit(
    do_not_specialize=["N", "M", "F", "big_bits", "scale_bits", "d", "I_lo", "n_tiles", "TR"]
)
def _dag_kernel(
    cost_ptr, D_ptr, skip_rows_ptr, up_ptr, left_ptr, diag_ptr, skip_ptr, x_ptr, y_ptr,
    out_ptr, fx_ptr, fy_ptr, win_ptr, N, M, F, big_bits, scale_bits, R, L, diag_weight, d, I_lo,
    n_tiles, TR, SOFT: tl.constexpr, STEPS: tl.constexpr, REVERSE: tl.constexpr,
    TL: tl.constexpr, SEEDED: tl.constexpr, SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr,
    BF: tl.constexpr,
):  # fmt: skip
    """y(s) = x(s) (+) the sum over steps of W(s, p) (x) y(p), predecessors p; with
    REVERSE, the transpose, over successors t of W(t, s) (x) y(t). Each edge's
    weight is stored at its successor.

    Three uses, by the pointers given:

    * the forward pass, with ``cost_ptr`` alone, or ``fx_ptr``, ``fy_ptr``
      and ``win_ptr`` (feature mode; see :func:`_cost`) in its place:
      min-plus, or soft-min-plus with ``SOFT``; every step's weight is the
      cost of the cell it enters, the diagonal's times ``diag_weight``, the
      ctc steps' skips gated by the (B, R) mask ``skip_rows_ptr``;
      D(0, 0) = c(0, 0);
    * the first-order backward, with ``cost_ptr`` or the features and the
      forward pass's ``D_ptr``, REVERSE: linear, x the gradient of D, the
      weights the shares, computed from D and the cost (soft with ``SOFT``)
      as each row is reached and kept for the rows above; ``out_ptr``, if not
      None, takes the cost's gradient when the diagonal is weighted;
    * the linear accumulation, with x and the weights of the steps in the
      set, the others None.

    A wavefront: the program (tile, b) of launch ``d`` is the tile
    ``I_lo + tile % n_tiles`` of tile row and ``d`` minus that of tile
    column, TR rows by TL columns, the grid mirrored with REVERSE. It reads
    only earlier launches' y, written whole. Row by row, the left step makes
    each row a scan of the maps x -> A (+) (C (x) x), from the right with
    REVERSE, applied to y of the column before the tile, whose weight is the
    row's own. The other rows' terms come from y already written, after a
    barrier; the first row's from the tile above. With ``D_ptr`` the shares
    of the rows below, kept in registers, start from the cells themselves.
    The neighbours' offsets are constants: offsets computed from REVERSE
    would cost a fifth of the time on long rows.
    """
    SHARES: tl.constexpr = D_ptr is not None
    FORWARD: tl.constexpr = (cost_ptr is not None or fx_ptr is not None) and not SHARES
    b = (tl.program_id(0) // n_tiles).to(tl.int64)
    i = I_lo + tl.program_id(0) % n_tiles
    j = d - i
    batch = b * R * L
    lanes = tl.arange(0, TL)
    pos = j * TL + lanes  # the column's place in the scan
    mask = pos < L
    cols = (L - 1 - pos) if REVERSE else pos
    first = i * TR  # the tile's first row's place
    # The row before it, the column before the tile in each row.
    above = (R - first) if REVERSE else (first - 1)
    edge = (L - j * TL) if REVERSE else (j * TL - 1)
    zero = _KERNEL_INF if FORWARD else 0.0  # the semiring's zero
    y_near = tl.load(y_ptr + batch + above * L + cols, mask=mask & (i > 0), other=zero)
    if SHARES:
        # The shares of the rows below, kept from earlier rows: row r + 1
        # straight down at j and diagonally at j + 1, row r + 2 by a skip at
        # j + 1. None below the last row.
        up_below, _, _, _ = _shares(
            D_ptr, cost_ptr, fx_ptr, fy_ptr, win_ptr, skip_rows_ptr, b, above, cols, R, L, N, M, F,
            big_bits, scale_bits, diag_weight, SOFT, STEPS, SHEAR, TRANSPOSE, BF,
        )  # fmt: skip
        _, _, diag_below_next, skip_below_next = _shares(
            D_ptr, cost_ptr, fx_ptr, fy_ptr, win_ptr, skip_rows_ptr, b, above, cols + 1, R, L, N,
            M, F, big_bits, scale_bits, diag_weight, SOFT, STEPS, SHEAR, TRANSPOSE, BF,
        )  # fmt: skip
        _, _, _, skip_below2_next = _shares(
            D_ptr, cost_ptr, fx_ptr, fy_ptr, win_ptr, skip_rows_ptr, b, above + 1, cols + 1, R, L,
            N, M, F, big_bits, scale_bits, diag_weight, SOFT, STEPS, SHEAR, TRANSPOSE, BF,
        )  # fmt: skip
    for k in range(0, tl.minimum(TR, R - first)):
        p = first + k
        r = (R - 1 - p) if REVERSE else p
        row = batch + r * L + cols
        rows_1 = p > 0
        # Recomputed per row: kept live, it costs registers that long rows lack.
        side = mask & (pos > 0)
        if SEEDED:
            left = tl.load(y_ptr + batch + r * L + edge, mask=j > 0, other=zero)
        if FORWARD:
            c = _cost(
                cost_ptr, fx_ptr, fy_ptr, win_ptr, b, r, cols, mask, R, L, N, M, F, big_bits,
                scale_bits, SHEAR, TRANSPOSE, BF,
            )  # fmt: skip
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
                ok &= p > 1
                if FORWARD:
                    ok &= tl.load(skip_rows_ptr + b * R + r) != 0
                    y = tl.load(_neighbour(y_ptr, row, L, 2, 1, REVERSE), mask=ok, other=zero)
                    skip_term = y + c
                    base = _min(base, skip_term, SOFT)
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
            base = tl.where((p == 0) & (pos == 0), c, base)
        elif SHARES:
            # Row r's shares at j, for the rows above, and at j + 1, whose left
            # share takes e(r, j + 1) into e(r, j).
            if fx_ptr is None:
                up_here, left_here, diag_here, _ = _shares(
                    D_ptr, cost_ptr, fx_ptr, fy_ptr, win_ptr, skip_rows_ptr, b, r, cols, R, L, N,
                    M, F, big_bits, scale_bits, diag_weight, SOFT, STEPS, SHEAR, TRANSPOSE, BF,
                )  # fmt: skip
                _, a, diag_next, skip_next = _shares(
                    D_ptr, cost_ptr, fx_ptr, fy_ptr, win_ptr, skip_rows_ptr, b, r, cols + 1, R, L,
                    N, M, F, big_bits, scale_bits, diag_weight, SOFT, STEPS, SHEAR, TRANSPOSE, BF,
                )  # fmt: skip
            else:
                # Each cost once: column j + 1's is the previous lane's (REVERSE),
                # passed through this row of y, which the row's result then
                # overwrites; the first lane's is the next tile's, computed here.
                c = _cost(
                    cost_ptr, fx_ptr, fy_ptr, win_ptr, b, r, cols, mask, R, L, N, M, F, big_bits,
                    scale_bits, SHEAR, TRANSPOSE, BF,
                )  # fmt: skip
                tl.store(y_ptr + row, c, mask=mask)
                tl.debug_barrier()
                c_next = tl.load(y_ptr + row + 1, mask=mask & (lanes > 0), other=0.0)
                first_lane = (lanes == 0) & (cols + 1 < L)
                c_edge = _cost(
                    cost_ptr, fx_ptr, fy_ptr, win_ptr, b, r, cols + 1, first_lane, R, L, N, M, F,
                    big_bits, scale_bits, SHEAR, TRANSPOSE, BF,
                )  # fmt: skip
                c_next = tl.where(lanes == 0, c_edge, c_next)
                up_here, left_here, diag_here, _ = _shares_of(
                    D_ptr, c, skip_rows_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS
                )
                _, a, diag_next, skip_next = _shares_of(
                    D_ptr, c_next, skip_rows_ptr, b, r, cols + 1, R, L, diag_weight, SOFT, STEPS
                )
        else:
            a = tl.load(
                _neighbour(left_ptr, row, L, 0 * REVERSE, 1 * REVERSE, REVERSE),
                mask=side,
                other=0.0,
            )
        # Seeded with the column before the tile; unseeded, the scan's other half is
        # left to the compiler to drop.
        if not FORWARD:
            mult, y_near = tl.associative_scan((a, base), 0, _compose_affine)
            if SEEDED:
                y_near += mult * left
        elif SOFT:
            y_near, scan_c = tl.associative_scan((base, a), 0, _compose_softmin)
            if SEEDED:
                y_near = _softmin(y_near, scan_c + left)
        else:
            y_near, scan_c = tl.associative_scan((base, a), 0, _compose_min)
            if SEEDED:
                y_near = tl.minimum(y_near, scan_c + left)
        if SHARES and fx_ptr is not None:
            tl.debug_barrier()  # every lane has read the costs passed through y
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


@triton.jit(do_not_specialize=["N", "M", "F", "big_bits", "scale_bits"])
def _backtrack_kernel(
    D_ptr, cost_ptr, fx_ptr, fy_ptr, win_ptr, ends_ptr, path_ptr, N, M, F, big_bits, scale_bits,
    R, L, P, diag_weight, STEPS: tl.constexpr, BLOCK: tl.constexpr, SHEAR: tl.constexpr,
    TRANSPOSE: tl.constexpr, BF: tl.constexpr,
):  # fmt: skip
    """Each pair's optimal DTW path, walked from its end cell back to (0, 0).

    Each cell's best step back is the first best of up, left and up-left,
    from D and the cost as the hard shares choose, so the path is the hard
    gradient's. Within a row the path runs left until a cell steps up or
    diagonally: the walk takes BLOCK cells of the row at a time and finds
    that exit as the last such cell by a max over the lanes, so a run of
    left steps costs a step per BLOCK cells. The cells, the end first, go to
    path (B, P, 2) as (row, column). One program per pair; each step waits
    on the last, but the walk stays in cache. (The ctc steps take one frame
    each, so their traceback is parallel instead: :func:`ctc_viterbi` and
    :func:`philtorch._trace_kernels.trace`.)
    """
    b = tl.program_id(0).to(tl.int64)
    batch = b * R * L
    out = path_ptr + b * P * 2
    i = tl.load(ends_ptr + 2 * b)
    j = tl.load(ends_ptr + 2 * b + 1)
    written = i * 0
    lanes = tl.arange(0, BLOCK)
    while i >= 0:
        cols = j - (BLOCK - 1) + lanes
        valid = cols >= 0
        cell = batch + i * L + cols
        c = _cost(
            cost_ptr, fx_ptr, fy_ptr, win_ptr, b, i, cols, valid, R, L, N, M, F, big_bits,
            scale_bits, SHEAR, TRANSPOSE, BF,
        )  # fmt: skip
        up = tl.load(D_ptr + cell - L, mask=valid & (i > 0), other=_KERNEL_INF) + c
        left = tl.load(D_ptr + cell - 1, mask=valid & (cols > 0), other=_KERNEL_INF) + c
        best = tl.minimum(up, left)
        if STEPS != _ORTHOGONAL:
            diag = (
                tl.load(D_ptr + cell - L - 1, mask=valid & (i > 0) & (cols > 0), other=_KERNEL_INF)
                + diag_weight * c
            )
            best = tl.minimum(best, diag)
        is_up = up == best
        is_left = (left == best) & ~is_up
        # The row's exit: the last cell that steps up or diagonally, or (0, 0),
        # which has no step back.
        leaves = valid & (~is_left | (best == _KERNEL_INF))
        exit_col = tl.max(tl.where(leaves, cols, -1), axis=0)
        found = exit_col >= 0
        # This block's cells of the path, from column j down to the exit.
        stop = tl.where(found, exit_col, j - BLOCK + 1)
        keep = valid & (cols >= stop)
        slot = out + 2 * (written + j - cols)
        tl.store(slot, i + 0 * cols, mask=keep)
        tl.store(slot + 1, cols, mask=keep)
        written += j - stop + 1
        at_exit = cols == exit_col
        exit_up = tl.max(tl.where(at_exit & is_up, 1, 0), axis=0) == 1
        origin = tl.max(tl.where(at_exit & (best == _KERNEL_INF), 1, 0), axis=0) == 1
        next_i = tl.where(found, tl.where(origin, -1, i - 1), i)
        j = tl.where(found, tl.where(exit_up, exit_col, exit_col - 1), j - BLOCK)
        i = next_i


def _feature_block(F: int, lanes: int) -> int:
    """The features a kernel reads at once per lane, within a register budget."""
    return max(1, min(8, 8192 // lanes, triton.next_power_of_2(F)))


def _features(feat, lanes: int):
    """feat = (x, y, window, big, scale, shear, transpose), or None for the cost
    tensor mode, as the kernels' arguments: (fx, fy, window, N, M, F, big_bits,
    scale_bits) and (SHEAR, TRANSPOSE, BF). ``big`` and ``scale`` go as the bits of
    a float64, as the kernels' float scalars are float32."""
    if feat is None:
        return (None, None, None, 0, 0, 0, 0, 0), (False, False, 1)
    x, y, window, big, scale, shear, transpose = feat
    x, y = x.contiguous(), y.contiguous()
    window = None if window is None else window.contiguous()
    bits = (struct.unpack("<q", struct.pack("<d", v))[0] for v in (big, scale))
    N, F = x.shape[1:]
    arguments = (x, y, window, N, y.shape[1], F, *bits)
    return arguments, (shear, transpose, _feature_block(F, lanes))


def _backtrack(D, cost, feat, ends, steps, diag_weight):
    D = D.contiguous()
    B, R, L = D.shape
    P = R + L - 1
    walk = ends.new_full((B, P, 2), -1)
    # Blocks of about twice the path's cells per row, L / R on rows of L >= R,
    # one more without the diagonal: one cell at a time on near-square grids
    # with it, where wider blocks would mostly be wasted. Measured on an RTX
    # 5060 Ti.
    per_row = L / R + (steps == "orthogonal")
    block = 1 if per_row < 1.5 else min(triton.next_power_of_2(int(2 * per_row)), 128)
    (fx, fy, window, *sizes), (shear, transpose, bf) = _features(feat, block)
    _backtrack_kernel[(B,)](
        D, None if cost is None else cost.contiguous(), fx, fy, window, ends.contiguous(), walk,
        *sizes, R, L, P, diag_weight, _STEPS[steps], block, shear, transpose, bf, num_warps=1,
    )  # fmt: skip
    # The walk runs from the end: each path's cells, reversed.
    count = (walk[..., 0] >= 0).sum(1, keepdim=True)
    t = torch.arange(P, device=D.device)
    path = walk.gather(1, (count - 1 - t).clamp(min=0)[..., None].expand(-1, -1, 2))
    return torch.where((t < count)[..., None], path, -1)


@torch.library.custom_op("philtorch::dtw_backtrack", mutates_args=())
def backtrack(D: Tensor, cost: Tensor, ends: Tensor, steps: str, diag_weight: float) -> Tensor:
    """Each pair's optimal path on the kernels' grid, from its end cell ``ends`` (B, 2)
    to (0, 0), as (row, column) pairs (B, R + L - 1, 2) in order, -1 past the path."""
    return _backtrack(D, cost, None, ends, steps, diag_weight)


@backtrack.register_fake
def _(D, cost, ends, steps, diag_weight):
    B, R, L = D.shape
    return ends.new_empty(B, R + L - 1, 2)


@torch.library.custom_op("philtorch::dtw_backtrack_features", mutates_args=())
def backtrack_features(
    D: Tensor, x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float,
    ends: Tensor, steps: str, diag_weight: float, shear: bool, transpose: bool,
) -> Tensor:  # fmt: skip
    """:func:`backtrack` with the costs of :func:`dtw_dp_features`."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _backtrack(D, None, feat, ends, steps, diag_weight)


@backtrack_features.register_fake
def _(D, x, y, window, big, scale, ends, steps, diag_weight, shear, transpose):
    B, R, L = D.shape
    return ends.new_empty(B, R + L - 1, 2)


def _num_warps(TL: int, backward: bool = False) -> int:
    """Warps per tile by its columns, measured on an RTX 5060 Ti.

    The backward kernel keeps more of each row live.
    """
    if not backward:
        warps = 4 if TL <= 256 else 8 if TL <= 1024 else 16
    else:
        warps = 4 if TL <= 128 else 8 if TL <= 256 else 16
    # A program has at most 1024 threads, and AMD's datacenter GPUs have
    # 64-thread warps.
    return min(warps, 16) if torch.version.hip else warps


def _tile(B: int, R: int, L: int, features: bool = False) -> tuple[int, int]:
    """The tile's rows and columns, measured on an RTX 5060 Ti: narrower tiles give
    a few pairs more programs per launch, wider ones fewer launches for many. With
    too few tiles side by side to pay for the launches, a tile is the whole grid,
    for rows of at most _MAX_ROW columns; with features, whose cells cost more,
    two side by side pay."""
    TL = min(256 if B < 8 else 1024, triton.next_power_of_2(L))
    n_cols = triton.cdiv(L, TL)
    side_by_side = B * min(triton.cdiv(R, 32), n_cols)
    if n_cols == 1 or (side_by_side < (2 if features else _MIN_TILES) and L <= _MAX_ROW):
        return R, triton.next_power_of_2(L)
    return 32, TL


# Outputs are allocated contiguous: torch.empty_like keeps a permuted input's
# strides, and the kernels write rows contiguously. Offsets within a pair are
# 32-bit, which keeps long rows fast; the batch offset is 64-bit.
_MAX_CELLS = 2**31 - 1


# The fewest tiles side by side, over the batch, worth a wavefront.
_MIN_TILES = 16
# The longest row a program holds whole: 2^18 lanes compile in about half a
# minute, once.
_MAX_ROW = 2**18
# The cells per program of _shares_kernel.
_SHARES_BLOCK = 1024


def _launch(
    B: int, R: int, L: int, cost, D, skip, up, left, diag, skip_share, x, y, out,
    diag_weight: float, soft: bool, steps: str, reverse: bool, backward: bool = False, feat=None,
):  # fmt: skip
    """Run _dag_kernel over the grid, one launch per tile anti-diagonal; ``feat``, as
    _features takes it, in place of ``cost``."""
    TR, TL = _tile(B, R, L, feat is not None)
    features, flags = _features(feat, TL)
    n_rows, n_cols = triton.cdiv(R, TR), triton.cdiv(L, TL)
    for d in range(n_rows + n_cols - 1):
        lo = max(0, d - n_cols + 1)
        n_tiles = min(n_rows - 1, d) - lo + 1
        _dag_kernel[(n_tiles * B,)](
            cost, D, skip, up, left, diag, skip_share, x, y, out, *features, R, L, diag_weight, d,
            lo, n_tiles, TR, soft, _STEPS[steps], reverse, TL, n_cols > 1, *flags,
            num_warps=_num_warps(TL, backward),
        )  # fmt: skip


def _empty(t: Tensor) -> Tensor:
    return torch.empty_like(t, memory_format=torch.contiguous_format)


def _dtw_backward(
    D: Tensor, cost: Tensor | None, skip: Tensor | None, grad: Tensor, soft: bool, steps: str,
    diag_weight: float, feat=None,
) -> Tensor:  # fmt: skip
    B, R, L = D.shape
    weighted = diag_weight != 1.0
    e = _empty(D)
    out = _empty(D) if weighted else e
    _launch(
        B, R, L, cost, D, skip, None, None, None, None, grad.contiguous(), e,
        out if weighted else None, diag_weight, soft, steps, True, backward=True, feat=feat,
    )  # fmt: skip
    return out


def _weigh(e: Tensor, up: Tensor, left: Tensor, diag: Tensor, diag_weight: float) -> Tensor:
    """The cost's gradient from e, with a weighted diagonal: e times the cell's weight."""
    none = (up + left + diag) == 0  # cell (0, 0)
    return e * torch.where(none, 1.0, up + left + diag_weight * diag)


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
        B, R, L, None, None, None, *weights, x.contiguous(), y, None, 1.0, False, steps, reverse
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
    return _forward(cost, skip, soft, steps, diag_weight)


def _forward(cost, skip, soft, steps, diag_weight, feat=None, grid=None):
    B, R, L = cost.shape if grid is None else grid
    if R * L > _MAX_CELLS:
        raise ValueError(f"the DTW kernels take at most 2^31 - 1 cells per pair, got {(B, R, L)}")
    D = (cost if feat is None else feat[0]).new_empty(B, R, L)
    _launch(
        B, R, L, None if cost is None else cost.contiguous(), None, skip, None, None, None, None,
        None, D, None, diag_weight, soft, steps, False, feat=feat,
    )  # fmt: skip
    return D


@torch.library.custom_op("philtorch::ctc_viterbi", mutates_args=())
def ctc_viterbi(cost: Tensor, skip: Tensor) -> tuple[Tensor, Tensor]:
    """The hard ctc forward pass over cost (B, R, L): D, and each cell's best step back,
    int8, 0 to stay in its row, 1 to advance from the row above, 2 to skip from two
    rows above, the first best under ties as the hard shares choose."""
    skip = skip.contiguous()
    D = _forward(cost, skip, False, "ctc", 1.0)
    B, R, L = D.shape
    # From D afterwards, over all cells at once: recorded in the forward pass,
    # they would add a barrier and a reload to each row's sequential step,
    # nearly doubling it on long rows.
    steps_back = torch.empty_like(D, dtype=torch.int8)
    _shares_kernel[(B * R, triton.cdiv(L, _SHARES_BLOCK))](
        D, cost.contiguous(), skip, steps_back, R, L, BLOCK=_SHARES_BLOCK, num_warps=4,
    )  # fmt: skip
    return D, steps_back


@ctc_viterbi.register_fake
def _(cost, skip):
    return _empty(cost), torch.empty_like(
        cost, dtype=torch.int8, memory_format=torch.contiguous_format
    )


@dtw_dp.register_fake
def _(cost, skip, soft, steps, diag_weight):
    return _empty(cost)


def _dtw_dp_setup(ctx, inputs, output):
    cost, skip, *ctx.options = inputs
    ctx.save_for_backward(cost, skip, output)


def _dtw_dp_backward(ctx, grad_D):
    cost, skip, D = ctx.saved_tensors
    return dtw_dp_backward(D, cost, skip, grad_D, *ctx.options), None, None, None, None


dtw_dp.register_autograd(_dtw_dp_backward, setup_context=_dtw_dp_setup)


@torch.library.custom_op("philtorch::dtw_dp_backward", mutates_args=())
def dtw_dp_backward(
    D: Tensor, cost: Tensor, skip: Tensor | None, grad: Tensor, soft: bool, steps: str,
    diag_weight: float,
) -> Tensor:  # fmt: skip
    """The gradient of cost from ``grad``, D's, through :func:`dtw_dp`, by the
    fused kernels; see the module docstring."""
    skip = None if skip is None else skip.contiguous()
    return _dtw_backward(D.contiguous(), cost.contiguous(), skip, grad, soft, steps, diag_weight)


@dtw_dp_backward.register_fake
def _(D, cost, skip, grad, soft, steps, diag_weight):
    return _empty(D)


def _decomposed_backward(D, cost, skip, grad, soft, steps, diag_weight):
    """:func:`dtw_dp_backward` from ops that autograd differentiates."""
    w_up, w_left, w_diag, w_skip = shares(D, cost, skip, soft, steps, diag_weight)
    e = accumulate(w_up, w_left, w_diag, w_skip, grad, True)
    return _weigh(e, w_up, w_left, w_diag, diag_weight) if diag_weight != 1.0 else e


def _dtw_dp_backward_setup(ctx, inputs, output):
    D, cost, skip, grad, *ctx.options = inputs
    ctx.save_for_backward(D, cost, skip, grad)


def _dtw_dp_backward_backward(ctx, grad_e):
    # The vector-Jacobian product of the decomposed gradient. It is taken
    # with respect to views of D, cost and grad, so that each gradient counts
    # only the paths through its own input, not cost's through D, while
    # staying connected to the inputs: its ops are differentiable, so
    # gradients exist to any order. (torch.func.vjp would take them as
    # independent too, but custom ops such as accumulate don't support it.)
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        inputs = [
            t.view_as(t) if t.requires_grad else t.detach().requires_grad_()
            for t in ctx.saved_tensors[:2] + ctx.saved_tensors[3:]
        ]
        e = _decomposed_backward(*inputs[:2], ctx.saved_tensors[2], inputs[2], *ctx.options)
        grad_D, grad_cost, grad_grad = torch.autograd.grad(
            e, inputs, grad_e, create_graph=create_graph, allow_unused=True
        )
    return grad_D, grad_cost, None, grad_grad, None, None, None


dtw_dp_backward.register_autograd(_dtw_dp_backward_backward, setup_context=_dtw_dp_backward_setup)


def _outside(window: Tensor, M: int) -> Tensor:
    """(B, N, M): the cells outside window (B, N, 2), the allowed columns [lo, hi] per row."""
    return (torch.arange(M, device=window.device) < window[..., :1]) | (
        torch.arange(M, device=window.device) > window[..., 1:]
    )


def _feature_cost(x, y, window, big, scale, shear, transpose, R, L):
    """The kernels' (B, R, L) cost grid in feature mode, from PyTorch ops that autograd
    differentiates: see :func:`_cost` for the cells."""
    N, M = x.size(1), y.size(1)
    cost = x.new_zeros(x.size(0), N, M)
    for f in range(x.size(2)):
        cost = cost + (x[:, :, None, f] - y[:, None, :, f]) ** 2
    cost = cost * scale
    if window is not None:
        cost = torch.where(_outside(window, M), big, cost)
    if shear:
        m = torch.arange(M, device=x.device)[:, None]
        cost = cost[
            :, (m + torch.arange(R if transpose else L, device=x.device)).clamp(max=N - 1), m
        ]
    return cost.mT if transpose else cost


@torch.library.custom_op("philtorch::dtw_dp_features", mutates_args=())
def dtw_dp_features(
    x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float, soft: bool,
    steps: str, diag_weight: float, shear: bool, transpose: bool, R: int, L: int,
) -> Tensor:  # fmt: skip
    """:func:`dtw_dp` over the (B, R, L) grid of the costs ``scale`` times the squared
    distances of x (B, N, F) and y (B, M, F), computed in the kernels (see the
    module docstring), outside ``window`` (B, N, 2), the allowed columns [lo, hi]
    of y per row of x, ``big``. A cell (r, l) is x's row n and y's row m: (a, s)
    = (l, r) with ``transpose``, else (r, l); n = a + s, clamped to N - 1, and m = a
    with ``shear``, else n = a and m = s. Only the steps "orthogonal" and
    "symmetric"."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _forward(None, None, soft, steps, diag_weight, feat, (x.size(0), R, L))


@dtw_dp_features.register_fake
def _(x, y, window, big, scale, soft, steps, diag_weight, shear, transpose, R, L):
    return x.new_empty(x.size(0), R, L)


def _dtw_dp_features_setup(ctx, inputs, output):
    x, y, window, *ctx.options = inputs
    ctx.save_for_backward(x, y, window, output)


def _dtw_dp_features_backward(ctx, grad_D):
    x, y, window, D = ctx.saved_tensors
    big, scale, soft, steps, diag_weight, shear, transpose, _, _ = ctx.options
    E = dtw_dp_backward_features(
        D, x, y, window, big, scale, grad_D, soft, steps, diag_weight, shear, transpose
    )
    # E, the cost's gradient on the grid, back to the (n, m) cells.
    N, M = x.size(1), y.size(1)
    if transpose:
        E = E.mT
    if shear:
        m = torch.arange(M, device=x.device)[:, None]
        cell = ((m + torch.arange(E.size(2), device=x.device)).clamp(max=N - 1) * M + m).flatten()
        E = E.new_zeros(E.size(0), N * M).index_add(1, cell, E.flatten(1)).view(-1, N, M)
    if window is not None:
        E = torch.where(_outside(window, M), 0.0, E)
    # From frames centred on each pair's mean: the products' difference then
    # cancels only their spread, not an offset they share.
    mean = torch.cat([x, y], 1).mean(1, keepdim=True)
    x, y = x - mean, y - mean
    grad_x = 2 * scale * (x * E.sum(2, keepdim=True) - E @ y)
    grad_y = 2 * scale * (y * E.sum(1).unsqueeze(-1) - E.mT @ x)
    return grad_x, grad_y, *(None,) * 10


dtw_dp_features.register_autograd(_dtw_dp_features_backward, setup_context=_dtw_dp_features_setup)


@torch.library.custom_op("philtorch::dtw_dp_backward_features", mutates_args=())
def dtw_dp_backward_features(
    D: Tensor, x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float,
    grad: Tensor, soft: bool, steps: str, diag_weight: float, shear: bool, transpose: bool,
) -> Tensor:  # fmt: skip
    """The gradient of the grid's cost from ``grad``, D's, through
    :func:`dtw_dp_features`, as :func:`dtw_dp_backward` gives it."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _dtw_backward(D.contiguous(), None, None, grad, soft, steps, diag_weight, feat)


@dtw_dp_backward_features.register_fake
def _(D, x, y, window, big, scale, grad, soft, steps, diag_weight, shear, transpose):
    return _empty(D)


def _dtw_dp_backward_features_setup(ctx, inputs, output):
    D, x, y, window, *ctx.options = inputs
    ctx.save_for_backward(D, x, y, window, inputs[6])


def _dtw_dp_backward_features_backward(ctx, grad_e):
    # As _dtw_dp_backward_backward, with the cost rebuilt from x and y.
    D, x, y, window, grad = ctx.saved_tensors
    big, scale, _, soft, steps, diag_weight, shear, transpose = ctx.options
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        D, x, y, grad = (
            t.view_as(t) if t.requires_grad else t.detach().requires_grad_()
            for t in (D, x, y, grad)
        )
        cost = _feature_cost(x, y, window, big, scale, shear, transpose, *D.shape[1:])
        e = _decomposed_backward(D, cost, None, grad, soft, steps, diag_weight)
        grad_D, grad_x, grad_y, grad_grad = torch.autograd.grad(
            e, (D, x, y, grad), grad_e, create_graph=create_graph, allow_unused=True
        )
    return grad_D, grad_x, grad_y, None, None, None, grad_grad, *(None,) * 5


dtw_dp_backward_features.register_autograd(
    _dtw_dp_backward_features_backward, setup_context=_dtw_dp_backward_features_setup
)
