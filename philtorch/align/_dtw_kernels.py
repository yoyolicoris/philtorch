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

Five kernels do all of it. ``_dag_kernel`` is the row-by-row recursion in
either semiring, min-plus (or soft-min-plus) for the forward pass and
linear for every accumulation, the first-order backward's included;
``_shares_kernel`` computes, for CTC's forced alignment, each cell's best
step back over all cells at once; with the ctc steps, both take the (B, R)
skip mask. ``_frontier_kernel`` is the recursion without storing D (below),
``_backtrack_kernel`` walks a DTW path back from its end cell, and, in feature
mode, ``_cost_kernel`` computes the costs the others read (below).

Feature mode. With the features x (B, N, F) and y (B, M, F), no (B, N, M)
cost is stored. Before the launches of a few tile anti-diagonals,
``_cost_kernel`` computes the costs of all their tiles' cells, in parallel
over blocks of cells, each block loading its rows of x and y once per chunk
of features: the squared distance from direct differences, which don't cancel
as |x|^2 + |y|^2 - 2 x.y does. They go to a scratch, a slice per tile, and the
kernels read them there as they read a cost tensor; that, not computing each
cell in the row loop, which reloads the tile's features every row with few
programs in flight, is what keeps large F near the speed of a matrix-product
cost. The scratch holds a chunk of anti-diagonals, at most _CHUNK and
_SCRATCH_CELLS cells, so that the memory is linear in the sides. The backward
and the reverse frontier also read the cost of the row above the tile and of
the column before it, which the slice holds as a halo (the backward's tile
rows are 31, so that a slice's 32 rows fill the cost blocks). A window of
allowed columns per row of x masks it by a large finite cost, and
``_hit_kernel`` flags the tiles with a cell in it, whose costs and recursion
the others skip. The shear, for the asymmetric steps, and the transpose, to
keep the rows the shorter side, map the kernels' cells to those of x and y.
Second derivatives rebuild the costs with :func:`grid_cost`, in the memory of
the grid.

Windows. A DTW, soft or hard, skips the tiles with no cell in the window, as
a band leaves most of the grid out, so that its work follows the window's
area, but for the tiles of a seeded grid's row 0: a skipped tile's D is inf,
not the costs ``big`` adds up to, and its gradient is zero. A path through
masked cells costs at least ``big``, which callers take for no path, and the
soft-min of inf with a finite D is that D, the weight of a ``big`` argument
underflowing to 0, so every other result is unchanged.

Seeded grids. In feature mode, with a seed (B, L), row 0 of D is given and
the rows below run from it, as the rows of a longer grid run from the row
above: a grid cut into blocks of rows ``row0`` on, each seeded with the last
row of the one before. The shares of row 0 are then all zero, so that the
backward's e there, g included, is the seed's gradient, and the cost's row 0
gets none.

Linear memory. :func:`frontier` runs the forward recursion, or the reverse one
from a cell to the end of a sub-grid (an "item"), over such a grid without
storing the cost or D, in the same wavefront of tiles: each tile hands its
last row to the tile below, its last column to the tile to its right, and the
rows are kept three tile rows deep, so that the memory is that of one row and
one column per item (Tralie and Dempsey, ISMIR 2020).
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
    """-log2(2^-x + 2^-y) as min(x, y) - log2(1 + 2^-|x - y|), or inf."""
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
    """x -> a x + b, the multiplier first: no slower than the other order."""
    return a_r * a_l, a_r * b_l + b_r


# The kernels take contiguous (B, R, L) tensors. In _dag_kernel, lanes past L
# hold padding; the scans run left to right, so it never reaches the real
# columns.


@triton.jit
def _cell(r, cols, N, SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr, row0=0):
    """x's row n and y's column m of cells (r, cols): see :func:`_cost`."""
    a = cols if TRANSPOSE else r + row0
    s = r + row0 if TRANSPOSE else cols
    n = tl.broadcast_to(tl.minimum(a + s if SHEAR else a, N - 1), cols.shape)
    return n, tl.broadcast_to(a if SHEAR else s, cols.shape)


@triton.jit
def _window_hit(
    win_ptr, b, r, step, rows, cols, valid, N, SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr,
    row0=0,
):  # fmt: skip
    """Whether a valid cell of the rows r, r + step, ... (``rows`` of them) is
    inside the window, from the window alone."""
    hit = tl.zeros(cols.shape, tl.int32)
    for k in range(0, rows):
        n, m = _cell(r + k * step, cols, N, SHEAR, TRANSPOSE, row0)
        at = win_ptr + (b * N + n) * 2
        inside = (m >= tl.load(at, mask=valid, other=0)) & (
            m <= tl.load(at + 1, mask=valid, other=0)
        )
        hit = tl.maximum(hit, (valid & inside).to(tl.int32))
    return tl.max(hit, axis=0) > 0


@triton.jit
def _cost(
    cost_ptr, fx_ptr, fy_ptr, win_ptr, b, r, cols, valid, R, L, N, M, F, big_bits, scale_bits,
    SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr, BF: tl.constexpr, row0=0,
):  # fmt: skip
    """The costs of cells (r, cols) of pair b, 0 where not valid: from the cost
    tensor, or in feature mode, with ``fx_ptr``, from the features.

    The cell is (a, s) = (cols, r + row0) with TRANSPOSE, else (r + row0, cols),
    of rows a and columns s; with SHEAR it is x's row a + s, clamped to N - 1,
    and y's a, else x's a and y's s. The cost is the squared distance times the
    scale (the bits of a float64: a kernel's float scalars are float32),
    ``big`` (likewise) outside the window, columns ``lo`` to ``hi`` of each row
    of x.
    """
    if fx_ptr is None:
        c = tl.load(cost_ptr + b * R * L + r * L + cols, mask=valid, other=0.0)
    else:
        n, m = _cell(r, cols, N, SHEAR, TRANSPOSE, row0)
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
def _tile_cost(
    tile_ptr, k, lanes, valid, TL: tl.constexpr, HALO: tl.constexpr, SHIFT: tl.constexpr = 0
):
    """Row k of a tile's costs, as :func:`_cost_kernel` left them in its slice
    of scratch, ``tile_ptr``: the cell of lane ``lanes + SHIFT``, 0 where not
    ``valid``. With HALO, the slice has the row k = -1 above the tile and the
    column -1 left of it (SHIFT = -1 reads it, or the lane before)."""
    return tl.load(
        tile_ptr + (k + HALO) * (TL + HALO) + HALO + SHIFT + lanes, mask=valid, other=0.0
    )


@triton.jit
def _frame(items_ptr, s, R, L):
    """The pair b and sub-grid (r0, c0, h, w) of item s of ``items_ptr``, or,
    without items, the whole grid of pair s."""
    if items_ptr is None:
        return s, s * 0, s * 0, R, L
    else:
        item = items_ptr + s * 5
        return (
            tl.load(item), tl.load(item + 1), tl.load(item + 2), tl.load(item + 3),
            tl.load(item + 4),
        )  # fmt: skip


@triton.jit(
    do_not_specialize=["N", "S", "R", "L", "d0", "n_rows", "n_cols", "TR", "P", "row0"]
)  # fmt: skip
def _hit_kernel(
    hit_ptr, items_ptr, win_ptr, N, S, R, L, d0, n_rows, n_cols, TR, P, row0,
    SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr, REVERSE: tl.constexpr, GIVEN: tl.constexpr,
    TL: tl.constexpr,
):  # fmt: skip
    """Whether each tile of the anti-diagonals d0, d0 + 1, ... (axis 1 of the
    grid) has a cell in the window, as :func:`_window_hit` says, to ``hit_ptr``
    at the slot (d - d0) * P plus the tile's program number in its launch: the
    tile decoded as in :func:`_dag_kernel` and :func:`_frontier_kernel`, whose
    grids are P wide (P >= the tiles of a diagonal); ``items_ptr`` as in
    :func:`_frame`. With GIVEN, the tiles holding row 0 are flagged whatever
    the window says: its D is the seed's, and its gradient the seed's."""
    pid = tl.program_id(0)
    dd = tl.program_id(1)
    d = d0 + dd
    lo = tl.maximum(0, d - n_cols + 1)
    n_tiles = tl.minimum(n_rows - 1, d) - lo + 1
    if pid < S * n_tiles:
        i = lo + pid % n_tiles
        j = d - i
        b, r0, c0, h, w = _frame(items_ptr, (pid // n_tiles).to(tl.int64), R, L)
        first = i * TR
        n = tl.where((first < h) & (j * TL < w), tl.minimum(TR, h - first), 0)
        pos = j * TL + tl.arange(0, TL)
        cols = c0 + ((w - 1 - pos) if REVERSE else pos)
        r_first = r0 + ((h - 1 - first) if REVERSE else first)
        hit = _window_hit(
            win_ptr, b, r_first, -1 if REVERSE else 1, n, cols, pos < w, N, SHEAR, TRANSPOSE, row0
        )  # fmt: skip
        if GIVEN:
            hit |= (n > 0) & (first + n == h if REVERSE else first == 0)
        tl.store(hit_ptr + dd * P + pid, hit.to(tl.int8))


@triton.jit(
    do_not_specialize=[
        "N", "M", "big_bits", "scale_bits", "S", "R", "L", "d0", "n_rows", "n_cols", "TR", "P",
        "row0",
    ]
)  # fmt: skip
def _cost_kernel(
    scr_ptr, hit_ptr, items_ptr, fx_ptr, fy_ptr, win_ptr, N, M, F, big_bits, scale_bits, S, R, L,
    d0, n_rows, n_cols, TR, P, row0, SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr,
    REVERSE: tl.constexpr, TL: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    BF: tl.constexpr,
):  # fmt: skip
    """The costs of the cells the tiles of the anti-diagonals d0, d0 + 1, ...
    (axis 2 of the grid) need, from the features, into ``scr_ptr``: the
    squared distance of the two rows from direct differences, as :func:`_cost`
    takes it, a block of BR rows by BC columns of one tile (axis 1) per
    program, each feature chunk of BF loaded once for the block.

    A tile's slice, at (slot + program number) of the (TR + HALO) by
    (TL + HALO) cells, holds its rows from the row above it (the tile's
    first, with HALO) and its columns from the column before it, in the
    order of processing; 0 where there is no cell. HALO, set by REVERSE,
    is what the backward and the reverse frontier read of the tile above and
    the tile to the left. Tiles ``hit_ptr`` (if given) says have no cell in
    the window are left as they are; see :func:`_hit_kernel` for the
    decoding and the slots.
    """
    HALO: tl.constexpr = REVERSE
    blocks_wide: tl.constexpr = (TL + HALO + BC - 1) // BC
    pid = tl.program_id(0)
    dd = tl.program_id(2)
    d = d0 + dd
    lo = tl.maximum(0, d - n_cols + 1)
    n_tiles = tl.minimum(n_rows - 1, d) - lo + 1
    if pid < S * n_tiles:
        i = lo + pid % n_tiles
        j = d - i
        b, r0, c0, h, w = _frame(items_ptr, (pid // n_tiles).to(tl.int64), R, L)
        live = (i * TR < h) & (j * TL < w)
        if hit_ptr is not None:
            live &= tl.load(hit_ptr + dd * P + pid) != 0
        if live:
            k = (tl.program_id(1) // blocks_wide) * BR + tl.arange(0, BR)
            e = (tl.program_id(1) % blocks_wide) * BC + tl.arange(0, BC)
            p = i * TR - HALO + k  # the cell's row and place in the scan
            pos = j * TL - HALO + e
            valid = ((p >= 0) & (p < h))[:, None] & ((pos >= 0) & (pos < w))[None, :]
            r = (r0 + ((h - 1 - p) if REVERSE else p) + row0)[:, None]
            cols = (c0 + ((w - 1 - pos) if REVERSE else pos))[None, :]
            # As in _cell, with the indices clamped into the features: the cells
            # not valid are 0, whatever they read. Without the shear, n and m
            # each follow one of r and cols, and the loads are shared.
            a = cols if TRANSPOSE else r
            s = r if TRANSPOSE else cols
            n = tl.maximum(tl.minimum(a + s if SHEAR else a, N - 1), 0)
            m = tl.maximum(tl.minimum(a if SHEAR else s, M - 1), 0)
            dtype = fx_ptr.dtype.element_ty
            c = tl.zeros([BR, BC], dtype)
            work = valid
            if win_ptr is not None:
                at = win_ptr + (b * N + n) * 2
                outside = (m < tl.load(at)) | (m > tl.load(at + 1))
                work = valid & ~outside
            # A block with no cell in the window is all ``big``.
            if tl.max(work.to(tl.int32)) > 0:
                f_x = fx_ptr + b * N * F + n[:, :, None].to(tl.int64) * F
                f_y = fy_ptr + b * M * F + m[:, :, None].to(tl.int64) * F
                for f0 in range(0, F, BF):
                    f = (f0 + tl.arange(0, BF))[None, None, :]
                    diff = tl.load(f_x + f, mask=f < F, other=0.0)
                    diff -= tl.load(f_y + f, mask=f < F, other=0.0)
                    c += tl.sum(diff * diff, axis=2)
            c *= scale_bits.to(tl.int64).to(tl.float64, bitcast=True).to(dtype)
            if win_ptr is not None:
                c = tl.where(
                    outside, big_bits.to(tl.int64).to(tl.float64, bitcast=True).to(dtype), c
                )
            c = tl.where(valid, c, 0.0)
            tile = scr_ptr + (dd * P + pid).to(tl.int64) * ((TR + HALO) * (TL + HALO))
            tl.store(
                tile + k[:, None] * (TL + HALO) + e[None, :], c,
                mask=(k < TR + HALO)[:, None] & (e < TL + HALO)[None, :],
            )  # fmt: skip


@triton.jit
def _shares(
    D_ptr, cost_ptr, skip_ptr, b, r, cols, R, L, diag_weight, SOFT: tl.constexpr,
    STEPS: tl.constexpr, GIVEN: tl.constexpr = False,
):  # fmt: skip
    """The shares of cells (r, cols) from their up, left, diagonal and skip.

    Zero for cells outside the grid and for predecessors outside it or not
    among the steps; with GIVEN, all zero in row 0, whose D is given.
    """
    valid = (cols >= 0) & (cols < L) & (r >= 0) & (r < R)
    c = tl.load(cost_ptr + b * R * L + r * L + cols, mask=valid, other=0.0)
    return _shares_of(D_ptr, c, skip_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS, GIVEN)


@triton.jit
def _shares_of(
    D_ptr, c, skip_ptr, b, r, cols, R, L, diag_weight, SOFT: tl.constexpr, STEPS: tl.constexpr,
    GIVEN: tl.constexpr = False,
):  # fmt: skip
    """:func:`_shares` from the cells' costs c."""
    valid = (cols >= 0) & (cols < L) & (r >= 0) & (r < R)
    offset = b * R * L + r * L + cols
    absent = tl.full(c.shape, _KERNEL_INF, c.dtype)
    up = absent
    if STEPS != _CTC:
        up = tl.load(D_ptr + offset - L, mask=valid & (r > 0), other=_KERNEL_INF) + c
    has_left = valid & (cols > 0)
    if GIVEN:
        has_left &= r > 0
    left = tl.load(D_ptr + offset - 1, mask=has_left, other=_KERNEL_INF) + c
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
    none = best == _KERNEL_INF  # cell (0, 0), row 0 if GIVEN, or off the grid
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
    """Every cell's best step back under the hard ctc steps, as int8, one
    program per BLOCK cells of a row: 0 to stay in its row, 1 to advance from
    the row above, 2 to skip from two rows above."""
    batch_row = tl.program_id(0)
    b = (batch_row // R).to(tl.int64)
    r = batch_row % R
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    _, _, diag, skip = _shares(
        D_ptr, cost_ptr, skip_ptr, b, r, cols, R, L, 1.0, False, _CTC
    )  # fmt: skip
    offset = b * R * L + r * L + cols
    tl.store(back_ptr + offset, (diag + 2 * skip).to(tl.int8), mask=cols < L)


@triton.jit
def _neighbour(ptr, row, L, ROWS: tl.constexpr, COLS: tl.constexpr, REVERSE: tl.constexpr):
    """ptr at ROWS rows and COLS columns after row's cell (REVERSE) or before.

    An edge's weight is stored at its successor, so it is
    ``_neighbour(ptr, row, L, ROWS * REVERSE, COLS * REVERSE, REVERSE)``: the
    neighbour with REVERSE, else row's own cell.
    """
    if REVERSE:
        return ptr + row + ROWS * L + COLS
    return ptr + row - ROWS * L - COLS


@triton.jit(do_not_specialize=["d", "I_lo", "n_tiles", "TR", "slot"])
def _dag_kernel(
    cost_ptr, D_ptr, skip_rows_ptr, up_ptr, left_ptr, diag_ptr, skip_ptr, x_ptr, y_ptr,
    out_ptr, scr_ptr, hit_ptr, R, L, diag_weight, d, I_lo, n_tiles, TR, slot, SOFT: tl.constexpr,
    STEPS: tl.constexpr, REVERSE: tl.constexpr, TL: tl.constexpr, SEEDED: tl.constexpr,
    GIVEN: tl.constexpr,
):  # fmt: skip
    """y(s) = x(s) (+) the sum over steps of W(s, p) (x) y(p), predecessors p;
    with REVERSE, the transpose, over successors t of W(t, s) (x) y(t). Each
    edge's weight is stored at its successor.

    Three uses, by the pointers given:

    * the forward pass, with ``cost_ptr`` alone, or ``scr_ptr`` and
      ``hit_ptr`` (feature mode; see :func:`_cost_kernel`) in its place:
      min-plus, or soft-min-plus with ``SOFT``; every step's weight is the
      cost of the cell it enters, the diagonal's times ``diag_weight``, the
      ctc steps' skips gated by the (B, R) mask ``skip_rows_ptr``;
      D(0, 0) = c(0, 0); with GIVEN, row 0 is the (B, L) ``x_ptr`` instead,
      and the grid starts from it;
    * the first-order backward, with ``cost_ptr`` or the scratch and the
      forward pass's ``D_ptr``, REVERSE: linear, x the gradient of D, the
      weights the shares, computed from D and the cost (soft with ``SOFT``)
      as each row is reached and kept for the rows above; ``out_ptr``, if not
      None, takes the cost's gradient when the diagonal is weighted; with
      GIVEN, row 0 has no predecessors, so its result is the gradient of the
      given row;
    * the linear accumulation, with x and the weights of the steps in the
      set, the others None.

    A wavefront: the program (tile, b) of launch ``d`` is the tile
    ``I_lo + tile % n_tiles`` of tile row and ``d`` minus that of tile
    column, TR rows by TL columns, the grid mirrored with REVERSE. It reads
    only earlier launches' y, written whole, and the costs its slice of the
    scratch holds, at ``slot`` plus its program number. Row by row, the left
    step makes each row a scan of the maps x -> A (+) (C (x) x), from the
    right with REVERSE, applied to y of the column before the tile, whose
    weight is the row's own. The other rows' terms come from y already
    written, after a barrier; the first row's from the tile above. With
    ``D_ptr`` the shares of the rows below, kept in registers, start from the
    cells themselves. The neighbours' offsets are constants: offsets computed
    from REVERSE would cost a fifth of the time on long rows.
    """
    SHARES: tl.constexpr = D_ptr is not None
    FEAT: tl.constexpr = scr_ptr is not None
    FORWARD: tl.constexpr = (cost_ptr is not None or FEAT) and not SHARES
    SKIP: tl.constexpr = hit_ptr is not None
    HALO: tl.constexpr = REVERSE
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
    n_rows = tl.minimum(TR, R - first)
    if FEAT:
        tile = scr_ptr + (slot + tl.program_id(0)).to(tl.int64) * ((TR + HALO) * (TL + HALO))
    if SKIP:
        # A tile with no cell in the window holds the semiring's zero: D = inf.
        r_first = (R - 1 - first) if REVERSE else first
        skipped = tl.load(hit_ptr + slot + tl.program_id(0)) == 0
        for k in range(0, tl.where(skipped, n_rows, 0)):
            row = batch + (r_first + (-k if REVERSE else k)) * L + cols
            tl.store(y_ptr + row, tl.full([TL], zero, y_ptr.dtype.element_ty), mask=mask)
        n_rows = tl.where(skipped, 0, n_rows)
    if SHARES:
        # The shares of the rows below, kept from earlier rows: row r + 1
        # straight down at j and diagonally at j + 1, row r + 2 by a skip at
        # j + 1. None below the last row.
        if FEAT:
            # The row above's costs, kept in the scratch's halo; no skips.
            up_below, _, _, _ = _shares_of(
                D_ptr, _tile_cost(tile, -1, lanes, mask, TL, HALO), skip_rows_ptr, b, above, cols,
                R, L, diag_weight, SOFT, STEPS, GIVEN,
            )  # fmt: skip
            _, _, diag_below_next, skip_below_next = _shares_of(
                D_ptr, _tile_cost(tile, -1, lanes, mask, TL, HALO, -1), skip_rows_ptr, b, above,
                cols + 1, R, L, diag_weight, SOFT, STEPS, GIVEN,
            )  # fmt: skip
            skip_below2_next = tl.zeros_like(skip_below_next)
        else:
            up_below, _, _, _ = _shares(
                D_ptr, cost_ptr, skip_rows_ptr, b, above, cols, R, L, diag_weight, SOFT, STEPS,
                GIVEN,
            )  # fmt: skip
            _, _, diag_below_next, skip_below_next = _shares(
                D_ptr, cost_ptr, skip_rows_ptr, b, above, cols + 1, R, L, diag_weight, SOFT, STEPS,
                GIVEN,
            )  # fmt: skip
            _, _, _, skip_below2_next = _shares(
                D_ptr, cost_ptr, skip_rows_ptr, b, above + 1, cols + 1, R, L, diag_weight, SOFT,
                STEPS, GIVEN,
            )  # fmt: skip
    for k in range(0, n_rows):
        p = first + k
        r = (R - 1 - p) if REVERSE else p
        row = batch + r * L + cols
        rows_1 = p > 0
        # Recomputed per row: kept live, it costs registers that long rows lack.
        side = mask & (pos > 0)
        if SEEDED:
            left = tl.load(y_ptr + batch + r * L + edge, mask=j > 0, other=zero)
        if FORWARD:
            if FEAT:
                c = _tile_cost(tile, k, lanes, mask, TL, HALO)
            else:
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
            if not FEAT:
                up_here, left_here, diag_here, _ = _shares(
                    D_ptr, cost_ptr, skip_rows_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS,
                    GIVEN,
                )  # fmt: skip
                _, a, diag_next, skip_next = _shares(
                    D_ptr, cost_ptr, skip_rows_ptr, b, r, cols + 1, R, L, diag_weight, SOFT, STEPS,
                    GIVEN,
                )  # fmt: skip
            else:
                # Column j + 1's cost is the previous lane's (REVERSE), the
                # first lane's the halo's.
                c = _tile_cost(tile, k, lanes, mask, TL, HALO)
                c_next = _tile_cost(tile, k, lanes, mask, TL, HALO, -1)
                up_here, left_here, diag_here, _ = _shares_of(
                    D_ptr, c, skip_rows_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS, GIVEN
                )
                _, a, diag_next, skip_next = _shares_of(
                    D_ptr, c_next, skip_rows_ptr, b, r, cols + 1, R, L, diag_weight, SOFT, STEPS,
                    GIVEN,
                )  # fmt: skip
        else:
            a = tl.load(
                _neighbour(left_ptr, row, L, 0 * REVERSE, 1 * REVERSE, REVERSE),
                mask=side,
                other=0.0,
            )
        # Seeded with the column before the tile; unseeded, the scan's other
        # half is left to the compiler to drop.
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
        if FORWARD and GIVEN:
            y_near = tl.where(p == 0, tl.load(x_ptr + b * L + cols, mask=mask, other=zero), y_near)
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


@triton.jit(do_not_specialize=["d", "I_lo", "n_tiles", "TR", "H", "slot"])
def _frontier_kernel(
    items_ptr, rows_ptr, cols_ptr, scr_ptr, hit_ptr, diag_weight, d, I_lo, n_tiles, TR, W, H,
    slot, SOFT: tl.constexpr, SYMMETRIC: tl.constexpr, REVERSE: tl.constexpr, TL: tl.constexpr,
    SEEDED: tl.constexpr,
):  # fmt: skip
    """:func:`frontier`'s wavefront, as :func:`_dag_kernel`'s forward pass, with
    REVERSE mirrored, but over the items (S, 5) of the grid of the costs in the
    scratch ``scr_ptr`` (see :func:`_cost_kernel`), at ``slot``.

    The program (item, tile) of launch ``d`` is the item's tile ``I_lo + tile %
    n_tiles`` of tile row, in the order of processing (the rows from the item's
    last with REVERSE, the columns likewise), and ``d`` minus that of tile
    column; it does nothing if the item has no such tile. Only these are kept,
    all in global memory: ``rows_ptr`` (S, 3, W) holds the rows of tile row
    i in slot i % 3, so that the tile reads the last row of the tile row above
    from slot (i - 1) % 3, the diagonal's corner from the tile to its left
    included, which tile row i + 2 is the first to overwrite, two launches after
    the last read. ``cols_ptr`` (S, H) holds, per processed row, the last
    column of the latest tile in it, for the tile to its right. A row's
    previous row, shifted by a lane for the diagonal, goes through the tile's
    own slot after a barrier, with the lane 0 from the left column of the
    previous row.

    With REVERSE the weight of a step is its successor's cost: the row's scan
    maps are min(T, c(next column) + x), with the next column's cost the
    scratch's, the tile edge's in its halo, and the row below's terms use its
    costs, kept.
    """
    SKIP: tl.constexpr = hit_ptr is not None
    HALO: tl.constexpr = REVERSE
    s = (tl.program_id(0) // n_tiles).to(tl.int64)
    i = I_lo + tl.program_id(0) % n_tiles
    j = d - i
    item = items_ptr + s * 5
    h = tl.load(item + 3)
    w = tl.load(item + 4)
    first = i * TR
    n_rows = tl.where((first < h) & (j * TL < w), tl.minimum(TR, h - first), 0).to(tl.int32)
    lanes = tl.arange(0, TL)
    pos = j * TL + lanes  # the column's place in the scan
    mask = pos < w
    own = (s * 3 + i % 3) * W
    above = (s * 3 + (i + 2) % 3) * W
    tile = scr_ptr + (slot + tl.program_id(0)).to(tl.int64) * ((TR + HALO) * (TL + HALO))
    y_near = tl.load(rows_ptr + above + pos, mask=mask & (i > 0), other=_KERNEL_INF)
    inf = tl.full([TL], _KERNEL_INF, rows_ptr.dtype.element_ty)
    left_prev = inf
    below = mask & (i > 0) & (n_rows > 0)  # not past the item: off its grid
    if SKIP:
        # A tile with no cell in the window hands on inf, and does nothing else.
        skipped = (tl.load(hit_ptr + slot + tl.program_id(0)) == 0) & (n_rows > 0)
        tl.store(rows_ptr + own + pos, inf, mask=mask & skipped)
        if SEEDED:
            for kk in range(0, tl.where(skipped, n_rows, 0)):
                tl.store(
                    cols_ptr + s * H + first + kk + lanes * 0, inf,
                    mask=(lanes == TL - 1) & ((j + 1) * TL < w),
                )  # fmt: skip
        n_rows = tl.where(skipped, 0, n_rows)
        below &= ~skipped
    if REVERSE:
        c_prev = _tile_cost(tile, -1, lanes, below, TL, HALO)
        c_prev_shift = _tile_cost(tile, -1, lanes, below, TL, HALO, -1)
    for kk in range(0, n_rows):
        g = first + kk
        c = _tile_cost(tile, kk, lanes, mask, TL, HALO)
        left = inf
        if SEEDED:
            left = tl.load(
                cols_ptr + s * H + g + lanes * 0, mask=(j > 0) & (lanes >= 0), other=_KERNEL_INF
            )
        tl.debug_barrier()
        if SYMMETRIC:
            src = tl.where(kk == 0, above, own)
            ok = mask & (pos > 0) & ((kk > 0) | (i > 0)) & ((kk == 0) | (lanes > 0))
            diag = tl.load(rows_ptr + src + pos - 1, mask=ok, other=_KERNEL_INF)
            diag = tl.where((kk > 0) & (lanes == 0), left_prev, diag)
        if REVERSE:
            c_shift = _tile_cost(tile, kk, lanes, mask, TL, HALO, -1)
            a = c_shift
            base = c_prev + y_near
            if SYMMETRIC:
                base = _min(base, diag_weight * c_prev_shift + diag, SOFT)
            base = tl.where((g == 0) & (pos == 0), 0.0, base)
        else:
            a = c
            base = c + y_near
            if SYMMETRIC:
                base = _min(base, diag + diag_weight * c, SOFT)
            base = tl.where((g == 0) & (pos == 0), c, base)
        if SOFT:
            y_near, scan_c = tl.associative_scan((base, a), 0, _compose_softmin)
            if SEEDED:
                y_near = _softmin(y_near, scan_c + left)
        else:
            y_near, scan_c = tl.associative_scan((base, a), 0, _compose_min)
            if SEEDED:
                y_near = tl.minimum(y_near, scan_c + left)
        tl.store(rows_ptr + own + pos, y_near, mask=mask)
        if SEEDED:
            tl.store(
                cols_ptr + s * H + g + lanes * 0, y_near,
                mask=(lanes == TL - 1) & ((j + 1) * TL < w),
            )  # fmt: skip
        left_prev = left
        if REVERSE:
            c_prev = c
            c_prev_shift = c_shift


@triton.jit(do_not_specialize=["N", "M", "big_bits", "scale_bits"])
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


def _features(feat, lanes: int):
    """feat = (x, y, window, big, scale, shear, transpose), or None for the cost
    tensor mode, as the kernels' arguments: (fx, fy, window, N, M, F, big_bits,
    scale_bits) and (SHEAR, TRANSPOSE, BF). ``big`` and ``scale`` go as the bits
    of a float64, as the kernels' float scalars are float32."""
    if feat is None:
        return (None, None, None, 0, 0, 0, 0, 0), (False, False, 1)
    x, y, window, big, scale, shear, transpose = feat
    x, y = x.contiguous(), y.contiguous()
    window = None if window is None else window.contiguous()
    bits = (struct.unpack("<q", struct.pack("<d", v))[0] for v in (big, scale))
    N, F = x.shape[1:]
    arguments = (x, y, window, N, y.shape[1], F, *bits)
    # The features a kernel reads at once per lane, within a register budget.
    return arguments, (shear, transpose, max(1, min(32, 8192 // lanes, triton.next_power_of_2(F))))


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
    """Each pair's optimal path on the kernels' grid, from its end cell ``ends``
    (B, 2) to (0, 0), as (row, column) pairs (B, R + L - 1, 2) in order, -1 past
    the path."""
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
    """The tile's rows and columns, measured on an RTX 5060 Ti: narrower tiles
    give a few pairs more programs per launch, wider ones fewer launches for
    many. With too few tiles side by side to pay for the launches, a tile is the
    whole grid, for rows of at most _MAX_ROW columns; with features, whose cells
    cost more, two side by side pay, and the tile's costs must fit the scratch,
    _SCRATCH_CELLS, which holds the costs of at least one anti-diagonal."""
    TL = min(256 if B < 8 else 1024, triton.next_power_of_2(L))
    n_cols = triton.cdiv(L, TL)
    side_by_side = B * min(triton.cdiv(R, 32), n_cols)
    if n_cols == 1 or (side_by_side < (2 if features else _MIN_TILES) and L <= _MAX_ROW):
        TL = triton.next_power_of_2(L)
        if not features or B * (R + 1) * (TL + 1) <= _SCRATCH_CELLS:
            return R, TL
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
# The costs of feature mode that a launch of _cost_kernel leaves in scratch,
# at most, but for one anti-diagonal's.
_SCRATCH_CELLS = 2**22
# The anti-diagonals whose costs one launch of _cost_kernel computes, at most.
_CHUNK = 8
# The fewest columns of y per chunk of _sheared_gradients, to keep its batch of
# matrix products small where the shear is narrow.
_MIN_COLUMNS = 64


def _halo_rows(TR: int) -> int:
    """A tile row of 32 with the halo row above it fills the 32 rows of
    _cost_kernel's blocks: 31 rows."""
    return 31 if TR == 32 else TR


def _diagonals(
    S: int, R: int, L: int, TR: int, TL: int, reverse: bool, feat, items=None, row0: int = 0,
    given: bool = False,
):  # fmt: skip
    """The tile anti-diagonals of S grids of R rows and L columns, in tiles of
    TR by TL, as (d, scratch, hit, slot) for the kernels' launches.

    In feature mode (``feat``, as _features takes it, with the ``items`` of
    _frame, if any), a chunk of anti-diagonals at a time, whose costs fit
    _SCRATCH_CELLS, first has its costs computed by _cost_kernel, in parallel
    over the blocks of all its tiles, into ``scratch``, and the tiles with a
    cell in the window flagged in ``hit`` (None without a window; with
    ``given``, the tiles of the seeded row 0 also): a tile's slice is at
    ``slot`` plus its program number. Without, the scratch is None.
    """
    n_rows, n_cols = triton.cdiv(R, TR), triton.cdiv(L, TL)
    n_diag = n_rows + n_cols - 1
    if feat is None:
        yield from ((d, None, None, 0) for d in range(n_diag))
        return
    (fx, fy, win, *sizes), (shear, transpose, _) = _features(feat, 1)
    BF = min(16, triton.next_power_of_2(sizes[2]))
    halo = reverse
    P = S * min(n_rows, n_cols)  # the most tiles of an anti-diagonal
    tile = (TR + halo) * (TL + halo)
    K = max(1, min(n_diag, _CHUNK, _SCRATCH_CELLS // (P * tile)))
    scratch = fx.new_empty(K * P * tile)
    hit = None if win is None else torch.empty(K * P, dtype=torch.int8, device=fx.device)
    BR, BC = 32, 64
    blocks = triton.cdiv(TR + halo, BR) * triton.cdiv(TL + halo, BC)
    for d0 in range(0, n_diag, K):
        k = min(K, n_diag - d0)
        if win is not None:
            _hit_kernel[(P, k)](
                hit, items, win, sizes[0], S, R, L, d0, n_rows, n_cols, TR, P, row0, shear,
                transpose, reverse, given, TL, num_warps=_num_warps(TL),
            )  # fmt: skip
        _cost_kernel[(P, blocks, k)](
            scratch, hit, items, fx, fy, win, *sizes, S, R, L, d0, n_rows, n_cols, TR, P, row0,
            shear, transpose, reverse, TL, BR, BC, BF, num_warps=4,
        )  # fmt: skip
        yield from ((d, scratch, hit, (d - d0) * P) for d in range(d0, d0 + k))


def _launch(
    B: int, R: int, L: int, cost, D, skip, up, left, diag, skip_share, x, y, out,
    diag_weight: float, soft: bool, steps: str, reverse: bool, backward: bool = False, feat=None,
    given: bool = False, row0: int = 0,
):  # fmt: skip
    """Run _dag_kernel over the grid, one launch per tile anti-diagonal;
    ``feat``, as _features takes it, in place of ``cost``, whose rows start at
    row ``row0`` of the features' grid; ``given`` as the kernel's GIVEN."""
    TR, TL = _tile(B, R, L, feat is not None)
    TR = _halo_rows(TR) if feat is not None and reverse else TR
    n_rows, n_cols = triton.cdiv(R, TR), triton.cdiv(L, TL)
    for d, scratch, hit, slot in _diagonals(B, R, L, TR, TL, reverse, feat, None, row0, given):
        lo = max(0, d - n_cols + 1)
        n_tiles = min(n_rows - 1, d) - lo + 1
        _dag_kernel[(n_tiles * B,)](
            cost, D, skip, up, left, diag, skip_share, x, y, out, scratch, hit, R, L, diag_weight,
            d, lo, n_tiles, TR, slot, soft, _STEPS[steps], reverse, TL, n_cols > 1, given,
            num_warps=_num_warps(TL, backward),
        )  # fmt: skip


def _frontier_tile(S: int, R: int, L: int) -> tuple[int, int]:
    """The tile of :func:`frontier` for S items of at most R rows and L columns:
    a tile row of R, one launch, where the columns fit in one tile and the
    costs in the scratch."""
    TL = min(256 if S < 8 else 1024, triton.next_power_of_2(L))
    whole = triton.cdiv(L, TL) == 1 and S * (R + 1) * (TL + 1) <= _SCRATCH_CELLS
    return (R if whole else 32), TL


@torch.library.custom_op("philtorch::dtw_frontier", mutates_args=())
def frontier(
    x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float, items: Tensor,
    soft: bool, steps: str, diag_weight: float, shear: bool, transpose: bool, reverse: bool,
    h_max: int, w_max: int,
) -> Tensor:  # fmt: skip
    """The last row of each item's accumulated costs, in linear memory.

    The cells of the grid have the costs of :func:`dtw_dp_features`, whose
    arguments x (P, N, F) through ``transpose`` are these. An item of
    ``items`` (S, 5), int64 on the device, is the pair p and the sub-grid of
    ``h`` rows from ``r0`` and ``w`` columns from ``l0``: (p, r0, l0, h, w).
    Forward, D over the sub-grid as :func:`dtw_dp_features` has it, from its
    first cell, and its last row is returned; with ``reverse``, G(s) is the
    cost of the best path from s to the sub-grid's last cell, each cell after s
    weighted by the step into it, and its first row is returned. Of shape
    (S, w_max), by columns in order, inf past an item's ``w``. Only the steps
    "orthogonal" and "symmetric". ``h_max`` and ``w_max`` bound the items' rows
    and columns, and ``w_max`` is the width of the result.
    """
    S = items.size(0)
    TR, TL = _frontier_tile(S, h_max, w_max)
    feat = (x, y, window, big, scale, shear, transpose)
    TR = _halo_rows(TR) if reverse else TR
    n_rows, n_cols = triton.cdiv(h_max, TR), triton.cdiv(w_max, TL)
    W = n_cols * TL
    rows = x.new_empty(S, 3, W)
    cols = x.new_empty(S, h_max)
    items = items.contiguous()
    for d, scratch, hit, slot in _diagonals(S, h_max, w_max, TR, TL, reverse, feat, items):
        lo = max(0, d - n_cols + 1)
        n_tiles = min(n_rows - 1, d) - lo + 1
        _frontier_kernel[(S * n_tiles,)](
            items, rows, cols, scratch, hit, diag_weight, d, lo, n_tiles, TR, W, h_max, slot, soft,
            steps == "symmetric", reverse, TL, n_cols > 1, num_warps=_num_warps(TL),
        )  # fmt: skip
    # Each item's last tile row holds the row.
    h, w = items[:, 3:].unbind(1)
    last = rows[torch.arange(S, device=x.device), (h - 1) // TR % 3]
    k = torch.arange(w_max, device=x.device)
    out = last.gather(1, ((w[:, None] - 1 - k) if reverse else k.expand(S, -1)).clamp(min=0))
    return torch.where(k < w[:, None], out, _INF)


@frontier.register_fake
def _(
    x, y, window, big, scale, items, soft, steps, diag_weight, shear, transpose, reverse, h_max,
    w_max,
):  # fmt: skip
    return x.new_empty(items.size(0), w_max)


def _empty(t: Tensor) -> Tensor:
    return torch.empty_like(t, memory_format=torch.contiguous_format)


def _dtw_backward(
    D: Tensor, cost: Tensor | None, skip: Tensor | None, grad: Tensor, soft: bool, steps: str,
    diag_weight: float, feat=None, seeded: bool = False, row0: int = 0,
) -> Tensor:  # fmt: skip
    B, R, L = D.shape
    weighted = diag_weight != 1.0
    e = _empty(D)
    out = _empty(D) if weighted else e
    _launch(
        B, R, L, cost, D, skip, None, None, None, None, grad.contiguous(), e,
        out if weighted else None, diag_weight, soft, steps, True, backward=True, feat=feat,
        given=seeded, row0=row0,
    )  # fmt: skip
    return out


def _weigh(e: Tensor, up: Tensor, left: Tensor, diag: Tensor, diag_weight: float) -> Tensor:
    """The cost's gradient from e, a diagonal weighted: e times its weight."""
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
    """y(i, j) = x(i, j) + W_up y(i - 1, j) + W_left y(i, j - 1)
    + W_diag y(i - 1, j - 1) + W_skip y(i - 2, j - 1), or with ``reverse`` its
    transpose.

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
    # or, the other way, at its predecessor times y at its successor.
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
    D: Tensor, cost: Tensor, skip: Tensor | None, soft: bool, steps: str, diag_weight: float,
    seeded: bool = False,
) -> list[Tensor | None]:  # fmt: skip
    """The shares of each cell from its up, left, up-left and skip predecessors.

    As the kernels compute them, with PyTorch ops that autograd differentiates:
    a softmax of the negated arguments, in base 2, for soft-DTW, a one-hot of
    the first best for DTW; zero for predecessors outside the grid, and all
    zero for cell (0, 0), and with ``seeded`` for row 0. None for the steps not
    in the set.
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
    weights = torch.where(outside, 0.0, weights)
    if seeded:
        weights = F.pad(weights[..., 1:, :], (0, 0, 1, 0))
    weights = iter(weights)
    return [next(weights) if p else None for p in present]


@torch.library.custom_op("philtorch::dtw_dp", mutates_args=())
def dtw_dp(cost: Tensor, skip: Tensor | None, soft: bool, steps: str, diag_weight: float) -> Tensor:
    """The accumulated costs D of cost (B, R, L) over the steps ``steps``, the
    diagonal weighted by ``diag_weight``, soft-min in base 2 with ``soft``;
    ``skip`` is the (B, R) mask of the rows the ctc steps may skip into, and
    None for the others. See the module docstring."""
    return _forward(cost, skip, soft, steps, diag_weight)


def _forward(cost, skip, soft, steps, diag_weight, feat=None, grid=None, seed=None, row0=0):
    B, R, L = cost.shape if grid is None else grid
    if R * L > _MAX_CELLS:
        raise ValueError(f"the DTW kernels take at most 2^31 - 1 cells per pair, got {(B, R, L)}")
    D = (cost if feat is None else feat[0]).new_empty(B, R, L)
    _launch(
        B, R, L, None if cost is None else cost.contiguous(), None, skip, None, None, None, None,
        None if seed is None else seed.contiguous(), D, None, diag_weight, soft, steps, False,
        feat=feat, given=seed is not None, row0=row0,
    )  # fmt: skip
    return D


@torch.library.custom_op("philtorch::ctc_viterbi", mutates_args=())
def ctc_viterbi(cost: Tensor, skip: Tensor) -> tuple[Tensor, Tensor]:
    """The hard ctc forward pass over cost (B, R, L): D, and each cell's best
    step back, int8, 0 to stay in its row, 1 to advance from the row above, 2 to
    skip from two rows above, the first best under ties as the hard shares
    choose."""
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


def _decomposed_backward(D, cost, skip, grad, soft, steps, diag_weight, seeded=False):
    """:func:`dtw_dp_backward` from ops that autograd differentiates."""
    w_up, w_left, w_diag, w_skip = shares(D, cost, skip, soft, steps, diag_weight, seeded)
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


def _cell_costs(x, y, window, big, scale, shear, transpose, p, r, col):
    """The costs of the grid's cells (r, col) of pairs p, as the kernels compute
    them, from PyTorch ops that autograd differentiates.

    p, r and col broadcast together; a cell is mapped as :func:`_cost` maps it.
    """
    a, s = (col, r) if transpose else (r, col)
    n = (a + s if shear else a).clamp(max=x.size(1) - 1)
    m = (a if shear else s).clamp(max=y.size(1) - 1)
    d = x[p, n] - y[p, m]
    c = (d * d).sum(-1) * scale
    if window is not None:
        lo, hi = window[p, n].unbind(-1)
        c = torch.where((m < lo) | (m > hi), big, c)
    return c


@torch.library.custom_op("philtorch::dtw_dp_features", mutates_args=())
def dtw_dp_features(
    x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float, soft: bool,
    steps: str, diag_weight: float, shear: bool, transpose: bool, R: int, L: int,
    seed: Tensor | None = None, row0: int = 0,
) -> Tensor:  # fmt: skip
    """:func:`dtw_dp` over the (B, R, L) grid of the costs ``scale`` times the
    squared distances of x (B, N, F) and y (B, M, F), computed in the kernels
    (see the module docstring), outside ``window`` (B, N, 2), the allowed
    columns [lo, hi] of y per row of x, ``big``. A cell (r, l) is x's row n and
    y's row m: (a, s) = (l, r + row0) with ``transpose``, else (r + row0, l);
    n = a + s, clamped to N - 1, and m = a with ``shear``, else n = a and
    m = s: the grid's rows are rows ``row0`` on of a larger one. ``seed``
    (B, L), if given, is D's row 0, as in the module docstring's seeded grids.
    Only the steps "orthogonal" and "symmetric"."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _forward(None, None, soft, steps, diag_weight, feat, (x.size(0), R, L), seed, row0)


@dtw_dp_features.register_fake
def _(
    x, y, window, big, scale, soft, steps, diag_weight, shear, transpose, R, L, seed=None, row0=0
):
    return x.new_empty(x.size(0), R, L)


def _dtw_dp_features_setup(ctx, inputs, output):
    x, y, window, *ctx.options, seed, ctx.row0 = inputs
    ctx.seeded = seed is not None
    ctx.save_for_backward(x, y, window, output)


def _sheared_gradients(x, y, window, scale, E, n0, m0):
    """:func:`_cost_gradients` for E (B, A, S), the gradient of the costs of
    the cells (m, s), x's row n0 + m + s, clamped to N - 1, and y's column
    m0 + m, in memory linear in E: not the dense (A + S - 1, A) block of x's
    rows by y's columns, of which only S diagonals are cells.

    The columns m go in chunks of C, each a dense block of the K = C + S - 1
    rows of x its cells cover by its C columns, with the products of all the
    chunks batched. A row of x is in two blocks at most, and they add up.
    """
    B, A, S = E.shape
    N, M = x.size(1), y.size(1)
    C = min(A, max(S, _MIN_COLUMNS))
    chunks, K = -(-A // C), C + S - 1
    E = F.pad(E, (0, 0, 0, chunks * C - A))
    # Cell (m, s) of chunk c is the block's row m + s, column m: offset
    # m (C + 1) + s C in rows of C.
    E = E.new_zeros(B, chunks, K, C).as_strided_scatter(
        E.reshape(B, chunks, C, S), (B, chunks, C, S), (chunks * K * C, K * C, C + 1, C)
    )
    n = (chunks - 1) * C + K

    def blocks(t):
        """The rows n0 on of t (B, N, Q), the last repeated past N, in the
        chunks' windows of K: (B, chunks, Q, K)."""
        t = torch.cat([t[:, n0:], t[:, -1:].expand(-1, max(0, n - N + n0), -1)], 1)
        return t[:, :n].unfold(1, K, C)

    if window is not None:
        cols = (
            m0
            + torch.arange(chunks, device=x.device)[:, None] * C
            + torch.arange(C, device=x.device)
        )[None]
        lo, hi = blocks(window).unbind(2)
        E.masked_fill_(cols[..., None, :] < lo[..., None], 0.0)
        E.masked_fill_(cols[..., None, :] > hi[..., None], 0.0)
    # From frames centred on each pair's mean, as the unsheared case.
    mean = torch.cat([x, y], 1).mean(1, keepdim=True)
    x, y = x - mean, F.pad(y[:, m0 : m0 + A] - mean, (0, 0, 0, chunks * C - A))
    x_rows, y_cols = blocks(x).mT, y.reshape(B, chunks, C, -1)
    grad_x = 2 * scale * (x_rows * E.sum(3, keepdim=True) - E @ y_cols)
    grad_y = 2 * scale * (y_cols * E.sum(2).unsqueeze(-1) - E.mT @ x_rows)
    # The blocks' rows back to x's, the overlaps added: a fold.
    grad_x = F.fold(
        grad_x.permute(0, 3, 2, 1).flatten(1, 2), (n, 1), (K, 1), stride=(C, 1)
    ).squeeze(3).mT  # fmt: skip
    k = min(n, N - n0)  # the cells past x's last row are clamped to it
    grad_x = torch.cat([grad_x[:, : k - 1], grad_x[:, k - 1 :].sum(1, keepdim=True)], 1)
    grad_x = F.pad(grad_x, (0, 0, n0, N - n0 - k))
    return grad_x, F.pad(grad_y.flatten(1, 2)[:, :A], (0, 0, m0, M - m0 - A))


def _cost_gradients(x, y, window, scale, shear, transpose, row0, E):
    """The gradients of x and y from E, the gradient of the grid's costs of
    :func:`dtw_dp_features` (B, R, L). E is masked in place, but for the
    shear's."""
    # E, the cost's gradient on the grid, back to the (n, m) cells it covers:
    # rows n0 to n1 of x, columns m0 on of y.
    N, M = x.size(1), y.size(1)
    if transpose:
        E = E.mT
    B, A, S = E.shape
    a0, s0 = (0, row0) if transpose else (row0, 0)
    if shear:
        return _sheared_gradients(x, y, window, scale, E, a0 + s0, a0)
    n0, n1, m0 = a0, a0 + A - 1, s0
    m1 = m0 + E.size(2)
    if window is not None:
        cols = torch.arange(m0, m1, device=x.device)
        lo, hi = window[:, n0 : n1 + 1].split(1, -1)
        E = E.masked_fill_((cols < lo) | (cols > hi), 0.0)
    # From frames centred on each pair's mean: the products' difference then
    # cancels only their spread, not an offset they share.
    mean = torch.cat([x, y], 1).mean(1, keepdim=True)
    x, y = x[:, n0 : n1 + 1] - mean, y[:, m0:m1] - mean
    grad_x = 2 * scale * (x * E.sum(2, keepdim=True) - E @ y)
    grad_y = 2 * scale * (y * E.sum(1).unsqueeze(-1) - E.mT @ x)
    grad_x, grad_y = F.pad(grad_x, (0, 0, n0, N - 1 - n1)), F.pad(grad_y, (0, 0, m0, M - m1))
    return grad_x, grad_y


def _dtw_dp_features_backward(ctx, grad_D):
    x, y, window, D = ctx.saved_tensors
    big, scale, soft, steps, diag_weight, shear, transpose, _, _ = ctx.options
    G = dtw_dp_backward_features(
        D, x, y, window, big, scale, grad_D, soft, steps, diag_weight, shear, transpose,
        ctx.seeded, ctx.row0,
    )  # fmt: skip
    # With a seed, G's row 0 is the seed's gradient; the cost's row 0 has none.
    E, grad_seed = (F.pad(G[:, 1:], (0, 0, 1, 0)), G[:, 0]) if ctx.seeded else (G, None)
    grad_x, grad_y = _cost_gradients(x, y, window, scale, shear, transpose, ctx.row0, E)
    return grad_x, grad_y, *(None,) * 10, grad_seed, None


dtw_dp_features.register_autograd(_dtw_dp_features_backward, setup_context=_dtw_dp_features_setup)


@torch.library.custom_op("philtorch::dtw_grid_cost", mutates_args=())
def grid_cost(
    x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float, shear: bool,
    transpose: bool, R: int, L: int, row0: int,
) -> Tensor:  # fmt: skip
    """The costs (B, R, L) of the cells of :func:`dtw_dp_features`'s grid, whose
    arguments x through ``row0`` are these.

    One feature at a time, from views of x and y, so that the memory is that of
    the grid, not F times it; the gradients of x and y are the matrix products
    of :func:`_cost_gradients`.
    """
    B, N, _ = x.shape
    # Of the cell (r, l): its rows r + row0, its columns l, or, for x's row
    # n = r + row0 + l, clamped to N - 1, the diagonals of the rows' windows.
    rows = slice(row0, row0 + R)
    a = (lambda t: t[:, None, :L]) if transpose else (lambda t: t[:, rows, None])
    s = (lambda t: t[:, rows, None]) if transpose else (lambda t: t[:, None, :L])

    def at_n(t):
        if not shear:
            return a(t)
        t = torch.cat([t, t[:, -1:].expand(-1, max(0, row0 + R + L - 1 - N))], 1)
        return t.unfold(1, L, 1)[:, rows]

    at_m = a if shear else s
    cost = x.new_zeros(B, R, L)
    for f in range(x.size(2)):
        d = at_n(x[..., f]) - at_m(y[..., f])
        cost.addcmul_(d, d)
    cost *= scale
    if window is not None:
        lo, hi = at_n(window[..., 0]), at_n(window[..., 1])
        m = at_m(torch.arange(y.size(1), device=x.device)[None])
        cost.masked_fill_((m < lo) | (m > hi), big)
    return cost


@grid_cost.register_fake
def _(x, y, window, big, scale, shear, transpose, R, L, row0):
    return x.new_empty(x.size(0), R, L)


def _grid_cost_setup(ctx, inputs, output):
    x, y, window, _, ctx.scale, ctx.shear, ctx.transpose, _, _, ctx.row0 = inputs
    ctx.save_for_backward(x, y, window)


def _grid_cost_backward(ctx, E):
    x, y, window = ctx.saved_tensors
    # _cost_gradients masks its E in place, and this one is not ours.
    E = E.clone() if window is not None else E
    grads = _cost_gradients(x, y, window, ctx.scale, ctx.shear, ctx.transpose, ctx.row0, E)
    return *grads, *(None,) * 8


grid_cost.register_autograd(_grid_cost_backward, setup_context=_grid_cost_setup)


@torch.library.custom_op("philtorch::dtw_dp_backward_features", mutates_args=())
def dtw_dp_backward_features(
    D: Tensor, x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float,
    grad: Tensor, soft: bool, steps: str, diag_weight: float, shear: bool, transpose: bool,
    seeded: bool, row0: int,
) -> Tensor:  # fmt: skip
    """The gradient of the grid's cost from ``grad``, D's, through
    :func:`dtw_dp_features`, as :func:`dtw_dp_backward` gives it."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _dtw_backward(
        D.contiguous(), None, None, grad, soft, steps, diag_weight, feat, seeded, row0
    )


@dtw_dp_backward_features.register_fake
def _(D, x, y, window, big, scale, grad, soft, steps, diag_weight, shear, transpose, seeded, row0):
    return _empty(D)


def _dtw_dp_backward_features_setup(ctx, inputs, output):
    D, x, y, window, *ctx.options = inputs
    ctx.save_for_backward(D, x, y, window, inputs[6])


def _dtw_dp_backward_features_backward(ctx, grad_e):
    # As _dtw_dp_backward_backward, with the cost rebuilt from x and y.
    D, x, y, window, grad = ctx.saved_tensors
    big, scale, _, soft, steps, diag_weight, shear, transpose, seeded, row0 = ctx.options
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        D, x, y, grad = (
            t.view_as(t) if t.requires_grad else t.detach().requires_grad_()
            for t in (D, x, y, grad)
        )
        cost = grid_cost(x, y, window, big, scale, shear, transpose, *D.shape[1:], row0)
        e = _decomposed_backward(D, cost, None, grad, soft, steps, diag_weight, seeded)
        grad_D, grad_x, grad_y, grad_grad = torch.autograd.grad(
            e, (D, x, y, grad), grad_e, create_graph=create_graph, allow_unused=True
        )
    return grad_D, grad_x, grad_y, None, None, None, grad_grad, *(None,) * 7


dtw_dp_backward_features.register_autograd(
    _dtw_dp_backward_features_backward, setup_context=_dtw_dp_backward_features_setup
)
