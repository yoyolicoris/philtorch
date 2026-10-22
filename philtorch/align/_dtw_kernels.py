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

Six kernels do all of it. ``_dag_kernel`` is the row-by-row recursion in
either semiring, min-plus (or soft-min-plus) for the forward pass and
linear for every accumulation, the first-order backward's included;
``_shares_kernel`` computes, for CTC's forced alignment, each cell's best
step back over all cells at once; with the ctc steps, both take the (B, R)
skip mask. ``_frontier_kernel`` is the recursion without storing D (below),
``_backtrack_kernel`` walks a DTW path back from its end cell, and, in feature
mode, ``_cost_kernel`` computes the costs the others read and ``_hit_kernel``
flags the tiles with a cell in the window (below).

Feature mode. With the features x (B, N, F) and y (B, M, F), no (B, N, M)
cost is stored. Before the launches of a few tile anti-diagonals,
``_cost_kernel`` computes the costs of all their tiles' cells, in parallel
over blocks of cells, each block loading its rows of x and y once per chunk
of features: the squared distance from direct differences, which don't cancel
as |x|^2 + |y|^2 - 2 x.y does. They go to a scratch, a slice per tile, and the
kernels read them there as they read a cost tensor; that, not computing each
cell in the row loop, which reloads the tile's features every row with few
programs in flight, is what keeps large F near the speed of a matrix-product
cost. The scratch holds a chunk of anti-diagonals, at most _CHUNK of them and
_SCRATCH_CELLS cells, but for one anti-diagonal's, so that the memory is linear
in the sides. The backward and the reverse frontier also read the cost of the
row above the tile and of the column before it, which the slice holds as a halo
(the backward's tile rows are 31, so that a slice's 32 rows fill the cost
blocks). A window of allowed columns per row of x masks it by a large finite
cost, and ``_hit_kernel`` flags the tiles with a cell in it, whose costs and
recursion the others skip. The shear, for the asymmetric steps, and the
transpose, to keep the rows the shorter side, map the kernels' cells to those
of x and y. Second derivatives rebuild the costs with :func:`grid_cost` (a
strip's with :func:`_strip_cost`), in the memory of the grid or strip.

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
from a cell to the end of a sub-grid (an "item"), over the feature mode's grid
without storing the cost or D, in the same wavefront of tiles: each tile hands
its last row to the tile below, its last column to the tile to its right, and
the rows are kept three tile rows deep, so that the memory is that of one row
and one column per item (Tralie and Dempsey, ISMIR 2020).

Strips. A grid of several tile columns, cut into blocks of rows, would fill
and drain the wavefront in each, so it is cut along the wavefront instead: a
strip is the tile anti-diagonals d0 to d1 - 1, which, one after the other,
launch as the whole grid's do. The tiles are those of both directions, TR = 31
rows (the backward's) by TL columns. A strip's buffer has the cell (r, c) at
row r + TR times its tile column, so that its tiles are whole rows of it and
only the left neighbours across a tile edge are TR rows more up; the kernels'
STRIP addressing is that skew, in memory linear in the strip. What a strip's
tiles read of the diagonals before it, its seed, is the last row of the one
before, its tiles' last columns, and the corner cells of the one before that:
L + TR (n_cols - 1) values per pair, kept by the checkpoint, and put in the
buffer's first TR + 1 rows. The backward of a strip runs the same wavefront
reversed, from the grid mirrored and padded to whole tiles, so that its tiles
are the forward ones, and then the diagonal of the seed, whose own shares are
0 as the given row's are: the gradient of the seed's cells is that of the
strip's cells that read them, and the corner cells' the diagonal step of the
strip's first cell. The costs and their gradients are the blocks of the
tile columns, each a grid of its own (:func:`_strip_gradients`), and second
derivatives are those of :func:`shares`, :func:`_strip_cost` and
:func:`accumulate` (with ``strip``) in the same skew.
"""


# Op map: the custom ops (torch.library), from the entry points in dtw.py and
# ctc.py down to the kernels; launchers in brackets, ">" for "calls".
#
#   dtw(cost), ctc_loss .... dtw_dp [_forward > _launch] > _dag_kernel
#   dtw_path(cost) ......... dtw_dp, then backtrack [_backtrack]
#                            > _backtrack_kernel
#   dtw(x=, y=), no grad ... frontier > _frontier_kernel
#   dtw(x=, y=), DTW with grad, and dtw_path(x=, y=):
#                            dtw_linear_path (dtw.py): _split > frontier,
#                            forward and reverse; _leaves > dtw_dp_features,
#                            backtrack_features; the distance is the path's
#                            cost, _path_cost, in PyTorch
#   dtw(x=, y=), soft-DTW with grad:
#                            dtw_dp_features [_forward > _launch]
#                            > _dag_kernel; with checkpoint, per block of rows
#                            (one tile column), or dtw_dp_strip [_launch] per
#                            strip (several)
#   soft_dtw_divergence .... dtw, three times
#   forced_align ........... ctc_viterbi [_forward] > _dag_kernel,
#                            _shares_kernel
#
# From frames, each launch of _dag_kernel or _frontier_kernel reads its costs
# from the scratch that [_diagonals] fills per chunk of anti-diagonals, with
# _hit_kernel (given a window), then _cost_kernel.
#
# Backward, as registered for autograd; each is itself differentiable:
#   dtw_dp ................. dtw_dp_backward [_dtw_backward > _launch]
#                            > _dag_kernel
#   dtw_dp_features ........ dtw_dp_backward_features (as above), then
#                            _cost_gradients for x and y
#   dtw_dp_strip ........... dtw_dp_backward_strip (as above), then
#                            _strip_gradients
#   dtw_dp_backward* ....... _vjp of _decomposed_backward: shares (PyTorch)
#                            and accumulate [_launch] > _dag_kernel, the costs
#                            rebuilt by grid_cost (or _strip_cost)
#   accumulate ............. accumulate, the other way
#   grid_cost .............. _cost_gradients
# Without autograd, their outputs being integers or taken without gradients:
# frontier, backtrack, backtrack_features, dtw_linear_path, ctc_viterbi.

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
# The block of cells per program of _cost_kernel: 32 rows, to fill a tile row of
# 31 with its halo.
_BR, _BC = 32, 64
_KERNEL_BR, _KERNEL_BC = tl.constexpr(_BR), tl.constexpr(_BC)


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
def _cell(r, cols, N, SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr):
    """x's row n and y's row m of cells (r, cols): see :func:`_cost`."""
    a = cols if TRANSPOSE else r
    s = r if TRANSPOSE else cols
    n = tl.broadcast_to(tl.minimum(a + s if SHEAR else a, N - 1), cols.shape)
    return n, tl.broadcast_to(a if SHEAR else s, cols.shape)


@triton.jit
def _unbits(bits, dtype):
    """The float64 whose bits are ``bits``, as ``dtype``: a kernel's float
    scalars are float32, so ``big`` and ``scale`` go as bits."""
    return bits.to(tl.int64).to(tl.float64, bitcast=True).to(dtype)


@triton.jit
def _outside(win_ptr, b, N, n, m, valid=None):
    """Whether y's row m is outside the window of x's row n, its allowed rows
    [lo, hi] of y, where ``valid``."""
    at = win_ptr + (b * N + n) * 2
    if valid is None:
        outside = (m < tl.load(at)) | (m > tl.load(at + 1))
    else:
        outside = (m < tl.load(at, mask=valid, other=0)) | (
            m > tl.load(at + 1, mask=valid, other=0)
        )
    return outside


@triton.jit
def _cost(
    cost_ptr, fx_ptr, fy_ptr, win_ptr, b, r, cols, valid, R, L, N, M, F, big_bits, scale_bits,
    SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr, BF: tl.constexpr,
):  # fmt: skip
    """The costs of cells (r, cols) of pair b, 0 where not valid: from the cost
    tensor, or in feature mode, with ``fx_ptr``, from the features.

    The cell is (a, s) = (cols, r) with TRANSPOSE, else (r, cols), of rows a
    and columns s; with SHEAR it is x's row a + s, clamped to N - 1, and y's a,
    else x's a and y's s. The cost is the squared distance times the scale,
    ``big`` outside the window (see :func:`_outside`), both as bits (see
    :func:`_unbits`).
    """
    if fx_ptr is None:
        c = tl.load(cost_ptr + b * R * L + r * L + cols, mask=valid, other=0.0)
    else:
        n, m = _cell(r, cols, N, SHEAR, TRANSPOSE)
        f_x = fx_ptr + b * N * F + n[:, None].to(tl.int64) * F
        f_y = fy_ptr + b * M * F + m[:, None].to(tl.int64) * F
        c = tl.zeros(cols.shape, fx_ptr.dtype.element_ty)
        for f0 in range(0, F, BF):
            f = (f0 + tl.arange(0, BF))[None, :]
            ok = valid[:, None] & (f < F)
            d = tl.load(f_x + f, mask=ok, other=0.0) - tl.load(f_y + f, mask=ok, other=0.0)
            c += tl.sum(d * d, axis=1)
        dtype = c.dtype
        c *= _unbits(scale_bits, dtype)
        if win_ptr is not None:
            c = tl.where(_outside(win_ptr, b, N, n, m, valid), _unbits(big_bits, dtype), c)
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


@triton.jit
def _tile_at(pid, d, R, L, TR, TL: tl.constexpr):
    """Whether program ``pid`` of a launch for the anti-diagonal d has a tile,
    and its tile row i, tile column j and item, as :func:`_dag_kernel` and
    :func:`_frontier_kernel` decode them: the grid is P = S min(n_rows,
    n_cols) wide, n_rows and n_cols being those of the R by L frame in tiles of
    TR by TL, and the item one of S: the pair, or an item of :func:`_frame`."""
    n_rows = (R + TR - 1) // TR
    n_cols = (L + TL - 1) // TL
    S = tl.num_programs(0) // tl.minimum(n_rows, n_cols)
    lo = tl.maximum(0, d - n_cols + 1)
    n_tiles = tl.minimum(n_rows - 1, d) - lo + 1
    i = lo + pid % n_tiles
    return pid < S * n_tiles, i, d - i, (pid // n_tiles).to(tl.int64)


@triton.jit(do_not_specialize=["N", "R", "L", "d0", "TR", "row0"])
def _hit_kernel(
    hit_ptr, items_ptr, win_ptr, N, R, L, d0, TR, row0, SHEAR: tl.constexpr,
    TRANSPOSE: tl.constexpr, REVERSE: tl.constexpr, GIVEN: tl.constexpr, TL: tl.constexpr,
):  # fmt: skip
    """Whether each tile of the anti-diagonals d0, d0 + 1, ... (axis 1 of the
    grid) has a valid cell inside the window, from the window alone, to
    ``hit_ptr`` at the slot (d - d0) * P plus the tile's program number in its
    launch, P being the grid's width: the tiles as in :func:`_tile_at`, and
    ``items_ptr`` as in :func:`_frame`.
    With GIVEN, the tiles holding row 0 are flagged whatever the window says:
    its D is the seed's, and its gradient the seed's."""
    pid = tl.program_id(0)
    dd = tl.program_id(1)
    live, i, j, item = _tile_at(pid, d0 + dd, R, L, TR, TL)
    if live:
        b, r0, c0, h, w = _frame(items_ptr, item, R, L)
        first = i * TR
        n = tl.where((first < h) & (j * TL < w), tl.minimum(TR, h - first), 0)
        pos = j * TL + tl.arange(0, TL)
        cols = c0 + ((w - 1 - pos) if REVERSE else pos)
        r_first = r0 + ((h - 1 - first) if REVERSE else first) + row0
        valid = pos < w
        seen = tl.zeros(cols.shape, tl.int32)
        for k in range(0, n):
            x_row, y_row = _cell(r_first + (-k if REVERSE else k), cols, N, SHEAR, TRANSPOSE)
            outside = _outside(win_ptr, b, N, x_row, y_row, valid)
            seen = tl.maximum(seen, (valid & ~outside).to(tl.int32))
        hit = tl.max(seen, axis=0) > 0
        if GIVEN:
            hit |= (n > 0) & (first + n == h if REVERSE else first == 0)
        tl.store(hit_ptr + dd * tl.num_programs(0) + pid, hit.to(tl.int8))


@triton.jit(do_not_specialize=["N", "M", "big_bits", "scale_bits", "R", "L", "d0", "TR", "row0"])
def _cost_kernel(
    scr_ptr, hit_ptr, items_ptr, fx_ptr, fy_ptr, win_ptr, N, M, F, big_bits, scale_bits, R, L, d0,
    TR, row0, SHEAR: tl.constexpr, TRANSPOSE: tl.constexpr, REVERSE: tl.constexpr,
    TL: tl.constexpr, BF: tl.constexpr,
):  # fmt: skip
    """The costs of the cells the tiles of the anti-diagonals d0, d0 + 1, ...
    (axis 2 of the grid) need, from the features, into ``scr_ptr``: the
    squared distance of the two rows from direct differences, as :func:`_cost`
    takes it, a block of _BR rows by _BC columns of one tile (axis 1) per
    program, each feature chunk of BF loaded once for the block.

    A tile's slice, at (slot + program number) of the (TR + HALO) by
    (TL + HALO) cells, holds its rows from the row above it (the tile's
    first, with HALO) and its columns from the column before it, in the
    order of processing; 0 where there is no cell. HALO, set by REVERSE,
    is what the backward and the reverse frontier read of the tile above and
    the tile to the left. Tiles ``hit_ptr`` (if given) says have no cell in
    the window are left as they are; see :func:`_hit_kernel` for the
    slots.
    """
    HALO: tl.constexpr = REVERSE
    blocks_wide: tl.constexpr = (TL + HALO + _KERNEL_BC - 1) // _KERNEL_BC
    P = tl.num_programs(0)
    pid = tl.program_id(0)
    dd = tl.program_id(2)
    on, i, j, item = _tile_at(pid, d0 + dd, R, L, TR, TL)
    if on:
        b, r0, c0, h, w = _frame(items_ptr, item, R, L)
        live = (i * TR < h) & (j * TL < w)
        if hit_ptr is not None:
            live &= tl.load(hit_ptr + dd * P + pid) != 0
        if live:
            k = (tl.program_id(1) // blocks_wide) * _KERNEL_BR + tl.arange(0, _KERNEL_BR)
            e = (tl.program_id(1) % blocks_wide) * _KERNEL_BC + tl.arange(0, _KERNEL_BC)
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
            c = tl.zeros([_KERNEL_BR, _KERNEL_BC], dtype)
            work = valid
            if win_ptr is not None:
                outside = _outside(win_ptr, b, N, n, m)
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
            c *= _unbits(scale_bits, dtype)
            if win_ptr is not None:
                c = tl.where(outside, _unbits(big_bits, dtype), c)
            c = tl.where(valid, c, 0.0)
            tile = scr_ptr + (dd * P + pid).to(tl.int64) * ((TR + HALO) * (TL + HALO))
            tl.store(
                tile + k[:, None] * (TL + HALO) + e[None, :], c,
                mask=(k < TR + HALO)[:, None] & (e < TL + HALO)[None, :],
            )  # fmt: skip


@triton.jit
def _cost_at(
    cost_ptr, tile, b, r, cols, k, lanes, mask, R, L, TL: tl.constexpr, HALO: tl.constexpr,
    SHIFT: tl.constexpr = 0,
):  # fmt: skip
    """The costs of cells (r, cols) of the tile's row k, in lanes ``lanes`` plus
    SHIFT: as :func:`_tile_cost` reads them from its slice of scratch, or, with
    ``cost_ptr``, from the cost tensor, 0 off the grid."""
    if cost_ptr is None:
        c = _tile_cost(tile, k, lanes, mask, TL, HALO, SHIFT)
    else:
        valid = (cols >= 0) & (cols < L) & (r >= 0) & (r < R)
        c = tl.load(cost_ptr + b * R * L + r * L + cols, mask=valid, other=0.0)
    return c


@triton.jit
def _shares(
    D_ptr, c, skip_ptr, b, r, cols, R, L, diag_weight, SOFT: tl.constexpr, STEPS: tl.constexpr,
    GIVEN: tl.constexpr = False, STRIP: tl.constexpr = False, RB=0, sbase=0, TR=0,
    TL: tl.constexpr = 1,
):  # fmt: skip
    """The shares of cells (r, cols), of costs c, from their up, left, diagonal
    and skip.

    Zero for cells outside the grid and for predecessors outside it or not
    among the steps. With STRIP, D is a strip's buffer of RB rows (see the
    module docstring), the cell (r, c) at row ``sbase`` + r + TR times its tile
    column: the cells of the seed's rows, as of other strips, have no shares.
    """
    valid = (cols >= 0) & (cols < L) & (r >= 0) & (r < R)
    edge = 0  # the step across a tile's left edge, in the skewed buffer, is TR rows more
    if STRIP:
        buf = r + (cols // TL) * TR + sbase
        valid &= (buf > TR) & (buf < RB)
        offset = b * RB * L + buf * L + cols
        edge = tl.where(cols % TL == 0, TR * L, 0)
    else:
        offset = b * R * L + r * L + cols
    absent = tl.full(c.shape, _KERNEL_INF, c.dtype)
    up = absent
    if STEPS != _CTC:
        up = tl.load(D_ptr + offset - L, mask=valid & (r > 0), other=_KERNEL_INF) + c
    has_left = valid & (cols > 0)
    if GIVEN:
        has_left &= r > 0
    left = tl.load(D_ptr + offset - 1 - edge, mask=has_left, other=_KERNEL_INF) + c
    diag = absent
    if STEPS != _ORTHOGONAL:
        up_left = tl.load(
            D_ptr + offset - L - 1 - edge, mask=valid & (r > 0) & (cols > 0), other=_KERNEL_INF
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
    """Every cell's best step back under the hard ctc steps, as int8 (see
    :func:`ctc_viterbi`), one program per BLOCK cells of a row."""
    batch_row = tl.program_id(0)
    b = (batch_row // R).to(tl.int64)
    r = batch_row % R
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = b * R * L + r * L + cols
    c = tl.load(cost_ptr + offset, mask=cols < L, other=0.0)
    _, _, diag, skip = _shares(D_ptr, c, skip_ptr, b, r, cols, R, L, 1.0, False, _CTC)
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


@triton.jit(do_not_specialize=["d", "TR", "slot", "RB", "sbase"])
def _dag_kernel(
    cost_ptr, D_ptr, skip_rows_ptr, up_ptr, left_ptr, diag_ptr, skip_ptr, x_ptr, y_ptr,
    out_ptr, scr_ptr, hit_ptr, R, L, diag_weight, d, TR, slot, RB, sbase, SOFT: tl.constexpr,
    STEPS: tl.constexpr, REVERSE: tl.constexpr, TL: tl.constexpr, SEEDED: tl.constexpr,
    GIVEN: tl.constexpr, STRIP: tl.constexpr,
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
    column, TR rows by TL columns, the grid mirrored with REVERSE; the
    anti-diagonal ``d`` of the R by L grid's tiles has ``n_tiles`` of them,
    from tile row ``I_lo``. It reads only earlier launches' y, written whole,
    and the costs its slice of the scratch holds, at ``slot`` plus its program
    number. Row by row, the left step makes each row a scan of the maps
    x -> A (+) (C (x) x), from the right with REVERSE, applied to y of the
    column before the tile, whose weight is the row's own. The other rows'
    terms come from y already written, after a barrier; the first row's from
    the tile above. With ``D_ptr`` the shares of the rows below, kept in
    registers, start from the cells themselves. The neighbours' offsets are
    constants: offsets computed from REVERSE would cost a fifth of the time on
    long rows.

    With STRIP, the feature mode's grid of R rows and L columns is a strip's
    buffer of RB rows (see the module docstring), the cell (r, c) at row
    ``sbase`` + r + TR times its tile column, the left neighbours TR rows up
    (down with REVERSE). With REVERSE the tiles are numbered in the grid padded
    to whole tiles, so that the mirrored tiles are the forward ones. A tile
    with no neighbour in the buffer, the last diagonal's with REVERSE, takes it
    as 0.
    """
    SHARES: tl.constexpr = D_ptr is not None
    FEAT: tl.constexpr = scr_ptr is not None
    FORWARD: tl.constexpr = (cost_ptr is not None or FEAT) and not SHARES
    SKIP: tl.constexpr = hit_ptr is not None
    HALO: tl.constexpr = REVERSE
    tile_rows = (R + TR - 1) // TR
    tile_cols = (L + TL - 1) // TL
    I_lo = tl.maximum(0, d - tile_cols + 1)
    n_tiles = tl.minimum(tile_rows - 1, d) - I_lo + 1
    Rp, Lp = R, L  # the frame the tiles are numbered in
    if STRIP and REVERSE:
        Rp, Lp = tile_rows * TR, tile_cols * TL
    b = (tl.program_id(0) // n_tiles).to(tl.int64)
    i = I_lo + tl.program_id(0) % n_tiles
    j = d - i
    lanes = tl.arange(0, TL)
    pos = j * TL + lanes  # the column's place in the scan
    cols = (Lp - 1 - pos) if REVERSE else pos
    first = i * TR  # the tile's first row's place
    # The row before it, the column before the tile in each row.
    above = (Rp - first) if REVERSE else (first - 1)
    edge = (Lp - j * TL) if REVERSE else (j * TL - 1)
    # The buffer's row of the tile's first row, and of the one before it.
    rb0 = (Rp - 1 - first) if REVERSE else first
    above_b = above
    batch = b * R * L
    mask = pos < L
    has_above = i > 0
    left_ok = j > 0
    shift = 0
    if STRIP:
        batch = b * RB * L
        mask = cols < L
        i_f = (tile_rows - 1 - i) if REVERSE else i
        j_f = (tile_cols - 1 - j) if REVERSE else j
        tile_row = (i_f + j_f) * TR + sbase
        rb0 = tile_row + (TR - 1 if REVERSE else 0)
        above_b = tile_row + TR if REVERSE else tile_row - 1
        has_above &= above_b < RB
        left_ok &= above_b < RB
        shift = TR if REVERSE else -TR
        # The step to the diagonal neighbour across the tile's left edge.
        lane_skew = tl.where(lanes == 0, TR * L if REVERSE else -TR * L, 0)
    zero = _KERNEL_INF if FORWARD else 0.0  # the semiring's zero
    y_near = tl.load(y_ptr + batch + above_b * L + cols, mask=mask & has_above, other=zero)
    n_rows = tl.minimum(TR, Rp - first)
    tile = None
    if FEAT:
        tile = scr_ptr + (slot + tl.program_id(0)).to(tl.int64) * ((TR + HALO) * (TL + HALO))
    if SKIP:
        # A tile with no cell in the window holds the semiring's zero: D = inf.
        skipped = tl.load(hit_ptr + slot + tl.program_id(0)) == 0
        for k in range(0, tl.where(skipped, n_rows, 0)):
            row = batch + (rb0 + (-k if REVERSE else k)) * L + cols
            tl.store(y_ptr + row, tl.full([TL], zero, y_ptr.dtype.element_ty), mask=mask)
            if SHARES and out_ptr is not None:
                tl.store(out_ptr + row, tl.full([TL], 0.0, out_ptr.dtype.element_ty), mask=mask)
        n_rows = tl.where(skipped, 0, n_rows)
    if SHARES:
        # The shares of the rows below, kept from earlier rows: row r + 1
        # straight down at j and diagonally at j + 1, row r + 2 by a skip at
        # j + 1. None below the last row. With features, the row above's costs
        # are kept in the scratch's halo, and there are no skips.
        c_above = _cost_at(cost_ptr, tile, b, above, cols, -1, lanes, mask, R, L, TL, HALO)
        c_above_next = _cost_at(
            cost_ptr, tile, b, above, cols + 1, -1, lanes, mask, R, L, TL, HALO, -1
        )
        up_below, _, _, _ = _shares(
            D_ptr, c_above, skip_rows_ptr, b, above, cols, R, L, diag_weight, SOFT, STEPS, GIVEN,
            STRIP, RB, sbase, TR, TL,
        )  # fmt: skip
        _, _, diag_below_next, skip_below_next = _shares(
            D_ptr, c_above_next, skip_rows_ptr, b, above, cols + 1, R, L, diag_weight, SOFT, STEPS,
            GIVEN, STRIP, RB, sbase, TR, TL,
        )  # fmt: skip
        skip_below2_next = tl.zeros_like(skip_below_next)
        if STEPS == _CTC:
            c_skip = _cost_at(
                cost_ptr, tile, b, above + 1, cols + 1, 0, lanes, mask, R, L, TL, HALO
            )
            _, _, _, skip_below2_next = _shares(
                D_ptr, c_skip, skip_rows_ptr, b, above + 1, cols + 1, R, L, diag_weight, SOFT, STEPS
            )
    for k in range(0, n_rows):
        p = first + k
        r = (Rp - 1 - p) if REVERSE else p
        rb = rb0 + (-k if REVERSE else k)
        row = batch + rb * L + cols
        rows_1 = p > 0
        # Recomputed per row: kept live, it costs registers that long rows lack.
        side = mask & (pos > 0)
        if SEEDED:
            left = tl.load(y_ptr + batch + (rb + shift) * L + edge, mask=left_ok, other=zero)
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
                up_ok = mask & rows_1
                if STRIP and REVERSE:
                    up_ok = mask & (rb + 1 < RB)
                w = tl.load(
                    _neighbour(up_ptr, row, L, 1 * REVERSE, 0 * REVERSE, REVERSE),
                    mask=up_ok,
                    other=0.0,
                )
                base += w * y_near
        if STEPS != _ORTHOGONAL:
            tl.debug_barrier()
            ok = side & rows_1
            diag_at = _neighbour(y_ptr, row, L, 1, 1, REVERSE)
            if STRIP:
                diag_at += lane_skew
                if REVERSE:
                    ok = mask & (cols + 1 < L) & (r + 1 < R)
                    ok &= rb + 1 + tl.where(lanes == 0, TR, 0) < RB
            if FORWARD:
                y = tl.load(diag_at, mask=ok, other=zero)
                term = y + diag_weight * c
                # The ctc steps have no up step: the diagonal starts the sum.
                base = term if STEPS == _CTC else _min(base, term, SOFT)
            else:
                if SHARES:
                    w = diag_below_next
                else:
                    w_at = _neighbour(diag_ptr, row, L, 1 * REVERSE, 1 * REVERSE, REVERSE)
                    if STRIP and REVERSE:
                        w_at += lane_skew
                    w = tl.load(w_at, mask=ok, other=0.0)
                base += w * tl.load(diag_at, mask=ok, other=0.0)
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
            # With features, column j + 1's cost is the previous lane's, the
            # first lane's the halo's.
            c = _cost_at(cost_ptr, tile, b, r, cols, k, lanes, mask, R, L, TL, HALO)
            c_next = _cost_at(cost_ptr, tile, b, r, cols + 1, k, lanes, mask, R, L, TL, HALO, -1)
            up_here, left_here, diag_here, _ = _shares(
                D_ptr, c, skip_rows_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS, GIVEN, STRIP,
                RB, sbase, TR, TL,
            )  # fmt: skip
            _, a, diag_next, skip_next = _shares(
                D_ptr, c_next, skip_rows_ptr, b, r, cols + 1, R, L, diag_weight, SOFT, STEPS, GIVEN,
                STRIP, RB, sbase, TR, TL,
            )  # fmt: skip
        else:
            a_at = _neighbour(left_ptr, row, L, 0 * REVERSE, 1 * REVERSE, REVERSE)
            a_ok = side
            if STRIP and REVERSE:
                a_at += lane_skew
                a_ok = mask & (cols + 1 < L) & (rb + tl.where(lanes == 0, TR, 0) < RB)
            a = tl.load(a_at, mask=a_ok, other=0.0)
        # Seeded with the column before the tile; unseeded, the scan's other
        # half is left to the compiler to drop.
        if not FORWARD:
            mult, y_near = tl.associative_scan((a, base), 0, _compose_affine)
            if SEEDED:
                y_near += mult * left
        else:
            y_near, scan_c = tl.associative_scan(
                (base, a), 0, _compose_softmin if SOFT else _compose_min
            )
            if SEEDED:
                y_near = _min(y_near, scan_c + left, SOFT)
        if FORWARD and GIVEN:
            y_near = tl.where(p == 0, tl.load(x_ptr + b * L + cols, mask=mask, other=zero), y_near)
        tl.store(y_ptr + row, y_near, mask=mask)
        if STRIP and REVERSE and STEPS != _ORTHOGONAL:
            # The strip's first diagonal's tiles' first cells, last in the
            # tile's order, give the corner cells their diagonal step.
            if SHARES:
                step = diag_here
            else:
                step = tl.load(diag_ptr + row, mask=mask, other=0.0)
            at = batch + cols - 1
            at_first = (lanes == TL - 1) & mask & (r > 0) & (cols > 0)
            at_first &= (k == TR - 1) & (tile_row == TR + 1)
            corner = y_near * step + tl.load(x_ptr + at, mask=at_first, other=0.0)
            tl.store(y_ptr + at, corner, mask=at_first)
            if SHARES and out_ptr is not None:
                tl.store(out_ptr + at, corner, mask=at_first)
        if SHARES:
            if out_ptr is not None:
                none = (up_here + left_here + diag_here) == 0  # cell (0, 0)
                factor = tl.where(none, 1.0, up_here + left_here + diag_weight * diag_here)
                tl.store(out_ptr + row, y_near * factor, mask=mask)
            up_below = up_here
            diag_below_next = diag_next
            skip_below2_next = skip_below_next
            skip_below_next = skip_next


@triton.jit(do_not_specialize=["d", "TR", "H", "slot"])
def _frontier_kernel(
    items_ptr, rows_ptr, cols_ptr, scr_ptr, hit_ptr, diag_weight, d, TR, W, H, slot,
    SOFT: tl.constexpr, SYMMETRIC: tl.constexpr, REVERSE: tl.constexpr, TL: tl.constexpr,
    SEEDED: tl.constexpr,
):  # fmt: skip
    """:func:`frontier`'s wavefront, as :func:`_dag_kernel`'s forward pass, with
    REVERSE mirrored, but over the items (S, 5) of the grid of the costs in the
    scratch ``scr_ptr`` (see :func:`_cost_kernel`), at ``slot``.

    The program (item, tile) of launch ``d`` is the item's tile ``I_lo + tile %
    n_tiles`` of tile row, in the order of processing (the rows from the item's
    last with REVERSE, the columns likewise), and ``d`` minus that of tile
    column, of the H rows by W columns' tiles of TR by TL; it does nothing if
    the item has no such tile. Only these are kept,
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
    n_cols = W // TL
    I_lo = tl.maximum(0, d - n_cols + 1)
    n_tiles = tl.minimum((H + TR - 1) // TR - 1, d) - I_lo + 1
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
        y_near, scan_c = tl.associative_scan(
            (base, a), 0, _compose_softmin if SOFT else _compose_min
        )
        if SEEDED:
            y_near = _min(y_near, scan_c + left, SOFT)
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
    R, L, diag_weight, STEPS: tl.constexpr, BLOCK: tl.constexpr, SHEAR: tl.constexpr,
    TRANSPOSE: tl.constexpr, BF: tl.constexpr,
):  # fmt: skip
    """Each pair's optimal DTW path, walked from its end cell back to (0, 0).

    Each cell's best step back is the first best of up, left and up-left,
    from D and the cost as :func:`_shares` chooses it, so the path is the hard
    gradient's. Within a row the path runs left until a cell steps up or
    diagonally: the walk takes BLOCK cells of the row at a time and finds
    that exit as the last such cell by a max over the lanes, so a run of
    left steps costs a step per BLOCK cells. The cells, the end first, go to
    path (B, R + L - 1, 2) as (row, column). One program per pair; each step
    waits on the last, but the walk stays in cache. (The ctc steps take one
    frame each, so their traceback is parallel instead: :func:`ctc_viterbi`
    and :func:`philtorch._trace_kernels.trace`.)
    """
    b = tl.program_id(0).to(tl.int64)
    out = path_ptr + b * (R + L - 1) * 2
    i = tl.load(ends_ptr + 2 * b)
    j = tl.load(ends_ptr + 2 * b + 1)
    written = i * 0
    lanes = tl.arange(0, BLOCK)
    while i >= 0:
        cols = j - (BLOCK - 1) + lanes
        valid = cols >= 0
        c = _cost(
            cost_ptr, fx_ptr, fy_ptr, win_ptr, b, i, cols, valid, R, L, N, M, F, big_bits,
            scale_bits, SHEAR, TRANSPOSE, BF,
        )  # fmt: skip
        w_up, w_left, w_diag, _ = _shares(
            D_ptr, c, None, b, i, cols, R, L, diag_weight, False, STEPS
        )
        # The row's exit: the last cell that steps up or diagonally, or (0, 0),
        # which has no step back.
        leaves = valid & (w_left == 0)
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
        exit_up = tl.max(tl.where(at_exit & (w_up != 0), 1, 0), axis=0) == 1
        origin = tl.max(tl.where(at_exit & (w_up + w_diag == 0), 1, 0), axis=0) == 1
        next_i = tl.where(found, tl.where(origin, -1, i - 1), i)
        j = tl.where(found, tl.where(exit_up, exit_col, exit_col - 1), j - BLOCK)
        i = next_i


def _features(feat):
    """feat = (x, y, window, big, scale, shear, transpose), or None for the cost
    tensor mode, as the kernels' arguments: (fx, fy, window, N, M, F, big_bits,
    scale_bits) and (SHEAR, TRANSPOSE). ``big`` and ``scale`` go as bits (see
    :func:`_unbits`)."""
    if feat is None:
        return (None, None, None, 0, 0, 0, 0, 0), (False, False)
    x, y, window, big, scale, shear, transpose = feat
    x, y = x.contiguous(), y.contiguous()
    window = None if window is None else window.contiguous()
    bits = (struct.unpack("<q", struct.pack("<d", v))[0] for v in (big, scale))
    N, F = x.shape[1:]
    return (x, y, window, N, y.shape[1], F, *bits), (shear, transpose)


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
    (fx, fy, window, *sizes), (shear, transpose) = _features(feat)
    # The features read at once per lane, within a register budget.
    bf = max(1, min(32, 8192 // block, triton.next_power_of_2(sizes[2])))
    _backtrack_kernel[(B,)](
        D, None if cost is None else cost.contiguous(), fx, fy, window, ends.contiguous(), walk,
        *sizes, R, L, diag_weight, _STEPS[steps], block, shear, transpose, bf, num_warps=1,
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


def _fake_path(D, *_, **__):
    B, R, L = D.shape
    return D.new_empty(B, R + L - 1, 2, dtype=torch.long)


backtrack.register_fake(_fake_path)


@torch.library.custom_op("philtorch::dtw_backtrack_features", mutates_args=())
def backtrack_features(
    D: Tensor, x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float,
    ends: Tensor, steps: str, diag_weight: float, shear: bool, transpose: bool,
) -> Tensor:  # fmt: skip
    """:func:`backtrack` with the costs of :func:`dtw_dp_features`."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _backtrack(D, None, feat, ends, steps, diag_weight)


backtrack_features.register_fake(_fake_path)


def _num_warps(TL: int, backward: bool = False) -> int:
    """Warps per tile by its columns, measured on an RTX 5060 Ti.

    The backward kernel keeps more of each row live.
    """
    if backward:
        return 4 if TL <= 128 else 8 if TL <= 256 else 16
    return 4 if TL <= 256 else 8 if TL <= 1024 else 16


def _tile(B: int, R: int, L: int, features: bool = False) -> tuple[int, int]:
    """The tile's rows and columns for B grids of R rows and L columns (at
    most, for :func:`frontier`'s items), measured on an RTX 5060 Ti: narrower
    tiles give a few pairs more programs per launch, wider ones fewer launches
    for many. With too few tiles side by side to pay for the launches, a tile
    is the whole grid, for rows of at most _MAX_ROW columns; with features,
    whose cells cost more, two side by side pay, and the tile's costs must fit
    the scratch, _SCRATCH_CELLS, which holds the costs of at least one
    anti-diagonal."""
    TL = min(256 if B < 8 else 1024, triton.next_power_of_2(L))
    n_cols = triton.cdiv(L, TL)
    side_by_side = B * min(triton.cdiv(R, _BR), n_cols)
    if n_cols == 1 or (side_by_side < (2 if features else _MIN_TILES) and L <= _MAX_ROW):
        TL = triton.next_power_of_2(L)
        if not features or B * (R + 1) * (TL + 1) <= _SCRATCH_CELLS:
            return R, TL
    return _BR, TL


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
# The fewest rows of y per chunk of _sheared_gradients, to keep its batch of
# matrix products small where the shear is narrow.
_MIN_ROWS = 64


def _halo_rows(TR: int) -> int:
    """A tile row of _BR with the halo row above it fills the _BR rows of
    _cost_kernel's blocks: _BR - 1 rows."""
    return _BR - 1 if TR == _BR else TR


def strip_tile(B: int, R: int, L: int) -> tuple[int, int, int] | None:
    """The tile rows and columns of the diagonal strips of a (B, R, L) grid
    from features: the grid's columns, and the backward's rows, so that the
    forward and the backward tiles are the same, and the number of tile
    anti-diagonals; None where the grid is one tile column, whose strips are
    blocks of rows."""
    # Concrete under torch.compile's dynamic shapes: the tile is chosen by them.
    TR, TL = _tile(int(B), int(R), int(L), True)
    TR, n_cols = _halo_rows(TR), triton.cdiv(int(L), TL)
    return None if n_cols == 1 else (TR, TL, triton.cdiv(int(R), TR) + n_cols - 1)


def _diagonals(
    S: int, R: int, L: int, TR: int, TL: int, reverse: bool, feat, items=None, row0: int = 0,
    given: bool = False, span: tuple[int, int] | None = None,
):  # fmt: skip
    """The tile anti-diagonals of S grids of R rows and L columns, in tiles of
    TR by TL, as (d, scratch, hit, slot, n_tiles) for the kernels' launches,
    n_tiles being the tiles of d in a grid: those from ``span[0]`` to
    ``span[1]`` - 1, if given, else all.

    In feature mode (``feat``, as _features takes it, with the ``items`` of
    _frame, if any), a chunk of anti-diagonals at a time, whose costs fit
    _SCRATCH_CELLS, first has its costs computed by _cost_kernel, in parallel
    over the blocks of all its tiles, into ``scratch``, and the tiles with a
    cell in the window flagged in ``hit`` (None without a window; with
    ``given``, the tiles of the seeded row 0 also): a tile's slice is at
    ``slot`` plus its program number. Without, the scratch is None.
    """
    if S == 0:
        return
    n_rows, n_cols = triton.cdiv(R, TR), triton.cdiv(L, TL)
    first, stop = (0, n_rows + n_cols - 1) if span is None else span

    def tiles(d):
        return min(n_rows - 1, d) - max(0, d - n_cols + 1) + 1

    if feat is None:
        yield from ((d, None, None, 0, tiles(d)) for d in range(first, stop))
        return
    (fx, fy, win, *sizes), (shear, transpose) = _features(feat)
    BF = min(16, triton.next_power_of_2(sizes[2]))
    halo = reverse
    P = S * min(n_rows, n_cols)  # the most tiles of an anti-diagonal
    tile = (TR + halo) * (TL + halo)
    K = max(1, min(stop - first, _CHUNK, _SCRATCH_CELLS // (P * tile)))
    scratch = fx.new_empty(K * P * tile)
    hit = None if win is None else torch.empty(K * P, dtype=torch.int8, device=fx.device)
    blocks = triton.cdiv(TR + halo, _BR) * triton.cdiv(TL + halo, _BC)
    for d0 in range(first, stop, K):
        k = min(K, stop - d0)
        if win is not None:
            _hit_kernel[(P, k)](
                hit, items, win, sizes[0], R, L, d0, TR, row0, shear, transpose, reverse, given,
                TL, num_warps=_num_warps(TL),
            )  # fmt: skip
        _cost_kernel[(P, blocks, k)](
            scratch, hit, items, fx, fy, win, *sizes, R, L, d0, TR, row0, shear, transpose,
            reverse, TL, BF, num_warps=4,
        )  # fmt: skip
        yield from ((d, scratch, hit, (d - d0) * P, tiles(d)) for d in range(d0, d0 + k))


def _launch(
    y: Tensor, steps: str, reverse: bool, *, cost=None, D=None, skip=None, weights=(None,) * 4,
    x=None, out=None, diag_weight: float = 1.0, soft: bool = False, feat=None,
    given: bool = False, row0: int = 0, strip: tuple[int, int] | None = None,
):  # fmt: skip
    """Run _dag_kernel over the grid of y (B, R, L), one launch per tile
    anti-diagonal, with the kernel's pointers by these names: ``weights`` are
    the up, left, diagonal and skip weights; ``feat``, as _features takes it,
    in place of ``cost``, whose rows start at row ``row0`` of the features'
    grid; ``given`` as the kernel's GIVEN. The backward is the launch with D.

    With ``strip`` = (R, d0), the tiles of the diagonals d0 on, as many as
    the buffer y (and D, x, out) (B, TR + 1 + (d1 - d0) TR, L) has rows for,
    of the grid's R rows, as the kernel's STRIP takes them (see
    :func:`strip_tile`; see the module docstring for the backward's)."""
    B, R, L = y.shape
    skip = None if skip is None else skip.contiguous()
    if strip is None:
        TR, TL = _tile(B, R, L, feat is not None)
        TR = _halo_rows(TR) if feat is not None and reverse else TR
    else:
        R, d0 = strip
        TR, TL, n_diag = strip_tile(B, R, L)
    n_rows, n_cols = triton.cdiv(R, TR), triton.cdiv(L, TL)
    frame, span, RB, sbase = (R, L), None, 0, 0
    if strip is not None:
        RB = y.size(1)
        d1, sbase = d0 + (RB - TR - 1) // TR, TR + 1 - d0 * TR
        span = (n_diag - d1, n_diag - d0 + (d0 > 0)) if reverse else (d0, d1)
        frame = (n_rows * TR, n_cols * TL) if reverse else (R, L)
    for d, scratch, hit, slot, n_tiles in _diagonals(
        B, *frame, TR, TL, reverse, feat, None, row0, given, span
    ):
        _dag_kernel[(n_tiles * B,)](
            cost, D, skip, *weights, x, y, out, scratch, hit, R, L, diag_weight, d, TR, slot, RB,
            sbase, soft, _STEPS[steps], reverse, TL, n_cols > 1, given, strip is not None,
            num_warps=_num_warps(TL, D is not None),
        )  # fmt: skip


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
    TR, TL = _tile(S, h_max, w_max, True)
    feat = (x, y, window, big, scale, shear, transpose)
    TR = _halo_rows(TR) if reverse else TR
    n_cols = triton.cdiv(w_max, TL)
    W = n_cols * TL
    rows = x.new_empty(S, 3, W)
    cols = x.new_empty(S, h_max)
    items = items.contiguous()
    for d, scratch, hit, slot, n_tiles in _diagonals(S, h_max, w_max, TR, TL, reverse, feat, items):
        _frontier_kernel[(S * n_tiles,)](
            items, rows, cols, scratch, hit, diag_weight, d, TR, W, h_max, slot, soft,
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


def _empty(t: Tensor, *_, **__) -> Tensor:
    return torch.empty_like(t, memory_format=torch.contiguous_format)


def _dtw_backward(
    D: Tensor, cost: Tensor | None, skip: Tensor | None, grad: Tensor, soft: bool, steps: str,
    diag_weight: float, feat=None, seeded: bool = False, row0: int = 0,
    strip: tuple[int, int] | None = None,
) -> Tensor:  # fmt: skip
    weighted = diag_weight != 1.0
    # A strip's buffer has cells that no tile writes: zeros.
    new = _empty if strip is None else lambda t: _empty(t).zero_()
    e = new(D)
    out = new(D) if weighted else e
    _launch(
        e, steps, True, cost=cost, D=D, skip=skip, x=grad.contiguous(),
        out=out if weighted else None, diag_weight=diag_weight, soft=soft, feat=feat,
        given=seeded, row0=row0, strip=strip,
    )  # fmt: skip
    return out


def _shift(
    t: Tensor, rows: int, cols: int, fill: float = 0.0, skew: tuple[int, int] | None = None
) -> Tensor:
    """t[..., i - rows, j - cols], filled where that is outside the grid.

    With ``skew`` = (TR, TL), t is a strip's buffer (see the module
    docstring): a step of one column, ``cols`` = 1, across a tile's left edge
    is TR rows more."""
    R, L = t.shape[-2:]
    out = F.pad(t, (cols, 0, rows, 0), value=fill)[..., :R, :L]
    if skew is None or not cols:
        return out
    edge = torch.arange(skew[1], L, skew[1], device=t.device)
    across = F.pad(t, (0, 0, rows + skew[0], 0), value=fill)[..., :R, edge - 1]
    return out.index_copy(-1, edge, across)


@torch.library.custom_op("philtorch::dtw_accumulate", mutates_args=())
def accumulate(
    w_up: Tensor | None, w_left: Tensor, w_diag: Tensor | None, w_skip: Tensor | None,
    x: Tensor, reverse: bool, strip: list[int] | None = None,
) -> Tensor:  # fmt: skip
    """y(i, j) = x(i, j) + W_up y(i - 1, j) + W_left y(i, j - 1)
    + W_diag y(i - 1, j - 1) + W_skip y(i - 2, j - 1), or with ``reverse`` its
    transpose.

    Each edge's weight is stored at its successor; the weights of absent
    steps are None.

    With ``strip`` = [R, d0], the steps up, left and up-left over the buffer
    x (B, TR + 1 + (d1 - d0) TR, L) of the strip d0 to d1 - 1 of a grid of R
    rows (see the module docstring), whose weights, in the seed's rows, are
    zero. Forward, the seed's rows of x are given: the cells of the strip
    start from them. With ``reverse``, the strip's cells pass their sums on to
    the seed's rows.
    """
    steps = "ctc" if w_up is None else "orthogonal" if w_diag is None else "symmetric"
    weights = tuple(None if w is None else w.contiguous() for w in (w_up, w_left, w_diag, w_skip))
    y = _empty(x)
    if strip is not None:
        y.zero_()  # cells that no tile writes
        if not reverse:
            TR = strip_tile(x.size(0), strip[0], x.size(2))[0]
            y[:, : TR + 1] = x[:, : TR + 1]  # the seed's rows are given
    _launch(y, steps, reverse, weights=weights, x=x.contiguous(), strip=strip)
    return y


@accumulate.register_fake
def _(w_up, w_left, w_diag, w_skip, x, reverse, strip=None):
    return _empty(x)


def _accumulate_setup(ctx, inputs, output):
    *weights, _, ctx.reverse, ctx.strip = inputs
    ctx.save_for_backward(*weights, output)


def _accumulate_backward(ctx, grad_y):
    *weights, y = ctx.saved_tensors
    # y = (I - W)^-1 x, so x's gradient is the accumulation the other way, and
    # an edge's is that gradient at its successor times y at its predecessor,
    # or, the other way, at its predecessor times y at its successor.
    grad_x = accumulate(*weights, grad_y, not ctx.reverse, ctx.strip)
    skew = None if ctx.strip is None else strip_tile(y.size(0), ctx.strip[0], y.size(2))[:2]
    # The (row, column) offsets of the up, left, up-left and skip steps.
    steps = ((1, 0), (0, 1), (1, 1), (2, 1))
    grads = [
        None if w is None
        else _shift(grad_x, *step, skew=skew) * y if ctx.reverse
        else grad_x * _shift(y, *step, skew=skew)
        for w, step in zip(weights, steps)
    ]  # fmt: skip
    return (*grads, grad_x, None, None)


accumulate.register_autograd(_accumulate_backward, setup_context=_accumulate_setup)


def shares(
    D: Tensor, cost: Tensor, skip: Tensor | None, soft: bool, steps: str, diag_weight: float,
    seeded: bool = False, strip: tuple[int, int] | None = None,
) -> list[Tensor | None]:  # fmt: skip
    """The shares of each cell from its up, left, up-left and skip predecessors.

    As the kernels compute them, with PyTorch ops that autograd differentiates:
    a softmax of the negated arguments, in base 2, for soft-DTW, a one-hot of
    the first best for DTW; zero for predecessors outside the grid, and all
    zero for cell (0, 0), and with ``seeded`` for row 0. None for the steps not
    in the set.

    With ``strip`` = (R, d0), D (B, RB, L) is a strip's buffer (see the module
    docstring) and cost (B, S, L) that of its rows, in the same skew; the
    shares of the seed's rows, and of cells off the grid, are zero.
    """
    present = (steps != "ctc", True, steps != "orthogonal", steps == "ctc")
    skew = None
    if strip is not None:
        R, d0 = strip
        TR, TL, _ = strip_tile(D.size(0), R, D.size(2))
        skew = (TR, TL)
        cost = F.pad(cost, (0, 0, TR + 1, 0))
    args = []
    if present[0]:
        args.append(_shift(D, 1, 0, _INF) + cost)
    args.append(_shift(D, 0, 1, _INF, skew) + cost)
    if present[2]:
        args.append(_shift(D, 1, 1, _INF, skew) + diag_weight * cost)
    if present[3]:
        args.append(torch.where(skip.bool()[..., None], _shift(D, 2, 1, _INF) + cost, _INF))
    args = torch.stack(args)
    if strip is not None:
        s = torch.arange(D.size(1), device=D.device)[:, None]
        r = s - (TR + 1) + d0 * TR - torch.arange(D.size(2), device=D.device) // TL * TR
        cell = (s > TR) & (r >= 0) & (r < R)
        ok = [m for m, p in zip((cell & (r > 0), cell, cell & (r > 0)), present) if p]
        args = torch.where(torch.stack(ok)[:, None], args, _INF)
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


dtw_dp.register_fake(_empty)


def _check_cells(B: int, R: int, L: int) -> None:
    """Check that B pairs' buffers of R by L cells have int32 offsets."""
    if R * L > _MAX_CELLS:
        raise ValueError(f"the DTW kernels take at most 2^31 - 1 cells per pair, got {(B, R, L)}")


def _forward(cost, skip, soft, steps, diag_weight, feat=None, grid=None, seed=None, row0=0):
    B, R, L = cost.shape if grid is None else grid
    _check_cells(B, R, L)
    D = (cost if feat is None else feat[0]).new_empty(B, R, L)
    _launch(
        D, steps, False, cost=None if cost is None else cost.contiguous(), skip=skip,
        x=None if seed is None else seed.contiguous(), diag_weight=diag_weight, soft=soft,
        feat=feat, given=seed is not None, row0=row0,
    )  # fmt: skip
    return D


@torch.library.custom_op("philtorch::ctc_viterbi", mutates_args=())
def ctc_viterbi(cost: Tensor, skip: Tensor) -> tuple[Tensor, Tensor]:
    """The hard ctc forward pass over cost (B, R, L): D, and each cell's best
    step back, int8, 0 to stay in its row, 1 to advance from the row above, 2 to
    skip from two rows above, the first best under ties as the hard shares
    choose."""
    cost, skip = cost.contiguous(), skip.contiguous()
    D = _forward(cost, skip, False, "ctc", 1.0)
    B, R, L = D.shape
    # From D afterwards, over all cells at once: recorded in the forward pass,
    # they would add a barrier and a reload to each row's sequential step,
    # nearly doubling it on long rows.
    steps_back = torch.empty_like(D, dtype=torch.int8)
    _shares_kernel[(B * R, triton.cdiv(L, _SHARES_BLOCK))](
        D, cost, skip, steps_back, R, L, BLOCK=_SHARES_BLOCK, num_warps=4,
    )  # fmt: skip
    return D, steps_back


@ctc_viterbi.register_fake
def _(cost, skip):
    return _empty(cost), torch.empty_like(
        cost, dtype=torch.int8, memory_format=torch.contiguous_format
    )


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
    return _dtw_backward(D.contiguous(), cost.contiguous(), skip, grad, soft, steps, diag_weight)


dtw_dp_backward.register_fake(_empty)


def _decomposed_backward(D, cost, skip, grad, soft, steps, diag_weight, seeded=False, strip=None):
    """:func:`dtw_dp_backward` from ops that autograd differentiates."""
    w_up, w_left, w_diag, w_skip = shares(D, cost, skip, soft, steps, diag_weight, seeded, strip)
    e = accumulate(w_up, w_left, w_diag, w_skip, grad, True, strip)
    if diag_weight == 1.0:
        return e
    # With a diagonal weighted, the cost's gradient is e times its weight.
    none = (w_up + w_left + w_diag) == 0  # cell (0, 0)
    return e * torch.where(none, 1.0, w_up + w_left + diag_weight * w_diag)


def _dtw_dp_backward_setup(ctx, inputs, output):
    D, cost, skip, grad, *ctx.options = inputs
    ctx.save_for_backward(D, cost, skip, grad)


def _vjp(decomposed, tensors, grad_e):
    """The vector-Jacobian product of ``decomposed(*tensors)``, a gradient
    from differentiable ops, with ``grad_e``, for the backward of an op that
    computes the same gradient by a kernel.

    It is taken with respect to views of the tensors, so that each gradient
    counts only the paths through its own input, not one tensor's through
    another, while staying connected to the inputs: the ops are
    differentiable, so gradients exist to any order. (torch.func.vjp would
    take them as independent too, but custom ops such as accumulate don't
    support it.)"""
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        inputs = [t.view_as(t) if t.requires_grad else t.detach().requires_grad_() for t in tensors]
        return torch.autograd.grad(
            decomposed(*inputs), inputs, grad_e, create_graph=create_graph, allow_unused=True
        )


def _dtw_dp_backward_backward(ctx, grad_e):
    D, cost, skip, grad = ctx.saved_tensors
    grad_D, grad_cost, grad_grad = _vjp(
        lambda D, cost, grad: _decomposed_backward(D, cost, skip, grad, *ctx.options),
        (D, cost, grad), grad_e,
    )  # fmt: skip
    return grad_D, grad_cost, None, grad_grad, None, None, None


dtw_dp_backward.register_autograd(_dtw_dp_backward_backward, setup_context=_dtw_dp_backward_setup)


def _cell_costs(x, y, window, big, scale, shear, transpose, p, r, col, valid=None):
    """The costs of the grid's cells (r, col) of pairs p, as the kernels compute
    them, from PyTorch ops that autograd differentiates.

    p, r and col broadcast together; a cell is mapped as :func:`_cost` maps it.
    Where ``valid`` is False, the frames' difference is zeroed before it is
    squared, so that infinite frames there give no NaN gradient.
    """
    a, s = (col, r) if transpose else (r, col)
    n = (a + s if shear else a).clamp(max=x.size(1) - 1)
    m = (a if shear else s).clamp(max=y.size(1) - 1)
    d = x[p, n] - y[p, m]
    if valid is not None:
        d = torch.where(valid[..., None], d, 0.0)
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
    rows [lo, hi] of y per row of x, ``big``. A cell (r, l) is mapped as
    :func:`_cost` maps (r + row0, l): the grid's rows are rows ``row0`` on of a
    larger one. ``seed`` (B, L), if given, is D's row 0, as in the module
    docstring's seeded grids. Only the steps "orthogonal" and "symmetric"."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _forward(None, None, soft, steps, diag_weight, feat, (x.size(0), R, L), seed, row0)


@dtw_dp_features.register_fake
def _(
    x, y, window, big, scale, soft, steps, diag_weight, shear, transpose, R, L, seed=None, row0=0
):
    return x.new_empty(x.size(0), R, L)


def _dtw_dp_features_setup(ctx, inputs, output):
    # Also the strip's: x through R and L, the seed, and row0 or d0, its last.
    x, y, window, *ctx.options, seed = inputs[:13]
    ctx.seeded, ctx.start = seed is not None, inputs[-1]
    ctx.save_for_backward(x, y, window, output)


def _products(x, y, E, scale):
    """The gradients of x (..., A, F) and y (..., S, F), the rows E (..., A, S)
    of the costs ``scale`` times the squared distances of the rows of x and y,
    have: sums of E's rows and columns times the frames, less matrix products,
    each row of x with y's of E's row."""
    grad_x = 2 * scale * (x * E.sum(-1, keepdim=True) - E @ y)
    grad_y = 2 * scale * (y * E.sum(-2).unsqueeze(-1) - E.mT @ x)
    return grad_x, grad_y


def _sheared_gradients(x, y, window, scale, E, n0, m0):
    """:func:`_cost_gradients` for E (B, A, S), the gradient of the costs of
    the cells (m, s), x's row n0 + m + s, clamped to N - 1, and y's row
    m0 + m, in memory linear in E: not the dense (A + S - 1, A) block of x's
    rows by y's rows, of which only S diagonals are cells.

    The rows m of y go in chunks of C, each a dense block of the K = C + S - 1
    rows of x its cells cover by its C rows of y, with the products of all the
    chunks batched. A row of x is in two blocks at most, and they add up.
    """
    B, A, S = E.shape
    N, M = x.size(1), y.size(1)
    C = min(A, max(S, _MIN_ROWS))
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
    x_rows, y_cols = blocks(x).mT, y.reshape(B, chunks, C, y.size(2))
    grad_x = 2 * scale * (x_rows * E.sum(3, keepdim=True) - E @ y_cols)
    grad_y = 2 * scale * (y_cols * E.sum(2).unsqueeze(-1) - E.mT @ x_rows)
    grad_x = _add_blocks(grad_x, (n0, C, K), N)
    return grad_x, F.pad(grad_y.flatten(1, 2)[:, :A], (0, 0, m0, M - m0 - A))


def _cost_gradients(x, y, window, scale, shear, transpose, row0, E):
    """The gradients of x and y from E, the gradient of the grid's costs of
    :func:`dtw_dp_features` (B, R, L). E is masked in place, but for the
    shear's."""
    # E, the cost's gradient on the grid, back to the (n, m) cells it covers:
    # rows n0 to n1 of x, rows m0 on of y.
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
        ctx.seeded, ctx.start,
    )  # fmt: skip
    # With a seed, G's row 0 is the seed's gradient; the cost's row 0 has none.
    E, grad_seed = (F.pad(G[:, 1:], (0, 0, 1, 0)), G[:, 0]) if ctx.seeded else (G, None)
    grad_x, grad_y = _cost_gradients(x, y, window, scale, shear, transpose, ctx.start, E)
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


dtw_dp_backward_features.register_fake(_empty)


def _dtw_dp_backward_features_setup(ctx, inputs, output):
    D, x, y, window, *ctx.options = inputs
    ctx.save_for_backward(D, x, y, window, inputs[6])


def _dtw_dp_backward_features_backward(ctx, grad_e):
    # As _dtw_dp_backward_backward, with the cost rebuilt from x and y; also
    # the strip's, whose options end in (R, L, d0), not (seeded, row0).
    D, x, y, window, grad = ctx.saved_tensors
    big, scale, _, soft, steps, diag_weight, shear, transpose, *grid = ctx.options

    def decomposed(D, x, y, grad):
        if len(grid) == 2:
            seeded, row0 = grid
            cost = grid_cost(x, y, window, big, scale, shear, transpose, *D.shape[1:], row0)
            return _decomposed_backward(D, cost, None, grad, soft, steps, diag_weight, seeded)
        R, L, d0 = grid
        TR, TL, _ = strip_tile(D.size(0), R, L)
        cost = _strip_cost(
            x, y, window, big, scale, shear, transpose, D.size(1) - TR - 1, TR, TL, d0, L
        )
        return _decomposed_backward(D, cost, None, grad, soft, steps, diag_weight, strip=(R, d0))

    grad_D, grad_x, grad_y, grad_grad = _vjp(decomposed, (D, x, y, grad), grad_e)
    return grad_D, grad_x, grad_y, None, None, None, grad_grad, *(None,) * (len(ctx.options) - 3)


dtw_dp_backward_features.register_autograd(
    _dtw_dp_backward_features_backward, setup_context=_dtw_dp_backward_features_setup
)


# Diagonal strips: see the module docstring.


def _seed_cells(L: int, TR: int, TL: int, device: torch.device) -> Tensor:
    """The flat indices of a strip's seed (see the module docstring) in the
    first TR + 1 rows of its buffer, the last rows of the buffer's, RB - TR - 1
    rows further, being the next strip's: row TR whole, then the tiles' last
    columns but the last tile's in rows 1 to TR - 1 and in row 0."""
    right = torch.arange(TL - 1, L - 1, TL, device=device)
    last_row = torch.arange(L, device=device) + TR * L
    rows = torch.arange(1, TR, device=device)[:, None] * L + right
    return torch.cat([last_row, rows.flatten(), right])


@torch.library.custom_op("philtorch::dtw_dp_strip", mutates_args=())
def dtw_dp_strip(
    x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float, soft: bool, steps: str,
    diag_weight: float, shear: bool, transpose: bool, R: int, L: int, seed: Tensor | None,
    step: int, d0: int,
) -> Tensor:  # fmt: skip
    """The strip of ``step`` tile anti-diagonals from d0 of
    :func:`dtw_dp_features`'s grid, whose arguments x through ``transpose``, R
    and L are these, as the buffer D (B, TR + 1 + step TR, L) of the strips'
    skew (see the module docstring); the last strip may be shorter.

    ``seed`` (B, n), if given, is the previous strip's, taken from its buffer
    at :func:`_seed_cells`, shifted by the buffer's rows past its first TR + 1.
    """
    B = x.size(0)
    TR, TL, n_diag = strip_tile(B, R, L)
    RB = TR + 1 + min(step, n_diag - d0) * TR
    _check_cells(B, RB, L)
    # The next seed gathers cells that no tile writes: zeros keep the outputs
    # deterministic, for opcheck.
    D = x.new_zeros(B, RB, L)
    if seed is not None:
        D.view(B, RB * L).scatter_(1, _seed_cells(L, TR, TL, x.device).expand(B, -1), seed)
    feat = (x, y, window, big, scale, shear, transpose)
    _launch(
        D, steps, False, diag_weight=diag_weight, soft=soft, feat=feat, strip=(R, d0)
    )  # fmt: skip
    return D


@dtw_dp_strip.register_fake
def _(x, y, window, big, scale, soft, steps, diag_weight, shear, transpose, R, L, seed, step, d0):
    TR, _, n_diag = strip_tile(x.size(0), R, L)
    return x.new_empty(x.size(0), TR + 1 + min(step, n_diag - d0) * TR, L)


def _dtw_dp_strip_backward(ctx, grad_D):
    x, y, window, D = ctx.saved_tensors
    big, scale, soft, steps, diag_weight, shear, transpose, R, L = ctx.options
    B = D.size(0)
    TR, TL, _ = strip_tile(B, R, L)
    G = dtw_dp_backward_strip(
        D, x, y, window, big, scale, grad_D, soft, steps, diag_weight, shear, transpose, R, L,
        ctx.start,
    )  # fmt: skip
    grad_x, grad_y = _strip_gradients(
        x, y, window, scale, shear, transpose, G[:, TR + 1 :], TR, TL, ctx.start
    )  # fmt: skip
    # The seed's gradient is that of its cells.
    grad_seed = None
    if ctx.seeded:
        grad_seed = (
            G[:, : TR + 1].flatten(1).gather(1, _seed_cells(L, TR, TL, x.device).expand(B, -1))
        )
    return grad_x, grad_y, *(None,) * 10, grad_seed, None, None


dtw_dp_strip.register_autograd(_dtw_dp_strip_backward, setup_context=_dtw_dp_features_setup)


@torch.library.custom_op("philtorch::dtw_dp_backward_strip", mutates_args=())
def dtw_dp_backward_strip(
    D: Tensor, x: Tensor, y: Tensor, window: Tensor | None, big: float, scale: float,
    grad: Tensor, soft: bool, steps: str, diag_weight: float, shear: bool, transpose: bool,
    R: int, L: int, d0: int,
) -> Tensor:  # fmt: skip
    """The gradient of a strip's costs from ``grad``, its buffer's, through
    :func:`dtw_dp_strip`, as :func:`dtw_dp_backward_features` gives it; the
    seed's rows hold the gradient of the seed."""
    feat = (x, y, window, big, scale, shear, transpose)
    return _dtw_backward(
        D.contiguous(), None, None, grad, soft, steps, diag_weight, feat, strip=(R, d0)
    )


dtw_dp_backward_strip.register_fake(_empty)


dtw_dp_backward_strip.register_autograd(
    _dtw_dp_backward_features_backward, setup_context=_dtw_dp_backward_features_setup
)


def _strip_frames(x, y, window, shear, transpose, S, TR, TL, d0, L):
    """The rows of x and y, and the window's columns, shifted to the block's,
    that the blocks (S, TL) of a strip's rows, one per tile column j, cover:
    the frames (B n_cols, rows, F) and the window (B n_cols, rows, 2), and the
    windows (first row of block 0, step between blocks, rows) of x's rows and
    y's.

    The block has the grid's rows (d0 - j) TR on and its columns j TL on, and a
    cell (r, c) is mapped as :func:`_cost` maps it."""
    rows, cols = (d0 * TR, -TR, S), (0, TL, TL)
    if shear:
        # x's row n = r + c, y's row m = r, or, transposed, m = c.
        windows = (d0 * TR, TL - TR, S + TL - 1), cols if transpose else rows
    else:
        windows = (cols, rows) if transpose else (rows, cols)
    j = torch.arange(triton.cdiv(L, TL), device=x.device)[:, None]
    (x_0, x_step, x_n), (y_0, y_step, y_n) = windows
    x_idx = x_0 + j * x_step + torch.arange(x_n, device=x.device)
    y_idx = y_0 + j * y_step + torch.arange(y_n, device=x.device)
    m_0 = y_0 + j * y_step  # y's row of the block's first
    x_idx, y_idx = x_idx.clamp(0, x.size(1) - 1), y_idx.clamp(0, y.size(1) - 1)
    win = None if window is None else (window[:, x_idx] - m_0[None, :, :, None]).flatten(0, 1)
    return x[:, x_idx].flatten(0, 1), y[:, y_idx].flatten(0, 1), win, windows


def _add_blocks(g, window, N):
    """The blocks' gradients g (B, n_cols, K, F) of the windows (first, step, K)
    of N rows, added up, the rows below 0 dropped and those past N - 1 added to
    the last, as their indices clamp: a fold, so the sums are in a fixed order.
    """
    first, step, K = window
    n_cols = g.size(1)
    if step < 0:  # in order of the rows
        g, first, step = g.flip(1), first + (n_cols - 1) * step, -step
    n = (n_cols - 1) * step + K
    g = F.fold(g.permute(0, 3, 2, 1).flatten(1, 2), (n, 1), (K, 1), stride=(step, 1))
    g = g.squeeze(3).mT[:, max(0, -first) :]
    # Rows from max(first, 0), then zeros past N - 1, whatever the window's end.
    g = F.pad(g, (0, 0, max(first, 0), N))
    return torch.cat([g[:, : N - 1], g[:, N - 1 :].sum(1, keepdim=True)], 1)


def _strip_gradients(x, y, window, scale, shear, transpose, E, TR, TL, d0):
    """:func:`_cost_gradients` for the E (B, S, L) of a strip's rows of the
    skewed buffer: each block of a tile column, a grid of its own, from its
    rows of x and y (see :func:`_strip_frames`), batched; their gradients add
    up."""
    B, S, L = E.shape
    n_cols = triton.cdiv(L, TL)
    frames_x, frames_y, win, windows = _strip_frames(
        x, y, window, shear, transpose, S, TR, TL, d0, L
    )
    E = F.pad(E, (0, n_cols * TL - L)).unflatten(2, (n_cols, TL)).transpose(1, 2).flatten(0, 1)
    grad_x, grad_y = _cost_gradients(
        frames_x, frames_y, win, scale, shear, False, 0, E.mT if transpose else E
    )
    return tuple(
        _add_blocks(g.unflatten(0, (B, n_cols)), w, t.size(1))
        for t, w, g in zip((x, y), windows, (grad_x, grad_y))
    )


def _strip_cost(x, y, window, big, scale, shear, transpose, S, TR, TL, d0, L):
    """The costs (B, S, L) of a strip's rows, as :func:`grid_cost` gives those
    of each block of a tile column, on the frames of :func:`_strip_frames`; the
    cells off the grid have the costs of some rows of x and y."""
    frames_x, frames_y, win, _ = _strip_frames(x, y, window, shear, transpose, S, TR, TL, d0, L)
    cost = grid_cost(
        frames_x, frames_y, win, big, scale, shear, False, *((TL, S) if transpose else (S, TL)), 0
    )
    cost = cost.mT if transpose else cost
    return cost.unflatten(0, (x.size(0), -1)).transpose(1, 2).flatten(2)[..., :L]
