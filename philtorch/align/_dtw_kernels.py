"""Triton kernels for DTW, soft-DTW and CTC, differentiable to any order.

The accumulated costs D of a cost matrix (B, R, L) follow, with soft-min for
soft-DTW,

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

* for first derivatives, :func:`_dtw_backward`, one kernel that computes the
  shares on the fly and scans each row from the right, or for a few long
  rows :func:`_dtw_backward_parallel`, which computes them over all cells
  first;
* :func:`dag_forward` (the linear accumulation over the grid, a kernel like
  the forward one) and PyTorch ops, which autograd differentiates again, for
  higher derivatives. dag_forward's own backward is dag_forward on the
  flipped grid.

Every kernel takes a ``STEPS`` constant, and with the ctc steps a pointer to
the (B, R) skip mask.

dtw_dp's backward takes the second when gradients of the gradient are
requested (``create_graph=True``) and the first otherwise.
"""

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
    """-log(e^-x + e^-y) as min(x, y) - log(1 + e^-|x - y|), and inf if both are."""
    smaller = tl.minimum(x, y)
    return tl.where(smaller == _KERNEL_INF, smaller, smaller - tl.log(1 + tl.exp(-tl.abs(x - y))))


@triton.jit
def _compose_min_plus(a_l, c_l, a_r, c_r):
    """Compose x -> min(A, C + x) maps, the left (earlier) one first."""
    return tl.minimum(a_r, c_r + a_l), c_r + c_l


@triton.jit
def _compose_softmin_plus(a_l, c_l, a_r, c_r):
    """As :func:`_compose_min_plus` with soft-min."""
    return _softmin(a_r, c_r + a_l), c_r + c_l


@triton.jit
def _compose_affine(a_l, b_l, a_r, b_r):
    """Compose x -> a x + b maps, the left (earlier) one first."""
    return a_r * a_l, a_r * b_l + b_r


@triton.jit
def _compose_affine_g(a_l, b_l, ga_l, gb_l, a_r, b_r, ga_r, gb_r):
    """As :func:`_compose_affine`, also carrying G, the segment without its last map."""
    return a_r * a_l, a_r * b_l + b_r, ga_r * a_l, ga_r * b_l + gb_r


# The kernels take contiguous (B, R, L) tensors, one program per batch item
# with a whole row in BLOCK >= L lanes. In the forward kernels, lanes past L
# hold padding; the scans run left to right, so it never reaches the real
# columns.


@triton.jit
def _min(x, y, SOFT: tl.constexpr):
    if SOFT:
        return _softmin(x, y)
    return tl.minimum(x, y)


@triton.jit
def _dtw_dp_kernel(
    cost_ptr, skip_ptr, D_ptr, R, L, diag_weight,
    SOFT: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0).to(tl.int64)
    batch = b * R * L
    cols = tl.arange(0, BLOCK)
    mask = cols < L
    c = tl.load(cost_ptr + batch + cols, mask=mask, other=0.0)
    prev = tl.cumsum(c, axis=0)
    tl.store(D_ptr + batch + cols, prev, mask=mask)
    for n in range(1, R):
        row = batch + n * L + cols
        c = tl.load(cost_ptr + row, mask=mask, other=0.0)
        if STEPS != _CTC:
            a = c + prev
        else:
            a = tl.full([BLOCK], _KERNEL_INF, c.dtype)
        if STEPS != _ORTHOGONAL:
            # Earlier rows at j - 1, written by every thread before the barrier.
            tl.debug_barrier()
            up_left = tl.load(D_ptr + row - L - 1, mask=mask & (cols > 0), other=_KERNEL_INF)
            a = _min(a, up_left + diag_weight * c, SOFT)
        if STEPS == _CTC:
            allowed = mask & (cols > 0) & (n > 1) & (tl.load(skip_ptr + b * R + n) != 0)
            skip = tl.load(D_ptr + row - 2 * L - 1, mask=allowed, other=_KERNEL_INF)
            a = _min(a, skip + c, SOFT)
        if SOFT:
            prev = tl.associative_scan((a, c), 0, _compose_softmin_plus)[0]
        else:
            prev = tl.associative_scan((a, c), 0, _compose_min_plus)[0]
        tl.store(D_ptr + row, prev, mask=mask)


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
        e_left = tl.exp(safe - left)
        zero = tl.zeros(c.shape, c.dtype)
        e_up = zero
        e_diag = zero
        e_skip = zero
        if STEPS != _CTC:
            e_up = tl.exp(safe - up)
        if STEPS != _ORTHOGONAL:
            e_diag = tl.exp(safe - diag)
        if STEPS == _CTC:
            e_skip = tl.exp(safe - skip)
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
def _dtw_backward_kernel(
    D_ptr, cost_ptr, skip_ptr, g_ptr, e_ptr, out_ptr, R, L, diag_weight,
    SOFT: tl.constexpr, STEPS: tl.constexpr, WEIGHTED: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """The cost's gradient, rows from the last, each row's lanes from its last column.

    Lane t holds column j = L - 1 - t, so e(i, j) = base(j) + a(j) e(i, j + 1)
    is a scan from lane 0 up. e is written to e_ptr for the next row's
    diagonal term, and the gradient, e times the cost's weight, to out_ptr,
    which may be e_ptr when no step is weighted.
    """
    b = tl.program_id(0).to(tl.int64)
    batch = b * R * L
    lanes = tl.arange(0, BLOCK)
    cols = L - 1 - lanes
    mask = lanes < L
    dtype = g_ptr.dtype.element_ty
    # The shares of the rows below, kept from earlier iterations: row r + 1
    # straight down at j and diagonally at j + 1, row r + 2 by a skip at
    # j + 1. None below the last row.
    e_below = tl.zeros([BLOCK], dtype)
    up_below = tl.zeros([BLOCK], dtype)
    diag_below_next = tl.zeros([BLOCK], dtype)
    skip_below_next = tl.zeros([BLOCK], dtype)
    skip_below2_next = tl.zeros([BLOCK], dtype)
    for k in range(0, R):
        r = R - 1 - k
        row = batch + r * L + cols
        base = tl.load(g_ptr + row, mask=mask, other=0.0)
        if STEPS != _CTC:
            base += up_below * e_below
        if STEPS != _ORTHOGONAL:
            # e at j + 1 of rows r + 1 and r + 2, written by every thread
            # before the barrier.
            tl.debug_barrier()
            next_col = mask & (cols + 1 < L)
            base += diag_below_next * tl.load(
                e_ptr + row + L + 1, mask=next_col & (r + 1 < R), other=0.0
            )
            if STEPS == _CTC:
                base += skip_below2_next * tl.load(
                    e_ptr + row + 2 * L + 1, mask=next_col & (r + 2 < R), other=0.0
                )
        # Row r's shares at j and at j + 1; from the right, e(r, j) gets
        # a(j) e(r, j + 1), with a(j) the share cell (r, j + 1) takes from its left.
        up_here, left_here, diag_here, skip_here = _shares(
            D_ptr, cost_ptr, skip_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS
        )
        _, a, diag_next, skip_next = _shares(
            D_ptr, cost_ptr, skip_ptr, b, r, cols + 1, R, L, diag_weight, SOFT, STEPS
        )
        e = tl.associative_scan((a, base), 0, _compose_affine)[1]
        tl.store(e_ptr + row, e, mask=mask)
        if WEIGHTED:
            none = (up_here + left_here + diag_here) == 0  # cell (0, 0)
            factor = tl.where(none, 1.0, up_here + left_here + diag_weight * diag_here)
            tl.store(out_ptr + row, e * factor, mask=mask)
        e_below = e
        up_below = up_here
        diag_below_next = diag_next
        skip_below2_next = skip_below_next
        skip_below_next = skip_next


@triton.jit
def _shares_kernel(
    D_ptr, cost_ptr, skip_ptr, up_ptr, left_ptr, diag_ptr, skip_share_ptr, R, L, diag_weight,
    SOFT: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """Every cell's shares, one program per BLOCK cells of a row."""
    batch_row = tl.program_id(0)
    b = (batch_row // R).to(tl.int64)
    r = batch_row % R
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    up, left, diag, skip = _shares(
        D_ptr, cost_ptr, skip_ptr, b, r, cols, R, L, diag_weight, SOFT, STEPS
    )
    offset = b * R * L + r * L + cols
    mask = cols < L
    tl.store(up_ptr + offset, up, mask=mask)
    tl.store(left_ptr + offset, left, mask=mask)
    tl.store(diag_ptr + offset, diag, mask=mask)
    if STEPS == _CTC:
        tl.store(skip_share_ptr + offset, skip, mask=mask)


@triton.jit
def _reverse_kernel(
    up_ptr, left_ptr, diag_ptr, skip_ptr, g_ptr, e_ptr, R, L,
    STEPS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """The reverse accumulation from stored shares, laid out as in _dtw_backward_kernel."""
    batch = tl.program_id(0).to(tl.int64) * R * L
    lanes = tl.arange(0, BLOCK)
    cols = L - 1 - lanes
    mask = lanes < L
    e_below = tl.zeros([BLOCK], g_ptr.dtype.element_ty)
    for k in range(0, R):
        r = R - 1 - k
        row = batch + r * L + cols
        below = mask & (r + 1 < R)
        base = tl.load(g_ptr + row, mask=mask, other=0.0)
        if STEPS != _CTC:
            base += e_below * tl.load(up_ptr + row + L, mask=below, other=0.0)
        if STEPS != _ORTHOGONAL:
            tl.debug_barrier()
            diagonal = below & (cols + 1 < L)
            base += tl.load(diag_ptr + row + L + 1, mask=diagonal, other=0.0) * tl.load(
                e_ptr + row + L + 1, mask=diagonal, other=0.0
            )
            if STEPS == _CTC:
                skip = diagonal & (r + 2 < R)
                base += tl.load(skip_ptr + row + 2 * L + 1, mask=skip, other=0.0) * tl.load(
                    e_ptr + row + 2 * L + 1, mask=skip, other=0.0
                )
        a = tl.load(left_ptr + row + 1, mask=mask & (cols + 1 < L), other=0.0)
        e = tl.associative_scan((a, base), 0, _compose_affine)[1]
        tl.store(e_ptr + row, e, mask=mask)
        e_below = e


@triton.jit
def _dag_forward_kernel(
    w_down_ptr, w_right_next_ptr, w_diag_next_ptr, w_skip_next_ptr, x_ptr, y_ptr, R, L,
    BLOCK: tl.constexpr,
):  # fmt: skip
    """y of the forward accumulation; the *_next weights are taken at column j + 1.

    Row by row, z(j) = W_right(j + 1) y(j) + W_diag(j + 1) y_prev(j) +
    W_skip(j + 1) y_prev2(j) is a scan of affine maps, and y(j) = base(j) +
    z(j - 1): G, each segment's composition without its last map, gives
    z(j - 1) at lane j. ``w_skip_next_ptr`` may be None.
    """
    batch = tl.program_id(0).to(tl.int64) * R * L
    cols = tl.arange(0, BLOCK)
    mask = cols < L
    y_prev = tl.zeros([BLOCK], x_ptr.dtype.element_ty)
    y_prev2 = tl.zeros([BLOCK], x_ptr.dtype.element_ty)
    identity_a = tl.full([BLOCK], 1.0, y_prev.dtype)
    identity_b = tl.zeros([BLOCK], y_prev.dtype)
    for i in range(0, R):
        row = batch + i * L + cols
        base = tl.load(x_ptr + row, mask=mask, other=0.0) + (
            tl.load(w_down_ptr + row, mask=mask, other=0.0) * y_prev
        )
        a = tl.load(w_right_next_ptr + row, mask=mask, other=0.0)
        b = a * base + tl.load(w_diag_next_ptr + row, mask=mask, other=0.0) * y_prev
        if w_skip_next_ptr is not None:
            b += tl.load(w_skip_next_ptr + row, mask=mask, other=0.0) * y_prev2
        z_left = tl.associative_scan((a, b, identity_a, identity_b), 0, _compose_affine_g)[3]
        y_prev2 = y_prev
        y_prev = base + z_left
        tl.store(y_ptr + row, y_prev, mask=mask)


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
        _dtw_backward_kernel, B, L,
        D, cost, skip, grad.contiguous(), e, out, R, L, diag_weight, soft, _STEPS[steps],
        weighted, num_warps=_num_warps(L, backward=True, diag=steps != "orthogonal"),
    )  # fmt: skip
    return out


def _dtw_backward_parallel(
    D: Tensor, cost: Tensor, skip: Tensor | None, grad: Tensor, soft: bool, steps: str,
    diag_weight: float,
) -> Tensor:  # fmt: skip
    """As :func:`_dtw_backward`, with the shares computed by a kernel over all cells.

    For few long rows: there the fused kernel computes every share in one
    program per batch item, leaving most of the GPU idle.
    """
    B, R, L = D.shape
    up, left, diag_share = _empty(D), _empty(D), _empty(D)
    skip_share = _empty(D) if steps == "ctc" else None
    block = 1024
    _shares_kernel[(B * R, triton.cdiv(L, block))](
        D, cost, skip, up, left, diag_share, skip_share, R, L, diag_weight, soft, _STEPS[steps],
        BLOCK=block, num_warps=4,
    )  # fmt: skip
    e = _empty(D)
    _launch(
        _reverse_kernel, B, L, up, left, diag_share, skip_share, grad.contiguous(), e, R, L,
        _STEPS[steps], num_warps=_num_warps(L),
    )  # fmt: skip
    if diag_weight != 1.0:
        none = (up + left + diag_share) == 0
        e = e * torch.where(none, 1.0, up + left + diag_weight * diag_share)
    return e


def _shift(t: Tensor, rows: int, cols: int, fill: float = 0.0) -> Tensor:
    """t[..., i - rows, j - cols], filled where that is outside the grid."""
    R, L = t.shape[-2:]
    return F.pad(t, (cols, 0, rows, 0), value=fill)[..., :R, :L]


def _flip(t: Tensor) -> Tensor:
    return t.flip(-2, -1)


@torch.library.custom_op("philtorch::dtw_dag_forward", mutates_args=())
def dag_forward(
    w_down: Tensor, w_right: Tensor, w_diag: Tensor, w_skip: Tensor | None, x: Tensor
) -> Tensor:
    """y(i, j) = x(i, j) + W_down y(i - 1, j) + W_right y(i, j - 1) + W_diag y(i - 1, j - 1)
    + W_skip y(i - 2, j - 1).

    Each edge's weight is stored at its successor (i, j); ``w_skip`` may be None.
    """
    B, R, L = x.shape

    def next_column(w):
        """Weights at column j + 1, zero past the last column."""
        return F.pad(w[..., 1:], (0, 1)).contiguous()

    y = _empty(x)
    _launch(
        _dag_forward_kernel, B, L,
        w_down.contiguous(), next_column(w_right), next_column(w_diag),
        None if w_skip is None else next_column(w_skip), x.contiguous(), y, R, L,
    )  # fmt: skip
    return y


@dag_forward.register_fake
def _(w_down, w_right, w_diag, w_skip, x):
    return _empty(x)


def dag_reverse(
    w_down: Tensor, w_right: Tensor, w_diag: Tensor, w_skip: Tensor | None, g: Tensor
) -> Tensor:
    """The reverse accumulation, dag_forward's transpose.

    It is dag_forward on the grid flipped in both directions, where each
    cell's successors become its predecessors. Each edge's weight moves from
    its successor to its predecessor before the flip, a shift by the step.
    """
    R, L = g.shape[-2:]

    def to_predecessor(w, rows, cols):
        return _flip(F.pad(w, (0, cols, 0, rows))[..., rows : rows + R, cols : cols + L])

    e = dag_forward(
        to_predecessor(w_down, 1, 0),
        to_predecessor(w_right, 0, 1),
        to_predecessor(w_diag, 1, 1),
        None if w_skip is None else to_predecessor(w_skip, 2, 1),
        _flip(g),
    )
    return _flip(e)


def _dag_forward_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4], output)


def _dag_forward_backward(ctx, grad_y):
    w_down, w_right, w_diag, w_skip, y = ctx.saved_tensors
    # y = (I - W)^-1 x, so x's gradient is the transposed accumulation, and an
    # edge's is that gradient at its successor times y at its predecessor.
    grad_x = dag_reverse(w_down, w_right, w_diag, w_skip, grad_y)
    return (
        grad_x * _shift(y, 1, 0),
        grad_x * _shift(y, 0, 1),
        grad_x * _shift(y, 1, 1),
        None if w_skip is None else grad_x * _shift(y, 2, 1),
        grad_x,
    )


dag_forward.register_autograd(_dag_forward_backward, setup_context=_dag_forward_setup)


def shares(
    D: Tensor, cost: Tensor, skip: Tensor | None, soft: bool, steps: str, diag_weight: float
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """The shares of each cell from its up, left, up-left and skip predecessors.

    As the kernels compute them, with PyTorch ops that autograd
    differentiates: a softmax of the negated arguments for soft-DTW, a
    one-hot of the first best for DTW; zero for predecessors outside the
    grid or not among the steps, and all zero for cell (0, 0). The skip
    shares are None without the ctc steps.
    """
    absent = torch.full_like(D, _INF)
    args = [
        absent if steps == "ctc" else _shift(D, 1, 0, _INF) + cost,
        _shift(D, 0, 1, _INF) + cost,
        absent if steps == "orthogonal" else _shift(D, 1, 1, _INF) + diag_weight * cost,
    ]
    if steps == "ctc":
        args.append(torch.where(skip.bool()[..., None], _shift(D, 2, 1, _INF) + cost, _INF))
    args = torch.stack(args)
    outside = torch.isinf(args)
    if soft:
        none = outside.all(0, keepdim=True)
        weights = torch.softmax(torch.where(none, 0.0, -args), dim=0)
    else:
        weights = F.one_hot(args.argmin(0), len(args)).movedim(-1, 0).to(D.dtype)
    weights = torch.where(outside, 0.0, weights)
    return weights[0], weights[1], weights[2], weights[3] if steps == "ctc" else None


@torch.library.custom_op("philtorch::dtw_dp", mutates_args=())
def dtw_dp(cost: Tensor, skip: Tensor | None, soft: bool, steps: str, diag_weight: float) -> Tensor:
    """The accumulated costs D of cost (B, R, L) over the steps ``steps``, the
    diagonal weighted by ``diag_weight``; ``skip`` is the (B, R) mask of the
    rows the ctc steps may skip into, and None for the others. See the module
    docstring."""
    B, R, L = cost.shape
    if R * L > _MAX_CELLS:
        raise ValueError(f"dtw handles at most 2^31 - 1 cells per pair, got {tuple(cost.shape)}")
    D = _empty(cost)
    _launch(
        _dtw_dp_kernel, B, L,
        cost.contiguous(), skip, D, R, L, diag_weight, soft, _STEPS[steps],
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
    e = dag_reverse(w_up, w_left, w_diag, w_skip, grad_D)
    if diag_weight != 1.0:
        none = (w_up + w_left + w_diag) == 0
        e = e * torch.where(none, 1.0, w_up + w_left + diag_weight * w_diag)
    return e, None, None, None, None


dtw_dp.register_autograd(_dtw_dp_backward, setup_context=_dtw_dp_setup)
