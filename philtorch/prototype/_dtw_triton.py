"""One-kernel row-wise DTW and soft-DTW in Triton, differentiable to any order.

The DTW grid is a DAG: cell (i, j) has predecessors (i - 1, j), (i, j - 1) and
(i - 1, j - 1). Two custom ops, each a Triton kernel that holds a whole row
of one batch item in registers and loops over the rows:

* ``dtw_dp(cost, soft, diag)`` -> D: the DTW recursion D = cost + min (or,
  for soft-DTW, softmin) over a cell's predecessors, without the diagonal
  step unless ``diag``.
* ``dag_forward(W_down, W_right, W_diag, x)`` -> y: the linear forward
  accumulation y(i, j) = x(i, j) + W_down(i, j) y(i - 1, j)
  + W_right(i, j) y(i, j - 1) + W_diag(i, j) y(i - 1, j - 1), with each edge's
  weight stored at its successor.

and ``dag_reverse(W_down, W_right, W_diag, g)`` -> e, its transpose,
e(i, j) = g(i, j) + W_down(i + 1, j) e(i + 1, j) + W_right(i, j + 1) e(i, j + 1)
+ W_diag(i + 1, j + 1) e(i + 1, j + 1), which is dag_forward on the grid flipped
in both directions: there each cell's successors are its predecessors, so
each edge's weight just moves to its predecessor before the flip.

Each row is one scan. The cross-row diagonal term would need a row shifted by
one column; instead each scan element also carries G, the composition of its
segment without the last map, which gives the scan's value at the previous
column at every position. Within a row:

* dtw_dp: s(j) = min(D_prev(j), D(j)) follows s(j) = min(alpha(j), c(j) + s(j - 1)),
  a scan of maps x -> min(A, C + x), and m(j) = min(D_prev(j), s(j - 1)).
  Without the diagonal step, D itself follows such maps, with no G.
* dag_forward: z(j) = W_right(j + 1) y(j) + W_diag(j + 1) y_prev(j) follows a
  scan of affine maps, and y(j) = base(j) + z(j - 1).

Derivatives. dag_forward is linear: its derivative with respect to its input
is its transpose, dag_reverse, which is dag_forward again, and with respect to
an edge weight, the product of the two ends' values, which needs only shifts.
So dag_forward's backward calls dag_forward. dtw_dp's derivative is
dag_reverse with the edge weights dD(succ) / dD(pred): a softmax over a cell's
negated predecessors for soft-DTW, and a one-hot of the best one for DTW,
computed from the stored D with
differentiable PyTorch ops. Recomputing them from D, not from the scan's own
minima, keeps them exact: the scan sums costs in a different order, so its
minima can differ from the stored D in the last bit. So every backward is
made of these ops and PyTorch ops, and differentiates again.

Each program handles one batch item, so only the warps per program are
tuned: from tuning the Helion versions of these kernels
(scripts/helion/dtw.py) on an RTX 5060 Ti, as chosen by the row length in
scripts/helion/_helion_aot_dtw_cuda_sm120.py.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

_INF = float("inf")
# Triton kernels can only read module constants made with tl.constexpr.
_KERNEL_INF = tl.constexpr(_INF)


@triton.jit
def _compose_min_plus(a_l, c_l, ga_l, gc_l, a_r, c_r, ga_r, gc_r):
    """Compose x -> min(A, C + x) maps, the left (earlier) one first.

    (A, C) is the whole segment's map, (GA, GC) the segment's without its
    last map; a single map's G is the identity, A = inf and C = 0.
    """
    return (
        tl.minimum(a_r, c_r + a_l),
        c_r + c_l,
        tl.minimum(ga_r, gc_r + a_l),
        gc_r + c_l,
    )


@triton.jit
def _softmin(x, y):
    """-log(e^-x + e^-y) as min(x, y) - log(1 + e^-|x - y|); wrong if both are inf."""
    return tl.minimum(x, y) - tl.log(1 + tl.exp(-tl.abs(x - y)))


@triton.jit
def _compose_softmin_plus(a_l, c_l, ga_l, gc_l, a_r, c_r, ga_r, gc_r):
    """As :func:`_compose_min_plus` with min replaced by softmin.

    Unguarded: inside these scans at most one of the two terms is inf (only G
    starts at the identity, and A is finite).
    """
    return _softmin(a_r, c_r + a_l), c_r + c_l, _softmin(ga_r, gc_r + a_l), gc_r + c_l


@triton.jit
def _compose_min_plus_whole(a_l, c_l, a_r, c_r):
    """:func:`_compose_min_plus` without G."""
    return tl.minimum(a_r, c_r + a_l), c_r + c_l


@triton.jit
def _compose_softmin_plus_whole(a_l, c_l, a_r, c_r):
    """:func:`_compose_softmin_plus` without G."""
    return _softmin(a_r, c_r + a_l), c_r + c_l


@triton.jit
def _compose_affine(a_l, b_l, ga_l, gb_l, a_r, b_r, ga_r, gb_r):
    """Compose x -> a x + b maps, the left (earlier) one first, with G as above."""
    return a_r * a_l, a_r * b_l + b_r, ga_r * a_l, ga_r * b_l + gb_r


# Both kernels take contiguous (B, R, L) tensors, one program per batch item
# with a whole row in BLOCK >= L lanes. Lanes past L hold padding; the scans
# run left to right, so it never reaches the real columns.


@triton.jit
def _dtw_dp_kernel(
    cost_ptr, D_ptr, R, L, SOFT: tl.constexpr, DIAG: tl.constexpr, BLOCK: tl.constexpr
):
    """D of cost, with steps up, left and, if DIAG, up-left."""
    batch = tl.program_id(0).to(tl.int64) * R * L
    cols = tl.arange(0, BLOCK)
    mask = cols < L
    c = tl.load(cost_ptr + batch + cols, mask=mask, other=0.0)
    prev = tl.cumsum(c, axis=0)
    tl.store(D_ptr + batch + cols, prev, mask=mask)
    identity_a = tl.full([BLOCK], _KERNEL_INF, c.dtype)
    identity_c = tl.zeros([BLOCK], c.dtype)
    for n in range(1, R):
        row = batch + n * L + cols
        c = tl.load(cost_ptr + row, mask=mask, other=0.0)
        if not DIAG:
            # D(j) = min(c(j) + prev(j), c(j) + D(j - 1)): a scan of the
            # maps x -> min(A, C + x), whose whole value is D itself.
            if SOFT:
                prev = tl.associative_scan((c + prev, c), 0, _compose_softmin_plus_whole)[0]
            else:
                prev = tl.associative_scan((c + prev, c), 0, _compose_min_plus_whole)[0]
        elif SOFT:
            # c + prev and prev are finite, so the plain form is safe.
            alpha = _softmin(prev, c + prev)
            s_left = tl.associative_scan(
                (alpha, c, identity_a, identity_c), 0, _compose_softmin_plus
            )[2]
            # s_left is inf only in column 0, where prev is finite.
            prev = c + _softmin(prev, s_left)
        else:
            alpha = tl.minimum(prev, c + prev)
            s_left = tl.associative_scan((alpha, c, identity_a, identity_c), 0, _compose_min_plus)[
                2
            ]
            prev = c + tl.minimum(prev, s_left)
        tl.store(D_ptr + row, prev, mask=mask)


@triton.jit
def _dag_forward_kernel(
    w_down_ptr, w_right_next_ptr, w_diag_next_ptr, x_ptr, y_ptr, R, L, BLOCK: tl.constexpr
):
    """y of the forward accumulation; the *_next weights are taken at column j + 1."""
    batch = tl.program_id(0).to(tl.int64) * R * L
    cols = tl.arange(0, BLOCK)
    mask = cols < L
    y_prev = tl.zeros([BLOCK], x_ptr.dtype.element_ty)
    identity_a = tl.full([BLOCK], 1.0, y_prev.dtype)
    identity_b = tl.zeros([BLOCK], y_prev.dtype)
    for i in range(0, R):
        row = batch + i * L + cols
        base = tl.load(x_ptr + row, mask=mask, other=0.0) + (
            tl.load(w_down_ptr + row, mask=mask, other=0.0) * y_prev
        )
        a = tl.load(w_right_next_ptr + row, mask=mask, other=0.0)
        b = a * base + tl.load(w_diag_next_ptr + row, mask=mask, other=0.0) * y_prev
        z_left = tl.associative_scan((a, b, identity_a, identity_b), 0, _compose_affine)[3]
        y_prev = base + z_left
        tl.store(y_ptr + row, y_prev, mask=mask)


def _dtw_dp_warps(L: int, soft: bool, diag: bool) -> int:
    """Tuned warps per program; the tuning's tree, without its near-ties."""
    if soft:
        return 8 if L <= 1024 else 16
    if L <= 128:
        return 16
    if L <= 1024:
        return 16 if (not diag and L > 256) else 8
    return 16 if L <= 4096 else 8


def _dag_forward_warps(L: int) -> int:
    return 1 if L <= 512 else 16 if L <= 2048 else 32


def _dtw_dp(cost: Tensor, soft: bool, diag: bool) -> Tensor:
    B, R, L = cost.shape
    D = torch.empty_like(cost)
    _dtw_dp_kernel[(B,)](
        cost, D, R, L, soft, diag, triton.next_power_of_2(L), num_warps=_dtw_dp_warps(L, soft, diag)
    )
    return D


def _dag_forward(w_down: Tensor, w_right_next: Tensor, w_diag_next: Tensor, x: Tensor) -> Tensor:
    B, R, L = x.shape
    y = torch.empty_like(x)
    _dag_forward_kernel[(B,)](
        w_down, w_right_next, w_diag_next, x, y, R, L, triton.next_power_of_2(L),
        num_warps=_dag_forward_warps(L),
    )  # fmt: skip
    return y


def _shift(t: Tensor, rows: int, cols: int, fill: float = 0.0) -> Tensor:
    """t[..., i - rows, j - cols], filled where that is outside the grid."""
    R, L = t.shape[-2:]
    return F.pad(t, (cols, 0, rows, 0), value=fill)[..., :R, :L]


def _up(t, fill=0.0):
    return _shift(t, 1, 0, fill)


def _left(t, fill=0.0):
    return _shift(t, 0, 1, fill)


def _up_left(t, fill=0.0):
    return _shift(t, 1, 1, fill)


@torch.library.custom_op("philtorch_prototype::dag_forward", mutates_args=())
def dag_forward(w_down: Tensor, w_right: Tensor, w_diag: Tensor, x: Tensor) -> Tensor:
    """The forward accumulation over the DTW grid; see the module docstring."""
    if x.numel() == 0:
        return torch.zeros_like(x)
    # Weights at column j + 1, zero past the last column.
    w_right_next = F.pad(w_right[..., 1:], (0, 1))
    w_diag_next = F.pad(w_diag[..., 1:], (0, 1))
    return _dag_forward(
        w_down.contiguous(), w_right_next.contiguous(), w_diag_next.contiguous(), x.contiguous()
    )


@dag_forward.register_fake
def _(w_down, w_right, w_diag, x):
    return torch.empty_like(x)


def _flip(t: Tensor) -> Tensor:
    return t.flip(-2, -1)


def dag_reverse(w_down: Tensor, w_right: Tensor, w_diag: Tensor, g: Tensor) -> Tensor:
    """The reverse accumulation over the DTW grid, dag_forward's transpose.

    It is dag_forward on the grid flipped in both directions, where each
    cell's successors become its predecessors. Each edge's weight moves from
    its successor to its predecessor before the flip, a shift by one cell.
    Built from dag_forward and PyTorch ops, so autograd differentiates it.
    """
    R, L = g.shape[-2:]

    def to_predecessor(w, rows, cols):
        return _flip(F.pad(w, (0, cols, 0, rows))[..., rows : rows + R, cols : cols + L])

    e = dag_forward(
        to_predecessor(w_down, 1, 0),
        to_predecessor(w_right, 0, 1),
        to_predecessor(w_diag, 1, 1),
        _flip(g),
    )
    return _flip(e)


def _weight_grads(downstream: Tensor, upstream: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """d<., result> / d W[succ <- pred] = downstream(succ) * upstream(pred)."""
    return downstream * _up(upstream), downstream * _left(upstream), downstream * _up_left(upstream)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3], output)


def _dag_forward_backward(ctx, grad_y):
    w_down, w_right, w_diag, y = ctx.saved_tensors
    # y = (I - W)^-1 x, so x's gradient is the transposed accumulation, and an
    # edge's is that gradient at its successor times y at its predecessor.
    grad_x = dag_reverse(w_down, w_right, w_diag, grad_y)
    return (*_weight_grads(grad_x, y), grad_x)


dag_forward.register_autograd(_dag_forward_backward, setup_context=_setup)


@torch.library.custom_op("philtorch_prototype::dtw_dp", mutates_args=())
def dtw_dp(cost: Tensor, soft: bool, diag: bool) -> Tensor:
    """The accumulated costs D over cost (B, R, L), with steps up, left and,
    if ``diag``, up-left."""
    if cost.numel() == 0:
        return torch.empty_like(cost)
    return _dtw_dp(cost.contiguous(), soft, diag)


@dtw_dp.register_fake
def _(cost, soft, diag):
    return torch.empty_like(cost)


def edge_weights(D: Tensor, soft: bool, diag: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """How much each cell's D took from each predecessor, stored at the cell.

    From the stored D alone, so the weights are exact for DTW, a one-hot of
    the best predecessor, and sum to 1 for soft-DTW, a softmax over the
    negated predecessors; predecessors outside the grid or along a step not
    taken get 0. Differentiable in D for soft-DTW; constant for DTW.
    """
    up_left = _up_left(D, _INF) if diag else torch.full_like(D, _INF)
    preds = torch.stack([_up(D, _INF), _left(D, _INF), up_left])
    outside = torch.isinf(preds)
    if soft:
        # Cell (0, 0) has no predecessor: give its softmax finite inputs.
        none = outside.all(0, keepdim=True)
        weights = torch.softmax(torch.where(none, 0.0, -preds), dim=0)
    else:
        weights = torch.nn.functional.one_hot(preds.argmin(0), 3).movedim(-1, 0).to(D.dtype)
    return tuple(torch.where(outside, 0.0, weights))


def _dtw_dp_setup(ctx, inputs, output):
    ctx.steps = inputs[1:]
    ctx.save_for_backward(output)


def _dtw_dp_backward(ctx, grad_D):
    (D,) = ctx.saved_tensors
    # D(succ) = cost(succ) + softmin over predecessors, and the weights are
    # dD(succ) / dD(pred), so the cost's gradient is the reverse accumulation.
    return dag_reverse(*edge_weights(D, *ctx.steps), grad_D), None, None


dtw_dp.register_autograd(_dtw_dp_backward, setup_context=_dtw_dp_setup)
