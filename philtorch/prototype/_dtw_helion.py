"""One-kernel row-wise DTW and soft-DTW in Helion, differentiable to any order.

The DTW grid is a DAG: cell (i, j) has predecessors (i - 1, j), (i, j - 1) and
(i - 1, j - 1). Two custom ops, each a Helion kernel that holds a whole row
of one batch item in registers and loops over the rows:

* ``dtw_dp(cost, soft)`` -> D: the DTW recursion D = cost + min (or, for
  soft-DTW, softmin) over a cell's predecessors.
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
"""

import os

import helion
import helion.language as hl
import torch
import torch.nn.functional as F
from torch import Tensor

_SETTINGS = {} if os.environ.get("HELION_AUTOTUNE_EFFORT") else {"autotune_effort": "quick"}
_INF = float("inf")


def _compose_min_plus(a_l, c_l, ga_l, gc_l, a_r, c_r, ga_r, gc_r):
    """Compose x -> min(A, C + x) maps, the left (earlier) one first.

    (A, C) is the whole segment's map, (GA, GC) the segment's without its
    last map; a single map's G is the identity, A = inf and C = 0.
    """
    return (
        torch.minimum(a_r, c_r + a_l),
        c_r + c_l,
        torch.minimum(ga_r, gc_r + a_l),
        gc_r + c_l,
    )


def _compose_softmin_plus(a_l, c_l, ga_l, gc_l, a_r, c_r, ga_r, gc_r):
    """As :func:`_compose_min_plus` with softmin(x, y) = -log(e^-x + e^-y).

    Written as min(x, y) - log(1 + e^-|x - y|): inside these scans at most one
    of x and y is inf (only G starts at the identity, and A is finite).
    """
    x, y = a_r, c_r + a_l
    a = torch.minimum(x, y) - torch.log(1 + torch.exp(-torch.abs(x - y)))
    gx, gy = ga_r, gc_r + a_l
    ga = torch.minimum(gx, gy) - torch.log(1 + torch.exp(-torch.abs(gx - gy)))
    return a, c_r + c_l, ga, gc_r + c_l


def _compose_affine(a_l, b_l, ga_l, gb_l, a_r, b_r, ga_r, gb_r):
    """Compose x -> a x + b maps, the left (earlier) one first, with G as above."""
    return a_r * a_l, a_r * b_l + b_r, ga_r * a_l, ga_r * b_l + gb_r


# static_shapes: the best configuration depends strongly on the row length.
@helion.kernel(**_SETTINGS, static_shapes=True)
def _dtw_dp_kernel(cost: Tensor, soft: hl.constexpr) -> tuple[Tensor, Tensor]:
    """D of cost (B, R, L), and the min-terms of rows 1, ...; row 0's are unset."""
    B, R, L = cost.shape
    D = torch.empty_like(cost)
    m_all = torch.empty_like(cost)
    for tile_b in hl.tile(B, block_size=1):
        prev = hl.cumsum(cost[tile_b, 0, :], dim=1)
        D[tile_b, 0, :] = prev
        for n in hl.grid(1, R):
            c = cost[tile_b, n, :]
            identity_a = torch.full_like(c, _INF)
            identity_c = torch.zeros_like(c)
            if soft:
                # c + prev and prev are finite, so the plain form is safe.
                alpha = torch.minimum(prev, c + prev) - torch.log(1 + torch.exp(-torch.abs(c)))
                s_left = hl.associative_scan(
                    _compose_softmin_plus, (alpha, c, identity_a, identity_c), dim=1
                )[2]
                # s_left is inf only in column 0, where prev is finite.
                m = torch.minimum(prev, s_left) - torch.log(
                    1 + torch.exp(-torch.abs(prev - s_left))
                )
            else:
                alpha = torch.minimum(prev, c + prev)
                s_left = hl.associative_scan(
                    _compose_min_plus, (alpha, c, identity_a, identity_c), dim=1
                )[2]
                m = torch.minimum(prev, s_left)
            prev = c + m
            D[tile_b, n, :] = prev
            m_all[tile_b, n, :] = m
    return D, m_all


@helion.kernel(**_SETTINGS, static_shapes=True)
def _dag_forward_kernel(
    w_down: Tensor, w_right_next: Tensor, w_diag_next: Tensor, x: Tensor
) -> Tensor:
    """y of the forward accumulation; the *_next weights are taken at column j + 1."""
    B, R, L = x.shape
    y = torch.empty_like(x)
    for tile_b in hl.tile(B, block_size=1):
        y_prev = torch.zeros_like(x[tile_b, 0, :])
        for i in hl.grid(R):
            base = x[tile_b, i, :] + w_down[tile_b, i, :] * y_prev
            a = w_right_next[tile_b, i, :]
            b = a * base + w_diag_next[tile_b, i, :] * y_prev
            identity_a = torch.ones_like(b)
            identity_b = torch.zeros_like(b)
            z_left = hl.associative_scan(_compose_affine, (a, b, identity_a, identity_b), dim=1)[3]
            y_prev = base + z_left
            y[tile_b, i, :] = y_prev
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
    return _dag_forward_kernel(
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
def dtw_dp(cost: Tensor, soft: bool) -> Tensor:
    """The accumulated costs D of the DTW recursion over cost (B, R, L)."""
    return _dtw_dp_kernel(cost.contiguous(), soft)[0]


@dtw_dp.register_fake
def _(cost, soft):
    return torch.empty_like(cost)


def edge_weights(D: Tensor, soft: bool) -> tuple[Tensor, Tensor, Tensor]:
    """How much each cell's D took from each predecessor, stored at the cell.

    From the stored D alone, so the weights are exact for DTW, a one-hot of
    the best predecessor, and sum to 1 for soft-DTW, a softmax over the
    negated predecessors; predecessors outside the grid get 0.
    Differentiable in D for soft-DTW; constant for DTW.
    """
    preds = torch.stack([_up(D, _INF), _left(D, _INF), _up_left(D, _INF)])
    outside = torch.isinf(preds)
    if soft:
        # Cell (0, 0) has no predecessor: give its softmax finite inputs.
        none = outside.all(0, keepdim=True)
        weights = torch.softmax(torch.where(none, 0.0, -preds), dim=0)
    else:
        weights = torch.nn.functional.one_hot(preds.argmin(0), 3).movedim(-1, 0).to(D.dtype)
    return tuple(torch.where(outside, 0.0, weights))


def _dtw_dp_setup(ctx, inputs, output):
    ctx.soft = inputs[1]
    ctx.save_for_backward(output)


def _dtw_dp_backward(ctx, grad_D):
    (D,) = ctx.saved_tensors
    # D(succ) = cost(succ) + softmin over predecessors, and the weights are
    # dD(succ) / dD(pred), so the cost's gradient is the reverse accumulation.
    return dag_reverse(*edge_weights(D, ctx.soft), grad_D), None


dtw_dp.register_autograd(_dtw_dp_backward, setup_context=_dtw_dp_setup)
